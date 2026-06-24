#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the latest 0603 experiment result.

Default behavior:
  1. find the most recently modified .npz result under common result roots;
  2. save an overview figure and summary CSV into <result_dir>/analysis;
  3. show the figure when a display is available, unless --no-show is used.
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-0603")

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOTS = [
    Path("/home/done/USTC/LJJ/franka_ros2_ws/results"),
    BASE_DIR / "results",
]


def setup_font():
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if os.path.exists(font_path):
        try:
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans"]
        except Exception:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def flatten_npz(path):
    raw = np.load(path, allow_pickle=True)
    data = {}
    for key in raw.files:
        value = raw[key]
        if value.dtype.kind in ("U", "S", "O"):
            data[key] = np.asarray(value).flatten()
        else:
            data[key] = np.asarray(value, dtype=float).flatten()
    return data


def fit_length(value, n, fill=0.0):
    arr = np.asarray(value).flatten()
    if len(arr) == n:
        return arr
    if len(arr) == 0:
        return np.full(n, fill, dtype=float)
    if len(arr) == 1:
        return np.full(n, arr[0], dtype=float)
    out = np.full(n, fill, dtype=float)
    m = min(n, len(arr))
    out[:m] = arr[:m]
    if m < n:
        out[m:] = arr[m - 1]
    return out


def numeric(data, key, n=None, fill=0.0):
    if key not in data:
        if n is None:
            return np.array([], dtype=float)
        return np.full(n, fill, dtype=float)
    arr = np.asarray(data[key], dtype=float).flatten()
    if n is not None:
        arr = fit_length(arr, n, fill=fill)
    return arr


def add_derived_fields(data):
    if "t" not in data:
        raise KeyError("结果文件缺少 t 字段")
    t = numeric(data, "t")
    n = len(t)
    if n == 0:
        raise ValueError("结果文件为空")
    data["t"] = t

    for key in ("pos_x", "pos_y", "pos_z", "F_measured", "F_desired", "alpha"):
        if key in data:
            data[key] = numeric(data, key, n)

    if "F_err" not in data and "F_measured" in data and "F_desired" in data:
        data["F_err"] = data["F_measured"] - data["F_desired"]
    else:
        data["F_err"] = numeric(data, "F_err", n)

    for axis in ("x", "y", "z"):
        pos_key = f"pos_{axis}"
        des_key = f"pos_des_{axis}"
        err_key = f"pos_err_{axis}"
        if des_key in data:
            data[des_key] = numeric(data, des_key, n)
        elif err_key in data and pos_key in data:
            data[des_key] = numeric(data, pos_key, n) - numeric(data, err_key, n)
        elif pos_key in data:
            data[des_key] = numeric(data, pos_key, n)

        if err_key in data:
            data[err_key] = numeric(data, err_key, n)
        elif pos_key in data and des_key in data:
            data[err_key] = numeric(data, pos_key, n) - numeric(data, des_key, n)
        else:
            data[err_key] = np.zeros(n)

    if "pos_err_norm" not in data:
        data["pos_err_norm"] = np.sqrt(
            data["pos_err_x"] ** 2 + data["pos_err_y"] ** 2 + data["pos_err_z"] ** 2
        )
    else:
        data["pos_err_norm"] = numeric(data, "pos_err_norm", n)

    data["error_rcm"] = np.abs(numeric(data, "error_rcm", n))
    data["K_hat"] = numeric(data, "K_hat", n, fill=np.nan)
    data["K_total"] = numeric(data, "K_total", n, fill=np.nan)
    data["K_r2"] = numeric(data, "K_r2", n, fill=np.nan)
    data["K_ef"] = numeric(data, "K_ef", n, fill=np.nan)
    data["K_sf"] = numeric(data, "K_sf", n, fill=np.nan)
    data["u_norm"] = numeric(data, "u_norm", n, fill=np.nan)
    return data


def latest_npz_from_roots(roots):
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(p for p in root.rglob("*.npz") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            "未找到 .npz 实验结果。请先运行实验，或用 --input 指定文件/目录。"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_input(input_path, roots):
    if input_path is None:
        return latest_npz_from_roots(roots)
    path = Path(input_path)
    if path.is_file():
        return path
    if path.is_dir():
        files = [p for p in path.glob("*.npz") if p.is_file()]
        if not files:
            files = [p for p in path.rglob("*.npz") if p.is_file()]
        if not files:
            raise FileNotFoundError(f"{path} 下没有 .npz 文件")
        return max(files, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(str(path))


def infer_strategy(path, data):
    if "arbitration_strategy" in data:
        values = np.asarray(data["arbitration_strategy"]).flatten()
        if len(values):
            text = str(values[0])
            if text and text != "nan":
                return text
    return path.stem.split("_t")[0]


def safe_mean(x):
    return float(np.nanmean(x)) if len(x) else np.nan


def safe_max(x):
    return float(np.nanmax(x)) if len(x) else np.nan


def compute_metrics(path, data):
    t = data["t"]
    force_err = data["F_err"]
    pos_err = data["pos_err_norm"]
    rcm_err = data["error_rcm"]
    alpha = data["alpha"] if "alpha" in data else np.full(len(t), np.nan)
    metrics = {
        "source_file": str(path),
        "strategy": infer_strategy(path, data),
        "samples": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "force_rmse_N": float(np.sqrt(np.nanmean(force_err ** 2))),
        "force_mae_N": safe_mean(np.abs(force_err)),
        "force_peak_abs_err_N": safe_max(np.abs(force_err)),
        "pos_rmse_mm": float(np.sqrt(np.nanmean(pos_err ** 2)) * 1000.0),
        "pos_peak_mm": safe_max(pos_err) * 1000.0,
        "rcm_rmse_mm": float(np.sqrt(np.nanmean(rcm_err ** 2)) * 1000.0),
        "rcm_peak_mm": safe_max(rcm_err) * 1000.0,
        "alpha_mean": safe_mean(alpha),
        "alpha_min": float(np.nanmin(alpha)) if len(alpha) else np.nan,
        "alpha_max": float(np.nanmax(alpha)) if len(alpha) else np.nan,
        "u_rms": float(np.sqrt(np.nanmean(data["u_norm"] ** 2))),
    }
    return metrics


def save_metrics_csv(path, metrics):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def plot_overview(data, metrics, title):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    t = data["t"]

    ax = axes[0, 0]
    ax.plot(t, data["pos_x"] * 1000, label="x")
    ax.plot(t, data["pos_y"] * 1000, label="y")
    ax.plot(t, data["pos_z"] * 1000, label="z")
    ax.plot(t, data["pos_des_x"] * 1000, "--", label="x_ref", alpha=0.7)
    ax.plot(t, data["pos_des_z"] * 1000, "--", label="z_ref", alpha=0.7)
    ax.set_ylabel("tool position (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    ax.plot(t, data["pos_err_norm"] * 1000, color="tab:purple")
    ax.set_ylabel("position error (mm)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, data["F_measured"], label="measured")
    ax.plot(t, data["F_desired"], "--", label="desired")
    ax.set_ylabel("force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(t, data["F_err"], color="tab:red")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_ylabel("force error (N)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    alpha_line, = ax.plot(t, data["alpha"], label="alpha")
    legend_handles = [alpha_line]
    if np.isfinite(data["K_hat"]).any():
        ax2 = ax.twinx()
        k_hat_line, = ax2.plot(t, data["K_hat"], color="tab:orange", alpha=0.65, label="K_hat")
        legend_handles.append(k_hat_line)
        ax2.set_ylabel("K_hat")
    ax.set_ylabel("alpha")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=9, loc="best")

    ax = axes[2, 1]
    ax.plot(t, data["error_rcm"] * 1000, label="RCM error")
    ax.plot(t, data["u_norm"], label="|tau|", alpha=0.8)
    ax.set_ylabel("RCM error (mm) / torque norm")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    text = (
        f"strategy={metrics['strategy']} | "
        f"force RMSE={metrics['force_rmse_N']:.4f} N | "
        f"pos RMSE={metrics['pos_rmse_mm']:.3f} mm | "
        f"RCM peak={metrics['rcm_peak_mm']:.3f} mm"
    )
    fig.suptitle(f"{title}\n{text}", fontsize=12)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="npz 文件或结果目录；默认寻找最近一次结果")
    ap.add_argument("--results-dir", action="append", default=[],
                    help="额外搜索的结果根目录，可重复传入")
    ap.add_argument("--output-dir", default=None, help="图像和 summary CSV 输出目录")
    ap.add_argument("--no-show", action="store_true", help="只保存图片，不弹出窗口")
    args = ap.parse_args()

    setup_font()
    roots = DEFAULT_RESULTS_ROOTS + [Path(p) for p in args.results_dir]
    npz_path = resolve_input(args.input, roots)
    data = add_derived_fields(flatten_npz(npz_path))
    metrics = compute_metrics(npz_path, data)

    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    fig = plot_overview(data, metrics, title=npz_path.name)
    png_path = output_dir / f"{stem}_overview.png"
    pdf_path = output_dir / f"{stem}_overview.pdf"
    csv_path = output_dir / f"{stem}_summary.csv"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    save_metrics_csv(csv_path, metrics)

    print(f"分析文件: {npz_path}")
    print(f"保存图像: {png_path}")
    print(f"保存统计: {csv_path}")
    print(
        f"force_RMSE={metrics['force_rmse_N']:.4f}N, "
        f"pos_RMSE={metrics['pos_rmse_mm']:.3f}mm, "
        f"RCM_peak={metrics['rcm_peak_mm']:.3f}mm"
    )

    if not args.no_show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
