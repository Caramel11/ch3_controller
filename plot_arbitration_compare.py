#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple arbitration strategies from 0603 experiment results.

Default behavior:
  1. find the latest result directory containing .npz files;
  2. group trials by arbitration_strategy or filename;
  3. save metric CSV files and comparison figures into <result_dir>/analysis_compare;
  4. show figures when a display is available, unless --no-show is used.
"""
import argparse
import csv
import os
import re
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

METRICS = [
    ("force_rmse_N", "Force RMSE (N)"),
    ("force_mae_N", "Force MAE (N)"),
    ("force_peak_abs_err_N", "Peak |Force Error| (N)"),
    ("pos_rmse_mm", "Position RMSE (mm)"),
    ("pos_peak_mm", "Peak Position Error (mm)"),
    ("rcm_peak_mm", "Peak RCM Error (mm)"),
    ("alpha_mean", "Mean Alpha"),
    ("u_rms", "Torque Norm RMS"),
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
        raise KeyError("缺少 t 字段")
    t = numeric(data, "t")
    n = len(t)
    if n < 2:
        raise ValueError("数据点过少")
    data["t"] = t

    for key in ("F_measured", "F_desired", "alpha", "u_norm"):
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
        if pos_key in data:
            data[pos_key] = numeric(data, pos_key, n)
        if des_key in data:
            data[des_key] = numeric(data, des_key, n)
        if err_key in data:
            data[err_key] = numeric(data, err_key, n)
        elif pos_key in data and des_key in data:
            data[err_key] = data[pos_key] - data[des_key]
        else:
            data[err_key] = np.zeros(n)

    if "pos_err_norm" not in data:
        data["pos_err_norm"] = np.sqrt(
            data["pos_err_x"] ** 2 + data["pos_err_y"] ** 2 + data["pos_err_z"] ** 2
        )
    else:
        data["pos_err_norm"] = numeric(data, "pos_err_norm", n)

    data["error_rcm"] = np.abs(numeric(data, "error_rcm", n))
    data["alpha"] = numeric(data, "alpha", n, fill=np.nan)
    data["u_norm"] = numeric(data, "u_norm", n, fill=np.nan)
    return data


def infer_strategy(path, data):
    if "arbitration_strategy" in data:
        values = np.asarray(data["arbitration_strategy"]).flatten()
        if len(values):
            text = str(values[0])
            if text and text != "nan":
                return text
    name = path.stem
    if "continuous_force_margin" in name:
        return "continuous_force_margin_alpha"
    if "online_priority" in name:
        return "online_priority_alpha"
    if "force_margin" in name:
        return "force_margin_alpha"
    if "coop_fuzzy" in name:
        return "coop_fuzzy_alpha"
    if "fixed" in name:
        match = re.search(r"fixed[_-]?(\d+)", name)
        return f"fixed_{match.group(1)}" if match else "fixed"
    return name.split("_t")[0]


def trial_metrics(path):
    data = add_derived_fields(flatten_npz(path))
    t = data["t"]
    force_err = data["F_err"]
    pos_err = data["pos_err_norm"]
    rcm_err = data["error_rcm"]
    alpha = data["alpha"]
    u_norm = data["u_norm"]
    dt = np.diff(t)
    dF = np.diff(data["F_measured"]) / np.maximum(dt, 1e-6) if "F_measured" in data else np.array([np.nan])
    return data, {
        "file": str(path),
        "strategy": infer_strategy(path, data),
        "duration_s": float(t[-1] - t[0]),
        "samples": int(len(t)),
        "force_rmse_N": float(np.sqrt(np.nanmean(force_err ** 2))),
        "force_mae_N": float(np.nanmean(np.abs(force_err))),
        "force_peak_abs_err_N": float(np.nanmax(np.abs(force_err))),
        "force_dF_p95_Nps": float(np.nanpercentile(np.abs(dF), 95)),
        "pos_rmse_mm": float(np.sqrt(np.nanmean(pos_err ** 2)) * 1000.0),
        "pos_peak_mm": float(np.nanmax(pos_err) * 1000.0),
        "rcm_rmse_mm": float(np.sqrt(np.nanmean(rcm_err ** 2)) * 1000.0),
        "rcm_peak_mm": float(np.nanmax(rcm_err) * 1000.0),
        "alpha_mean": float(np.nanmean(alpha)),
        "alpha_std": float(np.nanstd(alpha)),
        "u_rms": float(np.sqrt(np.nanmean(u_norm ** 2))),
    }


def latest_result_dir(roots):
    dirs = set()
    for root in roots:
        if root.exists():
            for npz in root.rglob("*.npz"):
                if npz.is_file():
                    dirs.add(npz.parent)
    if not dirs:
        raise FileNotFoundError("未找到 .npz 结果目录")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def collect_trials(input_path, roots):
    if input_path is None:
        root = latest_result_dir(roots)
    else:
        root = Path(input_path)
    files = sorted(root.glob("*.npz")) if root.is_dir() else [root]
    if not files:
        raise FileNotFoundError(f"{root} 下没有 .npz 文件")

    rows = []
    series = []
    for path in files:
        try:
            data, row = trial_metrics(path)
            rows.append(row)
            series.append((row["strategy"], path, data))
        except Exception as exc:
            print(f"跳过 {path}: {exc}")
    if not rows:
        raise RuntimeError(f"{root} 中没有可用实验数据")
    return root, rows, series


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    strategies = sorted({r["strategy"] for r in rows})
    out = []
    for strategy in strategies:
        group = [r for r in rows if r["strategy"] == strategy]
        row = {"strategy": strategy, "trials": len(group)}
        for metric, _ in METRICS:
            values = np.asarray([g[metric] for g in group], dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values))
        out.append(row)
    return out


def plot_metric_bars(summary_rows, output_dir):
    strategies = [r["strategy"] for r in summary_rows]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(strategies))
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, max(1, len(strategies))))
    for ax, (metric, label) in zip(axes, METRICS):
        means = [r[f"{metric}_mean"] for r in summary_rows]
        stds = [r[f"{metric}_std"] for r in summary_rows]
        ax.bar(x, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Arbitration Strategy Metric Comparison", fontsize=13)
    png = output_dir / "arbitration_metric_comparison.png"
    pdf = output_dir / "arbitration_metric_comparison.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    return fig, png


def plot_time_overlay(series, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    color_map = {}
    strategies = sorted({s for s, _, _ in series})
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(strategies))))
    for strategy, color in zip(strategies, palette):
        color_map[strategy] = color

    labels_seen = set()
    for strategy, path, data in series:
        label = strategy if strategy not in labels_seen else None
        labels_seen.add(strategy)
        t = data["t"]
        color = color_map[strategy]
        axes[0].plot(t, data["F_err"], color=color, alpha=0.75, label=label)
        axes[1].plot(t, data["pos_err_norm"] * 1000, color=color, alpha=0.75)
        axes[2].plot(t, data["error_rcm"] * 1000, color=color, alpha=0.75)
        axes[3].plot(t, data["alpha"], color=color, alpha=0.75)

    axes[0].set_ylabel("Force error (N)")
    axes[1].set_ylabel("Position error (mm)")
    axes[2].set_ylabel("RCM error (mm)")
    axes[3].set_ylabel("Alpha")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, ncol=2)
    fig.suptitle("Arbitration Strategy Time-Series Overlay", fontsize=13)
    png = output_dir / "arbitration_timeseries_overlay.png"
    pdf = output_dir / "arbitration_timeseries_overlay.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    return fig, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="包含多组 .npz 的目录；默认寻找最近结果目录")
    ap.add_argument("--results-dir", action="append", default=[],
                    help="额外搜索的结果根目录，可重复传入")
    ap.add_argument("--output-dir", default=None, help="输出目录，默认 <input>/analysis_compare")
    ap.add_argument("--no-show", action="store_true", help="只保存图片，不弹出窗口")
    args = ap.parse_args()

    setup_font()
    roots = DEFAULT_RESULTS_ROOTS + [Path(p) for p in args.results_dir]
    root, rows, series = collect_trials(args.input, roots)
    output_dir = Path(args.output_dir) if args.output_dir else root / "analysis_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = output_dir / "trial_metrics.csv"
    trial_fields = list(rows[0].keys())
    write_csv(trial_csv, rows, trial_fields)

    summary_rows = summarize(rows)
    summary_csv = output_dir / "strategy_summary.csv"
    summary_fields = list(summary_rows[0].keys())
    write_csv(summary_csv, summary_rows, summary_fields)

    fig1, png1 = plot_metric_bars(summary_rows, output_dir)
    fig2, png2 = plot_time_overlay(series, output_dir)

    print(f"分析目录: {root}")
    print(f"保存 trial 指标: {trial_csv}")
    print(f"保存策略汇总: {summary_csv}")
    print(f"保存对照图: {png1}")
    print(f"保存时序图: {png2}")
    for row in summary_rows:
        print(
            f"{row['strategy']}: trials={row['trials']}, "
            f"force_RMSE={row['force_rmse_N_mean']:.4f}N, "
            f"pos_RMSE={row['pos_rmse_mm_mean']:.3f}mm, "
            f"RCM_peak={row['rcm_peak_mm_mean']:.3f}mm"
        )

    if not args.no_show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
