#!/usr/bin/env python3
"""Analyze no-RCM Gazebo tracking data with posture diagnostics."""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _arr(data, name, default=None):
    if name in data:
        return np.asarray(data[name])
    if default is None:
        raise KeyError(name)
    return np.asarray(default)


def _stack(data, names):
    return np.column_stack([_arr(data, name) for name in names])


def _rms(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def _mae(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x)))


def _peak(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.max(np.abs(x)))


def _jitter(signal, t, window_sec=0.5):
    signal = np.asarray(signal, dtype=float)
    t = np.asarray(t, dtype=float)
    if signal.size < 3:
        return np.zeros_like(signal)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    n = max(3, int(round(window_sec / max(dt, 1e-6))))
    if n >= signal.size:
        return signal - np.mean(signal)
    kernel = np.ones(n, dtype=float) / n
    smooth = np.convolve(signal, kernel, mode="same")
    return signal - smooth


def _save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze(npz_path, output_dir=None):
    npz_path = Path(npz_path).resolve()
    data = np.load(npz_path, allow_pickle=True)
    if output_dir is None:
        output_dir = npz_path.parent / "analysis"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = _arr(data, "t")
    pos_err = _stack(data, ["pos_err_x", "pos_err_y", "pos_err_z"])
    pos = _stack(data, ["pos_x", "pos_y", "pos_z"])
    pos_des = _stack(data, ["pos_des_x", "pos_des_y", "pos_des_z"])
    force_err = _arr(data, "F_err")
    force = _arr(data, "F_measured")
    force_des = _arr(data, "F_desired")

    has_pose = "ori_err_deg" in data
    if has_pose:
        ori_err_deg = _arr(data, "ori_err_deg")
        front_err_deg = _arr(data, "front_axis_err_deg")
        omega_norm = _arr(data, "tool_omega_norm")
        euler = _stack(data, ["tool_roll", "tool_pitch", "tool_yaw"])
        euler_ref = _stack(data, ["tool_roll_ref", "tool_pitch_ref", "tool_yaw_ref"])
        q_err = _stack(data, [f"q_err{i}" for i in range(1, 8)])
        qd = _stack(data, [f"qd{i}" for i in range(1, 8)])
        q_err_inf = _arr(data, "q_err_inf")
        qd_norm = _arr(data, "qd_norm")
    else:
        ori_err_deg = front_err_deg = omega_norm = np.zeros_like(t)
        euler = euler_ref = np.zeros((t.size, 3))
        q_err = qd = np.zeros((t.size, 7))
        q_err_inf = qd_norm = np.zeros_like(t)

    # 位置和力跟踪
    plt.figure(figsize=(11, 8))
    ax = plt.subplot(3, 1, 1)
    ax.plot(t, pos[:, 0], label="x")
    ax.plot(t, pos_des[:, 0], "--", label="x_ref")
    ax.plot(t, pos[:, 1], label="y")
    ax.plot(t, pos_des[:, 1], "--", label="y_ref")
    ax.set_ylabel("x/y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    ax = plt.subplot(3, 1, 2)
    ax.plot(t, pos[:, 2], label="z")
    ax.plot(t, pos_des[:, 2], "--", label="z_ref")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = plt.subplot(3, 1, 3)
    ax.plot(t, pos_err[:, 0] * 1000, label="ex")
    ax.plot(t, pos_err[:, 1] * 1000, label="ey")
    ax.plot(t, pos_err[:, 2] * 1000, label="ez")
    ax.plot(t, np.linalg.norm(pos_err, axis=1) * 1000, label="norm")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    _save_plot(output_dir / "pose_position_tracking.png")

    plt.figure(figsize=(11, 6))
    ax = plt.subplot(2, 1, 1)
    ax.plot(t, force, label="F")
    ax.plot(t, force_des, "--", label="F_ref")
    ax.set_ylabel("force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = plt.subplot(2, 1, 2)
    ax.plot(t, force_err, label="F-F_ref")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force error (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_plot(output_dir / "pose_force_tracking.png")

    # 末端姿态
    plt.figure(figsize=(11, 8))
    ax = plt.subplot(3, 1, 1)
    ax.plot(t, ori_err_deg, label="SO(3) error")
    ax.plot(t, front_err_deg, label="front-axis error")
    ax.set_ylabel("angle error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = plt.subplot(3, 1, 2)
    labels = ["roll", "pitch", "yaw"]
    for i, label in enumerate(labels):
        ax.plot(t, np.degrees(euler[:, i]), label=label)
        ax.plot(t, np.degrees(euler_ref[:, i]), "--", linewidth=1.0, label=f"{label}_ref")
    ax.set_ylabel("Euler xyz (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    ax = plt.subplot(3, 1, 3)
    ax.plot(t, omega_norm, label="omega norm")
    ax.plot(t, _jitter(ori_err_deg, t), label="orientation high-pass")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rad/s or deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_plot(output_dir / "end_effector_orientation.png")

    # 关节冗余漂移
    plt.figure(figsize=(11, 8))
    ax = plt.subplot(2, 1, 1)
    for i in range(q_err.shape[1]):
        ax.plot(t, q_err[:, i], label=f"q{i+1}")
    ax.plot(t, q_err_inf, "k--", linewidth=1.0, label="inf")
    ax.set_ylabel("q - q_init (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    ax = plt.subplot(2, 1, 2)
    for i in range(qd.shape[1]):
        ax.plot(t, qd[:, i], label=f"dq{i+1}")
    ax.plot(t, qd_norm, "k--", linewidth=1.0, label="norm")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint velocity (rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    _save_plot(output_dir / "joint_posture_drift.png")

    # 抖动来源对照
    pos_jitter_mm = np.linalg.norm(
        np.column_stack([_jitter(pos_err[:, i], t) for i in range(3)]), axis=1
    ) * 1000.0
    force_jitter = _jitter(force_err, t)
    ori_jitter = _jitter(ori_err_deg, t)
    plt.figure(figsize=(11, 7))
    ax = plt.subplot(3, 1, 1)
    ax.plot(t, pos_jitter_mm)
    ax.set_ylabel("pos jitter (mm)")
    ax.grid(True, alpha=0.3)
    ax = plt.subplot(3, 1, 2)
    ax.plot(t, force_jitter)
    ax.set_ylabel("force jitter (N)")
    ax.grid(True, alpha=0.3)
    ax = plt.subplot(3, 1, 3)
    ax.plot(t, ori_jitter)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ori jitter (deg)")
    ax.grid(True, alpha=0.3)
    _save_plot(output_dir / "tracking_jitter_components.png")

    metrics = {
        "samples": int(t.size),
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "position_rmse_mm": 1000.0 * _rms(np.linalg.norm(pos_err, axis=1)),
        "position_peak_mm": 1000.0 * _peak(np.linalg.norm(pos_err, axis=1)),
        "x_rmse_mm": 1000.0 * _rms(pos_err[:, 0]),
        "y_rmse_mm": 1000.0 * _rms(pos_err[:, 1]),
        "z_rmse_mm": 1000.0 * _rms(pos_err[:, 2]),
        "force_rmse_N": _rms(force_err),
        "force_mae_N": _mae(force_err),
        "force_mean_error_N": float(np.mean(force_err)),
        "force_peak_error_N": _peak(force_err),
        "orientation_rmse_deg": _rms(ori_err_deg),
        "orientation_peak_deg": _peak(ori_err_deg),
        "front_axis_rmse_deg": _rms(front_err_deg),
        "front_axis_peak_deg": _peak(front_err_deg),
        "omega_rms_rad_s": _rms(omega_norm),
        "omega_peak_rad_s": _peak(omega_norm),
        "orientation_jitter_rms_deg": _rms(ori_jitter),
        "position_jitter_rms_mm": _rms(pos_jitter_mm),
        "force_jitter_rms_N": _rms(force_jitter),
        "q_err_inf_peak_rad": _peak(q_err_inf),
        "q_err_inf_mean_rad": float(np.mean(q_err_inf)),
        "qd_norm_rms_rad_s": _rms(qd_norm),
        "qd_norm_peak_rad_s": _peak(qd_norm),
    }

    csv_path = output_dir / "pose_tracking_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key, value in metrics.items():
            f.write(f"{key},{value}\n")

    report_path = output_dir / "pose_tracking_analysis_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Gazebo 有重力补偿 no-RCM 姿态/力位跟踪分析\n\n")
        f.write(f"- 数据文件: `{npz_path}`\n")
        f.write(f"- 样本数: {metrics['samples']}\n")
        f.write(f"- 扫描时长: {metrics['duration_s']:.2f} s\n\n")
        f.write("## 核心指标\n\n")
        f.write(f"- 位置 RMSE: {metrics['position_rmse_mm']:.2f} mm，峰值: {metrics['position_peak_mm']:.2f} mm\n")
        f.write(f"- 分轴 RMSE: x={metrics['x_rmse_mm']:.2f} mm, y={metrics['y_rmse_mm']:.2f} mm, z={metrics['z_rmse_mm']:.2f} mm\n")
        f.write(f"- 力 RMSE: {metrics['force_rmse_N']:.3f} N，平均误差: {metrics['force_mean_error_N']:+.3f} N，峰值误差: {metrics['force_peak_error_N']:.3f} N\n")
        f.write(f"- SO(3) 姿态误差 RMSE: {metrics['orientation_rmse_deg']:.3f} deg，峰值: {metrics['orientation_peak_deg']:.3f} deg\n")
        f.write(f"- 正面朝前轴误差 RMSE: {metrics['front_axis_rmse_deg']:.3f} deg，峰值: {metrics['front_axis_peak_deg']:.3f} deg\n")
        f.write(f"- 姿态抖动 RMS: {metrics['orientation_jitter_rms_deg']:.4f} deg，角速度 RMS: {metrics['omega_rms_rad_s']:.4f} rad/s\n")
        f.write(f"- 位置抖动 RMS: {metrics['position_jitter_rms_mm']:.3f} mm，力抖动 RMS: {metrics['force_jitter_rms_N']:.4f} N\n")
        f.write(f"- 最大关节偏移: {metrics['q_err_inf_peak_rad']:.3f} rad，平均无穷范数偏移: {metrics['q_err_inf_mean_rad']:.3f} rad\n\n")
        f.write("## 自主分析\n\n")
        if has_pose and metrics["front_axis_peak_deg"] < 3.0:
            f.write("- 末端正面朝前约束有效，front-axis 峰值误差小于 3 deg，没有发生明显不定向或翻转。\n")
        elif has_pose:
            f.write("- 末端正面朝前约束存在可见偏差，需要检查姿态环限幅、腕部关节正则和雅可比奇异附近的力矩竞争。\n")
        else:
            f.write("- 当前数据缺少姿态字段，无法判断正面朝前和姿态抖动。\n")
        if metrics["orientation_jitter_rms_deg"] < 0.1:
            f.write("- 姿态高频抖动很低，主要误差不是姿态环振荡造成的。\n")
        else:
            f.write("- 姿态高频抖动偏高，优先降低姿态积分或增加姿态阻尼，并检查力矩限幅是否频繁饱和。\n")
        if abs(metrics["force_mean_error_N"]) > 0.10:
            f.write("- 力跟踪存在稳态偏差，主要来自当前 z 方向力位仲裁偏保守，以及虚拟环境刚度估计/扫描 z 参考之间的静态不一致。\n")
        else:
            f.write("- 力跟踪平均偏差较小，主要误差来自刚度分段过渡和接触速度阻尼项。\n")
        if metrics["y_rmse_mm"] > metrics["z_rmse_mm"] * 3.0:
            f.write("- 位置误差主要集中在 x/y 扫描平面，说明 z 方向接触控制较稳，平面误差更多来自冗余姿态正则与笛卡尔位置任务之间的竞争。\n")
        if metrics["q_err_inf_peak_rad"] > 0.25:
            f.write("- 冗余关节相对初始姿态漂移较大，应提高零空间姿态约束或降低扫描阶段直接腕部正则的扰动。\n")
        else:
            f.write("- 冗余关节漂移在可接受范围内，未观察到明显机械臂姿态扭曲。\n")
        f.write("\n## 图表\n\n")
        for name in [
            "pose_position_tracking.png",
            "pose_force_tracking.png",
            "end_effector_orientation.png",
            "joint_posture_drift.png",
            "tracking_jitter_components.png",
        ]:
            f.write(f"- `{output_dir / name}`\n")

    return report_path, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="npz file or result directory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit(f"no npz files in {input_path}")
        input_path = files[-1]
    report, metrics = analyze(input_path, args.output_dir)
    print(f"report: {report}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
