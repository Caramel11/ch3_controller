#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gazebo mock-sensor test harness for 0603 *_real.py.

该脚本不伪造串口设备，而是从代码层面给 run_no_rcm_real.py /
run_with_rcm_real.py 注入一个与 DirectForceSensorInput 接口一致的
MockForceSensorInput。控制循环仍走 *_real.py 的传感器优先分支，
force_source 会记录为 sensor_mock。

典型用法:
  source /home/liu/franka_ws_1101/devel/setup.bash
  python3 test_real_mock_sensor_gazebo.py --mode with_rcm --trials 1
  python3 test_real_mock_sensor_gazebo.py --mode both --trials 1 --quick-scan-length 0.010
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import rospy
from panda_robot import PandaArm, PandaKinematics

import run_no_rcm_real as no_rcm_real
import run_with_rcm_real as with_rcm_real
from src.gt_controller import CooperativeGameController
from src.utils import DataLogger, VirtualStiffnessSurface
from src.env_estimator import EnvironmentEstimator


class MockForceSensorInput:
    """DirectForceSensorInput 兼容的 Gazebo 代码级 mock 传感器。"""

    source_name = "sensor_mock"

    def __init__(
        self,
        zones,
        freshness_timeout=0.05,
        force_axis=2,
        force_sign=1.0,
        noise_std=0.0,
        bias=0.0,
        scale=1.0,
        seed=7,
    ):
        self.venv = VirtualStiffnessSurface(zones)
        self.timeout = float(freshness_timeout)
        self.force_axis = int(force_axis)
        self.force_sign = float(force_sign)
        self.noise_std = float(noise_std)
        self.bias = float(bias)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)
        self._force = np.zeros(3)
        self._torque = np.zeros(3)
        self._stamp = None
        self._seq = 0

    def update_contact_state(self, x, delta, delta_dot):
        force = self.venv.compute_force(float(x), float(delta), float(delta_dot))
        force = max(0.0, self.scale * force + self.bias)
        if self.noise_std > 0.0:
            force = max(0.0, force + float(self.rng.normal(0.0, self.noise_std)))
        self._force[:] = 0.0
        self._force[self.force_axis] = self.force_sign * force
        self._torque[:] = 0.0
        self._stamp = time.time()
        self._seq += 1

    def available(self):
        return self._stamp is not None and (time.time() - self._stamp) <= self.timeout

    def age(self):
        return float("inf") if self._stamp is None else time.time() - self._stamp

    def seq(self):
        return self._seq

    def wait_for_data(self, timeout=2.0):
        return True

    def force_vector(self):
        return self._force.copy()

    def wrench_vector(self):
        return np.hstack([self._force, self._torque])

    def signed_axis_force(self):
        return self.force_sign * self._force[self.force_axis]

    def contact_force(self):
        return abs(self.signed_axis_force())

    def close(self):
        return None


def make_scheduler(module, name):
    if name not in module.STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Available: {sorted(module.STRATEGIES)}")
    return module.STRATEGIES[name]()


def configure_quick_surface(module, cfg, robot, kin_tool, kin_flange, mode, args):
    """把 mock 表面放在当前 tool 下方，避免 Gazebo 测试等待很久。"""
    if not args.quick_surface:
        return
    try:
        if hasattr(module, "INIT_JOINTS"):
            module.safe_move_to_joint_position(robot, module.INIT_JOINTS)
            rospy.sleep(cfg.settle_time)
    except Exception as exc:
        rospy.logwarn(f"Quick-surface pre-move failed; using current pose anyway: {exc}")

    rs = module.update_robot_state(kin_tool, kin_flange)
    tool_x = float(rs["tool_position"][0])
    z0 = float(rs["tool_position"][2])
    surface_z = z0 - float(args.surface_offset)
    if mode == "no_rcm":
        if args.localize_no_rcm_x:
            scan_len = cfg.scan_end_x - cfg.scan_start_x
            cfg.scan_start_x = tool_x
            cfg.scan_end_x = tool_x + scan_len
        cfg.approach_z = surface_z
        cfg.scan_z = surface_z - cfg.contact_penetration_depth
        rospy.loginfo(
            f"[{mode}] mock surface from current tool z={z0:.4f}: "
            f"x0={cfg.scan_start_x:.4f}, surface_z={cfg.approach_z:.4f}, "
            f"scan_z={cfg.scan_z:.4f}"
        )
    else:
        cfg.scan_z = surface_z
        cfg.approach_z = surface_z + 0.005
        rospy.loginfo(
            f"[{mode}] mock surface from current tool z={z0:.4f}: "
            f"scan_z={cfg.scan_z:.4f}, approach_z={cfg.approach_z:.4f}"
        )


def shorten_scan(cfg, args):
    if args.full_scan:
        return
    length = max(0.001, float(args.quick_scan_length))
    cfg.scan_end_x = cfg.scan_start_x + length


def analyze_npz(path):
    data = np.load(path, allow_pickle=True)
    t = np.asarray(data["t"], dtype=float)
    out = {
        "file": os.path.basename(path),
        "samples": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
    }
    for key, scale in (
        ("pos_err_norm", 1000.0),
        ("error_rcm", 1000.0),
        ("F_measured", 1.0),
        ("F_err", 1.0),
        ("u_norm", 1.0),
    ):
        if key in data.files:
            arr = np.asarray(data[key], dtype=float)
            if arr.size:
                out[f"{key}_rms"] = float(np.sqrt(np.mean(arr * arr)) * scale)
                out[f"{key}_max"] = float(np.max(np.abs(arr)) * scale)
                out[f"{key}_mean"] = float(np.mean(arr) * scale)
    if "force_source" in data.files:
        values, counts = np.unique(data["force_source"], return_counts=True)
        out["force_sources"] = {
            str(v): int(c) for v, c in zip(values.tolist(), counts.tolist())
        }
    return out


def save_summary(out_dir, rows):
    json_path = os.path.join(out_dir, "mock_sensor_summary.json")
    csv_path = os.path.join(out_dir, "mock_sensor_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    keys = sorted({k for row in rows for k in row.keys() if k != "force_sources"})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return json_path, csv_path


def run_mode(mode, robot, base_out_dir, args):
    module = no_rcm_real if mode == "no_rcm" else with_rcm_real
    cfg = module.Config()
    shorten_scan(cfg, args)

    kin_tool = PandaKinematics(robot, "panda_link11" if mode == "no_rcm" else "panda_link10")
    kin_flange = PandaKinematics(robot, "panda_link8")
    configure_quick_surface(module, cfg, robot, kin_tool, kin_flange, mode, args)

    ctrl = CooperativeGameController(control_mode=args.controller_mode)
    if mode == "no_rcm":
        ctrl.u_threshold = cfg.no_rcm_u_threshold
        ctrl.P_ori = cfg.no_rcm_P_ori
        ctrl.D_ori = cfg.no_rcm_D_ori
        ctrl.I_ori = cfg.no_rcm_I_ori
        ctrl.no_rcm_u_tool_limits = cfg.no_rcm_u_tool_limits

    Ke_vals = sorted(set(
        [zone[2] for zone in cfg.stiffness_zones]
        + [50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    ))
    ctrl.precompute_gains(alpha_grid=np.linspace(0.0, 1.0, 21), Ke_grid=Ke_vals)

    mode_dir = os.path.join(base_out_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    sensor = MockForceSensorInput(
        cfg.stiffness_zones,
        freshness_timeout=args.force_timeout,
        force_axis=cfg.force_axis,
        force_sign=args.force_sign,
        noise_std=args.force_noise_std,
        bias=args.force_bias,
        scale=args.force_scale,
        seed=args.seed,
    )

    rows = []
    for trial_id in range(args.trials):
        sched = make_scheduler(module, args.strategy)
        est = module.EnvironmentEstimator(
            theta_init=[cfg.estimator_initial_K, cfg.estimator_initial_B],
            alpha_lp=cfg.estimator_alpha_lp,
        ) if hasattr(cfg, "estimator_initial_K") else EnvironmentEstimator()
        logger = DataLogger()
        approach_logger = DataLogger() if mode == "with_rcm" else None
        ok = module.run_trial(
            robot, kin_tool, kin_flange,
            cfg, ctrl, sched, est, None, sensor,
            logger, trial_id,
            approach_logger=approach_logger,
        ) if mode == "with_rcm" else module.run_trial(
            robot, kin_tool, kin_flange,
            cfg, ctrl, sched, est, None, sensor,
            logger, trial_id,
        )

        if approach_logger is not None and approach_logger.count:
            approach_dir = os.path.join(mode_dir, "approach_debug")
            os.makedirs(approach_dir, exist_ok=True)
            apath = os.path.join(approach_dir, f"approach_{sched.name}_t{trial_id:02d}.npz")
            approach_logger.save(apath)
            row = analyze_npz(apath)
            row.update({"mode": mode, "trial": trial_id, "phase": "approach", "ok": bool(ok)})
            rows.append(row)

        if logger.count:
            path = os.path.join(mode_dir, f"{sched.name}_t{trial_id:02d}.npz")
            logger.save(path)
            row = analyze_npz(path)
            row.update({"mode": mode, "trial": trial_id, "phase": "scan", "ok": bool(ok)})
            rows.append(row)
        else:
            rows.append({
                "mode": mode, "trial": trial_id, "phase": "scan",
                "ok": bool(ok), "samples": 0, "duration_s": 0.0,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_rcm", "with_rcm", "both"], default="with_rcm")
    ap.add_argument("--strategy", default="continuous_force_margin")
    ap.add_argument("--controller-mode", default="are", choices=["are", "pareto_iter"])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--output-dir", default="/home/done/USTC/LJJ/franka_ros2_ws/results")
    ap.add_argument("--quick-surface", action="store_true", default=True)
    ap.add_argument("--no-quick-surface", dest="quick_surface", action="store_false")
    ap.add_argument("--surface-offset", type=float, default=0.004)
    ap.add_argument("--full-scan", action="store_true")
    ap.add_argument("--quick-scan-length", type=float, default=0.012)
    ap.add_argument("--localize-no-rcm-x", action="store_true", default=True)
    ap.add_argument("--no-localize-no-rcm-x", dest="localize_no_rcm_x", action="store_false")
    ap.add_argument("--force-timeout", type=float, default=0.05)
    ap.add_argument("--force-sign", type=float, default=1.0)
    ap.add_argument("--force-noise-std", type=float, default=0.0)
    ap.add_argument("--force-bias", type=float, default=0.0)
    ap.add_argument("--force-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rospy.init_node("mock_sensor_real_script_gazebo_test", anonymous=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"mock_real_gazebo_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    rospy.loginfo(f"Mock-sensor Gazebo test output: {out_dir}")

    robot = PandaArm()
    rospy.sleep(1.0)

    modes = ["no_rcm", "with_rcm"] if args.mode == "both" else [args.mode]
    rows = []
    for mode in modes:
        rows.extend(run_mode(mode, robot, out_dir, args))

    json_path, csv_path = save_summary(out_dir, rows)
    rospy.loginfo(f"Saved mock-sensor summary: {json_path}")
    rospy.loginfo(f"Saved mock-sensor summary CSV: {csv_path}")
    for row in rows:
        rospy.loginfo(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
