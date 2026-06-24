#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序 A: RCM 约束下的合作博弈力-位扫描实验
=============================================

控制链路:
  泄漏积分更新 → 4D-ARE 查表 → u_tool → RCM 杠杆 → τ

数学模型:
  M·ẍ + C·ẋ + K_v·(x−x_r) = u + f_ext      (含 K_v 基线刚度)
  ė_r1 = e_r2 − ε_r · e_r1                  (泄漏积分器)
  ė_f  = -(K_e − B_e·C/M)·e_r2 − B_e/M·e_f − B_e/M·u
  σ̇_f  = e_f − ε_f · σ_f                    (泄漏积分器)

用法:
  Terminal 1: roslaunch panda_simulator simulation.launch
  Terminal 2: python run_with_rcm.py --strategy all --trials 3 --use-virtual-env
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import rospy
from panda_robot import PandaArm, PandaKinematics

from src.gt_controller import CooperativeGameController
from src.alpha_scheduler_gt import (
    PhaseAwareFuzzyAlphaScheduler, ForceMarginFuzzyAlphaScheduler,
    FixedAlphaScheduler
)
from src.env_estimator import EnvironmentEstimator
from src.utils import VirtualStiffnessSurface, ForceSensorInput, DataLogger
from src.leaky_integrator import LeakyIntegrator
from src.robot_interface import (
    update_robot_state, safe_move_to_joint_position,
    compute_torque_with_rcm, INIT_JOINTS,
)


# ================================================================
# 实验配置
# ================================================================
class Config:
    # 扫描轨迹 (tool tip)
    scan_start_x = 0.23
    scan_end_x   = 0.32
    scan_y       = 0.0
    scan_z       = 0.2
    approach_z   = 0.05
    scan_vx      = 0.002

    # 力控
    F_desired    = 0.5
    F_min        = 0.3
    F_max        = 1.0
    force_axis   = 2

    # RCM
    trocar_position = np.array([0.3, 0, 0.235])
    tool_length = 0.525

    # 变刚度区
    # stiffness_zones = [
    #     (0.25, 0.29, 500,  5),    # 软
    #     (0.29, 0.33, 5000, 50),   # 硬
    #     (0.33, 0.37, 80,   2),    # 极软
    #     (0.37, 0.41, 500,  5),    # 软
    # ]

    stiffness_zones = [
        (0.23, 0.275, 250, 4),    # 低刚度
        (0.275, 0.32, 500, 8),    # 高刚度
    ]

    # 控制频率 (文档要求 1kHz; 仿真可用 100Hz)
    ctrl_rate = 100
    dt = 1.0 / ctrl_rate

    # 泄漏积分器 (标量)
    eps_r = 1.0
    eps_f = 2.0

    settle_time = 2.0


def read_contact_force(force_sensor, venv, x, delta, delta_dot):
    """
    获取当前接触力。

    优先使用新鲜的六维力传感器数据；若传感器未启动、话题无数据或超时，
    回退到原 Kelvin-Voigt 虚拟刚度环境。
    """
    if force_sensor is not None and force_sensor.available():
        return (
            force_sensor.contact_force(),
            force_sensor.wrench_vector(),
            "sensor",
            True,
        )

    F_z = venv.compute_force(x, delta, delta_dot) if venv is not None else 0.0
    return F_z, np.zeros(6), "virtual", False


# ================================================================
# 单次试验
# ================================================================
def run_trial(robot, kin_tool, kin_flange,
              cfg, ctrl, sched, est, venv, force_sensor, logger, trial_id):
    rate = rospy.Rate(cfg.ctrl_rate)
    dt = cfg.dt

    # 泄漏积分器状态 (仅 σ_f, e_r1 直接由 x−x_r 计算)
    sigma_f_int = LeakyIntegrator(eps=cfg.eps_f, dt=dt, dim=3)
    integ_euler = np.zeros(3)

    # 差分导数缓存
    prev_ef = 0.0
    prev_er = 0.0
    prev_K = 500.0

    est.reset()
    sched.reset()
    sigma_f_int.reset()

    rospy.loginfo(f"[Trial {trial_id}] {sched.name} (RCM)")

    # ---- Phase 0: 到起点 ----
    safe_move_to_joint_position(robot, INIT_JOINTS)
    rospy.sleep(cfg.settle_time)
    rs = update_robot_state(kin_tool, kin_flange)
    rospy.loginfo(f"  Tool at: {rs['tool_position']}")

    # ---- Phase 1: 下降接近 ----
    rospy.loginfo("  Phase 1: Approaching...")
    z_contact = None
    t0 = time.time()

    while not rospy.is_shutdown() and z_contact is None:
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        delta = max(0, cfg.scan_z - tp[2])
        F_z, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, -tv[2]
        )

        if abs(F_z) > 0.3 or tp[2] <= cfg.scan_z + 0.002:
            z_contact = tp[2]
            rospy.loginfo(
                f"  Contact at z={z_contact:.4f}, F={F_z:.3f}N "
                f"({force_source})"
            )
            break

        # 纯位控下降 (α=1) — flange-space 控制
        x_tool_ref = np.array([cfg.scan_start_x, cfg.scan_y, tp[2] - 0.002 * dt])
        xdot_tool_ref = np.array([0.0, 0.0, -0.002])

        # 自由空间: 力误差为 0, 力积分不更新
        e_f = np.zeros(3)
        sigma_f = sigma_f_int.get()

        tau, _, _, integ_euler, _, _ = compute_torque_with_rcm(
            ctrl, rs, kin_flange,
            x_tool_ref=x_tool_ref, xdot_tool_ref=xdot_tool_ref,
            e_f=e_f, sigma_f=sigma_f,
            alpha=1.0, K_e_hat=500,
            trocar_pos=cfg.trocar_position,
            length=cfg.tool_length,
            integ_euler=integ_euler, dt=dt,
        )
        robot.exec_torque_cmd(tau)
        rate.sleep()

        if time.time() - t0 > 20:
            rospy.logwarn("  Approach timeout")
            return False

    if z_contact is None:
        z_contact = cfg.scan_z

    # 重置积分器状态 (进入 Phase 2)
    sigma_f_int.reset()
    integ_euler = np.zeros(3)

    # ---- Phase 2: 恒力扫描 ----
    rospy.loginfo("  Phase 2: Scanning...")
    x_cur = cfg.scan_start_x
    t_scan = time.time()

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        t = time.time() - t_scan
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 接触力: 传感器优先, 不可用时回退到虚拟环境
        delta = max(0, z_contact - tp[2] + 0.003)
        F_z, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, -tv[2]
        )

        # 环境估计
        K_hat, B_hat = est.update(
            abs(F_z),
            max(0, z_contact - tp[2] + 0.003),
            -tv[2]
        )

        # tool 期望轨迹
        x_tool_ref = np.array([x_cur, cfg.scan_y, z_contact])
        xdot_tool_ref = np.array([cfg.scan_vx, 0.0, 0.0])

        # 力目标: 向上接触反力 (z+) 为正
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        # 力误差 + 力误差泄漏积分
        e_f_vec = F_meas - F_des
        sigma_f = sigma_f_int.update(e_f_vec)

        # 差分导数 (用于 α 调度)
        F_actual = abs(F_z)
        e_f_scalar = F_actual - cfg.F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt
        prev_ef = e_f_scalar

        e_r_scalar = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
        de_r = (e_r_scalar - prev_er) / dt
        prev_er = e_r_scalar

        dK = (K_hat - prev_K) / dt
        prev_K = K_hat

        # α 计算
        if isinstance(sched, FixedAlphaScheduler):
            alpha = sched.compute()
            phase_val = -1
        else:
            alpha = sched.compute(
                F_norm=abs(F_z), e_f=e_f_scalar, K_hat=K_hat,
                e_r=e_r_scalar, z_vel=tv[2],
                de_f=e_f_dot, dK=dK, de_r=de_r,
                F_desired=cfg.F_desired,
                F_min=cfg.F_min,
                F_max=cfg.F_max,
            )
            phase_val = sched.phase_detector.phase.value

        # 博弈控制 (flange-space, 内部计算 flange 期望与误差)
        tau, u_flange, K_eff, integ_euler, x_flange_ref, error = \
            compute_torque_with_rcm(
                ctrl, rs, kin_flange,
                x_tool_ref=x_tool_ref, xdot_tool_ref=xdot_tool_ref,
                e_f=e_f_vec, sigma_f=sigma_f,
                alpha=alpha, K_e_hat=K_hat,
                trocar_pos=cfg.trocar_position,
                length=cfg.tool_length,
                integ_euler=integ_euler, dt=dt,
            )
        robot.exec_torque_cmd(tau)

        # 推进扫描参考
        x_cur += cfg.scan_vx * dt

        # 6 个核心物理量
        pos_tool = tp.copy()
        pos_tool_des = x_tool_ref.copy()
        pos_err = pos_tool - pos_tool_des           # tool 位置误差 (3D)
        pos_err_norm = np.linalg.norm(pos_err)
        F_desired = cfg.F_desired
        F_err = F_actual - F_desired
        rcm_err = error[0]

        # 实时打印 (10 Hz, 每 10 步打一次)
        if logger.count % 10 == 0:
            rospy.loginfo(
                f"  t={t:5.2f}s | "
                f"tool=[{pos_tool[0]*1000:6.2f},{pos_tool[1]*1000:6.2f},{pos_tool[2]*1000:6.2f}]mm | "
                f"des=[{pos_tool_des[0]*1000:6.2f},{pos_tool_des[1]*1000:6.2f},{pos_tool_des[2]*1000:6.2f}]mm | "
                f"err={pos_err_norm*1000:5.2f}mm | "
                f"rcm={rcm_err*1000:5.2f}mm | "
                f"F={F_actual:.3f}N (des={F_desired:.3f}, err={F_err:+.3f}) | "
                f"α={alpha:.2f} | source={force_source}"
            )

        # 日志中的 e_r1 取 flange 误差范数 (新方案的反馈量)
        e_r1_flange_norm = np.linalg.norm(rs["flange_position"] - x_flange_ref)

        # 记录
        logger.log(
            t=t,
            pos=pos_tool,
            pos_des=pos_tool_des,
            pos_err=pos_err,
            F_measured=F_actual, F_desired=F_desired, F_err=F_err,
            wrench=wrench,
            force_source=force_source,
            sensor_available=int(sensor_available),
            e_f=e_f_scalar, e_r=e_r_scalar,
            sigma_f_norm=np.linalg.norm(sigma_f),
            e_r1_norm=e_r1_flange_norm,
            alpha=alpha, K_hat=K_hat,
            K_eff=K_eff,
            x_desired=x_cur,
            error_rcm=rcm_err, error_track=error[1],
            arbitration_strategy=sched.name,
            u_norm=np.linalg.norm(tau),
            phase=phase_val,
        )
        rate.sleep()

    # ---- Phase 3: 撤退 ----
    rospy.loginfo("  Phase 3: Retreating...")
    if hasattr(sched, 'set_retreat'):
        sched.set_retreat(True)
    safe_move_to_joint_position(robot, INIT_JOINTS)
    rospy.loginfo(f"  Done. {logger.count} samples.")
    return True


# ================================================================
# 策略工厂
# ================================================================
STRATEGIES = {
    'fixed_08': lambda: FixedAlphaScheduler(0.8),
    'fixed_05': lambda: FixedAlphaScheduler(0.5),
    'fixed_02': lambda: FixedAlphaScheduler(0.2),
    'coop_fuzzy': lambda: PhaseAwareFuzzyAlphaScheduler(dt=0.01),
    'force_margin': lambda: ForceMarginFuzzyAlphaScheduler(
        dt=0.01,
        F_min=Config.F_min,
        F_max=Config.F_max,
        F_desired=Config.F_desired,
    ),
}


# ================================================================
# 主入口
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='force_margin')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--gains-file', default=None)
    ap.add_argument('--force-topic', default='/force_sensor/wrench')
    ap.add_argument('--force-timeout', type=float, default=0.02,
                    help='force topic freshness window; 0.02s matches a 1kHz publisher with margin')
    ap.add_argument('--force-axis', type=int, default=2)
    ap.add_argument('--force-sign', type=float, default=1.0)
    ap.add_argument('--force-wait-timeout', type=float, default=2.0)
    ap.add_argument('--force-expected-rate', type=float, default=1000.0)
    ap.add_argument('--force-node-command-format', default='both')
    ap.add_argument('--force-node-data-source', default='0x33')
    ap.add_argument('--force-node-streaming', dest='force_node_streaming',
                    action='store_true', default=True)
    ap.add_argument('--no-force-node-streaming', dest='force_node_streaming',
                    action='store_false')
    ap.add_argument('--no-force-sensor', action='store_true')
    args = ap.parse_args()

    rospy.init_node("coop_gt_rcm")
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Cooperative Game Force-Position Experiment (RCM)")
    rospy.loginfo("  Model: 4D per-axis ARE + K_v baseline + leaky integrators")
    rospy.loginfo("=" * 60)

    robot = PandaArm()
    kin_tool = PandaKinematics(robot, "panda_link10")
    kin_flange = PandaKinematics(robot, "panda_link8")
    rospy.sleep(1.0)

    cfg = Config()
    ctrl = CooperativeGameController()

    # 增益表
    if args.gains_file and os.path.exists(args.gains_file):
        ctrl.load_gains(args.gains_file)
        rospy.loginfo(f"Loaded gains from {args.gains_file}")
    else:
        Ke_vals = sorted(set(
            [v[2] for v in cfg.stiffness_zones]
            + [50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500,
               2000, 3000, 5000]
        ))
        ctrl.precompute_gains(
            alpha_grid=np.linspace(0.0, 1.0, 21),
            Ke_grid=Ke_vals,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        gpath = os.path.join(args.output_dir, 'coop_gains_rcm.npy')
        ctrl.save_gains(gpath)
        rospy.loginfo(f"Saved gains to {gpath}")

    venv = VirtualStiffnessSurface(cfg.stiffness_zones)
    force_sensor = None
    if not args.no_force_sensor:
        force_sensor = ForceSensorInput(
            topic=args.force_topic,
            timeout=args.force_timeout,
            force_axis=args.force_axis,
            force_sign=args.force_sign,
        )
        got_first_frame = force_sensor.wait_for_data(args.force_wait_timeout)
        rospy.loginfo(
            f"Force sensor topic: {args.force_topic}, "
            f"axis={args.force_axis}, sign={args.force_sign}, "
            f"timeout={args.force_timeout}s; expected_rate={args.force_expected_rate:g}Hz; "
            f"node_format={args.force_node_command_format}, "
            f"data_source={args.force_node_data_source}, streaming={args.force_node_streaming}; "
            f"first_frame={got_first_frame}, seq={force_sensor.seq()}, "
            f"age={force_sensor.age():.4f}s; virtual env fallback enabled"
        )
    else:
        rospy.loginfo("Force sensor disabled; using virtual env fallback only")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = os.path.join(args.output_dir, f"rcm_{stamp}")
    os.makedirs(odir, exist_ok=True)

    names = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]
    for sn in names:
        s = STRATEGIES[sn]()
        rospy.loginfo(f"\n{'='*50}\n  Strategy: {s.name}\n{'='*50}")
        for t in range(args.trials):
            if rospy.is_shutdown():
                break
            lg = DataLogger()
            est = EnvironmentEstimator()
            s.reset()
            ok = run_trial(
                robot, kin_tool, kin_flange,
                cfg, ctrl, s, est, venv, force_sensor, lg, t,
            )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")


if __name__ == '__main__':
    main()
