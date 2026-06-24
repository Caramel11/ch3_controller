#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序 B: 无 RCM 约束的合作博弈力-位扫描实验
=============================================

控制链路:
  1. 读取 tool 端位姿/速度
  2. 虚拟 Kelvin-Voigt 环境生成接触力
  3. RLS 在线估计环境刚度 K_hat 与阻尼 B_hat
  4. alpha 调度器根据力误差/边界风险输出仲裁参数
  5. CooperativeGameController 查表得到 K_eff
  6. u_tool 直接作为 tool 端笛卡尔力
  7. τ = J_tool^T u_tool + 姿态保持力矩

与 RCM 模式的区别:
  - 使用 panda_link11 的 Jacobian (不是 flange)
  - 无杠杆变换
  - 姿态保持: PD+I 维持初始 tool frame 姿态
  - 扫描范围更大, 力目标更大

用法:
  Terminal 1: roslaunch panda_simulator simulation.launch
  Terminal 2: python run_no_rcm.py --strategy continuous_force_margin --controller-mode pareto_iter
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import rospy
from panda_robot import PandaArm, PandaKinematics

from src.gt_controller import CooperativeGameController
from src.alpha_scheduler_gt import (
    PhaseAwareFuzzyAlphaScheduler, ForceMarginFuzzyAlphaScheduler,
    ContinuousForceMarginFuzzyAlphaScheduler,
    OnlinePriorityAdaptationAlphaScheduler,
    FixedAlphaScheduler
)
from src.env_estimator import EnvironmentEstimator
from src.utils import VirtualStiffnessSurface, DataLogger, ContactDeltaDotEstimator
from src.leaky_integrator import LeakyIntegrator
from src.robot_interface import (
    update_robot_state, safe_move_to_joint_position,
    compute_torque_no_rcm, INIT_JOINTS,
)


class Config:
    """实验全局参数。

    当前版本以 Gazebo + 虚拟 Kelvin-Voigt 接触环境为主，所有几何参数、
    力目标、安全边界、控制频率和接触稳定判据都集中放在这里，便于复现实验。
    """

    # 扫描几何参数，单位均为 m；no-RCM 只约束 tool 端位置。
    # Phase 2 中 x 从 scan_start_x 匀速走到 scan_end_x，y/z 为名义扫描线。
    scan_start_x = 0.40      # 扫描起点 x
    scan_end_x   = 0.48      # 扫描终点 x
    scan_y       = 0.0       # 扫描线 y 坐标
    scan_z       = 0.298     # 固定 z 扫描参考；关闭力一致轨迹时使用
    # approach_z 是虚拟表面高度；tool 低于该高度时虚拟环境产生压入量。
    approach_z   = 0.30
    scan_vx      = 0.0015    # Phase 2 沿 x 的扫描速度, m/s；降低以减小动态滞后和力抖动
    force_consistent_scan_z = True  # 根据 F_desired/K_env 生成 z 参考，避免力/位目标冲突
    scan_z_min   = 0.2964    # 力一致 z 参考下限，防止低刚度估计导致过深压入
    scan_z_max   = 0.2984    # 力一致 z 参考上限，保留少量高刚度/过渡裕度
    scan_z_stiffness_floor = 250.0  # 生成 z 参考时使用的最小刚度, N/m

    # Phase 1 接近参数；接近阶段固定 alpha=1，只做位置主导下探。
    approach_speed = 0.007   # 下探参考速度, m/s；原 0.005，适度加快接近

    # 力目标与安全边界，单位 N；供 alpha 调度器和力误差归一化使用。
    F_desired    = 1.0       # 期望法向接触力
    F_min        = 0.3       # 低力边界，低于该值更偏向力控补偿
    F_max        = 2.0       # 高力边界，高于该值触发更保守的调度
    force_axis   = 2         # 力控制轴，2 表示 z 轴

    # 虚拟分段环境参数: (x_start, x_end, K_e, B_e)。
    # K_e 单位 N/m，B_e 单位 Ns/m，用于 Gazebo/mock 接触力与估计器对照。
    stiffness_zones = [
        (0.40, 0.44, 300, 5),    # 低刚度
        (0.44, 0.48, 500, 8),    # 高刚度
    ]

    # 控制周期与泄漏积分器参数；eps 越大，积分记忆衰减越快。
    ctrl_rate = 100         # 主控制循环频率, Hz
    dt = 1.0 / ctrl_rate
    eps_r = 1.0             # 位置误差积分泄漏系数
    eps_f = 2.0             # 力误差积分泄漏系数

    # 接触检测与稳定判据。Phase 1 先下探，满足力/高度条件后，
    # Phase 1.5 等力和位置波动收敛，正式数据从 Phase 2 才开始记录。
    settle_time = 2.0                  # 初始关节位姿到达后的静置时间, s
    contact_force_threshold = 0.3      # 判定触碰表面的最小接触力, N
    contact_z_tolerance = 0.001        # 接触高度判定容差, m
    contact_settle_time = 1.0          # 接触稳定滑动窗口长度, s
    contact_stable_z_std = 0.0003      # 稳定窗口内 z 标准差阈值, m
    contact_stable_force_std = 0.05    # 稳定窗口内力标准差阈值, N
    contact_stable_force_slope = 0.20  # 稳定窗口首尾力变化率阈值, N/s
    contact_force_blend_time = 0.5     # 接触后力控从 0 平滑引入的时间, s
    contact_delta_dot_filter_tau = 0.05  # 压入速度低通时间常数, s
    contact_delta_dot_limit = 0.02       # 压入速度限幅, m/s

    # 接触调整阶段，不写入实验数据；用于滤掉刚接触瞬态震荡。
    adjustment_min_time = 1.0          # 调整阶段最短持续时间, s
    adjustment_timeout = 8.0           # 调整阶段最长等待时间, s
    adjustment_stable_window = 0.8     # 调整稳定统计窗口, s
    adjustment_force_error = 0.15      # 允许的稳态力误差, N
    adjustment_pos_std = 0.0005        # 允许的位置误差标准差, m

    # Phase 3 退回初始关节角的超时；退回使用小步关节位置命令并打印进度。
    retreat_timeout = 25.0

    # z-only 仲裁下的 continuous_force_margin 默认参数。
    # x/y 已由 robot_interface 固定 alpha=1 严格位置跟踪；这里的 alpha 只作用
    # 于 z/压入深度，因此默认允许更低的 z 向位置权重以改善力跟踪。
    continuous_safe_tracking_alpha = 0.26
    continuous_safe_tracking_extra = 0.03
    continuous_force_balance_alpha = 0.26
    continuous_force_guard_alpha = 0.42
    continuous_low_force_guard_alpha = 0.12
    continuous_alpha_min = 0.10
    continuous_alpha_max = 0.46
    continuous_risk_margin_start = 0.20
    continuous_risk_margin_full = 0.04
    continuous_smooth_tau = 0.60
    continuous_stiffness_alpha_enabled = False
    continuous_stiffness_low_threshold = 250.0
    continuous_stiffness_high_threshold = 1000.0
    continuous_stiffness_low_alpha = 0.30
    continuous_stiffness_high_alpha = 0.75
    continuous_stiffness_blend = 0.0


def apply_benchmark_config(cfg, benchmark):
    """按命令行 benchmark 覆盖实验条件。

    default 保持原始力-位一致扫描；force_margin_challenge 刻意使用固定 z
    参考和强刚度变化，制造 fixed alpha 难以兼顾的力/位置冲突。stiffness_alpha_showcase
    进一步采用软/硬环境交替，用来检验 continuous_force_margin 是否能在稳态
    扫描中随估计刚度改变 z 向 alpha。
    """
    if benchmark == "default":
        return
    if benchmark not in ("force_margin_challenge", "stiffness_alpha_showcase"):
        raise ValueError(f"unknown benchmark: {benchmark}")

    if benchmark == "force_margin_challenge":
        cfg.force_consistent_scan_z = False
        cfg.scan_z = 0.298
        cfg.scan_vx = 0.0022
        cfg.F_desired = 1.0
        cfg.F_min = 0.70
        cfg.F_max = 1.32
        cfg.stiffness_zones = [
            (0.40, 0.418, 720, 8),
            (0.418, 0.444, 230, 4),
            (0.444, 0.462, 1150, 14),
            (0.462, 0.480, 420, 6),
        ]
        cfg.adjustment_force_error = 0.25
        cfg.adjustment_timeout = 5.0
        cfg.continuous_safe_tracking_alpha = 0.52
        cfg.continuous_safe_tracking_extra = 0.10
        cfg.continuous_force_balance_alpha = 0.42
        cfg.continuous_force_guard_alpha = 0.62
        cfg.continuous_low_force_guard_alpha = 0.16
        cfg.continuous_alpha_min = 0.12
        cfg.continuous_alpha_max = 0.68
        cfg.continuous_risk_margin_start = 0.24
        cfg.continuous_risk_margin_full = 0.04
        cfg.continuous_smooth_tau = 0.55
        return

    cfg.force_consistent_scan_z = False
    cfg.scan_z = 0.298
    cfg.scan_vx = 0.0018
    cfg.F_desired = 1.0
    cfg.F_min = 0.66
    cfg.F_max = 1.38
    cfg.stiffness_zones = [
        (0.40, 0.418, 260, 4),
        (0.418, 0.440, 780, 9),
        (0.440, 0.460, 220, 4),
        (0.460, 0.480, 1050, 12),
    ]
    cfg.adjustment_force_error = 0.25
    cfg.adjustment_timeout = 5.0
    cfg.continuous_safe_tracking_alpha = 0.42
    cfg.continuous_safe_tracking_extra = 0.08
    cfg.continuous_force_balance_alpha = 0.42
    cfg.continuous_force_guard_alpha = 0.68
    cfg.continuous_low_force_guard_alpha = 0.22
    cfg.continuous_alpha_min = 0.16
    cfg.continuous_alpha_max = 0.72
    cfg.continuous_risk_margin_start = 0.26
    cfg.continuous_risk_margin_full = 0.05
    cfg.continuous_smooth_tau = 0.25
    cfg.continuous_stiffness_alpha_enabled = True
    cfg.continuous_stiffness_low_threshold = 260.0
    cfg.continuous_stiffness_high_threshold = 760.0
    cfg.continuous_stiffness_low_alpha = 0.26
    cfg.continuous_stiffness_high_alpha = 0.62
    cfg.continuous_stiffness_blend = 0.86


def contact_delta_from_surface(cfg, z):
    """以 approach_z 作为虚拟表面高度，计算单向压入量 δ。

    z >= approach_z 表示未接触，返回 0；z < approach_z 表示压入表面，
    返回 approach_z - z。该函数是虚拟环境力计算的唯一压入量来源。
    """
    return max(0.0, cfg.approach_z - z)


def scan_z_reference(cfg, venv, x):
    """返回当前 x 位置的扫描 z 参考。

    在虚拟环境已知时，用 Kelvin-Voigt 静态关系 F=K(x)delta 反推
    delta_ref=F_desired/K(x)，让位置目标和恒力目标物理一致。若关闭该模式
    或没有虚拟环境，则退回固定 cfg.scan_z。
    """
    if not cfg.force_consistent_scan_z or venv is None:
        return float(cfg.scan_z)
    K_ref, _ = venv.get_stiffness(float(x))
    K_ref = max(float(K_ref), cfg.scan_z_stiffness_floor)
    z_ref = cfg.approach_z - cfg.F_desired / K_ref
    return float(np.clip(z_ref, cfg.scan_z_min, cfg.scan_z_max))


def run_trial(robot, kin_tool, kin_flange,
              cfg, ctrl, sched, est, venv, logger, trial_id):
    """运行一次完整 no-RCM 试验。

    Phase 1: 位置主导接近表面；
    Phase 2: 使用在线 alpha 和力位控制器扫描；
    Phase 3: 设置 retreat 状态并回到初始关节角。

    `ctrl.compute_control(...)` 的接口在 ARE 和 Pareto 迭代模式下完全相同，
    因此这里不需要感知具体控制器内部求解方法。
    """
    rate = rospy.Rate(cfg.ctrl_rate)
    dt = cfg.dt

    # σ_f 是力误差泄漏积分状态；integ_euler 是姿态 PD+I 的积分项。
    sigma_f_int = LeakyIntegrator(eps=cfg.eps_f, dt=dt, dim=3)
    integ_euler = np.zeros(3)

    # 保存上一控制周期的物理量，用于计算变化率输入。
    prev_ef = 0.0; prev_er = 0.0; prev_K = 500.0

    est.reset()
    sched.reset()
    sigma_f_int.reset()

    rospy.loginfo(f"[Trial {trial_id}] {sched.name} (NO-RCM)")

    safe_move_to_joint_position(robot, INIT_JOINTS)
    rospy.sleep(cfg.settle_time)
    rs = update_robot_state(kin_tool, kin_flange)
    ref_euler_fixed = rs["tool_rotation_euler"].copy()
    rospy.loginfo(f"  Tool at: {rs['tool_position']}, "
                  f"euler: {ref_euler_fixed}")

    # ---- Phase 1: 接近 ----
    # 接近阶段固定 alpha=1.0，以位置控制为主；力误差置零，避免未接触时
    # 因力目标造成不必要的下压命令。
    rospy.loginfo("  Phase 1: Approaching...")
    z_contact = None
    t0 = rospy.Time.now().to_sec()
    last_approach_time = t0
    approach_speed = cfg.approach_speed
    z_ref_approach = rs["tool_position"][2]
    xy_ref_approach = np.array([cfg.scan_start_x, cfg.scan_y])
    contact_z_threshold = scan_z_reference(cfg, venv, cfg.scan_start_x)
    approach_timeout = max(
        20.0,
        1.5 * max(0.0, z_ref_approach - contact_z_threshold) / approach_speed + 5.0,
    )
    approach_count = 0
    rospy.loginfo(
        f"  Approach target: z<={contact_z_threshold:.4f}m or |Fz|>0.300N; "
        f"surface_z={cfg.approach_z:.4f}m, scan_z0={contact_z_threshold:.4f}m, "
        f"xy_ref=[{xy_ref_approach[0]:.4f}, {xy_ref_approach[1]:.4f}], "
        f"z0={z_ref_approach:.4f}m, speed={approach_speed:.4f}m/s, "
        f"timeout={approach_timeout:.1f}s"
    )
    surface_touched = False
    settle_start = None
    settle_forces = []
    settle_z_values = []
    settle_window = max(3, int(cfg.contact_settle_time * cfg.ctrl_rate))

    while not rospy.is_shutdown() and z_contact is None:
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 当前版本默认使用虚拟接触环境。未接触时 F_z=0。
        F_z = 0.0
        if venv:
            F_z = venv.compute_force(
                tp[0], contact_delta_from_surface(cfg, tp[2]), -tv[2]
            )

        if not surface_touched and abs(F_z) > cfg.contact_force_threshold:
            surface_touched = True
            rospy.loginfo(
                f"  Surface contact detected at z={tp[2]:.4f}, F={F_z:.3f}N"
            )

        now = rospy.Time.now().to_sec()
        dt_approach = now - last_approach_time
        if dt_approach <= 0.0 or not np.isfinite(dt_approach):
            dt_approach = dt
        dt_approach = min(max(dt_approach, dt), 0.1)
        last_approach_time = now

        # 期望 z 缓慢下降；一旦触碰表面，不再让参考点继续穿过扫描高度。
        z_ref_approach -= approach_speed * dt_approach
        if surface_touched:
            z_ref_approach = max(z_ref_approach, contact_z_threshold)
        xdot_ref = np.array([0.0, 0.0, -approach_speed])
        if surface_touched and z_ref_approach <= contact_z_threshold:
            xdot_ref = np.zeros(3)
        x_ref_p1 = np.array([
            xy_ref_approach[0],
            xy_ref_approach[1],
            z_ref_approach,
        ])
        # 接近阶段只构造位置/速度误差；e_f 和 σ_f 均不参与控制。
        e_r1 = tp - x_ref_p1
        e_r2 = tv - xdot_ref
        e_f = np.zeros(3)
        sigma_f = sigma_f_int.get()

        # 仍走统一的 no-RCM 力矩装配函数，保持接口和扫描阶段一致。
        tau, _, _, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f, sigma_f,
            alpha=1.0, K_e_hat=500,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        robot.exec_torque_cmd(tau)
        rate.sleep()
        approach_count += 1

        if approach_count % 10 == 0:
            rospy.loginfo(
                f"  approaching | z={tp[2]:.4f}m, z_ref={z_ref_approach:.4f}m, "
                f"target_z<={contact_z_threshold:.4f}m, F={F_z:.3f}N, "
                f"vz={tv[2]:+.4f}m/s, touched={surface_touched}, "
                f"dt={dt_approach*1000:.1f}ms"
            )

        at_contact_height = tp[2] <= contact_z_threshold + cfg.contact_z_tolerance
        if surface_touched and at_contact_height:
            # 接触稳定判定: 到达接触高度后，统计滑动窗口内的力波动、
            # z 位置波动和力变化斜率，三者都足够小才进入扫描阶段。
            if settle_start is None:
                settle_start = rospy.Time.now().to_sec()
                settle_forces = []
                settle_z_values = []
                rospy.loginfo(
                    f"  Contact height reached at z={tp[2]:.4f}; settling..."
                )
            settle_forces.append(abs(F_z))
            settle_z_values.append(tp[2])
            if len(settle_forces) > settle_window:
                settle_forces.pop(0)
                settle_z_values.pop(0)

            settled_time = rospy.Time.now().to_sec() - settle_start
            force_std = float(np.std(settle_forces)) if len(settle_forces) > 1 else float("inf")
            z_std = float(np.std(settle_z_values)) if len(settle_z_values) > 1 else float("inf")
            z_err = abs(tp[2] - contact_z_threshold)
            force_slope = 0.0
            if len(settle_forces) > 1:
                force_slope = abs(settle_forces[-1] - settle_forces[0]) / max(
                    (len(settle_forces) - 1) * dt, dt
                )
            stable = (
                settled_time >= cfg.contact_settle_time
                and z_err <= cfg.contact_z_tolerance
                and z_std <= cfg.contact_stable_z_std
                and force_std <= cfg.contact_stable_force_std
                and force_slope <= cfg.contact_stable_force_slope
            )
            if approach_count % 10 == 0:
                rospy.loginfo(
                    f"  settling | t={settled_time:.2f}s, z_err={z_err*1000:.2f}mm, "
                    f"z_std={z_std*1000:.3f}mm, F={abs(F_z):.3f}N, F_std={force_std:.3f}, "
                    f"F_slope={force_slope:.3f}N/s"
                )
            if stable:
                z_contact = tp[2]
                rospy.loginfo(
                    f"  Contact settled at z={z_contact:.4f}, F={abs(F_z):.3f}N, "
                    f"z_err={z_err*1000:.2f}mm, z_std={z_std*1000:.3f}mm"
                )
                break
        else:
            settle_start = None
            settle_forces = []
            settle_z_values = []

        if rospy.Time.now().to_sec() - t0 > approach_timeout:
            rospy.logwarn(
                f"  Approach timeout "
                f"(z={tp[2]:.4f}, z_ref={z_ref_approach:.4f}, "
                f"target_z<={contact_z_threshold:.4f}, F={F_z:.3f}N, "
                f"touched={surface_touched}, "
                f"timeout={approach_timeout:.1f}s)"
            )
            return False

    if z_contact is None:
        z_contact = contact_z_threshold

    sigma_f_int.reset()
    integ_euler = np.zeros(3)
    rospy.loginfo(
        f"  Contact completed at z={z_contact:.4f}; "
        f"scan_z_ref0={contact_z_threshold:.4f}, surface_z={cfg.approach_z:.4f}"
    )

    # ---- Phase 1.5: 接触调整 ----
    # 刚接触后先在扫描起点保持 x 不动，让力控、刚度估计和姿态积分项收敛。
    # 这一段只执行控制，不写入主 logger；正式扫描的数据从 Phase 2 的 t=0 开始。
    rospy.loginfo("  Phase 1.5: Contact adjustment (not logged)...")
    x_cur = cfg.scan_start_x
    adjust_t0 = time.time()
    adjust_last_log = 0.0
    adjust_forces = []
    adjust_z_values = []
    adjust_pos_errors = []
    adjust_window = max(3, int(cfg.adjustment_stable_window * cfg.ctrl_rate))
    delta_dot_est = ContactDeltaDotEstimator(
        tau=cfg.contact_delta_dot_filter_tau,
        limit=cfg.contact_delta_dot_limit,
    )

    while not rospy.is_shutdown():
        adjust_t = time.time() - adjust_t0
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        delta = contact_delta_from_surface(cfg, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, dt, raw_delta_dot=-tv[2]
        )
        F_z = venv.compute_force(tp[0], delta, delta_dot) if venv else 0.0
        K_hat, B_hat = est.update(abs(F_z), delta, delta_dot)

        z_scan_ref = scan_z_reference(cfg, venv, x_cur)
        x_ref = np.array([x_cur, cfg.scan_y, z_scan_ref])
        xdot_ref = np.zeros(3)
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        e_r1 = tp - x_ref
        e_r2 = tv - xdot_ref
        e_f_vec = F_meas - F_des
        force_blend = min(1.0, max(0.0, adjust_t / cfg.contact_force_blend_time))
        e_f_vec_ctrl = force_blend * e_f_vec
        sigma_f = sigma_f_int.update(e_f_vec_ctrl)

        F_actual = abs(F_z)
        e_f_scalar = F_actual - cfg.F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt; prev_ef = e_f_scalar
        e_r_scalar = abs(tp[2] - z_scan_ref)
        de_r = (e_r_scalar - prev_er) / dt; prev_er = e_r_scalar
        dK = (K_hat - prev_K) / dt; prev_K = K_hat

        if isinstance(sched, FixedAlphaScheduler):
            alpha = sched.compute()
        else:
            alpha = sched.compute(
                F_norm=abs(F_z), e_f=e_f_scalar, K_hat=K_hat,
                e_r=e_r_scalar, z_vel=tv[2],
                de_f=e_f_dot, dK=dK, de_r=de_r,
                F_desired=cfg.F_desired, F_min=cfg.F_min, F_max=cfg.F_max,
                tracking_boost_enabled=True,
            )

        tau, _, _, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f_vec_ctrl, sigma_f,
            alpha=alpha, K_e_hat=K_hat,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        robot.exec_torque_cmd(tau)

        adjust_forces.append(F_actual)
        adjust_z_values.append(tp[2])
        adjust_pos_errors.append(np.linalg.norm(tp - x_ref))
        if len(adjust_forces) > adjust_window:
            adjust_forces.pop(0)
            adjust_z_values.pop(0)
            adjust_pos_errors.pop(0)

        force_std = float(np.std(adjust_forces)) if len(adjust_forces) > 1 else float("inf")
        z_std = float(np.std(adjust_z_values)) if len(adjust_z_values) > 1 else float("inf")
        pos_std = float(np.std(adjust_pos_errors)) if len(adjust_pos_errors) > 1 else float("inf")
        force_slope = 0.0
        if len(adjust_forces) > 1:
            force_slope = abs(adjust_forces[-1] - adjust_forces[0]) / max(
                (len(adjust_forces) - 1) * dt, dt
            )
        stable = (
            adjust_t >= cfg.adjustment_min_time
            and force_blend >= 1.0
            and abs(F_actual - cfg.F_desired) <= cfg.adjustment_force_error
            and force_std <= cfg.contact_stable_force_std
            and force_slope <= cfg.contact_stable_force_slope
            and z_std <= cfg.contact_stable_z_std
            and pos_std <= cfg.adjustment_pos_std
        )

        if adjust_t - adjust_last_log >= 0.5:
            adjust_last_log = adjust_t
            rospy.loginfo(
                f"  adjusting | t={adjust_t:.2f}s, F={F_actual:.3f}N, "
                f"F_std={force_std:.3f}, F_slope={force_slope:.3f}N/s, "
                f"z_std={z_std*1000:.3f}mm, pos_std={pos_std*1000:.3f}mm, "
                f"blend={force_blend:.2f}, K={K_hat:.1f}, B={B_hat:.2f}"
            )

        if stable:
            rospy.loginfo(
                f"  Adjustment settled after {adjust_t:.2f}s; "
                f"discarded {len(adjust_forces)} recent adjustment samples from scan log."
            )
            break
        if adjust_t >= cfg.adjustment_timeout:
            rospy.logwarn(
                f"  Adjustment timeout after {adjust_t:.2f}s; start scan without "
                f"logging the contact transient (F_std={force_std:.3f}, "
                f"F_slope={force_slope:.3f}N/s, z_std={z_std*1000:.3f}mm)."
            )
            break
        rate.sleep()

    # ---- Phase 2: 恒力扫描 ----
    # 扫描阶段 x 方向按 scan_vx 匀速推进，z 方向由力位控制器自动调节，
    # 目标是在跟踪扫描路径的同时把接触反力维持在 F_desired 附近。
    rospy.loginfo("  Phase 2: Scanning...")
    t_scan = time.time()

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        t = time.time() - t_scan
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 计算当前虚拟接触力，并用同一压入量送入环境估计器。
        delta = contact_delta_from_surface(cfg, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, dt, raw_delta_dot=-tv[2]
        )
        if venv:
            K_env_true, B_env_true = venv.get_stiffness(tp[0])
        else:
            K_env_true, B_env_true = 0.0, 0.0
        F_z = 0.0
        if venv:
            F_z = venv.compute_force(tp[0], delta, delta_dot)

        K_hat, B_hat = est.update(
            abs(F_z),
            delta,
            delta_dot,
        )

        # 当前期望 tool 位姿: x 随时间增长，y/z 固定。
        z_scan_ref = scan_z_reference(cfg, venv, x_cur)
        x_ref = np.array([x_cur, cfg.scan_y, z_scan_ref])
        xdot_ref = np.array([cfg.scan_vx, 0, 0])

        # 力目标: +z 方向为上 (接触反力方向)
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        # 控制器状态量:
        # e_r1/e_r2 是位置和速度误差；e_f_vec 是三轴力误差。
        # z 轴力误差符号采用 F_meas - F_des，与论文风险指标一致。
        e_r1 = tp - x_ref
        e_r2 = tv - xdot_ref
        e_f_vec = F_meas - F_des

        # 接触调整阶段已经完成力控渐入，正式扫描从完整力控开始。
        force_blend = 1.0
        e_f_vec_ctrl = force_blend * e_f_vec
        sigma_f = sigma_f_int.update(e_f_vec_ctrl)

        F_actual = abs(F_z)
        F_desired = cfg.F_desired
        e_f_scalar = F_actual - F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt; prev_ef = e_f_scalar
        e_r_scalar = abs(tp[2] - z_scan_ref)
        de_r = (e_r_scalar - prev_er) / dt; prev_er = e_r_scalar
        dK = (K_hat - prev_K) / dt; prev_K = K_hat

        # alpha 仲裁器输入统一为可观测物理量。FixedAlphaScheduler 作为对照组，
        # 不需要阶段检测和风险计算。
        if isinstance(sched, FixedAlphaScheduler):
            alpha = sched.compute()
            phase_val = -1
        else:
            alpha = sched.compute(
                F_norm=abs(F_z), e_f=e_f_scalar, K_hat=K_hat,
                e_r=e_r_scalar, z_vel=tv[2],
                de_f=e_f_dot, dK=dK, de_r=de_r,
                F_desired=cfg.F_desired, F_min=cfg.F_min, F_max=cfg.F_max,
                tracking_boost_enabled=True,
            )
            phase_val = sched.phase_detector.phase.value

        # no-RCM 力矩装配: ctrl 产生 tool 端笛卡尔力，robot_interface 再用
        # J_tool^T 映射到 7 维关节力矩，并叠加姿态保持项。
        tau, u_tool, K_eff, integ_euler, error = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f_vec_ctrl, sigma_f,
            alpha=alpha, K_e_hat=K_hat,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        robot.exec_torque_cmd(tau)
        x_cur += cfg.scan_vx * dt

        pos_tool = tp.copy()
        pos_tool_des = x_ref.copy()
        pos_err = pos_tool - pos_tool_des
        pos_err_norm = np.linalg.norm(pos_err)
        F_err = F_actual - F_desired

        if logger.count % 10 == 0:
            rospy.loginfo(
                f"  t={t:5.2f}s | "
                f"tool=[{pos_tool[0]*1000:6.2f},{pos_tool[1]*1000:6.2f},{pos_tool[2]*1000:6.2f}]mm | "
                f"des=[{pos_tool_des[0]*1000:6.2f},{pos_tool_des[1]*1000:6.2f},{pos_tool_des[2]*1000:6.2f}]mm | "
                f"err={pos_err_norm*1000:5.2f}mm | "
                f"F={F_actual:.3f}N (des={F_desired:.3f}, err={F_err:+.3f}) | "
                f"K={K_hat:7.1f} B={B_hat:5.2f} | "
                f"e_f={e_f_scalar:+.3f} e_r={e_r_scalar*1000:5.2f}mm | "
                f"alpha={alpha:.2f} phase={phase_val}"
            )

        # 保存所有关键状态，后续可直接从 npz 统计力 RMSE、位置误差和 alpha 曲线。
        logger.log(
            t=t, pos=pos_tool,
            pos_des=pos_tool_des,
            pos_err=pos_err,
            F_measured=F_actual, F_desired=F_desired,
            F_err=F_err,
            wrench=np.zeros(6),
            force_source='virtual' if venv else 'none',
            sensor_available=0,
            e_f=e_f_scalar, e_r=e_r_scalar,
            sigma_f_norm=np.linalg.norm(sigma_f),
            e_r1_norm=np.linalg.norm(e_r1),
            alpha=alpha, K_hat=K_hat,
            K_hat_raw=getattr(est, "K_observed", K_hat),
            B_hat=B_hat,
            delta=delta, delta_dot=delta_dot,
            delta_dot_raw=delta_dot_raw, delta_dot_fd=delta_dot_fd,
            K_env_true=K_env_true, B_env_true=B_env_true,
            K_eff=K_eff,
            x_desired=x_cur,
            error_rcm=0.0, error_track=error[1],
            arbitration_strategy=sched.name,
            u_norm=np.linalg.norm(tau),
            phase=phase_val,
        )
        rate.sleep()

    rospy.loginfo("  Phase 3: Retreating...")
    # 退回阶段通知调度器进入 RETREAT，使 alpha 回到位置优先。
    if hasattr(sched, 'set_retreat'):
        sched.set_retreat(True)
    retreat_ok = safe_move_to_joint_position(
        robot, INIT_JOINTS,
        timeout=cfg.retreat_timeout,
        prefer_streaming=True,
        log_prefix="retreat",
    )
    if not retreat_ok:
        rospy.logwarn("  Retreat did not reach INIT_JOINTS within timeout; leaving trial safely.")
    rospy.loginfo(f"  Done. {logger.count} samples.")
    return True


STRATEGIES = {
    # 策略表保持原项目风格: 命令行传入 strategy 名称即可实例化调度器。
    # 当前推荐组合: continuous_force_margin + --controller-mode pareto_iter。
    'fixed_08': lambda cfg: FixedAlphaScheduler(0.8),
    'fixed_05': lambda cfg: FixedAlphaScheduler(0.5),
    'fixed_02': lambda cfg: FixedAlphaScheduler(0.2),
    'coop_fuzzy': lambda cfg: PhaseAwareFuzzyAlphaScheduler(dt=0.01),
    'force_margin': lambda cfg: ForceMarginFuzzyAlphaScheduler(
        dt=0.01,
        F_min=cfg.F_min,
        F_max=cfg.F_max,
        F_desired=cfg.F_desired,
    ),
    'continuous_force_margin': lambda cfg: ContinuousForceMarginFuzzyAlphaScheduler(
        dt=0.01,
        F_min=cfg.F_min,
        F_max=cfg.F_max,
        F_desired=cfg.F_desired,
        safe_tracking_alpha=cfg.continuous_safe_tracking_alpha,
        safe_tracking_extra=cfg.continuous_safe_tracking_extra,
        force_balance_alpha=cfg.continuous_force_balance_alpha,
        force_guard_alpha=cfg.continuous_force_guard_alpha,
        low_force_guard_alpha=cfg.continuous_low_force_guard_alpha,
        alpha_min=cfg.continuous_alpha_min,
        alpha_max=cfg.continuous_alpha_max,
        risk_margin_start=cfg.continuous_risk_margin_start,
        risk_margin_full=cfg.continuous_risk_margin_full,
        smooth_tau=cfg.continuous_smooth_tau,
        stiffness_alpha_enabled=cfg.continuous_stiffness_alpha_enabled,
        stiffness_low_threshold=cfg.continuous_stiffness_low_threshold,
        stiffness_high_threshold=cfg.continuous_stiffness_high_threshold,
        stiffness_low_alpha=cfg.continuous_stiffness_low_alpha,
        stiffness_high_alpha=cfg.continuous_stiffness_high_alpha,
        stiffness_blend=cfg.continuous_stiffness_blend,
    ),
    'online_priority': lambda cfg: OnlinePriorityAdaptationAlphaScheduler(
        dt=0.01,
        F_min=cfg.F_min,
        F_max=cfg.F_max,
        F_desired=cfg.F_desired,
    ),
}


def auto_plot_results(result_dir, no_show=False):
    """实验结束后自动生成最近结果图；多文件时额外生成策略对照图。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.abspath(result_dir)
    latest_script = os.path.join(script_dir, "plot_latest_result.py")
    compare_script = os.path.join(script_dir, "plot_arbitration_compare.py")
    common_args = ["--input", result_dir]
    if no_show:
        common_args.append("--no-show")

    npz_files = [
        name for name in os.listdir(result_dir)
        if name.endswith(".npz") and os.path.isfile(os.path.join(result_dir, name))
    ]
    if not npz_files:
        rospy.logwarn(f"No npz result files found for plotting: {result_dir}")
        return

    for script in (latest_script,):
        try:
            subprocess.run([sys.executable, script] + common_args, check=False)
        except Exception as exc:
            rospy.logwarn(f"Auto plot failed for {script}: {exc}")

    if len(npz_files) > 1:
        try:
            subprocess.run([sys.executable, compare_script] + common_args, check=False)
        except Exception as exc:
            rospy.logwarn(f"Auto arbitration comparison failed: {exc}")


def main():
    """命令行入口。

    当前 0603 版本默认使用:
      --strategy continuous_force_margin --controller-mode pareto_iter
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='continuous_force_margin')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--use-virtual-env', dest='use_virtual_env',
                    action='store_true', default=True,
                    help='启用虚拟接触环境 (默认启用)')
    ap.add_argument('--no-virtual-env', dest='use_virtual_env',
                    action='store_false',
                    help='关闭虚拟接触环境')
    ap.add_argument(
        '--output-dir',
        default='/home/done/USTC/LJJ/franka_ros2_ws/results',
        help='实验结果根目录，默认保存到工作空间级 results，便于绘图脚本自动查找',
    )
    ap.add_argument('--gains-file', default=None)
    ap.add_argument('--controller-mode', default='are',
                    choices=['are', 'pareto_iter'],
                    help='are: 原 4D ARE 查表; pareto_iter: Algorithm 2 迭代 P1/P2')
    ap.add_argument('--no-auto-plot', action='store_true',
                    help='实验结束后不自动调用绘图脚本')
    ap.add_argument('--plot-no-show', action='store_true',
                    help='自动绘图时只保存图像，不弹出 matplotlib 窗口')
    ap.add_argument(
        '--benchmark',
        default='default',
        choices=['default', 'force_margin_challenge', 'stiffness_alpha_showcase'],
        help='实验场景: default 为原始力位一致扫描; force_margin_challenge 为固定 z + 强刚度变化压力测试; stiffness_alpha_showcase 为刚度敏感 alpha 展示场景',
    )
    args = ap.parse_args()

    rospy.init_node("coop_gt_no_rcm")
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Cooperative Game Force-Position Experiment (NO RCM)")
    rospy.loginfo("=" * 60)

    robot = PandaArm()
    kin_tool = PandaKinematics(robot, "panda_link11")
    kin_flange = PandaKinematics(robot, "panda_link8")
    rospy.sleep(1.0)

    cfg = Config()
    apply_benchmark_config(cfg, args.benchmark)
    rospy.loginfo(
        f"Benchmark: {args.benchmark}; F_des={cfg.F_desired:.3f}N, "
        f"F_bounds=[{cfg.F_min:.3f}, {cfg.F_max:.3f}]N, "
        f"scan_vx={cfg.scan_vx:.4f}m/s, force_consistent_z={cfg.force_consistent_scan_z}, "
        f"stiffness_zones={cfg.stiffness_zones}"
    )
    rospy.loginfo(
        f"Continuous z-alpha params: safe={cfg.continuous_safe_tracking_alpha:.2f}, "
        f"balance={cfg.continuous_force_balance_alpha:.2f}, "
        f"high_guard={cfg.continuous_force_guard_alpha:.2f}, "
        f"low_guard={cfg.continuous_low_force_guard_alpha:.2f}, "
        f"range=[{cfg.continuous_alpha_min:.2f}, {cfg.continuous_alpha_max:.2f}], "
        f"tau={cfg.continuous_smooth_tau:.2f}s, "
        f"K_alpha_enabled={cfg.continuous_stiffness_alpha_enabled}, "
        f"K_range=[{cfg.continuous_stiffness_low_threshold:.1f}, "
        f"{cfg.continuous_stiffness_high_threshold:.1f}]N/m, "
        f"alpha_K=[{cfg.continuous_stiffness_low_alpha:.2f}, "
        f"{cfg.continuous_stiffness_high_alpha:.2f}], "
        f"K_blend={cfg.continuous_stiffness_blend:.2f}"
    )

    # control_mode='are' 使用原始 4D ARE 查表；
    # control_mode='pareto_iter' 使用 Algorithm 2 迭代生成同形状增益表。
    ctrl = CooperativeGameController(control_mode=args.controller_mode)
    rospy.loginfo(f"Controller mode: {args.controller_mode}")

    need_precompute = True
    if args.gains_file and os.path.exists(args.gains_file):
        ctrl.load_gains(args.gains_file)
        need_precompute = not ctrl.has_precomputed_gains()
        if need_precompute:
            rospy.logwarn(
                f"Gains file {args.gains_file} does not contain "
                f"{args.controller_mode} data; recomputing."
            )

    if need_precompute:
        # K_e 网格覆盖实验刚度区间和常见估计值，RLS 输出落在网格之间时
        # get_gain 会双线性插值，避免增益突变。
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
        gains_name = 'coop_gains_no_rcm.npy'
        if args.controller_mode == 'pareto_iter':
            gains_name = 'coop_gains_no_rcm_pareto_iter.npy'
        ctrl.save_gains(os.path.join(args.output_dir, gains_name))

    venv = VirtualStiffnessSurface(cfg.stiffness_zones) \
        if args.use_virtual_env else None
    rospy.loginfo(
        f"Virtual environment: {'enabled' if venv is not None else 'disabled'}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = os.path.join(args.output_dir, f"no_rcm_{stamp}")
    os.makedirs(odir, exist_ok=True)

    names = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]
    for sn in names:
        s = STRATEGIES[sn](cfg)
        rospy.loginfo(f"\n{'='*50}\n  Strategy: {s.name}\n{'='*50}")
        for t in range(args.trials):
            if rospy.is_shutdown():
                break
            lg = DataLogger()
            est = EnvironmentEstimator()
            s.reset()
            ok = run_trial(
                robot, kin_tool, kin_flange,
                cfg, ctrl, s, est, venv, lg, t,
            )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")
    if not args.no_auto_plot:
        auto_plot_results(odir, no_show=args.plot_no_show)


if __name__ == '__main__':
    main()
