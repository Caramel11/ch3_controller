#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序 A: RCM 约束下的合作博弈力-位扫描实验
=============================================

控制链路:
  泄漏积分更新 → ARE/Pareto 增益查表 → u_tool → RCM 杠杆 → τ

数学模型:
  M·ẍ + C·ẋ + K_v·(x−x_r) = u + f_ext      (含 K_v 基线刚度)
  ė_r1 = e_r2 − ε_r · e_r1                  (泄漏积分器)
  ė_f  = -(K_e − B_e·C/M)·e_r2 − B_e/M·e_f − B_e/M·u
  σ̇_f  = e_f − ε_f · σ_f                    (泄漏积分器)

用法:
  Terminal 1: roslaunch panda_simulator simulation.launch
  Terminal 2: python run_with_rcm.py --strategy continuous_force_margin --controller-mode pareto_iter
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
from src.utils import (
    VirtualStiffnessSurface, ForceSensorInput, DataLogger,
    ContactDeltaDotEstimator, AlphaCommandLimiter,
)
from src.leaky_integrator import LeakyIntegrator
from src.robot_interface import (
    update_robot_state, safe_move_to_joint_position,
    compute_torque_with_rcm, INIT_JOINTS,
    compute_position_rcm,
)


# ================================================================
# 实验配置
# ================================================================
class Config:
    # 扫描几何参数，单位均为 m；with-RCM 中这些量描述 tool tip。
    # Phase 2 中 x 从 scan_start_x 走到 scan_end_x，RCM 约束由 trocar_position 给出。
    scan_start_x = 0.30      # 扫描起点 tool-tip x
    scan_end_x   = 0.35      # 扫描终点 tool-tip x
    scan_y       = 0.0       # 扫描线 y 坐标
    scan_z       = 0.2       # 名义扫描 z，高度会在接触后由 scan_z_ref 自适应修正
    approach_z   = 0.05      # 接近阶段虚拟/名义表面高度
    scan_vx      = 0.0004    # Phase 2 沿 x 的扫描速度, m/s

    # 力目标与 alpha 调度边界，单位 N。
    F_desired    = 0.5       # 期望法向接触力
    F_min        = 0.3       # 低力边界，低于该值更偏向力控补偿
    F_max        = 1.0       # 高力边界，高于该值更偏向位置/安全
    force_axis   = 2         # 力控制轴，2 表示 z 轴

    # RCM 几何参数。trocar_position 是基坐标系下的穿刺点位置；
    # tool_length 是 flange 到 tool tip 的等效工具长度。
    trocar_position = np.array([0.3, 0, 0.235])
    tool_length = 0.525

    # 虚拟变刚度区: 每项为 (x_start, x_end, K_e, B_e)。
    # K_e 单位 N/m，B_e 单位 Ns/m，用于 Gazebo/mock 接触力与估计器对照。
    # stiffness_zones = [
    #     (0.25, 0.29, 500,  5),    # 软
    #     (0.29, 0.33, 5000, 50),   # 硬
    #     (0.33, 0.37, 80,   2),    # 极软
    #     (0.37, 0.41, 500,  5),    # 软
    # ]

    stiffness_zones = [
        (0.30, 0.325, 250, 4),    # 低刚度
        (0.325, 0.35, 500, 8),    # 高刚度
    ]

    # 控制频率与泄漏积分器。eps 越大，积分记忆衰减越快。
    ctrl_rate = 100         # 主控制循环频率, Hz
    dt = 1.0 / ctrl_rate
    eps_r = 1.0             # 位置误差积分泄漏系数
    eps_f = 2.0             # 力误差积分泄漏系数

    # Phase 1 接近参数。先低速对齐 x/y，再沿 z 下降；速度会被
    # approach_speed_scale 根据跟踪误差和 RCM 误差继续缩放。
    settle_time = 2.0                  # 初始关节位姿到达后的静置时间, s
    approach_xy_speed = 0.0014         # x/y 对齐参考速度, m/s
    approach_z_speed = 0.00042         # z 下降参考速度, m/s；原 0.00028，适度加快
    approach_min_speed_scale = 0.70    # 接近软边界时的最低速度比例
    approach_xy_tolerance = 0.001      # x/y 对齐完成阈值, m
    approach_force_threshold = 0.3     # 判定接触的最小接触力, N
    approach_z_margin = 0.002          # z 接近目标的安全余量, m
    approach_timeout = 80.0            # 接近阶段基础超时, s，会按起始距离自动放大
    approach_surface_margin = 0.0005   # 虚拟表面判定余量, m
    approach_follow_tolerance = 0.0012 # tool 跟踪误差软限, m
    approach_rcm_soft_limit = 0.0025   # RCM 误差软限，超过后降速, m
    approach_rcm_warn = 0.004          # RCM 误差告警阈值, m
    approach_tau_norm_limit = 3.0      # 接近阶段关节力矩范数限幅, Nm
    approach_retarget_log_interval = 50  # 接近阶段重定向日志间隔, 控制周期数
    approach_start_speed_limit = 0.003 # 接近开始前关节/笛卡尔速度稳定阈值, m/s
    approach_start_settle_hold = 0.5   # 进入接近前需要连续稳定的时间, s
    approach_start_settle_timeout = 8.0  # 等待初始速度稳定的超时, s

    # Phase 1.5/2 接触力与 z_ref 自适应参数。
    scan_contact_bias = 0.0010         # 接触预载压入偏置, m
    scan_z_ref_rate = 0.00008          # scan_z_ref 自适应变化率限幅, m/s
    scan_z_ref_min_offset = -0.0012    # 相对接触平面最深目标偏移, m
    scan_z_ref_max_offset = 0.0004     # 相对接触平面最浅目标偏移, m
    scan_force_deadband = 0.05         # 力误差死区, N
    scan_force_error_limit = 0.35      # 控制用力误差限幅, N
    scan_sigma_force_limit = 0.08      # 力误差积分限幅, Ns
    contact_delta_dot_filter_tau = 0.05  # 压入速度低通时间常数, s
    contact_delta_dot_limit = 0.02       # 压入速度限幅, m/s

    # alpha 与 RCM 安全调度。RCM 误差越接近暂停阈值，alpha 越偏位置/RCM。
    scan_min_alpha_when_rcm_soft = 0.85  # RCM 软区内 alpha 下限上界
    scan_alpha_filter_tau = 0.25         # alpha 一阶平滑时间常数, s
    scan_alpha_rate_limit = 0.8          # alpha 变化率限幅, 1/s
    scan_rcm_recovery_enter = 0.0050     # 进入 RCM 恢复模式阈值, m
    scan_rcm_recovery_exit = 0.0028      # 退出 RCM 恢复模式阈值, m
    scan_rcm_recovery_alpha = 0.95       # RCM 恢复模式 alpha
    scan_rcm_recovery_timeout = 12.0     # RCM 恢复最长持续时间, s
    scan_stop_after_rcm_recovery = True  # 恢复后是否结束当前扫描

    # Phase 2 慢速/暂停/中止阈值。
    scan_timeout = 260.0                 # 正式扫描最长时间, s
    scan_track_slow_error = 0.0040       # 跟踪误差超过后降速, m
    scan_track_pause_error = 0.0070      # 跟踪误差超过后暂停推进, m
    scan_rcm_slow_error = 0.0035         # RCM 误差超过后降速并抬高 alpha, m
    scan_rcm_pause_error = 0.0065        # RCM 误差超过后暂停推进, m
    scan_force_slow = 0.75               # 接触力超过后降速, N
    scan_force_pause = 1.10              # 接触力超过后暂停推进, N
    scan_tau_norm_limit = 2.5            # 扫描阶段关节力矩范数限幅, Nm
    scan_abort_pos_error = 0.0080        # 位置误差硬中止阈值, m
    scan_abort_rcm_error = 0.0075        # RCM 误差硬中止阈值, m
    scan_abort_force = 1.20              # 接触力硬中止阈值, N

    # Phase 1.5 接触调整参数；该阶段不写入主实验数据。
    adjustment_min_time = 1.0            # 调整阶段最短持续时间, s
    adjustment_timeout = 10.0            # 调整阶段最长等待时间, s
    adjustment_stable_window = 0.8       # 稳定统计窗口, s
    adjustment_force_error = 0.15        # 允许的稳态力误差, N
    adjustment_force_std = 0.05          # 力标准差阈值, N
    adjustment_force_slope = 0.20        # 力变化率阈值, N/s
    adjustment_pos_std = 0.0005          # 位置误差标准差阈值, m
    adjustment_rcm_std = 0.00025         # RCM 误差标准差阈值, m
    adjustment_rcm_limit = scan_rcm_slow_error  # 调整完成时允许的 RCM 误差, m
    adjustment_abort_force = max(F_max, scan_abort_force)  # 调整阶段力硬中止阈值, N

    # continuous_force_margin 的 with-RCM 专用默认参数。
    # RCM 场景下高力风险常常伴随穿刺点误差/姿态耦合振荡，因此默认保持较高
    # 位置权重；低力风险时才小幅释放给力控。
    continuous_safe_tracking_alpha = 0.86
    continuous_safe_tracking_extra = 0.04
    continuous_force_balance_alpha = 0.86
    continuous_force_guard_alpha = 0.94
    continuous_low_force_guard_alpha = 0.70
    continuous_alpha_min = 0.05
    continuous_alpha_max = 0.95
    continuous_risk_margin_start = 0.35
    continuous_risk_margin_full = 0.10
    continuous_smooth_tau = 0.16
    continuous_stiffness_alpha_enabled = False
    continuous_stiffness_low_threshold = 250.0
    continuous_stiffness_high_threshold = 1000.0
    continuous_stiffness_low_alpha = 0.45
    continuous_stiffness_high_alpha = 0.90
    continuous_stiffness_blend = 0.0
    continuous_stiffness_zref_enabled = False
    continuous_stiffness_zref_gain = 0.0
    continuous_stiffness_zref_max_scale = 1.0


def apply_benchmark_config(cfg, benchmark):
    """按 benchmark 覆盖 with-RCM 实验环境和仲裁参数。

    default 保持原始 `run_with_rcm.py` 设置；force_margin_rcm_challenge
    用更短扫描、强刚度突变和更紧的力边界制造 fixed alpha 难以同时兼顾
    RCM、力边界和位置跟踪的压力测试。
    """
    if benchmark == "default":
        return
    if benchmark == "stiffness_alpha_showcase":
        cfg.scan_start_x = 0.30
        cfg.scan_end_x = 0.332
        cfg.scan_vx = 0.00045
        cfg.F_desired = 0.52
        cfg.F_min = 0.36
        cfg.F_max = 0.90
        cfg.stiffness_zones = [
            (0.300, 0.308, 260, 4),
            (0.308, 0.316, 780, 10),
            (0.316, 0.324, 190, 3),
            (0.324, 0.332, 980, 12),
        ]

        cfg.scan_z_ref_rate = 0.000060
        cfg.scan_z_ref_min_offset = -0.0017
        cfg.scan_z_ref_max_offset = 0.00050
        cfg.scan_force_deadband = 0.030
        cfg.scan_force_error_limit = 0.32
        cfg.scan_sigma_force_limit = 0.08
        cfg.scan_timeout = 110.0

        cfg.scan_min_alpha_when_rcm_soft = 0.78
        cfg.scan_alpha_filter_tau = 0.18
        cfg.scan_alpha_rate_limit = 1.20
        cfg.scan_track_slow_error = 0.0037
        cfg.scan_track_pause_error = 0.0065
        cfg.scan_rcm_slow_error = 0.0030
        cfg.scan_rcm_pause_error = 0.0055
        cfg.scan_rcm_recovery_enter = 0.0047
        cfg.scan_rcm_recovery_exit = 0.0025
        cfg.scan_rcm_recovery_alpha = 0.96
        cfg.scan_rcm_recovery_timeout = 8.0
        cfg.scan_stop_after_rcm_recovery = False
        cfg.scan_force_slow = 0.90
        cfg.scan_force_pause = 1.55
        cfg.scan_abort_pos_error = 0.0085
        cfg.scan_abort_rcm_error = 0.0072
        cfg.scan_abort_force = 1.75
        cfg.adjustment_abort_force = max(cfg.F_max, cfg.scan_abort_force)

        cfg.adjustment_force_error = 0.12
        cfg.adjustment_timeout = 7.0
        cfg.adjustment_rcm_limit = cfg.scan_rcm_slow_error

        cfg.continuous_safe_tracking_alpha = 0.68
        cfg.continuous_safe_tracking_extra = 0.05
        cfg.continuous_force_balance_alpha = 0.68
        cfg.continuous_force_guard_alpha = 0.90
        cfg.continuous_low_force_guard_alpha = 0.44
        cfg.continuous_alpha_min = 0.38
        cfg.continuous_alpha_max = 0.93
        cfg.continuous_risk_margin_start = 0.30
        cfg.continuous_risk_margin_full = 0.06
        cfg.continuous_smooth_tau = 0.11
        cfg.continuous_stiffness_alpha_enabled = True
        cfg.continuous_stiffness_low_threshold = 240.0
        cfg.continuous_stiffness_high_threshold = 420.0
        cfg.continuous_stiffness_low_alpha = 0.50
        cfg.continuous_stiffness_high_alpha = 0.88
        cfg.continuous_stiffness_blend = 0.88
        cfg.continuous_stiffness_zref_enabled = True
        cfg.continuous_stiffness_zref_gain = 3.0
        cfg.continuous_stiffness_zref_max_scale = 4.0
        return
    if benchmark != "force_margin_rcm_challenge":
        raise ValueError(f"unknown benchmark: {benchmark}")

    cfg.scan_start_x = 0.30
    cfg.scan_end_x = 0.328
    cfg.scan_vx = 0.00050
    cfg.F_desired = 0.55
    cfg.F_min = 0.38
    cfg.F_max = 0.82
    cfg.stiffness_zones = [
        (0.300, 0.311, 350, 4),
        (0.311, 0.320, 180, 4),
        (0.320, 0.328, 900, 12),
    ]

    cfg.scan_z_ref_rate = 0.00010
    cfg.scan_z_ref_min_offset = -0.0014
    cfg.scan_z_ref_max_offset = 0.00045
    cfg.scan_force_deadband = 0.035
    cfg.scan_force_error_limit = 0.30
    cfg.scan_sigma_force_limit = 0.07
    cfg.scan_timeout = 90.0

    cfg.scan_min_alpha_when_rcm_soft = 0.92
    cfg.scan_alpha_filter_tau = 0.22
    cfg.scan_alpha_rate_limit = 0.75
    cfg.scan_track_slow_error = 0.0035
    cfg.scan_track_pause_error = 0.0062
    cfg.scan_rcm_slow_error = 0.0028
    cfg.scan_rcm_pause_error = 0.0052
    cfg.scan_rcm_recovery_enter = 0.0045
    cfg.scan_rcm_recovery_exit = 0.0024
    cfg.scan_rcm_recovery_alpha = 0.96
    cfg.scan_rcm_recovery_timeout = 7.0
    cfg.scan_stop_after_rcm_recovery = False
    cfg.scan_force_slow = 0.90
    cfg.scan_force_pause = 1.30
    cfg.scan_abort_pos_error = 0.0075
    cfg.scan_abort_rcm_error = 0.0065
    cfg.scan_abort_force = 1.40
    cfg.adjustment_abort_force = max(cfg.F_max, cfg.scan_abort_force)

    cfg.adjustment_force_error = 0.12
    cfg.adjustment_timeout = 7.0
    cfg.adjustment_rcm_limit = cfg.scan_rcm_slow_error

    cfg.continuous_safe_tracking_alpha = 0.90
    cfg.continuous_safe_tracking_extra = 0.04
    cfg.continuous_force_balance_alpha = 0.90
    cfg.continuous_force_guard_alpha = 0.96
    cfg.continuous_low_force_guard_alpha = 0.82
    cfg.continuous_alpha_min = 0.82
    cfg.continuous_alpha_max = 0.98
    cfg.continuous_smooth_tau = 0.20


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


def slew_toward(current, target, max_step):
    """按最大步长推进参考点，避免目标位置阶跃。"""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    dist = np.linalg.norm(delta)
    if dist <= max_step or dist < 1e-12:
        return target.copy()
    return current + delta * (max_step / dist)


def clamp_norm(vec, limit):
    """限制向量范数，保持方向不变。"""
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= limit or norm < 1e-12:
        return vec
    return vec * (float(limit) / norm)


def approach_speed_scale(follow_err, rcm_err, cfg):
    """接近阶段靠近 RCM/跟踪软边界时自动降速。"""
    follow_ratio = follow_err / max(cfg.approach_follow_tolerance, 1e-9)
    rcm_ratio = rcm_err / max(cfg.approach_rcm_soft_limit, 1e-9)
    margin = max(follow_ratio, rcm_ratio)
    if margin <= 0.5:
        return 1.0
    if margin >= 1.0:
        return cfg.approach_min_speed_scale
    scale = 1.0 - (margin - 0.5) / 0.5 * (1.0 - cfg.approach_min_speed_scale)
    return float(np.clip(scale, cfg.approach_min_speed_scale, 1.0))


def rcm_alpha_floor(rcm_err, cfg):
    """连续提高 alpha 下限，避免 RCM soft 阈值附近二值跳变。"""
    span = max(cfg.scan_rcm_pause_error - cfg.scan_rcm_slow_error, 1e-9)
    ratio = np.clip((rcm_err - cfg.scan_rcm_slow_error) / span, 0.0, 1.0)
    weight = ratio * ratio * (3.0 - 2.0 * ratio)
    return float(cfg.scan_min_alpha_when_rcm_soft * weight)


def stiffness_z_ref_rate_scale(cfg, sched, K_hat):
    """continuous_force_margin 专用的刚度感知 z_ref 速率门控。"""
    if (
        isinstance(sched, FixedAlphaScheduler)
        or not cfg.continuous_stiffness_zref_enabled
        or not np.isfinite(K_hat)
    ):
        return 1.0
    k_gate = np.clip(
        (float(K_hat) - cfg.continuous_stiffness_low_threshold)
        / max(
            cfg.continuous_stiffness_high_threshold
            - cfg.continuous_stiffness_low_threshold,
            1e-9,
        ),
        0.0,
        1.0,
    )
    k_gate = k_gate * k_gate * (3.0 - 2.0 * k_gate)
    scale = 1.0 + cfg.continuous_stiffness_zref_gain * k_gate
    return float(np.clip(scale, 1.0, cfg.continuous_stiffness_zref_max_scale))


def wait_for_tool_settle(robot, kin_tool, kin_flange, cfg, rate):
    """进入 approach 前等待当前位置速度变小，避免带残余速度开始接近。"""
    start = time.time()
    stable_start = None
    hold_joints = None
    try:
        hold_joints = robot.angles()
    except Exception:
        hold_joints = None

    while not rospy.is_shutdown():
        rs = update_robot_state(kin_tool, kin_flange)
        speed = float(np.linalg.norm(rs["tool_position_velocity"]))
        now = time.time()
        if speed <= cfg.approach_start_speed_limit:
            if stable_start is None:
                stable_start = now
            if now - stable_start >= cfg.approach_start_settle_hold:
                rospy.loginfo(
                    f"  Tool settled before approach: |v|={speed*1000:.2f}mm/s"
                )
                return rs
        else:
            stable_start = None

        if hold_joints is not None:
            try:
                robot.exec_position_cmd(hold_joints)
            except Exception:
                pass

        if now - start >= cfg.approach_start_settle_timeout:
            rospy.logwarn(
                f"  Tool did not fully settle before approach "
                f"(|v|={speed*1000:.2f}mm/s); continuing with rate-limited reference."
            )
            return rs
        rate.sleep()


# ================================================================
# 单次试验
# ================================================================
def run_trial(robot, kin_tool, kin_flange,
              cfg, ctrl, sched, est, venv, force_sensor,
              logger, trial_id, approach_logger=None):
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
    rs = wait_for_tool_settle(robot, kin_tool, kin_flange, cfg, rate)
    rospy.loginfo(f"  Tool at: {rs['tool_position']}")

    # ---- Phase 1: 下降接近 ----
    rospy.loginfo("  Phase 1: Approaching...")
    z_contact = None
    t0 = time.time()
    last_approach_time = t0
    approach_ref = rs["tool_position"].copy()
    target_xy = np.array([cfg.scan_start_x, cfg.scan_y], dtype=float)
    if approach_ref[2] <= cfg.scan_z + cfg.approach_z_margin:
        contact_z_threshold = approach_ref[2] - cfg.approach_surface_margin
        rospy.logwarn(
            f"  Current tool z={approach_ref[2]:.4f}m is already below "
            f"configured scan_z={cfg.scan_z:.4f}m; using local approach "
            f"threshold z={contact_z_threshold:.4f}m to avoid immediate "
            "contact-triggered scan start."
        )
    else:
        contact_z_threshold = cfg.scan_z
    approach_count = 0
    approach_distance_xy = np.linalg.norm(approach_ref[:2] - target_xy)
    approach_height = max(0.0, approach_ref[2] - contact_z_threshold)
    approach_timeout = max(
        cfg.approach_timeout,
        1.5 * approach_distance_xy / max(cfg.approach_xy_speed, 1e-9)
        + 1.5 * approach_height / max(cfg.approach_z_speed, 1e-9)
        + 10.0,
    )
    rospy.loginfo(
        f"  Approach reference starts at tool={approach_ref}, "
        f"target_xy=[{target_xy[0]:.4f}, {target_xy[1]:.4f}], "
        f"contact_z={contact_z_threshold:.4f}, "
        f"xy_speed={cfg.approach_xy_speed:.4f}m/s, "
        f"z_speed={cfg.approach_z_speed:.4f}m/s, timeout={approach_timeout:.1f}s"
    )

    while not rospy.is_shutdown() and z_contact is None:
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]
        flange_pos = rs["flange_position"]
        rcm_now = np.linalg.norm(
            cfg.trocar_position
            - compute_position_rcm(tp, flange_pos, cfg.trocar_position)
        )

        delta = max(0, contact_z_threshold - tp[2])
        F_z, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, -tv[2]
        )

        now = time.time()
        loop_dt = now - last_approach_time
        if loop_dt <= 0.0 or not np.isfinite(loop_dt):
            loop_dt = dt
        loop_dt = min(max(loop_dt, dt), 0.1)
        last_approach_time = now

        prev_ref = approach_ref.copy()
        xy_ref_err = np.linalg.norm(approach_ref[:2] - target_xy)
        xy_actual_err = np.linalg.norm(tp[:2] - target_xy)
        xy_aligned = (
            xy_ref_err <= cfg.approach_xy_tolerance
            and xy_actual_err <= cfg.approach_xy_tolerance
        )

        height_margin = min(cfg.approach_z_margin, 0.0001)
        force_contact = abs(F_z) > cfg.approach_force_threshold
        height_contact = tp[2] <= contact_z_threshold + height_margin
        if force_contact and not xy_aligned:
            rospy.logwarn(
                f"  Force contact reached before xy alignment "
                f"(xy_err={xy_actual_err*1000:.2f}mm, "
                f"rcm={rcm_now*1000:.2f}mm, F={F_z:.3f}N). "
                "Stopping this trial instead of dragging laterally under contact."
            )
            return False
        if height_contact and not xy_aligned:
            contact_z_threshold = min(
                contact_z_threshold,
                tp[2] - cfg.approach_surface_margin,
            )
            if approach_count % cfg.approach_retarget_log_interval == 0:
                rospy.logwarn(
                    f"  Approach height threshold reached before xy alignment "
                    f"without force contact; retarget contact_z to "
                    f"{contact_z_threshold:.4f}m and continue alignment "
                    f"(xy_err={xy_actual_err*1000:.2f}mm, "
                    f"rcm={rcm_now*1000:.2f}mm)."
                )
            height_contact = False
        if (force_contact or height_contact) and xy_aligned:
            z_contact = tp[2]
            rospy.loginfo(
                f"  Contact at z={z_contact:.4f}, F={F_z:.3f}N "
                f"({force_source}), rcm={rcm_now*1000:.2f}mm"
            )
            break

        follow_err = np.linalg.norm(tp - approach_ref)
        can_advance = (
            follow_err <= cfg.approach_follow_tolerance
            and rcm_now <= cfg.approach_rcm_soft_limit
        )
        speed_scale = approach_speed_scale(follow_err, rcm_now, cfg)

        if not can_advance:
            approach_stage = "settle_rcm"
        elif xy_aligned:
            approach_stage = "descend"
            approach_ref[:2] = target_xy
            approach_ref[2] = max(
                contact_z_threshold,
                approach_ref[2] - cfg.approach_z_speed * speed_scale * loop_dt,
            )
        else:
            approach_stage = "align_xy"
            approach_ref[:2] = slew_toward(
                approach_ref[:2],
                target_xy,
                cfg.approach_xy_speed * speed_scale * loop_dt,
            )

        # 纯位控接近 (α=1) — RCM flange-space 控制。参考轨迹显式限速，
        # 避免从当前 tool 位姿瞬间跳到扫描起点。
        x_tool_ref = approach_ref.copy()
        xdot_tool_ref = (approach_ref - prev_ref) / loop_dt

        # 自由空间: 力误差为 0, 力积分不更新
        e_f = np.zeros(3)
        sigma_f = sigma_f_int.get()

        tau, _, K_eff, integ_euler, _, error = compute_torque_with_rcm(
            ctrl, rs, kin_flange,
            x_tool_ref=x_tool_ref, xdot_tool_ref=xdot_tool_ref,
            e_f=e_f, sigma_f=sigma_f,
            alpha=1.0, K_e_hat=500,
            trocar_pos=cfg.trocar_position,
            length=cfg.tool_length,
            integ_euler=integ_euler, dt=dt,
        )
        tau = clamp_norm(tau, cfg.approach_tau_norm_limit)
        robot.exec_torque_cmd(tau)
        rate.sleep()
        approach_count += 1
        if approach_logger is not None:
            pos_err = tp - x_tool_ref
            approach_logger.log(
                t=time.time() - t0,
                pos=tp.copy(),
                pos_des=x_tool_ref.copy(),
                pos_err=pos_err,
                F_measured=abs(F_z),
                F_desired=0.0,
                F_err=abs(F_z),
                wrench=wrench,
                force_source=force_source,
                sensor_available=int(sensor_available),
                e_f=0.0,
                e_r=np.linalg.norm(pos_err),
                sigma_f_norm=0.0,
                e_r1_norm=follow_err,
                alpha=1.0,
                K_hat=500.0,
                K_eff=K_eff,
                x_desired=x_tool_ref[0],
                error_rcm=error[0],
                error_track=error[1],
                arbitration_strategy=sched.name,
                u_norm=np.linalg.norm(tau),
                phase=-2,
            )

        if approach_count % 10 == 0:
            rospy.loginfo(
                f"  approaching({approach_stage}) | "
                f"tool=[{tp[0]*1000:6.2f},{tp[1]*1000:6.2f},{tp[2]*1000:6.2f}]mm | "
                f"ref=[{x_tool_ref[0]*1000:6.2f},{x_tool_ref[1]*1000:6.2f},{x_tool_ref[2]*1000:6.2f}]mm | "
                f"|v|={np.linalg.norm(tv)*1000:.2f}mm/s | "
                f"follow={follow_err*1000:.2f}mm xy_err={xy_actual_err*1000:.2f}mm | "
                f"rcm={error[0]*1000:.2f}mm | "
                f"scale={speed_scale:.2f} | F={F_z:.3f}N"
            )
        if error[0] > cfg.approach_rcm_warn and approach_count % 50 == 0:
            rospy.logwarn(
                f"  Approach RCM error is high: {error[0]*1000:.2f}mm. "
                "Reference motion remains rate-limited; check the initial pose "
                "if this persists."
            )

        if time.time() - t0 > approach_timeout:
            rospy.logwarn(
                f"  Approach timeout "
                f"(stage={approach_stage}, tool={tp}, ref={x_tool_ref}, "
                f"F={F_z:.3f}N, source={force_source}, rcm={error[0]:.4f}m)"
            )
            return False

    if z_contact is None:
        z_contact = cfg.scan_z

    # 重置积分器状态 (进入 Phase 2)
    sigma_f_int.reset()
    integ_euler = np.zeros(3)

    # ---- Phase 1.5: 接触调整 ----
    # 刚接触后先在扫描起点保持 x 不动，让恒力、RCM 和自适应 scan_z_ref 收敛；
    # 该阶段不写入主 logger，避免把接触震荡混入正式实验数据。
    rospy.loginfo("  Phase 1.5: Contact/RCM adjustment (not logged)...")
    x_cur = cfg.scan_start_x
    scan_z_ref = z_contact
    prev_scan_z_ref = scan_z_ref
    scan_rcm_recovery = False
    scan_rcm_recovery_t0 = None
    adjust_t0 = time.time()
    adjust_last_time = adjust_t0
    adjust_last_log = 0.0
    adjust_forces = []
    adjust_pos_errors = []
    adjust_rcm_errors = []
    adjust_window = max(3, int(cfg.adjustment_stable_window * cfg.ctrl_rate))
    delta_dot_est = ContactDeltaDotEstimator(
        tau=cfg.contact_delta_dot_filter_tau,
        limit=cfg.contact_delta_dot_limit,
    )
    alpha_limiter = AlphaCommandLimiter(
        tau=cfg.scan_alpha_filter_tau,
        max_rate=cfg.scan_alpha_rate_limit,
        initial=0.5,
    )

    while not rospy.is_shutdown():
        adjust_t = time.time() - adjust_t0
        if adjust_t > cfg.scan_timeout:
            rospy.logwarn("  Adjustment exceeded scan timeout; stopping trial.")
            return False
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]
        flange_pos = rs["flange_position"]

        now = time.time()
        loop_dt = now - adjust_last_time
        if loop_dt <= 0.0 or not np.isfinite(loop_dt):
            loop_dt = dt
        loop_dt = min(max(loop_dt, dt), 0.1)
        adjust_last_time = now

        delta = max(0, z_contact - tp[2] + cfg.scan_contact_bias)
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, loop_dt, raw_delta_dot=-tv[2]
        )
        F_z, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, delta_dot
        )
        K_hat, B_hat = est.update(abs(F_z), delta, delta_dot)

        F_actual = abs(F_z)
        if force_source == "virtual" and venv is not None:
            K_for_z, _ = venv.get_stiffness(x_cur)
        else:
            K_for_z = K_hat
        K_for_z = float(np.clip(K_for_z, 50.0, 5000.0))
        scan_z_target = z_contact + cfg.scan_contact_bias - cfg.F_desired / K_for_z
        scan_z_target = float(np.clip(
            scan_z_target,
            z_contact + cfg.scan_z_ref_min_offset,
            z_contact + cfg.scan_z_ref_max_offset,
        ))
        z_ref_scale = stiffness_z_ref_rate_scale(cfg, sched, K_hat)
        z_step = cfg.scan_z_ref_rate * z_ref_scale * loop_dt
        prev_scan_z_ref = scan_z_ref
        scan_z_ref += float(np.clip(scan_z_target - scan_z_ref, -z_step, z_step))

        rcm_now = np.linalg.norm(
            cfg.trocar_position
            - compute_position_rcm(tp, flange_pos, cfg.trocar_position)
        )
        scan_z_vel = (scan_z_ref - prev_scan_z_ref) / loop_dt
        xy_track_err = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
        pos_track_err = np.linalg.norm(tp - np.array([x_cur, cfg.scan_y, scan_z_ref]))

        x_tool_ref = np.array([x_cur, cfg.scan_y, scan_z_ref])
        xdot_tool_ref = np.array([0.0, 0.0, scan_z_vel])

        if (
            pos_track_err > cfg.scan_abort_pos_error
            or rcm_now > cfg.scan_abort_rcm_error
            or F_actual > cfg.adjustment_abort_force
        ):
            rospy.logwarn(
                "  Adjustment safety limit reached; stopping before formal scan "
                f"(pos_err={pos_track_err*1000:.2f}mm, "
                f"rcm={rcm_now*1000:.2f}mm, F={F_actual:.3f}N)."
            )
            return False

        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, F_actual])
        e_f_raw_vec = F_meas - F_des
        e_f_control = np.sign(e_f_raw_vec[2]) * max(
            abs(e_f_raw_vec[2]) - cfg.scan_force_deadband,
            0.0,
        )
        e_f_control = float(np.clip(
            e_f_control,
            -cfg.scan_force_error_limit,
            cfg.scan_force_error_limit,
        ))
        e_f_vec = np.array([0.0, 0.0, e_f_control])
        sigma_f = np.clip(
            sigma_f_int.update(e_f_vec),
            -cfg.scan_sigma_force_limit,
            cfg.scan_sigma_force_limit,
        )

        e_f_scalar = F_actual - cfg.F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt
        prev_ef = e_f_scalar
        e_r_scalar = pos_track_err
        de_r = (e_r_scalar - prev_er) / dt
        prev_er = e_r_scalar
        dK = (K_hat - prev_K) / dt
        prev_K = K_hat

        if isinstance(sched, FixedAlphaScheduler):
            alpha = sched.compute()
        else:
            alpha = sched.compute(
                F_norm=abs(F_z), e_f=e_f_scalar, K_hat=K_hat,
                e_r=e_r_scalar, z_vel=tv[2],
                de_f=e_f_dot, dK=dK, de_r=de_r,
                F_desired=cfg.F_desired,
                F_min=cfg.F_min,
                F_max=cfg.F_max,
            )
            alpha = max(alpha, rcm_alpha_floor(rcm_now, cfg))
        alpha = alpha_limiter.update(alpha, loop_dt)

        tau, _, _, integ_euler, _, error = compute_torque_with_rcm(
            ctrl, rs, kin_flange,
            x_tool_ref=x_tool_ref, xdot_tool_ref=xdot_tool_ref,
            e_f=e_f_vec, sigma_f=sigma_f,
            alpha=alpha, K_e_hat=K_hat,
            trocar_pos=cfg.trocar_position,
            length=cfg.tool_length,
            integ_euler=integ_euler, dt=dt,
        )
        tau = clamp_norm(tau, cfg.scan_tau_norm_limit)
        robot.exec_torque_cmd(tau)

        adjust_forces.append(F_actual)
        adjust_pos_errors.append(pos_track_err)
        adjust_rcm_errors.append(error[0])
        if len(adjust_forces) > adjust_window:
            adjust_forces.pop(0)
            adjust_pos_errors.pop(0)
            adjust_rcm_errors.pop(0)

        force_std = float(np.std(adjust_forces)) if len(adjust_forces) > 1 else float("inf")
        pos_std = float(np.std(adjust_pos_errors)) if len(adjust_pos_errors) > 1 else float("inf")
        rcm_std = float(np.std(adjust_rcm_errors)) if len(adjust_rcm_errors) > 1 else float("inf")
        force_slope = 0.0
        if len(adjust_forces) > 1:
            force_slope = abs(adjust_forces[-1] - adjust_forces[0]) / max(
                (len(adjust_forces) - 1) * dt, dt
            )
        stable = (
            adjust_t >= cfg.adjustment_min_time
            and abs(F_actual - cfg.F_desired) <= cfg.adjustment_force_error
            and force_std <= cfg.adjustment_force_std
            and force_slope <= cfg.adjustment_force_slope
            and pos_std <= cfg.adjustment_pos_std
            and rcm_std <= cfg.adjustment_rcm_std
            and error[0] <= cfg.adjustment_rcm_limit
        )

        if adjust_t - adjust_last_log >= 0.5:
            adjust_last_log = adjust_t
            rospy.loginfo(
                f"  adjusting | t={adjust_t:.2f}s, F={F_actual:.3f}N, "
                f"F_std={force_std:.3f}, F_slope={force_slope:.3f}N/s, "
                f"err={pos_track_err*1000:.2f}mm, rcm={error[0]*1000:.2f}mm, "
                f"rcm_std={rcm_std*1000:.3f}mm, z_ref={scan_z_ref:.4f}, "
                f"alpha={alpha:.2f}, K={K_hat:.1f}, B={B_hat:.2f}"
            )

        if stable:
            rospy.loginfo(
                f"  Adjustment settled after {adjust_t:.2f}s; formal scan logging starts now."
            )
            break
        if adjust_t >= cfg.adjustment_timeout:
            rospy.logwarn(
                f"  Adjustment timeout after {adjust_t:.2f}s; start scan without "
                f"logging contact transient (F_std={force_std:.3f}, "
                f"rcm={error[0]*1000:.2f}mm, rcm_std={rcm_std*1000:.3f}mm)."
            )
            break
        rate.sleep()

    # ---- Phase 2: 恒力扫描 ----
    rospy.loginfo("  Phase 2: Scanning...")
    t_scan = time.time()
    last_scan_time = t_scan

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        t = time.time() - t_scan
        if t > cfg.scan_timeout:
            rospy.logwarn(
                f"  Scan timeout at x={x_cur:.4f}m; saving partial data."
            )
            break
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]
        flange_pos = rs["flange_position"]

        now = time.time()
        loop_dt = now - last_scan_time
        if loop_dt <= 0.0 or not np.isfinite(loop_dt):
            loop_dt = dt
        loop_dt = min(max(loop_dt, dt), 0.1)
        last_scan_time = now

        # 接触力: 传感器优先, 不可用时回退到虚拟环境
        delta = max(0, z_contact - tp[2] + cfg.scan_contact_bias)
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, loop_dt, raw_delta_dot=-tv[2]
        )
        if venv is not None:
            K_env_true, B_env_true = venv.get_stiffness(tp[0])
        else:
            K_env_true, B_env_true = 0.0, 0.0
        F_z, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, delta_dot
        )

        # 环境估计
        K_hat, B_hat = est.update(
            abs(F_z),
            delta,
            delta_dot
        )

        F_actual = abs(F_z)
        if force_source == "virtual" and venv is not None:
            K_for_z, _ = venv.get_stiffness(x_cur)
        else:
            K_for_z = K_hat
        K_for_z = float(np.clip(K_for_z, 50.0, 5000.0))
        scan_z_target = z_contact + cfg.scan_contact_bias - cfg.F_desired / K_for_z
        scan_z_target = float(np.clip(
            scan_z_target,
            z_contact + cfg.scan_z_ref_min_offset,
            z_contact + cfg.scan_z_ref_max_offset,
        ))
        z_ref_scale = stiffness_z_ref_rate_scale(cfg, sched, K_hat)
        z_step = cfg.scan_z_ref_rate * z_ref_scale * loop_dt
        prev_scan_z_ref = scan_z_ref
        scan_z_ref += float(np.clip(scan_z_target - scan_z_ref, -z_step, z_step))

        rcm_now = np.linalg.norm(
            cfg.trocar_position
            - compute_position_rcm(tp, flange_pos, cfg.trocar_position)
        )
        if scan_rcm_recovery:
            if (
                rcm_now <= cfg.scan_rcm_recovery_exit
                and cfg.F_min <= F_actual <= cfg.F_max
            ):
                scan_rcm_recovery = False
                scan_rcm_recovery_t0 = None
                rospy.loginfo(
                    f"  RCM recovery complete "
                    f"(rcm={rcm_now*1000:.2f}mm, F={F_actual:.3f}N)."
                )
                if cfg.scan_stop_after_rcm_recovery:
                    rospy.loginfo(
                        "  Ending scan after successful RCM recovery; "
                        "saving stable partial trial."
                    )
                    break
        elif rcm_now >= cfg.scan_rcm_recovery_enter:
            scan_rcm_recovery = True
            scan_rcm_recovery_t0 = t
            rospy.logwarn(
                f"  Enter RCM recovery: rcm={rcm_now*1000:.2f}mm, "
                f"F={F_actual:.3f}N. Pausing scan and unloading contact."
            )

        if scan_rcm_recovery:
            unload_z = z_contact + cfg.scan_z_ref_max_offset
            scan_z_ref += float(np.clip(unload_z - scan_z_ref, -z_step, z_step))
            if (
                scan_rcm_recovery_t0 is not None
                and t - scan_rcm_recovery_t0 > cfg.scan_rcm_recovery_timeout
                and rcm_now > cfg.scan_rcm_recovery_enter
            ):
                rospy.logwarn(
                    "  RCM recovery timeout; stopping scan before oscillation "
                    f"(rcm={rcm_now*1000:.2f}mm, F={F_actual:.3f}N)."
                )
                break

        scan_z_vel = (scan_z_ref - prev_scan_z_ref) / loop_dt
        xy_track_err = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
        pos_track_err = np.linalg.norm(tp - np.array([x_cur, cfg.scan_y, scan_z_ref]))

        def _slowdown(value, slow_at, pause_at):
            if value <= slow_at:
                return 1.0
            if value >= pause_at:
                return 0.0
            return float((pause_at - value) / max(pause_at - slow_at, 1e-9))

        scan_speed_scale = min(
            _slowdown(pos_track_err, cfg.scan_track_slow_error,
                      cfg.scan_track_pause_error),
            _slowdown(rcm_now, cfg.scan_rcm_slow_error,
                      cfg.scan_rcm_pause_error),
            _slowdown(F_actual, cfg.scan_force_slow,
                      cfg.scan_force_pause),
        )
        if scan_rcm_recovery:
            scan_speed_scale = 0.0
        scan_vx_cmd = cfg.scan_vx * scan_speed_scale

        # tool 期望轨迹
        x_tool_ref = np.array([x_cur, cfg.scan_y, scan_z_ref])
        xdot_tool_ref = np.array([scan_vx_cmd, 0.0, scan_z_vel])

        if (
            pos_track_err > cfg.scan_abort_pos_error
            or rcm_now > cfg.scan_abort_rcm_error
            or F_actual > cfg.scan_abort_force
        ):
            rospy.logwarn(
                "  Scan safety limit reached; stopping scan before "
                f"oscillation grows (pos_err={pos_track_err*1000:.2f}mm, "
                f"rcm={rcm_now*1000:.2f}mm, F={F_actual:.3f}N)."
            )
            break

        # 力目标: 向上接触反力 (z+) 为正
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        # 力误差 + 力误差泄漏积分
        e_f_raw_vec = F_meas - F_des
        e_f_control = np.sign(e_f_raw_vec[2]) * max(
            abs(e_f_raw_vec[2]) - cfg.scan_force_deadband,
            0.0,
        )
        e_f_control = float(np.clip(
            e_f_control,
            -cfg.scan_force_error_limit,
            cfg.scan_force_error_limit,
        ))
        if scan_rcm_recovery:
            e_f_control = 0.0
            sigma_f_int.reset()
        e_f_vec = np.array([0.0, 0.0, e_f_control])
        sigma_f = np.clip(
            sigma_f_int.update(e_f_vec),
            -cfg.scan_sigma_force_limit,
            cfg.scan_sigma_force_limit,
        )

        # 差分导数 (用于 α 调度)
        e_f_scalar = F_actual - cfg.F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt
        prev_ef = e_f_scalar

        e_r_scalar = pos_track_err
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
            alpha = max(alpha, rcm_alpha_floor(rcm_now, cfg))
            if scan_rcm_recovery:
                alpha = max(alpha, cfg.scan_rcm_recovery_alpha)
        alpha = alpha_limiter.update(alpha, loop_dt)

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
        tau = clamp_norm(tau, cfg.scan_tau_norm_limit)
        robot.exec_torque_cmd(tau)

        # 推进扫描参考
        x_cur += scan_vx_cmd * dt

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
                f"α={alpha:.2f} | vx={scan_vx_cmd*1000:.2f}mm/s | "
                f"source={force_source}"
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
            K_hat_raw=getattr(est, "K_observed", K_hat),
            B_hat=B_hat,
            delta=delta, delta_dot=delta_dot,
            delta_dot_raw=delta_dot_raw, delta_dot_fd=delta_dot_fd,
            K_env_true=K_env_true, B_env_true=B_env_true,
            scan_z_ref=scan_z_ref,
            contact_plane_z=z_contact,
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
        approach_dir = os.path.join(result_dir, "approach_debug")
        approach_npz = []
        if os.path.isdir(approach_dir):
            approach_npz = [
                name for name in os.listdir(approach_dir)
                if name.endswith(".npz") and os.path.isfile(os.path.join(approach_dir, name))
            ]
        if not approach_npz:
            rospy.logwarn(f"No npz result files found for plotting: {result_dir}")
            return
        rospy.logwarn(
            f"No scan npz files found; plotting approach debug data: {approach_dir}"
        )
        common_args = ["--input", approach_dir]
        if no_show:
            common_args.append("--no-show")
        try:
            subprocess.run([sys.executable, latest_script] + common_args, check=False)
        except Exception as exc:
            rospy.logwarn(f"Auto approach plot failed: {exc}")
        return

    try:
        subprocess.run([sys.executable, latest_script] + common_args, check=False)
    except Exception as exc:
        rospy.logwarn(f"Auto plot failed: {exc}")

    if len(npz_files) > 1:
        try:
            subprocess.run([sys.executable, compare_script] + common_args, check=False)
        except Exception as exc:
            rospy.logwarn(f"Auto arbitration comparison failed: {exc}")


# ================================================================
# 主入口
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='continuous_force_margin')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--output-dir', default='results')
    ap.add_argument('--gains-file', default=None)
    ap.add_argument('--controller-mode', default='are',
                    choices=['are', 'pareto_iter'],
                    help='are: 原 4D ARE 查表; pareto_iter: Algorithm 2 迭代 P1/P2')
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
    ap.add_argument('--no-auto-plot', action='store_true',
                    help='实验结束后不自动调用绘图脚本')
    ap.add_argument('--plot-no-show', action='store_true',
                    help='自动绘图时只保存图像，不弹出 matplotlib 窗口')
    ap.add_argument(
        '--benchmark',
        default='default',
        choices=['default', 'force_margin_rcm_challenge', 'stiffness_alpha_showcase'],
        help='实验场景: default 为原 run_with_rcm 设置; force_margin_rcm_challenge 为强刚度变化 RCM 压力测试; stiffness_alpha_showcase 用于凸显 alpha 随刚度自适应',
    )
    args = ap.parse_args()

    rospy.init_node("coop_gt_rcm", anonymous=True)
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Cooperative Game Force-Position Experiment (RCM)")
    rospy.loginfo("  Model: ARE/Pareto per-axis gains + K_v baseline + leaky integrators")
    rospy.loginfo("=" * 60)

    robot = PandaArm()
    kin_tool = PandaKinematics(robot, "panda_link10")
    kin_flange = PandaKinematics(robot, "panda_link8")
    rospy.sleep(1.0)

    cfg = Config()
    apply_benchmark_config(cfg, args.benchmark)
    rospy.loginfo(
        f"Benchmark: {args.benchmark}; F_des={cfg.F_desired:.3f}N, "
        f"F_bounds=[{cfg.F_min:.3f}, {cfg.F_max:.3f}]N, "
        f"scan=[{cfg.scan_start_x:.4f}, {cfg.scan_end_x:.4f}]m, "
        f"scan_vx={cfg.scan_vx:.4f}m/s, stiffness_zones={cfg.stiffness_zones}"
    )
    rospy.loginfo(
        f"RCM thresholds: slow={cfg.scan_rcm_slow_error*1000:.2f}mm, "
        f"pause={cfg.scan_rcm_pause_error*1000:.2f}mm, "
        f"recovery_enter={cfg.scan_rcm_recovery_enter*1000:.2f}mm, "
        f"abort={cfg.scan_abort_rcm_error*1000:.2f}mm"
    )
    rospy.loginfo(
        f"Continuous alpha params: safe={cfg.continuous_safe_tracking_alpha:.2f}, "
        f"extra={cfg.continuous_safe_tracking_extra:.2f}, "
        f"balance={cfg.continuous_force_balance_alpha:.2f}, "
        f"high_guard={cfg.continuous_force_guard_alpha:.2f}, "
        f"low_guard={cfg.continuous_low_force_guard_alpha:.2f}, "
        f"range=[{cfg.continuous_alpha_min:.2f}, {cfg.continuous_alpha_max:.2f}], "
        f"tau={cfg.continuous_smooth_tau:.2f}s, "
        f"K_alpha={cfg.continuous_stiffness_alpha_enabled}, "
        f"K_range=[{cfg.continuous_stiffness_low_threshold:.0f}, "
        f"{cfg.continuous_stiffness_high_threshold:.0f}]"
    )
    ctrl = CooperativeGameController(control_mode=args.controller_mode)
    rospy.loginfo(f"Controller mode: {args.controller_mode}")

    # 增益表
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
        gains_name = 'coop_gains_rcm.npy'
        if args.controller_mode == 'pareto_iter':
            gains_name = 'coop_gains_rcm_pareto_iter.npy'
        gpath = os.path.join(args.output_dir, gains_name)
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
        s = STRATEGIES[sn](cfg)
        rospy.loginfo(f"\n{'='*50}\n  Strategy: {s.name}\n{'='*50}")
        for t in range(args.trials):
            if rospy.is_shutdown():
                break
            lg = DataLogger()
            approach_lg = DataLogger()
            est = EnvironmentEstimator()
            s.reset()
            ok = run_trial(
                robot, kin_tool, kin_flange,
                cfg, ctrl, s, est, venv, force_sensor,
                lg, t, approach_logger=approach_lg,
            )
            if approach_lg.count:
                approach_dir = os.path.join(odir, "approach_debug")
                os.makedirs(approach_dir, exist_ok=True)
                approach_lg.save(
                    os.path.join(approach_dir, f"approach_{s.name}_t{t:02d}.npz")
                )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")
    if not args.no_auto_plot:
        auto_plot_results(odir, no_show=args.plot_no_show)


if __name__ == '__main__':
    main()
