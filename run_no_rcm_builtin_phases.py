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
from scipy.spatial.transform import Rotation

from src.gt_controller import CooperativeGameController
from src.alpha_scheduler_gt import (
    PhaseAwareFuzzyAlphaScheduler, ForceMarginFuzzyAlphaScheduler,
    ContinuousForceMarginFuzzyAlphaScheduler,
    OnlinePriorityAdaptationAlphaScheduler,
    FixedAlphaScheduler
)
from src.env_estimator import EnvironmentEstimator
from src.utils import (
    VirtualStiffnessSurface, DataLogger, ContactDeltaDotEstimator,
    FirstOrderLowPass, VectorRateLimiter,
)
from src.leaky_integrator import LeakyIntegrator
from src.robot_interface import (
    update_robot_state, safe_move_to_joint_position,
    compute_torque_no_rcm, fixed_downward_tool_euler,
    tool_visual_front_axis_local, INIT_JOINTS,
)
from ch3_controller.builtin_phase_control import BuiltinPhaseController


def prime_gravity_compensation(robot, seconds=1.0, rate_hz=200.0):
    """Publish zero control torque so the ROS 2 adapter immediately adds gravity."""
    if not hasattr(robot, "exec_torque_cmd"):
        rospy.sleep(seconds)
        return

    period = 1.0 / max(float(rate_hz), 1.0)
    tau_zero = np.zeros(len(INIT_JOINTS), dtype=float)
    if hasattr(robot, "wait_until_ready"):
        robot.wait_until_ready(timeout=2.0)

    deadline = time.time() + float(seconds)
    while time.time() < deadline and not rospy.is_shutdown():
        robot.exec_torque_cmd(tau_zero)
        if hasattr(robot, "spin_once"):
            robot.spin_once(timeout_sec=0.0)
        time.sleep(period)


def orientation_diagnostics(robot, rs, ref_euler, nominal_joints):
    """Return joint and SO(3) posture diagnostics for one control sample."""
    q = np.asarray(robot.angles(), dtype=float).reshape(-1)
    qd = np.asarray(getattr(robot, "_qd_meas", np.zeros_like(q)), dtype=float).reshape(-1)
    q_nom = np.asarray(nominal_joints, dtype=float).reshape(-1)
    n = min(q.size, qd.size, q_nom.size)
    q = q[:n]
    qd = qd[:n]
    q_err = q - q_nom[:n]

    tool_euler = np.asarray(rs["tool_rotation_euler"], dtype=float)
    tool_omega = np.asarray(rs["tool_rotation_euler_velocity"], dtype=float)
    ref_euler = np.asarray(ref_euler, dtype=float)
    R_cur = Rotation.from_euler("xyz", tool_euler).as_matrix()
    R_ref = Rotation.from_euler("xyz", ref_euler).as_matrix()
    ori_err = Rotation.from_matrix(R_ref @ R_cur.T).as_rotvec()

    # “正面朝前”按 trocar 可视正面轴诊断，而不是裸 tool-x 轴。
    front_local = tool_visual_front_axis_local()
    front_dot = float(np.clip(np.dot(R_cur @ front_local, R_ref @ front_local), -1.0, 1.0))
    front_axis_err_deg = float(np.degrees(np.arccos(front_dot)))
    return {
        "q": q,
        "qd": qd,
        "q_err": q_err,
        "qd_norm": float(np.linalg.norm(qd)),
        "tool_euler": tool_euler,
        "tool_euler_ref": ref_euler,
        "ori_err_rotvec": ori_err,
        "tool_omega": tool_omega,
        "front_axis_err_deg": front_axis_err_deg,
    }


def blend_euler_reference(start_euler, target_euler, blend):
    """SO(3) interpolation from start_euler to target_euler."""
    blend = float(np.clip(blend, 0.0, 1.0))
    R_start = Rotation.from_euler("xyz", np.asarray(start_euler, dtype=float))
    R_target = Rotation.from_euler("xyz", np.asarray(target_euler, dtype=float))
    delta = R_target * R_start.inv()
    R_blend = Rotation.from_rotvec(blend * delta.as_rotvec()) * R_start
    return R_blend.as_euler("xyz", degrees=False)


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
    scan_vx      = 0.0016    # Phase 2 沿 x 的扫描速度, m/s；在稳定姿态下适度加快扫描
    force_consistent_scan_z = True  # 根据 F_desired/K_env 生成 z 参考，避免力/位目标冲突
    scan_z_min   = 0.2964    # 力一致 z 参考下限，防止低刚度估计导致过深压入
    scan_z_max   = 0.2984    # 力一致 z 参考上限，保留少量高刚度/过渡裕度
    scan_z_stiffness_floor = 250.0  # 生成 z 参考时使用的最小刚度, N/m

    # Phase 1 接近参数；参考 ROS1 0526/0603 的 phase 切换做法：
    # 先限速对齐 x/y，再按当前 tool 实际高度小步下探，避免 z 参考长期跑飞。
    approach_xy_speed = 0.012
    approach_z_speed = 0.007
    approach_min_speed_scale = 0.80
    approach_xy_tolerance = 0.0015
    approach_xy_descend_tolerance = 0.010
    approach_contact_xy_tolerance = 0.012
    approach_xy_hold_z_tolerance = 0.025
    approach_z_margin = 0.002
    approach_z_lag_limit = 0.030
    approach_follow_tolerance = 0.018
    approach_precontact_clearance = 0.035
    approach_contact_probe_depth = 0.018
    approach_extra_down_gap = 0.006
    approach_extra_down_kp = 450.0
    approach_extra_down_force_limit = 8.0
    approach_u_threshold = 35.0
    approach_u_tool_limits = np.array([18.0, 18.0, 18.0])
    approach_tau_norm_limit = 35.0
    approach_start_speed_limit = 0.003
    approach_start_settle_hold = 0.5
    approach_start_settle_timeout = 8.0
    approach_timeout = 240.0
    approach_orientation_prealign_time = 8.0
    approach_orientation_ramp_time = 6.0
    approach_position_error_limit = np.array([0.045, 0.045, 0.050])
    approach_velocity_error_limit = np.array([0.08, 0.08, 0.08])

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
    contact_settle_force_min = 0.12    # 轻接触稳定后允许进入力控调整, N
    contact_settle_z_error_max = 0.004 # 轻接触稳定允许的扫描 z 参考偏差, m
    contact_settle_time = 1.0          # 接触稳定滑动窗口长度, s
    contact_stable_z_std = 0.0003      # 稳定窗口内 z 标准差阈值, m
    contact_stable_force_std = 0.05    # 稳定窗口内力标准差阈值, N
    contact_stable_force_slope = 0.20  # 稳定窗口首尾力变化率阈值, N/s
    contact_force_blend_time = 0.5     # 接触后力控从 0 平滑引入的时间, s
    contact_delta_dot_filter_tau = 0.05  # 压入速度低通时间常数, s
    contact_delta_dot_limit = 0.02       # 压入速度限幅, m/s
    estimator_initial_K = 400.0          # 初始/滤波刚度估计, N/m
    force_filter_tau = 0.0               # 0 表示不滤波；虚拟环境本身无传感噪声
    stiffness_filter_tau = 0.0           # 0 表示控制使用原始 K_hat
    torque_rate_limit = 0.0              # 0 表示不做额外力矩限速
    scan_position_error_limit = 0.0      # 0 表示不裁剪位置误差
    scan_velocity_error_limit = 0.0      # 0 表示不裁剪速度误差
    scan_force_error_soft_limit = 0.0    # 0 表示不裁剪力误差
    scan_z_gain_khat_max = 0.0           # 0 表示不裁剪控制用 K_hat

    # 接触调整阶段，不写入实验数据；用于滤掉刚接触瞬态震荡。
    adjustment_min_time = 1.0          # 调整阶段最短持续时间, s
    adjustment_timeout = 8.0           # 调整阶段最长等待时间, s
    adjustment_stable_window = 0.8     # 调整稳定统计窗口, s
    adjustment_force_error = 0.15      # 允许的稳态力误差, N
    adjustment_pos_std = 0.0005        # 允许的位置误差标准差, m
    adjustment_xy_error = 0.0025       # 进入扫描前允许的 xy 稳态误差, m

    # Phase 3 退回初始关节角的超时；退回使用小步关节位置命令并打印进度。
    retreat_timeout = 16.0

    # 副本版本: 初始化、粗下降和退回使用 ros2_control 自带轨迹控制器。
    builtin_init_duration = 4.5
    builtin_coarse_descent_duration = 7.0
    builtin_retreat_duration = 7.0
    builtin_goal_timeout_margin = 6.0
    builtin_coarse_clearance = 0.075

    # Gazebo effort 接口下的 no-RCM 末端笛卡尔姿态控制参数。
    # 姿态参考固定为可视正面朝世界 +x、tool y 轴朝世界 -z：
    # 即“正面向前、末端竖直向下”。不再叠加零空间关节姿态控制器。
    no_rcm_u_threshold = 24.0
    no_rcm_P_ori = 10.0
    no_rcm_D_ori = 3.0
    no_rcm_I_ori = 0.0
    no_rcm_u_tool_limits = np.array([18.0, 18.0, 4.0])
    no_rcm_xy_stiffness_boost = 420.0
    no_rcm_xy_damping_boost = 14.0
    no_rcm_u_rot_limit = 4.0
    no_rcm_u_rot_rate_limit = 24.0
    no_rcm_omega_limit = 1.2
    no_rcm_omega_filter_tau = 0.06
    no_rcm_nominal_joints = np.array(INIT_JOINTS, dtype=float)
    no_rcm_nullspace_enabled = False
    no_rcm_null_kp = 5.0
    no_rcm_null_kd = 1.8
    no_rcm_null_tau_max = 3.0
    no_rcm_null_position_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.5])
    no_rcm_null_velocity_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 2.5])
    no_rcm_direct_posture_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0])
    no_rcm_direct_posture_kp = 3.0
    no_rcm_direct_posture_kd = 1.0
    no_rcm_direct_posture_tau_max = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 2.5])
    no_rcm_direct_posture_scan_enabled = False

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


def apply_benchmark_config(cfg, benchmark):
    """按命令行 benchmark 覆盖实验条件。

    default 保持原始力-位一致扫描；force_margin_challenge 刻意使用固定 z
    参考和强刚度变化，制造 fixed alpha 难以兼顾的力/位置冲突，用来检验
    continuous_force_margin 是否能根据力边界和位置误差连续调整 alpha。
    """
    if benchmark == "default":
        return
    if benchmark != "force_margin_challenge":
        raise ValueError(f"unknown benchmark: {benchmark}")

    cfg.force_consistent_scan_z = False
    cfg.scan_z = 0.298
    cfg.scan_vx = 0.0018
    cfg.F_desired = 1.0
    cfg.F_min = 0.55
    cfg.F_max = 1.45
    cfg.stiffness_zones = [
        (0.40, 0.425, 500, 5),
        (0.425, 0.452, 250, 4),
        (0.452, 0.480, 1000, 10),
    ]
    cfg.adjustment_force_error = 0.25
    cfg.adjustment_timeout = 5.0
    cfg.continuous_safe_tracking_alpha = 0.24
    cfg.continuous_safe_tracking_extra = 0.02
    cfg.continuous_force_balance_alpha = 0.24
    cfg.continuous_force_guard_alpha = 0.38
    cfg.continuous_low_force_guard_alpha = 0.12
    cfg.continuous_alpha_min = 0.10
    cfg.continuous_alpha_max = 0.42
    cfg.continuous_risk_margin_start = 0.18
    cfg.continuous_risk_margin_full = 0.04
    cfg.continuous_smooth_tau = 0.80


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


def slew_toward(current, target, max_step):
    """Move a vector reference toward target without a step change."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-12:
        return target.copy()
    return current + delta * (float(max_step) / dist)


def clamp_norm(vec, limit):
    """Limit vector norm while preserving direction."""
    vec = np.asarray(vec, dtype=float)
    limit = float(limit)
    norm = float(np.linalg.norm(vec))
    if limit <= 0.0 or norm <= limit or norm < 1e-12:
        return vec
    return vec * (limit / norm)


def approach_speed_scale(follow_err, cfg):
    """Slow the approach reference when the tool lags behind it."""
    ratio = float(follow_err) / max(float(cfg.approach_follow_tolerance), 1e-9)
    if ratio <= 0.5:
        return 1.0
    if ratio >= 1.0:
        return float(cfg.approach_min_speed_scale)
    scale = 1.0 - (ratio - 0.5) / 0.5 * (1.0 - float(cfg.approach_min_speed_scale))
    return float(np.clip(scale, cfg.approach_min_speed_scale, 1.0))


def run_trial(robot, kin_tool, kin_flange,
              cfg, ctrl, sched, est, venv, logger, trial_id,
              builtin_phases=None):
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
    force_filter = FirstOrderLowPass(cfg.force_filter_tau, initial=0.0)
    stiffness_filter = FirstOrderLowPass(
        cfg.stiffness_filter_tau, initial=cfg.estimator_initial_K
    )
    tau_limiter = VectorRateLimiter(cfg.torque_rate_limit)

    # 保存上一控制周期的物理量，用于计算变化率输入。
    prev_ef = 0.0; prev_er = 0.0; prev_K = cfg.estimator_initial_K

    est.reset()
    sched.reset()
    sigma_f_int.reset()
    for attr in ("_no_rcm_omega_filtered", "_no_rcm_u_rot_limited"):
        if hasattr(ctrl, attr):
            delattr(ctrl, attr)

    rospy.loginfo(f"[Trial {trial_id}] {sched.name} (NO-RCM)")

    if builtin_phases is not None:
        rospy.loginfo("  Built-in Phase INIT: joint trajectory to INIT_JOINTS...")
        if not builtin_phases.switch_to_trajectory():
            rospy.logwarn("  Failed to activate built-in trajectory controller.")
            return False
        if not builtin_phases.move_joints(
            INIT_JOINTS,
            duration=cfg.builtin_init_duration,
            timeout=cfg.builtin_init_duration + cfg.builtin_goal_timeout_margin,
        ):
            rospy.logwarn("  Built-in INIT trajectory failed.")
            return False
    else:
        if not safe_move_to_joint_position(robot, INIT_JOINTS, timeout=25.0):
            rospy.logwarn("  Initial joint move failed; abort trial before Cartesian control.")
            return False
    rospy.sleep(cfg.settle_time)
    rs = update_robot_state(kin_tool, kin_flange)
    ref_euler_fixed = fixed_downward_tool_euler()
    ref_euler_start = rs["tool_rotation_euler"].copy()
    rospy.loginfo(f"  Tool at: {rs['tool_position']}, "
                  f"down_forward_euler_ref: {ref_euler_fixed}, "
                  f"initial_tool_euler: {rs['tool_rotation_euler']}")

    # ---- Phase 0: 姿态预对齐 / 自带粗下降 ----
    if builtin_phases is not None:
        coarse_pos = np.array([
            cfg.scan_start_x,
            cfg.scan_y,
            cfg.approach_z + cfg.builtin_coarse_clearance,
        ])
        rospy.loginfo(
            "  Built-in Phase COARSE: trajectory to "
            f"[{coarse_pos[0]:.4f}, {coarse_pos[1]:.4f}, {coarse_pos[2]:.4f}]m..."
        )
        if not builtin_phases.move_tool_coarse(
            coarse_pos,
            ref_euler_fixed,
            duration=cfg.builtin_coarse_descent_duration,
            timeout=cfg.builtin_coarse_descent_duration + cfg.builtin_goal_timeout_margin,
        ):
            rospy.logwarn("  Built-in coarse descent failed.")
            return False
        builtin_phases.wait_after_switch(0.3)
        rs = update_robot_state(kin_tool, kin_flange)
        ref_euler_start = rs["tool_rotation_euler"].copy()
        rospy.loginfo(
            f"  Built-in coarse done. Tool at {rs['tool_position']}, "
            f"euler={rs['tool_rotation_euler']}"
        )
        if not builtin_phases.switch_to_effort():
            rospy.logwarn("  Failed to switch back to no-RCM effort controller.")
            return False
        prime_gravity_compensation(robot, seconds=0.8, rate_hz=200.0)
    else:
        # 在当前位置保持 tool tip，不下探；先把姿态从初始值平滑切到指定
        # [-90, 0, -45] deg，避免姿态阶跃在接触阶段耦合成 z 向弹跳。
        rospy.loginfo("  Phase 0: Orientation pre-align...")
        prealign_t0 = rospy.Time.now().to_sec()
        hold_pos = rs["tool_position"].copy()
        while not rospy.is_shutdown():
            rs = update_robot_state(kin_tool, kin_flange)
            tp = rs["tool_position"]
            tv = rs["tool_position_velocity"]
            now = rospy.Time.now().to_sec()
            elapsed = now - prealign_t0
            ramp = elapsed / max(float(cfg.approach_orientation_prealign_time), dt)
            ref_euler_prealign = blend_euler_reference(
                ref_euler_start, ref_euler_fixed, ramp
            )
            e_r1 = tp - hold_pos
            if np.all(np.asarray(cfg.approach_position_error_limit) > 0.0):
                e_r1 = np.clip(
                    e_r1,
                    -cfg.approach_position_error_limit,
                    cfg.approach_position_error_limit,
                )
            e_r2 = tv
            if np.all(np.asarray(cfg.approach_velocity_error_limit) > 0.0):
                e_r2 = np.clip(
                    e_r2,
                    -cfg.approach_velocity_error_limit,
                    cfg.approach_velocity_error_limit,
                )
            tau, _, _, integ_euler, _ = compute_torque_no_rcm(
                ctrl, rs, kin_tool,
                e_r1, e_r2, np.zeros(3), sigma_f_int.get(),
                alpha=1.0, K_e_hat=500,
                ref_euler_fixed=ref_euler_prealign,
                integ_euler=integ_euler, dt=dt,
            )
            if cfg.torque_rate_limit > 0.0:
                tau = tau_limiter.update(tau, dt)
            robot.exec_torque_cmd(tau)
            if int(elapsed * cfg.ctrl_rate) % max(1, int(cfg.ctrl_rate)) == 0:
                posture = orientation_diagnostics(
                    robot, rs, ref_euler_fixed, cfg.no_rcm_nominal_joints
                )
                rospy.loginfo(
                    f"  pre-align | t={elapsed:.1f}s, "
                    f"pos_err={np.linalg.norm(tp - hold_pos)*1000:.2f}mm, "
                    f"ori={np.degrees(np.linalg.norm(posture['ori_err_rotvec'])):.2f}deg, "
                    f"front={posture['front_axis_err_deg']:.2f}deg, "
                    f"qd={posture['qd_norm']:.3f}rad/s"
                )
            if elapsed >= cfg.approach_orientation_prealign_time:
                break
            rate.sleep()

    sigma_f_int.reset()
    integ_euler = np.zeros(3)
    tau_limiter.reset()

    # ---- Phase 1: 接近 ----
    # 接近阶段固定 alpha=1.0，以位置控制为主；力误差置零，避免未接触时
    # 因力目标造成不必要的下压命令。
    rospy.loginfo("  Phase 1: Approaching...")
    z_contact = None
    t0 = rospy.Time.now().to_sec()
    last_approach_time = t0
    rs = update_robot_state(kin_tool, kin_flange)
    approach_ref = rs["tool_position"].copy()
    target_xy = np.array([cfg.scan_start_x, cfg.scan_y], dtype=float)
    contact_z_threshold = scan_z_reference(cfg, venv, cfg.scan_start_x)
    z_distance = max(0.0, approach_ref[2] - contact_z_threshold)
    approach_timeout = max(
        float(cfg.approach_timeout),
        1.25 * z_distance / max(float(cfg.approach_z_speed), 1e-6) + 15.0,
    )
    approach_count = 0
    rospy.loginfo(
        f"  Approach target: z<={contact_z_threshold:.4f}m or "
        f"|Fz|>{cfg.contact_force_threshold:.3f}N; "
        f"surface_z={cfg.approach_z:.4f}m, scan_z0={contact_z_threshold:.4f}m, "
        f"xy_target=[{target_xy[0]:.4f}, {target_xy[1]:.4f}], "
        f"z0={approach_ref[2]:.4f}m, xy_speed={cfg.approach_xy_speed:.4f}m/s, "
        f"z_speed={cfg.approach_z_speed:.4f}m/s, "
        f"timeout={approach_timeout:.1f}s"
    )

    scan_u_threshold = ctrl.u_threshold
    scan_u_tool_limits = np.asarray(ctrl.no_rcm_u_tool_limits, dtype=float).copy()
    ctrl.u_threshold = max(float(ctrl.u_threshold), float(cfg.approach_u_threshold))
    ctrl.no_rcm_u_tool_limits = np.asarray(cfg.approach_u_tool_limits, dtype=float)
    rospy.loginfo(
        f"  Approach limits: u_threshold={ctrl.u_threshold:.1f}, "
        f"u_tool_limits={ctrl.no_rcm_u_tool_limits}, "
        f"tau_norm_limit={cfg.approach_tau_norm_limit:.1f}"
    )

    start_wait_t0 = rospy.Time.now().to_sec()
    stable_since = None
    hold_pos = approach_ref.copy()
    while not rospy.is_shutdown():
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]
        now = rospy.Time.now().to_sec()
        speed = float(np.linalg.norm(tv))
        if speed <= cfg.approach_start_speed_limit:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= cfg.approach_start_settle_hold:
                break
        else:
            stable_since = None

        e_r1 = tp - hold_pos
        if np.all(np.asarray(cfg.approach_position_error_limit) > 0.0):
            e_r1 = np.clip(
                e_r1,
                -cfg.approach_position_error_limit,
                cfg.approach_position_error_limit,
            )
        e_r2 = tv
        if np.all(np.asarray(cfg.approach_velocity_error_limit) > 0.0):
            e_r2 = np.clip(
                e_r2,
                -cfg.approach_velocity_error_limit,
                cfg.approach_velocity_error_limit,
            )
        tau, _, _, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, np.zeros(3), sigma_f_int.get(),
            alpha=1.0, K_e_hat=500,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        tau = clamp_norm(tau, cfg.approach_tau_norm_limit)
        if cfg.torque_rate_limit > 0.0:
            tau = tau_limiter.update(tau, dt)
        robot.exec_torque_cmd(tau)
        if now - start_wait_t0 >= cfg.approach_start_settle_timeout:
            rospy.logwarn(
                f"  Approach start settle timeout; continue with |v|={speed:.4f}m/s."
            )
            break
        rate.sleep()
    rs = update_robot_state(kin_tool, kin_flange)
    approach_ref = rs["tool_position"].copy()
    last_approach_time = rospy.Time.now().to_sec()

    while not rospy.is_shutdown() and z_contact is None:
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 当前版本默认使用虚拟接触环境。未接触时 F_z=0。
        F_raw = 0.0
        if venv:
            F_raw = venv.compute_force(
                tp[0], contact_delta_from_surface(cfg, tp[2]), -tv[2]
            )
        F_z = float(force_filter.update(F_raw, dt))

        now = rospy.Time.now().to_sec()
        dt_approach = now - last_approach_time
        if dt_approach <= 0.0 or not np.isfinite(dt_approach):
            dt_approach = dt
        dt_approach = min(max(dt_approach, dt), 0.1)
        last_approach_time = now

        xy_ref_err = float(np.linalg.norm(approach_ref[:2] - target_xy))
        xy_actual_err = float(np.linalg.norm(tp[:2] - target_xy))
        xy_fine_aligned = (
            xy_ref_err <= cfg.approach_xy_tolerance
            and xy_actual_err <= 2.0 * cfg.approach_xy_tolerance
        )
        xy_descend_ready = (
            xy_ref_err <= cfg.approach_xy_descend_tolerance
            and xy_actual_err <= cfg.approach_xy_descend_tolerance
        )
        xy_contact_ready = xy_actual_err <= cfg.approach_contact_xy_tolerance
        force_contact = abs(F_z) > cfg.contact_force_threshold
        height_contact = tp[2] <= contact_z_threshold + cfg.approach_z_margin
        if force_contact or height_contact:
            if not xy_contact_ready:
                rospy.logwarn(
                    f"  Contact before xy alignment "
                    f"(xy_actual={xy_actual_err*1000:.2f}mm, F={F_z:.3f}N, "
                    f"z={tp[2]:.4f}m). Stop trial to avoid lateral dragging."
                )
                ctrl.u_threshold = scan_u_threshold
                ctrl.no_rcm_u_tool_limits = scan_u_tool_limits
                return False
            z_contact = tp[2]
            rospy.loginfo(
                f"  Contact handoff at z={z_contact:.4f}, F={abs(F_z):.3f}N, "
                f"xy_err={xy_actual_err*1000:.2f}mm"
            )
            break

        prev_ref = approach_ref.copy()
        follow_err = float(np.linalg.norm(tp - approach_ref))
        speed_scale = approach_speed_scale(follow_err, cfg)
        slow_suffix = "_slow" if follow_err > cfg.approach_follow_tolerance else ""
        z_step = cfg.approach_z_speed * speed_scale * dt_approach
        probe_floor = contact_z_threshold - cfg.approach_contact_probe_depth
        z_lag_floor = max(probe_floor, tp[2] - cfg.approach_z_lag_limit)
        if xy_descend_ready:
            approach_stage = "descend"
            if xy_fine_aligned:
                approach_ref[:2] = target_xy
            else:
                approach_ref[:2] = slew_toward(
                    approach_ref[:2],
                    target_xy,
                    cfg.approach_xy_speed * speed_scale * dt_approach,
                )
            z_candidate = min(approach_ref[2] - z_step, tp[2] - z_step)
            approach_ref[2] = max(z_lag_floor, z_candidate)
        else:
            approach_stage = "align_xy"
            if xy_actual_err > cfg.approach_xy_hold_z_tolerance:
                approach_ref[:2] = slew_toward(
                    approach_ref[:2],
                    target_xy,
                    cfg.approach_xy_speed * speed_scale * dt_approach,
                )
                approach_ref[2] = tp[2]
            else:
                approach_ref[:2] = slew_toward(
                    approach_ref[:2],
                    target_xy,
                    cfg.approach_xy_speed * speed_scale * dt_approach,
                )
                precontact_floor = max(
                    contact_z_threshold + cfg.approach_precontact_clearance,
                    z_lag_floor,
                )
                z_candidate = min(approach_ref[2] - z_step, tp[2] - z_step)
                approach_ref[2] = max(precontact_floor, z_candidate)
        approach_stage += slow_suffix

        x_ref_p1 = approach_ref.copy()
        xdot_ref = (approach_ref - prev_ref) / dt_approach
        if approach_stage.startswith("descend"):
            xdot_ref[2] = -cfg.approach_z_speed * speed_scale
        elif not approach_stage.startswith("align_xy"):
            xdot_ref[:] = 0.0

        # 接近阶段只构造位置/速度误差；e_f 和 σ_f 均不参与控制。
        e_r1 = tp - x_ref_p1
        if np.all(np.asarray(cfg.approach_position_error_limit) > 0.0):
            e_r1 = np.clip(
                e_r1,
                -cfg.approach_position_error_limit,
                cfg.approach_position_error_limit,
            )
        e_r2 = tv - xdot_ref
        if np.all(np.asarray(cfg.approach_velocity_error_limit) > 0.0):
            e_r2 = np.clip(
                e_r2,
                -cfg.approach_velocity_error_limit,
                cfg.approach_velocity_error_limit,
            )
        e_f = np.zeros(3)
        sigma_f = sigma_f_int.get()
        ramp = (now - t0) / max(float(cfg.approach_orientation_ramp_time), dt)
        ref_euler_approach = blend_euler_reference(
            ref_euler_start, ref_euler_fixed, ramp
        )

        # 仍走统一的 no-RCM 力矩装配函数，保持接口和扫描阶段一致。
        tau, u_tool, K_eff, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f, sigma_f,
            alpha=1.0, K_e_hat=500,
            ref_euler_fixed=ref_euler_approach,
            integ_euler=integ_euler, dt=dt,
        )
        z_track_gap = float(tp[2] - x_ref_p1[2])
        extra_u_tool = np.zeros(3)
        if (
            F_z < cfg.contact_force_threshold
            and approach_stage.startswith("descend")
            and z_track_gap > cfg.approach_extra_down_gap
        ):
            extra = cfg.approach_extra_down_kp * (
                z_track_gap - cfg.approach_extra_down_gap
            )
            extra_u_tool[2] = -min(cfg.approach_extra_down_force_limit, extra)
            J_tool = np.array(kin_tool.jacobian())
            tau = tau + (J_tool[:3, :].T @ extra_u_tool).flatten()
            u_tool = u_tool + extra_u_tool
        tau = clamp_norm(tau, cfg.approach_tau_norm_limit)
        if cfg.torque_rate_limit > 0.0:
            tau = tau_limiter.update(tau, dt_approach)
        robot.exec_torque_cmd(tau)
        rate.sleep()
        approach_count += 1

        if approach_count % 10 == 0:
            rospy.loginfo(
                f"  approaching({approach_stage}) | "
                f"tool=[{tp[0]*1000:6.2f},{tp[1]*1000:6.2f},{tp[2]*1000:6.2f}]mm, "
                f"ref=[{x_ref_p1[0]*1000:6.2f},{x_ref_p1[1]*1000:6.2f},{x_ref_p1[2]*1000:6.2f}]mm, "
                f"target_z<={contact_z_threshold:.4f}m, F={F_z:.3f}N, "
                f"follow={follow_err*1000:.2f}mm, xy={xy_actual_err*1000:.2f}mm, "
                f"scale={speed_scale:.2f}, vz={tv[2]:+.4f}m/s, "
                f"u_z={u_tool[2]:+.2f}N, extra_z={extra_u_tool[2]:+.2f}N, "
                f"Kp={K_eff[0]:.1f}, "
                f"dt={dt_approach*1000:.1f}ms"
            )

        if rospy.Time.now().to_sec() - t0 > approach_timeout:
            rospy.logwarn(
                f"  Approach timeout "
                f"(stage={approach_stage}, z={tp[2]:.4f}, ref={approach_ref}, "
                f"target_z<={contact_z_threshold:.4f}, F={F_z:.3f}N, "
                f"follow={follow_err:.4f}, xy={xy_actual_err:.4f}, "
                f"timeout={approach_timeout:.1f}s)"
            )
            ctrl.u_threshold = scan_u_threshold
            ctrl.no_rcm_u_tool_limits = scan_u_tool_limits
            return False

    if z_contact is None:
        z_contact = contact_z_threshold

    ctrl.u_threshold = scan_u_threshold
    ctrl.no_rcm_u_tool_limits = scan_u_tool_limits
    sigma_f_int.reset()
    integ_euler = np.zeros(3)
    tau_limiter.reset()
    force_filter.reset(F_z)
    stiffness_filter.reset(cfg.estimator_initial_K)
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
    adjust_last_time = adjust_t0
    adjust_last_log = 0.0
    adjust_forces = []
    adjust_z_values = []
    adjust_pos_errors = []
    adjust_xy_errors = []
    adjust_window = max(3, int(cfg.adjustment_stable_window * cfg.ctrl_rate))
    delta_dot_est = ContactDeltaDotEstimator(
        tau=cfg.contact_delta_dot_filter_tau,
        limit=cfg.contact_delta_dot_limit,
    )

    while not rospy.is_shutdown():
        now_adjust = time.time()
        adjust_t = now_adjust - adjust_t0
        scan_dt = now_adjust - adjust_last_time
        if scan_dt <= 0.0 or not np.isfinite(scan_dt):
            scan_dt = dt
        scan_dt = min(max(scan_dt, dt), 0.1)
        adjust_last_time = now_adjust

        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        delta = contact_delta_from_surface(cfg, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, scan_dt, raw_delta_dot=-tv[2]
        )
        F_raw = venv.compute_force(tp[0], delta, delta_dot) if venv else 0.0
        F_z = float(force_filter.update(F_raw, scan_dt))
        K_hat_raw, B_hat = est.update(abs(F_z), delta, delta_dot)
        K_hat = float(stiffness_filter.update(K_hat_raw, scan_dt))
        K_hat_ctrl = min(K_hat, cfg.scan_z_gain_khat_max) \
            if cfg.scan_z_gain_khat_max > 0.0 else K_hat

        z_scan_ref = scan_z_reference(cfg, venv, x_cur)
        x_ref = np.array([x_cur, cfg.scan_y, z_scan_ref])
        xdot_ref = np.zeros(3)
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        e_r1 = tp - x_ref
        if cfg.scan_position_error_limit > 0.0:
            e_r1 = np.clip(
                e_r1,
                -cfg.scan_position_error_limit,
                cfg.scan_position_error_limit,
            )
        e_r2 = tv - xdot_ref
        if cfg.scan_velocity_error_limit > 0.0:
            e_r2 = np.clip(
                e_r2,
                -cfg.scan_velocity_error_limit,
                cfg.scan_velocity_error_limit,
            )
        e_f_vec = F_meas - F_des
        if cfg.scan_force_error_soft_limit > 0.0:
            e_f_vec = np.clip(
                e_f_vec,
                -cfg.scan_force_error_soft_limit,
                cfg.scan_force_error_soft_limit,
            )
        force_blend = min(1.0, max(0.0, adjust_t / cfg.contact_force_blend_time))
        e_f_vec_ctrl = force_blend * e_f_vec
        sigma_f = sigma_f_int.update(e_f_vec_ctrl)

        F_actual = abs(F_z)
        e_f_scalar = F_actual - cfg.F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt; prev_ef = e_f_scalar
        e_r_scalar = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
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
            alpha=alpha, K_e_hat=K_hat_ctrl,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        if cfg.torque_rate_limit > 0.0:
            tau = tau_limiter.update(tau, scan_dt)
        robot.exec_torque_cmd(tau)

        adjust_forces.append(F_actual)
        adjust_z_values.append(tp[2])
        adjust_pos_errors.append(np.linalg.norm(tp - x_ref))
        adjust_xy_errors.append(np.linalg.norm(tp[:2] - x_ref[:2]))
        if len(adjust_forces) > adjust_window:
            adjust_forces.pop(0)
            adjust_z_values.pop(0)
            adjust_pos_errors.pop(0)
            adjust_xy_errors.pop(0)

        force_std = float(np.std(adjust_forces)) if len(adjust_forces) > 1 else float("inf")
        z_std = float(np.std(adjust_z_values)) if len(adjust_z_values) > 1 else float("inf")
        pos_std = float(np.std(adjust_pos_errors)) if len(adjust_pos_errors) > 1 else float("inf")
        xy_err = float(adjust_xy_errors[-1]) if adjust_xy_errors else float("inf")
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
            and xy_err <= cfg.adjustment_xy_error
        )

        if adjust_t - adjust_last_log >= 0.5:
            adjust_last_log = adjust_t
            rospy.loginfo(
                f"  adjusting | t={adjust_t:.2f}s, F={F_actual:.3f}N, "
                f"F_std={force_std:.3f}, F_slope={force_slope:.3f}N/s, "
                f"z_std={z_std*1000:.3f}mm, pos_std={pos_std*1000:.3f}mm, "
                f"xy_err={xy_err*1000:.2f}mm, "
                f"blend={force_blend:.2f}, K={K_hat_ctrl:.1f}, "
                f"rawK={K_hat_raw:.1f}, B={B_hat:.2f}"
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
                f"F_slope={force_slope:.3f}N/s, z_std={z_std*1000:.3f}mm, "
                f"xy_err={xy_err*1000:.2f}mm)."
            )
            break
        rate.sleep()

    # ---- Phase 2: 恒力扫描 ----
    # 扫描阶段 x 方向按 scan_vx 匀速推进，z 方向由力位控制器自动调节，
    # 目标是在跟踪扫描路径的同时把接触反力维持在 F_desired 附近。
    ctrl.no_rcm_direct_posture_enabled = bool(cfg.no_rcm_direct_posture_scan_enabled)
    rospy.loginfo(
        "  Direct wrist posture regularization "
        f"{'enabled' if ctrl.no_rcm_direct_posture_enabled else 'disabled'} for scanning."
    )
    rospy.loginfo("  Phase 2: Scanning...")
    t_scan = time.time()
    last_scan_time = t_scan

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        now_scan = time.time()
        t = now_scan - t_scan
        scan_dt = now_scan - last_scan_time
        if scan_dt <= 0.0 or not np.isfinite(scan_dt):
            scan_dt = dt
        scan_dt = min(max(scan_dt, dt), 0.1)
        last_scan_time = now_scan
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 计算当前虚拟接触力，并用同一压入量送入环境估计器。
        delta = contact_delta_from_surface(cfg, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, scan_dt, raw_delta_dot=-tv[2]
        )
        if venv:
            K_env_true, B_env_true = venv.get_stiffness(tp[0])
        else:
            K_env_true, B_env_true = 0.0, 0.0
        F_raw = venv.compute_force(tp[0], delta, delta_dot) if venv else 0.0
        F_z = float(force_filter.update(F_raw, scan_dt))

        K_hat_raw, B_hat = est.update(
            abs(F_z),
            delta,
            delta_dot,
        )
        K_hat = float(stiffness_filter.update(K_hat_raw, scan_dt))
        K_hat_ctrl = min(K_hat, cfg.scan_z_gain_khat_max) \
            if cfg.scan_z_gain_khat_max > 0.0 else K_hat

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
        if cfg.scan_position_error_limit > 0.0:
            e_r1 = np.clip(
                e_r1,
                -cfg.scan_position_error_limit,
                cfg.scan_position_error_limit,
            )
        e_r2 = tv - xdot_ref
        if cfg.scan_velocity_error_limit > 0.0:
            e_r2 = np.clip(
                e_r2,
                -cfg.scan_velocity_error_limit,
                cfg.scan_velocity_error_limit,
            )
        e_f_vec = F_meas - F_des
        if cfg.scan_force_error_soft_limit > 0.0:
            e_f_vec = np.clip(
                e_f_vec,
                -cfg.scan_force_error_soft_limit,
                cfg.scan_force_error_soft_limit,
            )

        # 接触调整阶段已经完成力控渐入，正式扫描从完整力控开始。
        force_blend = 1.0
        e_f_vec_ctrl = force_blend * e_f_vec
        sigma_f = sigma_f_int.update(e_f_vec_ctrl)

        F_actual = abs(F_z)
        F_desired = cfg.F_desired
        e_f_scalar = F_actual - F_desired
        e_f_dot = (e_f_scalar - prev_ef) / dt; prev_ef = e_f_scalar
        e_r_scalar = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
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
            alpha=alpha, K_e_hat=K_hat_ctrl,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        if cfg.torque_rate_limit > 0.0:
            tau = tau_limiter.update(tau, scan_dt)
        robot.exec_torque_cmd(tau)
        x_cur += cfg.scan_vx * scan_dt

        pos_tool = tp.copy()
        pos_tool_des = x_ref.copy()
        pos_err = pos_tool - pos_tool_des
        pos_err_norm = np.linalg.norm(pos_err)
        F_err = F_actual - F_desired
        posture = orientation_diagnostics(
            robot, rs, ref_euler_fixed, cfg.no_rcm_nominal_joints
        )

        if logger.count % 10 == 0:
            rospy.loginfo(
                f"  t={t:5.2f}s | "
                f"tool=[{pos_tool[0]*1000:6.2f},{pos_tool[1]*1000:6.2f},{pos_tool[2]*1000:6.2f}]mm | "
                f"des=[{pos_tool_des[0]*1000:6.2f},{pos_tool_des[1]*1000:6.2f},{pos_tool_des[2]*1000:6.2f}]mm | "
                f"err={pos_err_norm*1000:5.2f}mm | "
                f"F={F_actual:.3f}N (des={F_desired:.3f}, err={F_err:+.3f}, raw={F_raw:.3f}) | "
                f"K={K_hat_ctrl:7.1f} rawK={K_hat_raw:7.1f} B={B_hat:5.2f} | "
                f"e_f={e_f_scalar:+.3f} e_r={e_r_scalar*1000:5.2f}mm | "
                f"ori={np.degrees(np.linalg.norm(posture['ori_err_rotvec'])):.2f}deg "
                f"front={posture['front_axis_err_deg']:.2f}deg | "
                f"qerr={np.max(np.abs(posture['q_err'])):.3f}rad | "
                f"alpha={alpha:.2f} phase={phase_val}"
            )

        # 保存所有关键状态，后续可直接从 npz 统计力 RMSE、位置误差和 alpha 曲线。
        logger.log(
            t=t, pos=pos_tool,
            pos_des=pos_tool_des,
            pos_err=pos_err,
            F_measured=F_actual, F_desired=F_desired,
            F_err=F_err,
            F_raw=F_raw,
            wrench=np.zeros(6),
            force_source='virtual' if venv else 'none',
            sensor_available=0,
            e_f=e_f_scalar, e_r=e_r_scalar,
            sigma_f_norm=np.linalg.norm(sigma_f),
            e_r1_norm=np.linalg.norm(e_r1),
            alpha=alpha, K_hat=K_hat,
            K_hat_raw=K_hat_raw,
            K_hat_ctrl=K_hat_ctrl,
            B_hat=B_hat,
            delta=delta, delta_dot=delta_dot,
            delta_dot_raw=delta_dot_raw, delta_dot_fd=delta_dot_fd,
            K_env_true=K_env_true, B_env_true=B_env_true,
            K_eff=K_eff,
            x_desired=x_cur,
            error_rcm=0.0, error_track=error[1],
            arbitration_strategy=sched.name,
            u_norm=np.linalg.norm(tau),
            force_blend=force_blend,
            contact_plane_z=cfg.approach_z,
            scan_z_ref=z_scan_ref,
            phase=phase_val,
            q=posture["q"],
            qd=posture["qd"],
            q_err=posture["q_err"],
            qd_norm=posture["qd_norm"],
            tool_euler=posture["tool_euler"],
            tool_euler_ref=posture["tool_euler_ref"],
            ori_err_rotvec=posture["ori_err_rotvec"],
            tool_omega=posture["tool_omega"],
            front_axis_err_deg=posture["front_axis_err_deg"],
        )
        rate.sleep()

    rospy.loginfo("  Phase 3: Retreating...")
    # 退回阶段通知调度器进入 RETREAT，使 alpha 回到位置优先。
    if hasattr(sched, 'set_retreat'):
        sched.set_retreat(True)
    if builtin_phases is not None:
        if not builtin_phases.switch_to_trajectory():
            retreat_ok = False
        else:
            retreat_ok = builtin_phases.move_joints(
                INIT_JOINTS,
                duration=cfg.builtin_retreat_duration,
                timeout=cfg.builtin_retreat_duration + cfg.builtin_goal_timeout_margin,
            )
    else:
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
        choices=['default', 'force_margin_challenge'],
        help='实验场景: default 为原始力位一致扫描; force_margin_challenge 为固定 z + 强刚度变化压力测试',
    )
    args = ap.parse_args()

    rospy.init_node("coop_gt_no_rcm")
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Cooperative Game Force-Position Experiment (NO RCM)")
    rospy.loginfo("=" * 60)

    robot = PandaArm()
    prime_gravity_compensation(robot, seconds=1.0, rate_hz=200.0)
    builtin_phases = BuiltinPhaseController(robot, robot)
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
        f"tau={cfg.continuous_smooth_tau:.2f}s"
    )

    # control_mode='are' 使用原始 4D ARE 查表；
    # control_mode='pareto_iter' 使用 Algorithm 2 迭代生成同形状增益表。
    ctrl = CooperativeGameController(control_mode=args.controller_mode)
    ctrl.u_threshold = cfg.no_rcm_u_threshold
    ctrl.P_ori = cfg.no_rcm_P_ori
    ctrl.D_ori = cfg.no_rcm_D_ori
    ctrl.I_ori = cfg.no_rcm_I_ori
    ctrl.no_rcm_u_tool_limits = cfg.no_rcm_u_tool_limits
    ctrl.no_rcm_xy_stiffness_boost = cfg.no_rcm_xy_stiffness_boost
    ctrl.no_rcm_xy_damping_boost = cfg.no_rcm_xy_damping_boost
    ctrl.no_rcm_u_rot_limit = cfg.no_rcm_u_rot_limit
    ctrl.no_rcm_u_rot_rate_limit = cfg.no_rcm_u_rot_rate_limit
    ctrl.no_rcm_omega_limit = cfg.no_rcm_omega_limit
    ctrl.no_rcm_omega_filter_tau = cfg.no_rcm_omega_filter_tau
    ctrl.no_rcm_nominal_joints = cfg.no_rcm_nominal_joints
    ctrl.no_rcm_nullspace_enabled = cfg.no_rcm_nullspace_enabled
    ctrl.no_rcm_null_kp = cfg.no_rcm_null_kp
    ctrl.no_rcm_null_kd = cfg.no_rcm_null_kd
    ctrl.no_rcm_null_tau_max = cfg.no_rcm_null_tau_max
    ctrl.no_rcm_null_position_weights = cfg.no_rcm_null_position_weights
    ctrl.no_rcm_null_velocity_weights = cfg.no_rcm_null_velocity_weights
    ctrl.no_rcm_direct_posture_enabled = False
    ctrl.no_rcm_direct_posture_weights = cfg.no_rcm_direct_posture_weights
    ctrl.no_rcm_direct_posture_kp = cfg.no_rcm_direct_posture_kp
    ctrl.no_rcm_direct_posture_kd = cfg.no_rcm_direct_posture_kd
    ctrl.no_rcm_direct_posture_tau_max = cfg.no_rcm_direct_posture_tau_max
    rospy.loginfo(f"Controller mode: {args.controller_mode}")
    rospy.loginfo(
        f"NO-RCM Gazebo gains: u_threshold={ctrl.u_threshold:.1f}, "
        f"ori_gains=({ctrl.P_ori:.1f},{ctrl.D_ori:.1f},{ctrl.I_ori:.1f}), "
        f"u_tool_limits={cfg.no_rcm_u_tool_limits}, "
        f"xy_boost=({cfg.no_rcm_xy_stiffness_boost:.1f},"
        f"{cfg.no_rcm_xy_damping_boost:.1f}), "
        f"u_rot_limit={cfg.no_rcm_u_rot_limit:.1f}, "
        f"u_rot_rate_limit={cfg.no_rcm_u_rot_rate_limit:.1f}, "
        f"omega_limit={cfg.no_rcm_omega_limit:.1f}, "
        f"omega_filter_tau={cfg.no_rcm_omega_filter_tau:.2f}, "
        f"null=({cfg.no_rcm_null_kp:.1f},{cfg.no_rcm_null_kd:.1f},"
        f"{cfg.no_rcm_null_tau_max:.1f}, enabled={cfg.no_rcm_nullspace_enabled})"
    )

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
                builtin_phases=builtin_phases,
            )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")
    if not args.no_auto_plot:
        auto_plot_results(odir, no_show=args.plot_no_show)


if __name__ == '__main__':
    main()
