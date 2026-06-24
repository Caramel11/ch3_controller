#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序 B: 无 RCM 约束的合作博弈力-位扫描实验 (ROS 1 真机版)
===========================================================

控制链路:
  1. 读取 tool 端位姿/速度
  2. 实物六维力传感器优先生成接触力，必要时可显式启用虚拟环境回退
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
  python run_no_rcm_real.py --strategy continuous_force_margin --controller-mode pareto_iter \
      --force-port /dev/ttyUSB0 --force-tare-on-start
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
    VirtualStiffnessSurface, DataLogger,
    FirstOrderLowPass, VectorRateLimiter, ContactDeltaDotEstimator,
)
from src.force_sensor_direct import DirectForceSensorInput
from src.leaky_integrator import LeakyIntegrator
from src.robot_interface import (
    update_robot_state, safe_move_to_joint_position,
    compute_torque_no_rcm, fixed_downward_tool_euler, INIT_JOINTS,
)


class Config:
    """实验全局参数。

    当前真机版本参考 tests/no_rcm_controller_ros2/no_rcm_ros2_real_node.py
    的实物实验几何、接触区域、刚度估计和力控限幅；控制器接口仍保持
    0603/run_no_rcm.py 的 ROS 1 PandaArm 路径。
    """

    # 扫描几何参数，单位均为 m；no-RCM 真机入口只约束 tool 端位置。
    # Phase 2 中 x 从 scan_start_x 匀速走到 scan_end_x，y 为扫描线横向坐标。
    scan_start_x = 0.43      # 扫描起点 x
    scan_end_x   = 0.50      # 扫描终点 x
    scan_y       = 0.0       # 扫描线 y 坐标
    # approach_z 是实物接触检测平面；接触后 scan_z 会更新为
    # 实际接触平面 z_contact - contact_penetration_depth。
    approach_z   = 0.05     # 名义表面高度，仅用于接触检测参考
    contact_penetration_depth = 0.005  # 触碰后目标压入深度, m
    scan_z       = approach_z - contact_penetration_depth
    approach_gain_khat = 3000.0  # 接近阶段等效环境刚度上限, N/m
    scan_vx      = 0.002     # Phase 2 沿 x 的扫描速度, m/s
    scan_static_wait_time = 0.5  # 正式扫描前静态等待/调整最短时间, s
    scan_position_error_limit = 0.0040  # 位置误差软限, m
    scan_velocity_error_limit = 0.015   # 速度误差软限, m/s
    scan_force_output_limit = 8.0       # 力控通道输出限幅, N
    scan_force_error_soft_limit = 1.0   # 力误差软饱和边界, N
    scan_force_output_rate_limit = 25.0 # 力控输出变化率限幅, N/s
    scan_z_gain_khat_max = 700.0        # z 向控制使用的刚度估计上限, N/m

    # 力目标、调度边界与刚度估计初始化。
    F_desired    = 3.0       # 期望法向接触力, N
    F_min        = 0.3       # alpha 调度的低力边界, N
    F_max        = 10.0      # alpha 调度和安全监测的高力边界, N
    force_axis   = 2         # 力控制轴，2 表示 z 轴
    estimator_initial_K = 1500.0  # 初始环境刚度估计, N/m
    estimator_initial_B = 5.0     # 初始环境阻尼估计, Ns/m
    estimator_alpha_lp = 0.08     # 估计值低通更新系数

    # 分段环境参数: 真机版仅用于可选虚拟环境回退和离线结果兼容。
    # 每项为 (x_start, x_end, K_e, B_e)，K_e 单位 N/m，B_e 单位 Ns/m。
    stiffness_zones = [
        (0.40, 0.44, 300, 5),
        (0.44, 0.48, 500, 8),
    ]

    # 控制周期与泄漏积分器参数；eps 越大，积分记忆衰减越快。
    ctrl_rate = 100         # 主控制循环频率, Hz
    dt = 1.0 / ctrl_rate
    eps_r = 1.0             # 位置误差积分泄漏系数
    eps_f = 1.2             # 力误差积分泄漏系数

    # 接触检测与稳定判据。Phase 1 下探到触碰，Phase 1.5 等接触稳定，
    # 正式扫描数据从 Phase 2 才开始记录。
    settle_time = 2.0                  # 初始关节位姿到达后的静置时间, s
    contact_force_threshold = 0.2      # 判定触碰表面的最小接触力, N
    contact_z_tolerance = 0.0032       # 接触高度判定容差, m
    contact_settle_time = 1.0          # 接触稳定滑动窗口长度, s
    contact_stable_z_std = 0.0003      # 稳定窗口内 z 标准差阈值, m
    contact_stable_force_std = 0.05    # 稳定窗口内力标准差阈值, N
    contact_stable_force_slope = 0.20  # 稳定窗口首尾力变化率阈值, N/s
    contact_penetration_ramp_time = 1.0  # 触碰后压入深度 S 曲线时间, s
    contact_force_blend_time = 0.1       # 力控通道平滑引入时间, s
    contact_delta_dot_filter_tau = 0.05  # 压入速度低通时间常数, s
    contact_delta_dot_limit = 0.03       # 压入速度限幅, m/s

    # 接触调整阶段，不写入实验数据；用于滤掉刚接触瞬态震荡。
    adjustment_min_time = scan_static_wait_time
    adjustment_timeout = 8.0           # 调整阶段最长等待时间, s
    adjustment_stable_window = 0.8     # 调整稳定统计窗口, s
    adjustment_force_error = 0.35      # 允许的稳态力误差, N
    adjustment_pos_std = 0.0005        # 允许的位置误差标准差, m

    # 滤波、限幅和 no-RCM 姿态控制参数。
    force_filter_tau = 0.02       # 传感器力一阶低通时间常数, s
    stiffness_filter_tau = 0.12   # 控制用 K_hat 低通时间常数, s
    torque_rate_limit = 120.0     # 关节力矩命令变化率限幅, Nm/s
    loop_warn_period = 0.08       # 控制循环超时告警阈值, s
    no_rcm_u_threshold = 20.0     # no-RCM 笛卡尔控制输出软限
    retreat_timeout = 25.0        # Phase 3 小步关节退回超时, s
    no_rcm_P_ori = 12.0           # tool 姿态比例增益
    no_rcm_D_ori = 2.0            # tool 姿态微分增益
    no_rcm_I_ori = 5.0            # tool 姿态积分增益
    no_rcm_alpha_floor = 0.65     # 真机 alpha 下限，避免过度力控
    no_rcm_u_tool_limits = np.array([8.0, 8.0, 6.0])  # tool 输出逐轴限幅

    # Phase 1 下探参数；真实机械臂速度不要一次调太大。
    approach_speed = 0.03       # 下探参考速度, m/s；原 0.002，适度加快
    approach_extra_down_gap = 0.004       # z 跟踪落后超过该值才补下压力, m
    approach_extra_down_kp = 250.0        # 接近阶段额外下压力比例增益, N/m
    approach_extra_down_force_limit = 4.0 # 额外下压力限幅, N
    approach_timeout_min = 160.0  # 接近阶段最小超时, s
    approach_timeout_margin = 8.0 # 按距离估算超时后的额外余量, s
    allow_sensorless_approach = False  # 真机默认必须有力传感/虚拟力来源
    sensor_unavailable_abort_time = 2.0


def smoothstep01(s):
    """0-1 S 曲线，用于真机接触后平滑压入。"""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


def contact_delta_from_surface(surface_z, z):
    """以给定接触平面作为表面高度，计算单向压入量 δ。

    z >= surface_z 表示未接触，返回 0；z < surface_z 表示压入表面，
    返回 surface_z - z。真机扫描中 surface_z 来自实际接触瞬间位置。
    """
    return max(0.0, float(surface_z) - float(z))


def read_contact_force(force_sensor, venv, x, delta, delta_dot):
    """
    获取当前接触力。

    实物串口传感器新鲜时优先使用；仅在命令行显式启用虚拟环境时回退到
    Kelvin-Voigt 虚拟力。contact_force() 保持“z 方向下压力取正”的约定。
    """
    if force_sensor is not None and hasattr(force_sensor, "update_contact_state"):
        force_sensor.update_contact_state(x, delta, delta_dot)

    if force_sensor is not None and force_sensor.available():
        return (
            force_sensor.contact_force(),
            force_sensor.wrench_vector(),
            getattr(force_sensor, "source_name", "sensor"),
            True,
        )

    F_z = venv.compute_force(x, delta, delta_dot) if venv is not None else 0.0
    return (
        F_z,
        np.zeros(6),
        "virtual" if venv is not None else "none",
        False,
    )


def run_trial(robot, kin_tool, kin_flange,
              cfg, ctrl, sched, est, venv, force_sensor, logger, trial_id):
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
    stiffness_filter = FirstOrderLowPass(cfg.stiffness_filter_tau, initial=cfg.estimator_initial_K)
    tau_limiter = VectorRateLimiter(cfg.torque_rate_limit)
    last_wall_time = time.time()

    # 保存上一控制周期的物理量，用于计算变化率输入。
    prev_ef = 0.0; prev_er = 0.0; prev_K = cfg.estimator_initial_K

    est.reset()
    sched.reset()
    sigma_f_int.reset()

    rospy.loginfo(f"[Trial {trial_id}] {sched.name} (NO-RCM)")

    safe_move_to_joint_position(robot, INIT_JOINTS)
    rospy.sleep(cfg.settle_time)
    rs = update_robot_state(kin_tool, kin_flange)
    ref_euler_fixed = fixed_downward_tool_euler()
    rospy.loginfo(f"  Tool at: {rs['tool_position']}, "
                  f"fixed downward euler ref: {ref_euler_fixed}")

    # ---- Phase 1: 接近 ----
    # 接近阶段固定 alpha=1.0，以位置控制为主；力误差置零，避免未接触时
    # 因力目标造成不必要的下压命令。
    rospy.loginfo("  Phase 1: Approaching...")
    z_contact = None
    t0 = rospy.Time.now().to_sec()
    last_approach_time = t0
    z_ref_approach = rs["tool_position"][2]
    xy_ref_approach = np.array([cfg.scan_start_x, cfg.scan_y])
    surface_z = cfg.approach_z
    scan_z_ref = surface_z - cfg.contact_penetration_depth
    approach_timeout = max(
        cfg.approach_timeout_min,
        max(0.0, z_ref_approach - scan_z_ref)
        / max(cfg.approach_speed, 1e-6)
        + cfg.approach_timeout_margin,
    )
    approach_count = 0
    rospy.loginfo(
        f"  Approach target: contact force>{cfg.contact_force_threshold:.3f}N "
        f"and z<=surface_z={surface_z:.4f}m; "
        f"nominal_scan_z={scan_z_ref:.4f}m, "
        f"xy_ref=[{xy_ref_approach[0]:.4f}, {xy_ref_approach[1]:.4f}], "
        f"z0={z_ref_approach:.4f}m, speed={cfg.approach_speed:.4f}m/s, "
        f"timeout={approach_timeout:.1f}s"
    )
    surface_touched = False
    contact_plane_z = None
    penetration_t0 = None
    settle_start = None
    settle_forces = []
    settle_z_values = []
    settle_window = max(3, int(cfg.contact_settle_time * cfg.ctrl_rate))
    sensor_unavailable_since = None
    no_progress_start = None
    no_progress_z = None

    while not rospy.is_shutdown() and z_contact is None:
        now_wall = time.time()
        wall_dt = now_wall - last_wall_time
        last_wall_time = now_wall
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        now = rospy.Time.now().to_sec()
        dt_approach = now - last_approach_time
        if dt_approach <= 0.0 or not np.isfinite(dt_approach):
            dt_approach = dt
        dt_approach = min(max(dt_approach, dt), 0.1)
        last_approach_time = now

        force_surface_z = contact_plane_z if contact_plane_z is not None else surface_z
        F_raw, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0],
            contact_delta_from_surface(force_surface_z, tp[2]),
            -tv[2],
        )
        F_z = float(force_filter.update(F_raw, dt_approach))

        sensor_ready = sensor_available or venv is not None or cfg.allow_sensorless_approach
        if not sensor_ready:
            if sensor_unavailable_since is None:
                sensor_unavailable_since = now
            if now - sensor_unavailable_since > cfg.sensor_unavailable_abort_time:
                rospy.logwarn(
                    f"  Abort approach: force sensor unavailable for "
                    f"{cfg.sensor_unavailable_abort_time:.1f}s "
                    f"(z={tp[2]:.4f}, source={force_source})"
                )
                return False
        else:
            sensor_unavailable_since = None

        contact_by_force = sensor_ready and abs(F_z) > cfg.contact_force_threshold
        contact_by_height = tp[2] <= surface_z
        if not surface_touched and contact_by_force:
            surface_touched = True
            contact_plane_z = float(tp[2])
            scan_z_ref = contact_plane_z - cfg.contact_penetration_depth
            penetration_t0 = now
            z_contact = contact_plane_z
            rospy.loginfo(
                f"  Surface contact detected at z={contact_plane_z:.4f}, "
                f"F={F_z:.3f}N (raw={F_raw:.3f}, {force_source}); "
                f"height_ok={contact_by_height}, scan_z={scan_z_ref:.4f}, "
                f"penetration={cfg.contact_penetration_depth*1000:.1f}mm. "
                "Handing off to contact adjustment."
            )
            break

        # 期望 z 缓慢下降；接触后使用 S 曲线平滑压入到 scan_z_ref。
        if surface_touched and penetration_t0 is not None and contact_plane_z is not None:
            s = smoothstep01((now - penetration_t0) / max(cfg.contact_penetration_ramp_time, 1e-9))
            z_ref_approach = contact_plane_z - s * cfg.contact_penetration_depth
            z_ref_approach = max(scan_z_ref, z_ref_approach)
            z_dot_ref = (
                -cfg.contact_penetration_depth / max(cfg.contact_penetration_ramp_time, 1e-9)
                if s < 1.0 else 0.0
            )
        elif sensor_ready:
            z_ref_approach -= cfg.approach_speed * dt_approach
            z_ref_approach = max(z_ref_approach, scan_z_ref)
            z_dot_ref = -cfg.approach_speed
        else:
            z_ref_approach = max(z_ref_approach, tp[2])
            z_dot_ref = 0.0

        if surface_touched:
            z_ref_approach = max(z_ref_approach, scan_z_ref)
        xy_ref_control = xy_ref_approach if sensor_ready else tp[:2]
        xdot_ref = np.array([0.0, 0.0, z_dot_ref])
        if surface_touched and z_ref_approach <= scan_z_ref:
            xdot_ref = np.zeros(3)
        x_ref_p1 = np.array([
            xy_ref_control[0],
            xy_ref_control[1],
            z_ref_approach,
        ])
        # 接近阶段只构造位置/速度误差；e_f 和 σ_f 均不参与控制。
        e_r1 = tp - x_ref_p1
        e_r2 = tv - xdot_ref
        e_f = np.zeros(3)
        sigma_f = sigma_f_int.get()

        # 仍走统一的 no-RCM 力矩装配函数，保持接口和扫描阶段一致。
        tau, u_tool, K_eff, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f, sigma_f,
            alpha=1.0, K_e_hat=cfg.approach_gain_khat,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        z_track_gap = tp[2] - z_ref_approach
        extra_u_tool = np.zeros(3)
        if (
            not surface_touched
            and sensor_ready
            and z_track_gap > cfg.approach_extra_down_gap
        ):
            extra = cfg.approach_extra_down_kp * (
                z_track_gap - cfg.approach_extra_down_gap
            )
            extra_u_tool[2] = -min(cfg.approach_extra_down_force_limit, extra)
            J_tool = np.array(kin_tool.jacobian())
            tau = tau + (J_tool[:3, :].T @ extra_u_tool).flatten()
            u_tool = u_tool + extra_u_tool
        tau = tau_limiter.update(tau, dt_approach)
        robot.exec_torque_cmd(tau)
        if wall_dt > cfg.loop_warn_period:
            rospy.logwarn_throttle(
                1.0,
                f"  NO-RCM wall-loop slow in approach: wall_dt={wall_dt*1000:.1f}ms "
                f"(target={dt*1000:.1f}ms)"
            )
        rate.sleep()
        approach_count += 1

        if approach_count % 10 == 0:
            rospy.loginfo(
                f"  approaching | z={tp[2]:.4f}m, z_ref={z_ref_approach:.4f}m, "
                f"target_z<={scan_z_ref:.4f}m, F={F_z:.3f}N, raw={F_raw:.3f}N, "
                f"source={force_source}, sensor={sensor_available}, "
                f"force_hit={contact_by_force}, height_hit={contact_by_height}, "
                f"vz={tv[2]:+.4f}m/s, u_z={u_tool[2]:+.3f}N, "
                f"|tau|={np.linalg.norm(tau):.3f}Nm, Kp={K_eff[0]:.1f}, "
                f"touched={surface_touched}, "
                f"dt={dt_approach*1000:.1f}ms"
            )

        if (
            not surface_touched
            and sensor_ready
            and z_track_gap > 0.010
            and abs(tv[2]) < 0.002
        ):
            if no_progress_start is None:
                no_progress_start = now
                no_progress_z = float(tp[2])
            elif now - no_progress_start > 4.0:
                z_motion = abs(float(tp[2]) - no_progress_z)
                if z_motion < 0.001:
                    rospy.logwarn(
                        f"  Abort approach: tool is not following descending z_ref "
                        f"for {now - no_progress_start:.1f}s "
                        f"(z={tp[2]:.4f}, z_ref={z_ref_approach:.4f}, "
                        f"gap={z_track_gap*1000:.1f}mm, moved={z_motion*1000:.2f}mm, "
                        f"u_z={u_tool[2]:+.3f}N, |tau|={np.linalg.norm(tau):.3f}Nm, "
                        f"Kp={K_eff[0]:.1f}). Check torque-controller response "
                        "or increase approach position authority."
                    )
                    return False
                no_progress_start = now
                no_progress_z = float(tp[2])
        else:
            no_progress_start = None
            no_progress_z = None

        at_contact_height = tp[2] <= scan_z_ref + cfg.contact_z_tolerance
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
            z_err = abs(tp[2] - scan_z_ref)
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
                    f"scan_z_ref={scan_z_ref:.4f}, z_err={z_err*1000:.2f}mm, "
                    f"z_std={z_std*1000:.3f}mm"
                )
                break
        else:
            settle_start = None
            settle_forces = []
            settle_z_values = []

        if (
            not surface_touched
            and sensor_ready
            and z_ref_approach <= scan_z_ref + 1e-6
            and tp[2] <= scan_z_ref + cfg.contact_z_tolerance
        ):
            rospy.logwarn(
                f"  Abort approach: reached nominal scan_z_ref={scan_z_ref:.4f}m "
                f"without force contact (z={tp[2]:.4f}, F={F_z:.3f}N, "
                f"raw={F_raw:.3f}N, source={force_source}, "
                f"sensor={sensor_available}). Check force axis/sign, tare, "
                "or approach_z/contact_penetration_depth."
            )
            return False

        if rospy.Time.now().to_sec() - t0 > approach_timeout:
            rospy.logwarn(
                f"  Approach timeout "
                f"(z={tp[2]:.4f}, z_ref={z_ref_approach:.4f}, "
                f"target_z<={scan_z_ref:.4f}, F={F_z:.3f}N, "
                f"source={force_source}, sensor={sensor_available}, "
                f"touched={surface_touched}, "
                f"timeout={approach_timeout:.1f}s)"
            )
            return False

    if z_contact is None:
        z_contact = scan_z_ref

    sigma_f_int.reset()
    integ_euler = np.zeros(3)
    tau_limiter.reset()
    force_filter.reset(F_z)
    stiffness_filter.reset(cfg.estimator_initial_K)
    rospy.loginfo(
        f"  Contact completed at z={z_contact:.4f}; "
        f"scan_z_ref={scan_z_ref:.4f}, surface_z={surface_z:.4f}"
    )

    # ---- Phase 1.5: 接触调整 ----
    # 真机刚接触后的力控/滤波/限幅瞬态只用于收敛，不进入正式实验数据。
    rospy.loginfo("  Phase 1.5: Contact adjustment (not logged)...")
    x_cur = cfg.scan_start_x
    adjust_t0 = time.time()
    adjust_last_time = adjust_t0
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
        now_scan = time.time()
        adjust_t = now_scan - adjust_t0
        scan_dt = now_scan - adjust_last_time
        if scan_dt <= 0.0 or not np.isfinite(scan_dt):
            scan_dt = dt
        scan_dt = min(max(scan_dt, dt), 0.1)
        adjust_last_time = now_scan

        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        delta = contact_delta_from_surface(contact_plane_z or surface_z, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, scan_dt, raw_delta_dot=-tv[2]
        )
        if venv is not None:
            K_env_true, B_env_true = venv.get_stiffness(tp[0])
        else:
            K_env_true, B_env_true = 0.0, 0.0
        F_raw, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, delta_dot
        )
        F_z = float(force_filter.update(F_raw, scan_dt))
        if not sensor_available and venv is None and not cfg.allow_sensorless_approach:
            rospy.logwarn(
                f"  Force sensor data lost during adjustment "
                f"(source={force_source}, F={F_z:.3f}N)."
            )
            return False

        K_hat_raw, B_hat = est.update(abs(F_z), delta, delta_dot)
        K_hat = float(stiffness_filter.update(K_hat_raw, scan_dt))
        K_hat_ctrl = min(K_hat, cfg.scan_z_gain_khat_max) if cfg.scan_z_gain_khat_max > 0.0 else K_hat

        x_ref = np.array([x_cur, cfg.scan_y, scan_z_ref])
        xdot_ref = np.zeros(3)
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        e_r1 = np.clip(
            tp - x_ref,
            -cfg.scan_position_error_limit,
            cfg.scan_position_error_limit,
        )
        e_r2 = np.clip(
            tv - xdot_ref,
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
            )

        tau, _, _, integ_euler, _ = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f_vec_ctrl, sigma_f,
            alpha=alpha, K_e_hat=K_hat_ctrl,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        tau = tau_limiter.update(tau, scan_dt)
        robot.exec_torque_cmd(tau)

        pos_err_norm = np.linalg.norm(tp - x_ref)
        adjust_forces.append(F_actual)
        adjust_z_values.append(tp[2])
        adjust_pos_errors.append(pos_err_norm)
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
                f"  adjusting | t={adjust_t:.2f}s, F={F_actual:.3f}N "
                f"(raw={F_raw:.3f}, {force_source}), F_std={force_std:.3f}, "
                f"F_slope={force_slope:.3f}N/s, z_std={z_std*1000:.3f}mm, "
                f"pos_std={pos_std*1000:.3f}mm, blend={force_blend:.2f}, "
                f"K={K_hat_ctrl:.1f} rawK={K_hat_raw:.1f} B={B_hat:.2f}"
            )

        if stable:
            rospy.loginfo(
                f"  Adjustment settled after {adjust_t:.2f}s; formal scan logging starts now."
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
    scan_motion_t0 = t_scan
    last_scan_time = t_scan

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        t = time.time() - t_scan
        now_scan = time.time()
        scan_dt = now_scan - last_scan_time
        if scan_dt <= 0.0 or not np.isfinite(scan_dt):
            scan_dt = dt
        scan_dt = min(max(scan_dt, dt), 0.1)
        last_scan_time = now_scan
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        delta = contact_delta_from_surface(contact_plane_z or surface_z, tp[2])
        delta_dot, delta_dot_raw, delta_dot_fd = delta_dot_est.update(
            delta, scan_dt, raw_delta_dot=-tv[2]
        )
        if venv is not None:
            K_env_true, B_env_true = venv.get_stiffness(tp[0])
        else:
            K_env_true, B_env_true = 0.0, 0.0
        F_raw, wrench, force_source, sensor_available = read_contact_force(
            force_sensor, venv, tp[0], delta, delta_dot
        )
        F_z = float(force_filter.update(F_raw, scan_dt))
        if not sensor_available and venv is None and not cfg.allow_sensorless_approach:
            rospy.logwarn(
                f"  Force sensor data lost during scan "
                f"(source={force_source}, F={F_z:.3f}N). Stop and save partial data."
            )
            break

        K_hat_raw, B_hat = est.update(
            abs(F_z),
            delta,
            delta_dot,
        )
        K_hat = float(stiffness_filter.update(K_hat_raw, scan_dt))

        # 当前期望 tool 位姿: x 随时间增长，y/z 固定。
        scan_active = now_scan >= scan_motion_t0
        vx_cmd = cfg.scan_vx if scan_active else 0.0
        x_ref = np.array([x_cur, cfg.scan_y, scan_z_ref])
        xdot_ref = np.array([vx_cmd, 0, 0])

        # 力目标: +z 方向为上 (接触反力方向)
        F_des = np.array([0.0, 0.0, cfg.F_desired])
        F_meas = np.array([0.0, 0.0, abs(F_z)])

        # 控制器状态量:
        # e_r1/e_r2 是位置和速度误差；e_f_vec 是三轴力误差。
        # z 轴力误差符号采用 F_meas - F_des，与论文风险指标一致。
        e_r1 = np.clip(
            tp - x_ref,
            -cfg.scan_position_error_limit,
            cfg.scan_position_error_limit,
        )
        e_r2 = np.clip(
            tv - xdot_ref,
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
            )
            phase_val = sched.phase_detector.phase.value

        # no-RCM 力矩装配: ctrl 产生 tool 端笛卡尔力，robot_interface 再用
        # J_tool^T 映射到 7 维关节力矩，并叠加姿态保持项。
        K_hat_ctrl = K_hat
        if cfg.scan_z_gain_khat_max > 0.0:
            K_hat_ctrl = min(K_hat_ctrl, cfg.scan_z_gain_khat_max)

        tau, u_tool, K_eff, integ_euler, error = compute_torque_no_rcm(
            ctrl, rs, kin_tool,
            e_r1, e_r2, e_f_vec_ctrl, sigma_f,
            alpha=alpha, K_e_hat=K_hat_ctrl,
            ref_euler_fixed=ref_euler_fixed,
            integ_euler=integ_euler, dt=dt,
        )
        tau = tau_limiter.update(tau, scan_dt)
        robot.exec_torque_cmd(tau)
        x_cur += vx_cmd * scan_dt

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
                f"F={F_actual:.3f}N (des={F_desired:.3f}, err={F_err:+.3f}, raw={F_raw:.3f}) | "
                f"K={K_hat_ctrl:7.1f} rawK={K_hat_raw:7.1f} B={B_hat:5.2f} | "
                f"e_f={e_f_scalar:+.3f} e_r={e_r_scalar*1000:5.2f}mm | "
                f"alpha={alpha:.2f} phase={phase_val} source={force_source}"
            )

        # 保存所有关键状态，后续可直接从 npz 统计力 RMSE、位置误差和 alpha 曲线。
        logger.log(
            t=t, pos=pos_tool,
            pos_des=pos_tool_des,
            pos_err=pos_err,
            F_measured=F_actual, F_desired=F_desired,
            F_err=F_err,
            F_raw=F_raw,
            wrench=wrench,
            force_source=force_source,
            sensor_available=int(sensor_available),
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
            contact_plane_z=contact_plane_z if contact_plane_z is not None else surface_z,
            scan_z_ref=scan_z_ref,
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
    'continuous_force_margin': lambda: ContinuousForceMarginFuzzyAlphaScheduler(
        dt=0.01,
        F_min=Config.F_min,
        F_max=Config.F_max,
        F_desired=Config.F_desired,
    ),
    'online_priority': lambda: OnlinePriorityAdaptationAlphaScheduler(
        dt=0.01,
        F_min=Config.F_min,
        F_max=Config.F_max,
        F_desired=Config.F_desired,
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

    当前 0603 真机版本默认使用:
      --strategy continuous_force_margin --controller-mode pareto_iter
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='continuous_force_margin')
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--use-virtual-env', dest='use_virtual_env',
                    action='store_true', default=False,
                    help='启用虚拟接触环境回退 (真机默认关闭)')
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
    ap.add_argument('--force-port', default='/dev/ttyUSB0')
    ap.add_argument('--force-baudrate', type=int, default=460800)
    ap.add_argument('--force-serial-timeout', type=float, default=0.05)
    ap.add_argument('--force-timeout', type=float, default=0.02,
                    help='direct sensor freshness window; 0.02s matches 1kHz streaming with margin')
    ap.add_argument('--force-wait-timeout', type=float, default=2.0)
    ap.add_argument('--force-axis', type=int, default=Config.force_axis)
    ap.add_argument('--force-sign', type=float, default=1.0)
    ap.add_argument('--force-command-format', default='both')
    ap.add_argument('--force-data-source', default='0x33')
    ap.add_argument('--force-streaming', dest='force_streaming',
                    action='store_true', default=True)
    ap.add_argument('--no-force-streaming', dest='force_streaming',
                    action='store_false')
    ap.add_argument('--force-poll-hz', type=float, default=100.0)
    ap.add_argument('--force-output-units', choices=['N', 'kgf'], default='N')
    ap.add_argument('--force-tare-on-start', dest='force_tare_on_start',
                    action='store_true', default=True)
    ap.add_argument('--no-force-tare-on-start', dest='force_tare_on_start',
                    action='store_false')
    ap.add_argument('--force-tare-settle-s', type=float, default=1.0)
    ap.add_argument('--allow-sensorless-approach', action='store_true',
                    help='允许无传感器数据时继续下降，仅用于离线/虚拟调试')
    ap.add_argument('--no-force-sensor', action='store_true')
    ap.add_argument('--no-auto-plot', action='store_true',
                    help='实验结束后不自动调用绘图脚本')
    ap.add_argument('--plot-no-show', action='store_true',
                    help='自动绘图时只保存图像，不弹出 matplotlib 窗口')
    args = ap.parse_args()

    rospy.init_node("coop_gt_no_rcm_real", anonymous=True)
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Cooperative Game Force-Position Experiment (NO RCM)")
    rospy.loginfo("=" * 60)

    robot = PandaArm()
    kin_tool = PandaKinematics(robot, "panda_link11")
    kin_flange = PandaKinematics(robot, "panda_link8")
    rospy.sleep(1.0)

    cfg = Config()
    cfg.allow_sensorless_approach = bool(args.allow_sensorless_approach)

    # control_mode='are' 使用原始 4D ARE 查表；
    # control_mode='pareto_iter' 使用 Algorithm 2 迭代生成同形状增益表。
    ctrl = CooperativeGameController(control_mode=args.controller_mode)
    ctrl.u_threshold = cfg.no_rcm_u_threshold
    ctrl.P_ori = cfg.no_rcm_P_ori
    ctrl.D_ori = cfg.no_rcm_D_ori
    ctrl.I_ori = cfg.no_rcm_I_ori
    ctrl.no_rcm_u_tool_limits = cfg.no_rcm_u_tool_limits
    rospy.loginfo(f"Controller mode: {args.controller_mode}")
    rospy.loginfo(
        f"NO-RCM real safety gains: u_threshold={ctrl.u_threshold:.1f}N, "
        f"ori_gains=({ctrl.P_ori:.1f},{ctrl.D_ori:.1f},{ctrl.I_ori:.1f}), "
        f"u_tool_limits={cfg.no_rcm_u_tool_limits}"
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
               2000, 3000, 5000, cfg.estimator_initial_K]
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
        f"Virtual environment fallback: {'enabled' if venv is not None else 'disabled'}"
    )
    force_sensor = None
    if not args.no_force_sensor:
        data_source = int(str(args.force_data_source), 0)
        try:
            force_sensor = DirectForceSensorInput(
                port=args.force_port,
                baudrate=args.force_baudrate,
                serial_timeout=args.force_serial_timeout,
                freshness_timeout=args.force_timeout,
                force_axis=args.force_axis,
                force_sign=args.force_sign,
                command_format=args.force_command_format,
                data_source_cmd=data_source,
                use_streaming=args.force_streaming,
                output_units=args.force_output_units,
                tare_on_start=args.force_tare_on_start,
                tare_settle_s=args.force_tare_settle_s,
                poll_hz=args.force_poll_hz,
            )
        except Exception as exc:
            rospy.logerr(f"Direct force sensor init failed: {exc}")
            force_sensor = None
        if force_sensor is not None:
            got_first_frame = force_sensor.wait_for_data(args.force_wait_timeout)
            rospy.loginfo(
                f"Direct force sensor: {force_sensor.port} @ {force_sensor.baudrate}, "
                f"axis={force_sensor.force_axis}, sign={force_sensor.force_sign}, "
                f"timeout={force_sensor.timeout}s, command_format={force_sensor.command_format}, "
                f"data_source=0x{data_source:02X}, streaming={force_sensor.use_streaming}, "
                f"detected_format={force_sensor.detected_format}; "
                f"first_frame={got_first_frame}, seq={force_sensor.seq()}, "
                f"age={force_sensor.age():.4f}s"
            )
    else:
        rospy.logwarn(
            "Direct force sensor disabled; force will be zero unless "
            "--use-virtual-env is enabled."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = os.path.join(args.output_dir, f"no_rcm_real_{stamp}")
    os.makedirs(odir, exist_ok=True)

    names = list(STRATEGIES.keys()) if args.strategy == 'all' else [args.strategy]
    for sn in names:
        s = STRATEGIES[sn]()
        rospy.loginfo(f"\n{'='*50}\n  Strategy: {s.name}\n{'='*50}")
        for t in range(args.trials):
            if rospy.is_shutdown():
                break
            lg = DataLogger()
            est = EnvironmentEstimator(
                theta_init=[cfg.estimator_initial_K, cfg.estimator_initial_B],
                alpha_lp=cfg.estimator_alpha_lp,
            )
            s.reset()
            ok = run_trial(
                robot, kin_tool, kin_flange,
                cfg, ctrl, s, est, venv, force_sensor, lg, t,
            )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")
    if not args.no_auto_plot:
        auto_plot_results(odir, no_show=args.plot_no_show)
    if force_sensor is not None:
        force_sensor.close()


if __name__ == '__main__':
    main()
