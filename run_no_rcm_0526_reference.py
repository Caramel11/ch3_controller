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
  Terminal 2: python run_no_rcm.py --strategy online_priority --controller-mode pareto_iter
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
    ContinuousForceMarginFuzzyAlphaScheduler,
    OnlinePriorityAdaptationAlphaScheduler,
    FixedAlphaScheduler
)
from src.env_estimator import EnvironmentEstimator
from src.utils import VirtualStiffnessSurface, DataLogger
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

    # 扫描轨迹: tool 端沿 x 方向匀速从 scan_start_x 运动到 scan_end_x。
    scan_start_x = 0.40
    scan_end_x   = 0.48
    scan_y       = 0.0
    scan_z       = 0.298
    # approach_z 是虚拟表面高度；tool 低于该高度时产生压入量。
    approach_z   = 0.30
    scan_vx      = 0.002

    # 力目标与安全边界，供 online_priority 等 alpha 调度器使用。
    F_desired    = 1.0
    F_min        = 0.3
    F_max        = 2.0
    force_axis   = 2

    # 分段环境参数: (x_start, x_end, K_e, B_e)。
    stiffness_zones = [
    (0.40, 0.44, 300, 5),    # 低刚度
    (0.44, 0.48, 500, 8),    # 高刚度
    ]

    # 控制周期与泄漏积分器参数。
    ctrl_rate = 100
    dt = 1.0 / ctrl_rate
    eps_r = 1.0
    eps_f = 2.0
    # 接触检测和接触稳定窗口参数。
    settle_time = 2.0
    contact_force_threshold = 0.3
    contact_z_tolerance = 0.001
    contact_settle_time = 1.0
    contact_stable_z_std = 0.0003
    contact_stable_force_std = 0.05
    contact_stable_force_slope = 0.20
    contact_force_blend_time = 0.5


def contact_delta_from_surface(cfg, z):
    """以 approach_z 作为虚拟表面高度，计算单向压入量 δ。

    z >= approach_z 表示未接触，返回 0；z < approach_z 表示压入表面，
    返回 approach_z - z。该函数是虚拟环境力计算的唯一压入量来源。
    """
    return max(0.0, cfg.approach_z - z)


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
    approach_speed = 0.005
    z_ref_approach = rs["tool_position"][2]
    xy_ref_approach = np.array([cfg.scan_start_x, cfg.scan_y])
    contact_z_threshold = cfg.scan_z
    approach_timeout = max(
        20.0,
        1.5 * max(0.0, z_ref_approach - contact_z_threshold) / approach_speed + 5.0,
    )
    approach_count = 0
    rospy.loginfo(
        f"  Approach target: z<={contact_z_threshold:.4f}m or |Fz|>0.300N; "
        f"surface_z={cfg.approach_z:.4f}m, scan_z={cfg.scan_z:.4f}m, "
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
        z_contact = cfg.scan_z

    sigma_f_int.reset()
    integ_euler = np.zeros(3)
    rospy.loginfo(
        f"  Contact completed at z={z_contact:.4f}; "
        f"scan_z_ref={cfg.scan_z:.4f}, surface_z={cfg.approach_z:.4f}"
    )

    # ---- Phase 2: 恒力扫描 ----
    # 扫描阶段 x 方向按 scan_vx 匀速推进，z 方向由力位控制器自动调节，
    # 目标是在跟踪扫描路径的同时把接触反力维持在 F_desired 附近。
    rospy.loginfo("  Phase 2: Scanning...")
    x_cur = cfg.scan_start_x
    t_scan = time.time()

    while not rospy.is_shutdown() and x_cur < cfg.scan_end_x:
        t = time.time() - t_scan
        rs = update_robot_state(kin_tool, kin_flange)
        tp = rs["tool_position"]
        tv = rs["tool_position_velocity"]

        # 计算当前虚拟接触力，并用同一压入量送入环境估计器。
        F_z = 0.0
        if venv:
            F_z = venv.compute_force(
                tp[0], contact_delta_from_surface(cfg, tp[2]), -tv[2]
            )

        K_hat, B_hat = est.update(
            abs(F_z),
            contact_delta_from_surface(cfg, tp[2]),
            -tv[2],
        )

        # 当前期望 tool 位姿: x 随时间增长，y/z 固定。
        x_ref = np.array([x_cur, cfg.scan_y, cfg.scan_z])
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

        # 刚进入扫描的前 contact_force_blend_time 秒内渐进引入力控制，
        # 防止接触瞬间的力误差直接造成控制力突变。
        force_blend = min(1.0, max(0.0, t / cfg.contact_force_blend_time))
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
    safe_move_to_joint_position(robot, INIT_JOINTS)
    rospy.loginfo(f"  Done. {logger.count} samples.")
    return True


STRATEGIES = {
    # 策略表保持原项目风格: 命令行传入 strategy 名称即可实例化调度器。
    # 当前推荐组合: online_priority + --controller-mode pareto_iter。
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


def main():
    """命令行入口。

    默认策略仍保留旧值以兼容历史脚本。当前版本推荐显式使用:
      --strategy online_priority --controller-mode pareto_iter
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='online_priority')
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
                cfg, ctrl, s, est, venv, lg, t,
            )
            if ok:
                lg.save(os.path.join(odir, f"{s.name}_t{t:02d}.npz"))

    rospy.loginfo(f"\nResults → {odir}")


if __name__ == '__main__':
    main()
