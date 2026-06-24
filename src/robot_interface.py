"""
机器人接口公共模块
====================

所有与机器人交互的底层操作:
  - 状态读取 (FK + Jacobian)
  - 安全控制器切换 (effort ↔ position)
  - RCM 几何计算
  - 姿态 PD+I 控制
  - 力矩装配 (u_cart → τ)
"""
import numpy as np
import rospy
from scipy.spatial.transform import Rotation


INIT_JOINTS = (
    -0.03572926, -0.71236292, -0.05355629,
    -2.31286173,  0.04212054,  1.5332542,   0.71300622,
)


# ================================================================
# 机器人状态读取
# ================================================================
def update_robot_state(kin_tool, kin_flange):
    """读取 tool 和 flange 的位姿与速度。

    no-RCM 控制主要使用 tool_position/tool_position_velocity；
    flange 字段保留给 RCM 版本和共享接口使用。
    """
    tp = kin_tool.forward_position_kinematics()
    tool_pos = np.array(tp[:3], dtype=np.float64)
    tq = (tp[3], tp[4], tp[5], tp[6])
    tool_euler = Rotation.from_quat(tq).as_euler("xyz", degrees=False)
    tv = np.array(list(kin_tool.forward_velocity_kinematics()))

    fp = kin_flange.forward_position_kinematics()
    flange_pos = np.array(fp[:3], dtype=np.float64)
    fq = (fp[3], fp[4], fp[5], fp[6])
    flange_euler = Rotation.from_quat(fq).as_euler("xyz", degrees=False)
    fv = np.array(list(kin_flange.forward_velocity_kinematics()))

    return {
        "tool_position": tool_pos,
        "tool_rotation_euler": tool_euler,
        "tool_position_velocity": tv[:3],
        "tool_rotation_euler_velocity": tv[3:6],
        "flange_position": flange_pos,
        "flange_rotation_euler": flange_euler,
        "flange_position_velocity": fv[:3],
        "flange_rotation_euler_velocity": fv[3:6],
    }


# ================================================================
# 安全控制器切换
# ================================================================
def _stream_joint_position(robot, joints, timeout=10.0, tolerance=0.01,
                           max_step=0.004, rate_hz=100, log_prefix="move"):
    """用小步关节位置命令退回目标位姿，并在日志中报告进度。"""
    tgt = np.array(joints, dtype=np.float64)
    rate = rospy.Rate(rate_hz)
    t0 = rospy.get_time()
    last_log = t0

    while not rospy.is_shutdown():
        now = rospy.get_time()
        cur = np.array(robot.angles(), dtype=np.float64)
        err = tgt - cur
        err_inf = float(np.max(np.abs(err)))
        err_norm = float(np.linalg.norm(err))

        if err_inf <= tolerance:
            rospy.loginfo(
                f"  {log_prefix}: reached joint target "
                f"(err_inf={err_inf:.4f}rad, err_norm={err_norm:.4f}rad)."
            )
            return True
        if now - t0 >= timeout:
            rospy.logwarn(
                f"  {log_prefix}: timeout after {timeout:.1f}s "
                f"(err_inf={err_inf:.4f}rad, err_norm={err_norm:.4f}rad)."
            )
            return False
        if now - last_log >= 1.0:
            last_log = now
            rospy.loginfo(
                f"  {log_prefix}: moving to joint target, "
                f"err_inf={err_inf:.4f}rad, err_norm={err_norm:.4f}rad"
            )

        step = np.clip(0.08 * err, -max_step, max_step)
        robot.exec_position_cmd((cur + step).tolist())
        rate.sleep()

    return False


def safe_move_to_joint_position(robot, joints, timeout=20.0,
                                prefer_streaming=False, log_prefix="safe move"):
    """effort → position 控制器切换的安全包装。

    先发送零力矩，再尝试有界的 trajectory 关节运动；必要时退回到
    小步位置命令。真机 retreat 可设置 prefer_streaming=True，避免
    MoveIt/action 阻塞时外层没有进度日志。
    """
    try:
        robot.exec_torque_cmd([0.0] * 7)
    except Exception:
        pass
    rospy.sleep(0.3)

    is_ros2_effort_adapter = hasattr(robot, "add_gravity_compensation")
    if is_ros2_effort_adapter:
        try:
            rospy.loginfo(f"  {log_prefix}: using ROS 2 effort joint move.")
            robot.move_to_joint_position(joints, timeout=timeout, use_moveit=False)
            cur = np.array(robot.angles(), dtype=np.float64)
            err_inf = float(np.max(np.abs(np.array(joints, dtype=np.float64) - cur)))
            if err_inf <= 0.05:
                rospy.loginfo(
                    f"  {log_prefix}: ROS 2 joint move accepted "
                    f"(err_inf={err_inf:.4f}rad)."
                )
                return True
            rospy.logwarn(
                f"  {log_prefix}: ROS 2 joint move did not converge "
                f"(err_inf={err_inf:.4f}rad)."
            )
            return False
        except Exception:
            rospy.logwarn(
                f"  {log_prefix}: ROS 2 joint move failed.",
                exc_info=True,
            )
            return False

    if prefer_streaming:
        return _stream_joint_position(
            robot, joints, timeout=timeout, log_prefix=log_prefix
        )

    try:
        rospy.loginfo(f"  {log_prefix}: trying trajectory joint move...")
        robot.move_to_joint_position(joints, timeout=min(timeout, 5.0), use_moveit=False)
        cur = np.array(robot.angles(), dtype=np.float64)
        err_inf = float(np.max(np.abs(np.array(joints, dtype=np.float64) - cur)))
        if err_inf <= 0.01:
            rospy.loginfo(f"  {log_prefix}: trajectory joint move complete.")
            return True
        rospy.logwarn(
            f"  {log_prefix}: trajectory move returned with "
            f"err_inf={err_inf:.4f}rad; switching to streaming fallback."
        )
    except Exception:
        rospy.logwarn(
            f"  {log_prefix}: trajectory joint move failed; "
            "switching to streaming fallback.",
            exc_info=True,
        )

    return _stream_joint_position(
        robot, joints, timeout=timeout, log_prefix=f"{log_prefix} fallback"
    )


# ================================================================
# RCM 几何
# ================================================================
def compute_position_rcm(tool_pos, flange_pos, trocar_pos):
    """器械轴线上距 trocar 最近的点 (RCM 投影点)。

    no-RCM 主入口不会使用该函数，但它保留在共享 robot_interface 中供 RCM
    控制代码复用。
    """
    v = tool_pos - flange_pos
    w = trocar_pos - flange_pos
    v_sq = np.dot(v, v)
    if v_sq < 1e-10:
        return flange_pos.copy()
    return flange_pos + (np.dot(v, w) / v_sq) * v


def tool_to_flange_full_ref(x_tool_ref, xdot_tool_ref, trocar_pos, length):
    """
    从 tool 期望位置/速度正向映射到 flange 期望位置/速度/姿态

    几何关系 (RCM 约束):
      x_f = x_t + L · n̂,         n̂ = (p_trocar − x_t) / r,  r = ‖p_trocar − x_t‖

    速度由对时间求导得 (推导见文档):
      ẋ_f = (I − (L/r)(I − n̂ n̂ᵀ)) · ẋ_t

      其中 (I − n̂ n̂ᵀ) 为垂直工具轴方向的投影算子。
      物理含义:
        - 沿工具轴 (轴向) 的 ẋ_t → flange 同方向同速
        - 垂直工具轴 (横向) 的 ẋ_t → flange 反向、按 (L/r − 1) 倍缩放
          (这是 trocar 杠杆的几何效应)

    姿态: 工具轴方向 z_dir = −n̂ (从 trocar 指向工具尖端)
    选定 y_ref = [0,−1,0] 为参考方向, Gram-Schmidt 构造正交基。

    Parameters
    ----------
    x_tool_ref    : (3,)   tool 期望位置
    xdot_tool_ref : (3,)   tool 期望速度
    trocar_pos    : (3,)   trocar 固定点
    length        : float  工具长度 L

    Returns
    -------
    x_flange_ref    : (3,)   flange 期望位置
    xdot_flange_ref : (3,)   flange 期望速度
    euler_ref       : (3,)   flange 期望姿态 (xyz Euler)
    """
    d = trocar_pos - x_tool_ref
    r = np.linalg.norm(d)
    if r < 1e-6:
        # 退化情况: tool 在 trocar 上, 几何不定
        return x_tool_ref.copy(), np.zeros(3), np.zeros(3)

    n_hat = d / r

    # 位置
    x_flange_ref = x_tool_ref + length * n_hat

    # 速度: 投影算子映射
    P_perp = np.eye(3) - np.outer(n_hat, n_hat)
    M_vel = np.eye(3) - (length / r) * P_perp
    xdot_flange_ref = M_vel @ xdot_tool_ref

    # 姿态
    z_dir = -n_hat
    y_ref = np.array([0.0, -1.0, 0.0])
    y_dir = y_ref - np.dot(y_ref, z_dir) * z_dir
    y_norm = np.linalg.norm(y_dir)
    y_dir = y_dir / y_norm if y_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    x_dir = np.cross(y_dir, z_dir)
    R_mat = np.column_stack([x_dir, y_dir, z_dir])
    euler_ref = Rotation.from_matrix(R_mat).as_euler("xyz", degrees=False)

    return x_flange_ref, xdot_flange_ref, euler_ref


def fixed_downward_tool_euler(y_ref=None):
    """
    构造 no-RCM 模式的固定末端姿态参考。

    no-RCM 仿真使用用户指定的固定姿态 [-90, 0, -45] deg：
    末端竖直向下，并让可见正面朝向正前方。y_ref 参数保留为兼容旧调用。
    """
    return Rotation.from_euler(
        "xyz", np.deg2rad([-90.0, 0.0, -45.0])
    ).as_euler("xyz", degrees=False)


def tool_visual_front_axis_local():
    """trocar 可视正面在 fr3_link11/tool frame 下的近似方向。"""
    return np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)


def euler_angle_diff(a, b):
    """带周期性处理的欧拉角差"""
    diff = b - a
    wrapped = diff % (2 * np.pi)
    wrapped[wrapped > np.pi] -= 2 * np.pi
    return wrapped


# ================================================================
# 姿态 PD+I 控制
# ================================================================
def compute_u_rotation(robot_state_euler, robot_state_euler_vel,
                       ref_euler, integ_euler, dt,
                       P_ori=20.0, D_ori=1.0, I_ori=30.0,
                       integ_limit=0.08, omega_limit=None):
    """独立于博弈框架的姿态控制。

    Gazebo effort 接口下直接做 Euler 角相减容易在姿态耦合处抖动。
    这里使用 world-aligned SO(3) 旋转向量作为姿态误差，和 Pinocchio
    LOCAL_WORLD_ALIGNED Jacobian 的角速度/角力矩约定一致。
    """
    R_cur = Rotation.from_euler("xyz", robot_state_euler).as_matrix()
    R_ref = Rotation.from_euler("xyz", ref_euler).as_matrix()
    e_rot = Rotation.from_matrix(R_ref @ R_cur.T).as_rotvec()
    omega = np.asarray(robot_state_euler_vel, dtype=float)
    if omega_limit is not None:
        limit = float(omega_limit)
        omega_norm = float(np.linalg.norm(omega))
        if limit > 0.0 and omega_norm > limit:
            omega = omega * limit / omega_norm
    integ_new = integ_euler + e_rot * dt
    if integ_limit is not None:
        integ_new = np.clip(integ_new, -float(integ_limit), float(integ_limit))
    u_rot = P_ori * e_rot - D_ori * omega + I_ori * integ_new
    return u_rot, integ_new


def _filter_vector_state(owner, attr, value, dt, tau):
    """一阶低通滤波，状态挂在控制器对象上，避免扩大调用接口。"""
    value = np.asarray(value, dtype=float)
    tau = float(tau)
    if tau <= 0.0 or dt <= 0.0:
        setattr(owner, attr, value.copy())
        return value
    prev = getattr(owner, attr, None)
    if prev is None:
        filtered = value.copy()
    else:
        prev = np.asarray(prev, dtype=float)
        if prev.shape != value.shape:
            filtered = value.copy()
        else:
            beta = float(dt) / (tau + float(dt))
            filtered = prev + beta * (value - prev)
    setattr(owner, attr, filtered.copy())
    return filtered


def _rate_limit_vector_state(owner, attr, value, dt, rate_limit):
    value = np.asarray(value, dtype=float)
    rate_limit = float(rate_limit)
    if rate_limit <= 0.0 or dt <= 0.0:
        setattr(owner, attr, value.copy())
        return value
    prev = getattr(owner, attr, None)
    if prev is None:
        limited = value.copy()
    else:
        prev = np.asarray(prev, dtype=float)
        if prev.shape != value.shape:
            limited = value.copy()
        else:
            step = value - prev
            step_norm = float(np.linalg.norm(step))
            max_step = rate_limit * float(dt)
            if step_norm > max_step > 0.0:
                step = step * max_step / step_norm
            limited = prev + step
    setattr(owner, attr, limited.copy())
    return limited


def _kinematics_joint_state(kin_tool):
    robot = getattr(kin_tool, "robot", None)
    if robot is None:
        robot = getattr(kin_tool, "_robot", None)
    if robot is None:
        return None, None
    try:
        q = np.asarray(robot.angles(), dtype=float).reshape(-1)
    except Exception:
        q = getattr(robot, "_q_meas", None)
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
    qd = getattr(robot, "_qd_meas", None)
    if qd is not None:
        qd = np.asarray(qd, dtype=float).reshape(-1)
    elif q is not None:
        qd = np.zeros_like(q)
    return q, qd


def _nullspace_posture_torque(ctrl, kin_tool, J):
    if not bool(getattr(ctrl, "no_rcm_nullspace_enabled", False)):
        return None
    if not hasattr(ctrl, "no_rcm_nominal_joints"):
        return None
    q, qd = _kinematics_joint_state(kin_tool)
    if q is None or qd is None:
        return None
    q_nom = np.asarray(ctrl.no_rcm_nominal_joints, dtype=float).reshape(-1)
    n = min(q.size, qd.size, q_nom.size, J.shape[1])
    if n <= 0:
        return None
    q = q[:n]
    qd = qd[:n]
    q_nom = q_nom[:n]
    kp = float(getattr(ctrl, "no_rcm_null_kp", 3.0))
    kd = float(getattr(ctrl, "no_rcm_null_kd", 1.2))
    tau_raw = kp * (q_nom - q) - kd * qd

    Jn = np.asarray(J[:, :n], dtype=float)
    # Torque-space null projection: keep posture correction from generating
    # a first-order Cartesian wrench at the tool task Jacobian.
    proj = np.eye(n) - Jn.T @ np.linalg.pinv(Jn.T, rcond=1e-3)
    tau_null = proj @ tau_raw
    max_abs = float(getattr(ctrl, "no_rcm_null_tau_max", 2.0))
    if max_abs > 0.0:
        tau_null = np.clip(tau_null, -max_abs, max_abs)
    return tau_null


def _direct_posture_torque(ctrl, kin_tool):
    if not bool(getattr(ctrl, "no_rcm_direct_posture_enabled", True)):
        return None
    if not hasattr(ctrl, "no_rcm_direct_posture_weights"):
        return None
    q, qd = _kinematics_joint_state(kin_tool)
    if q is None or qd is None:
        return None
    q_nom = np.asarray(ctrl.no_rcm_nominal_joints, dtype=float).reshape(-1)
    n = min(q.size, qd.size, q_nom.size)
    if n <= 0:
        return None
    q = q[:n]
    qd = qd[:n]
    q_nom = q_nom[:n]
    weights = np.asarray(ctrl.no_rcm_direct_posture_weights, dtype=float).reshape(-1)
    if weights.size == 1:
        weights = np.full(n, float(weights[0]))
    else:
        weights = np.pad(weights[:n], (0, max(0, n - weights.size)))[:n]
    kp = float(getattr(ctrl, "no_rcm_direct_posture_kp", 4.0))
    kd = float(getattr(ctrl, "no_rcm_direct_posture_kd", 1.6))
    tau = weights * (kp * (q_nom - q) - kd * qd)
    max_abs = np.asarray(getattr(ctrl, "no_rcm_direct_posture_tau_max", 2.0), dtype=float)
    if max_abs.size == 1:
        tau = np.clip(tau, -float(max_abs), float(max_abs))
    else:
        max_abs = np.pad(max_abs[:n], (0, max(0, n - max_abs.size)))[:n]
        tau = np.clip(tau, -max_abs, max_abs)
    return tau


# ================================================================
# 力矩装配 — RCM 模式 (flange-space 控制)
# ================================================================
def compute_torque_with_rcm(ctrl, robot_state, kin_flange,
                            x_tool_ref, xdot_tool_ref,
                            e_f, sigma_f,
                            alpha, K_e_hat,
                            trocar_pos, length,
                            integ_euler, dt):
    """
    RCM 模式 (flange-space 控制)

    控制流程:
      1. tool 期望 (x_tool_ref, ẋ_tool_ref) → RCM 正映射
         → flange 期望 (x_flange_ref, ẋ_flange_ref, euler_ref)
      2. flange 实测 − flange 期望 → e_r1_flange, e_r2_flange
      3. 博弈控制律 u_flange = -K_eff·[e_r1_flange; e_r2_flange; e_f; σ_f]
         注: 力相关项 e_f, σ_f 不做杠杆变换, 直接累加到 u_flange
      4. u_flange + u_rot → J_flange^T → τ

    与旧方案的差异:
      旧: tool 误差 → u_tool → 杠杆缩放 → u_flange
      新: tool 期望 → flange 期望 → flange 误差 → u_flange (无杠杆缩放)

    Parameters
    ----------
    ctrl          : CooperativeGameController
    robot_state   : dict       update_robot_state() 返回值
    kin_flange    : Kinematics flange (panda_link8) 运动学
    x_tool_ref    : (3,)       tool 期望位置
    xdot_tool_ref : (3,)       tool 期望速度
    e_f           : (3,)       力误差 F_meas − F_des (tool 空间)
    sigma_f       : (3,)       力误差泄漏积分 (tool 空间)
    alpha         : float      仲裁参数
    K_e_hat       : float      RLS 估计环境刚度
    trocar_pos    : (3,)       trocar 固定点
    length        : float      工具长度
    integ_euler   : (3,)       姿态积分项
    dt            : float      控制周期

    Returns
    -------
    j_torque        : (7,)
    u_flange        : (3,)     flange 笛卡尔控制力
    K_eff           : (4,)     当前激活增益
    integ_euler_new : (3,)
    x_flange_ref    : (3,)     flange 期望位置 (用于日志)
    error           : (2,)     [RCM 误差, tool 跟踪误差]
    """
    tool_pos = robot_state["tool_position"]
    flange_pos = robot_state["flange_position"]
    flange_vel = robot_state["flange_position_velocity"]

    # 1. tool 期望 → flange 期望 (位置、速度、姿态)
    x_flange_ref, xdot_flange_ref, euler_ref = tool_to_flange_full_ref(
        x_tool_ref, xdot_tool_ref, trocar_pos, length
    )

    # 2. flange 空间的位置/速度误差
    e_r1_flange = flange_pos - x_flange_ref
    e_r2_flange = flange_vel - xdot_flange_ref

    # 3. 博弈控制律 (flange-space, 力误差直接累加无杠杆变换)。
    # 力位仲裁只作用于 z/压入深度方向；x/y 始终使用 alpha=1 严格位置跟踪。
    if hasattr(ctrl, "compute_control_axis_alpha"):
        u_flange, K_axes = ctrl.compute_control_axis_alpha(
            e_r1_flange, e_r2_flange, e_f, sigma_f,
            np.array([1.0, 1.0, alpha]), K_e_hat
        )
        K_eff = K_axes[2]
    else:
        u_flange, K_eff = ctrl.compute_control(
            e_r1_flange, e_r2_flange, e_f, sigma_f, alpha, K_e_hat
        )

    # 4. 姿态控制 (flange 姿态)
    u_rot, integ_euler_new = compute_u_rotation(
        robot_state["flange_rotation_euler"],
        robot_state["flange_rotation_euler_velocity"],
        euler_ref, integ_euler, dt,
        P_ori=ctrl.P_ori, D_ori=ctrl.D_ori, I_ori=ctrl.I_ori,
    )

    # 5. 装配 wrench + 限幅
    u_cart = np.hstack([u_flange, u_rot])
    cart_norm = np.linalg.norm(u_cart)
    if cart_norm > ctrl.u_threshold:
        u_cart = u_cart * ctrl.u_threshold / cart_norm

    # 6. Jacobian → 关节力矩
    J = np.array(kin_flange.jacobian())
    j_torque = np.clip(
        (J.T @ u_cart).flatten(),
        -ctrl.tau_max, ctrl.tau_max,
    )

    # 7. 误差日志
    p_rcm = compute_position_rcm(tool_pos, flange_pos, trocar_pos)
    error_rcm = np.linalg.norm(trocar_pos - p_rcm)
    error_track = np.linalg.norm(tool_pos - x_tool_ref)

    return (j_torque, u_flange, K_eff, integ_euler_new,
            x_flange_ref, np.array([error_rcm, error_track]))


# ================================================================
# 力矩装配 — 无 RCM 模式
# ================================================================
def compute_torque_no_rcm(ctrl, robot_state, kin_tool,
                          e_r1, e_r2, e_f, sigma_f,
                          alpha, K_e_hat,
                          ref_euler_fixed, integ_euler, dt):
    """no-RCM 力矩装配函数。

    输入误差均在 tool 坐标对应的笛卡尔空间中表达。函数内部先调用
    ctrl.compute_control(...) 得到 tool 端笛卡尔力 u_tool，再用 tool Jacobian
    映射到关节力矩，并叠加姿态保持项。
    """
    tool_pos = robot_state["tool_position"]

    # 1. 平动控制: 力位仲裁只作用于 z/压入深度方向；
    # x/y 始终使用 alpha=1 严格位置跟踪，避免 z 向力控降权同步削弱扫描轨迹。
    if hasattr(ctrl, "compute_control_axis_alpha"):
        u_tool, K_axes = ctrl.compute_control_axis_alpha(
            e_r1, e_r2, e_f, sigma_f, np.array([1.0, 1.0, alpha]), K_e_hat
        )
        K_eff = K_axes[2]
    else:
        u_tool, K_eff = ctrl.compute_control(e_r1, e_r2, e_f, sigma_f, alpha, K_e_hat)
    xy_kp = float(getattr(ctrl, "no_rcm_xy_stiffness_boost", 0.0))
    xy_kd = float(getattr(ctrl, "no_rcm_xy_damping_boost", 0.0))
    if xy_kp > 0.0 or xy_kd > 0.0:
        # Keep z force arbitration untouched while giving x/y enough authority
        # to remove lateral residuals before and during scanning.
        e_r1_arr = np.asarray(e_r1, dtype=float).reshape(3)
        e_r2_arr = np.asarray(e_r2, dtype=float).reshape(3)
        u_tool[:2] += -(xy_kp * e_r1_arr[:2] + xy_kd * e_r2_arr[:2])
    if hasattr(ctrl, "no_rcm_u_tool_limits"):
        # 可选的三轴限幅钩子，当前控制器默认没有设置该属性。
        limits = np.asarray(ctrl.no_rcm_u_tool_limits, dtype=float)
        if limits.shape == (3,):
            u_tool = np.clip(u_tool, -limits, limits)

    # 末端笛卡尔姿态控制: 姿态误差在 SO(3) 上计算，再通过 tool 6D Jacobian
    # 的角速度/角力矩行映射到关节力矩，不叠加零空间关节姿态控制器。
    omega_raw = robot_state["tool_rotation_euler_velocity"]
    omega_for_damping = _filter_vector_state(
        ctrl,
        "_no_rcm_omega_filtered",
        omega_raw,
        dt,
        getattr(ctrl, "no_rcm_omega_filter_tau", 0.0),
    )
    u_rot, integ_euler_new = compute_u_rotation(
        robot_state["tool_rotation_euler"],
        omega_for_damping,
        ref_euler_fixed, integ_euler, dt,
        P_ori=ctrl.P_ori, D_ori=ctrl.D_ori, I_ori=ctrl.I_ori,
        omega_limit=getattr(ctrl, "no_rcm_omega_limit", None),
    )
    if hasattr(ctrl, "no_rcm_u_rot_limit"):
        rot_limit = float(ctrl.no_rcm_u_rot_limit)
        rot_norm = float(np.linalg.norm(u_rot))
        if rot_limit > 0.0 and rot_norm > rot_limit:
            u_rot = u_rot * rot_limit / rot_norm
    u_rot = _rate_limit_vector_state(
        ctrl,
        "_no_rcm_u_rot_limited",
        u_rot,
        dt,
        getattr(ctrl, "no_rcm_u_rot_rate_limit", 0.0),
    )

    # 2. 将平动力和姿态力矩拼成 6D wrench，并做整体范数限幅。
    u_cart = np.hstack([u_tool, u_rot])
    cart_norm = np.linalg.norm(u_cart)
    if cart_norm > ctrl.u_threshold:
        u_cart = u_cart * ctrl.u_threshold / cart_norm

    # 3. tool Jacobian 映射到关节力矩；no-RCM 不做 trocar 杠杆变换。
    J = np.array(kin_tool.jacobian())
    j_torque = np.clip(
        (J.T @ u_cart).flatten(),
        -ctrl.tau_max, ctrl.tau_max,
    )
    tau_null = _nullspace_posture_torque(ctrl, kin_tool, J)
    if tau_null is not None and tau_null.shape == j_torque.shape:
        j_torque = np.clip(j_torque + tau_null, -ctrl.tau_max, ctrl.tau_max)
    tau_direct = _direct_posture_torque(ctrl, kin_tool)
    if tau_direct is not None and tau_direct.shape == j_torque.shape:
        j_torque = np.clip(j_torque + tau_direct, -ctrl.tau_max, ctrl.tau_max)

    error_track = np.linalg.norm(e_r1)
    return (j_torque, u_tool, K_eff, integ_euler_new,
            np.array([0.0, error_track]))
