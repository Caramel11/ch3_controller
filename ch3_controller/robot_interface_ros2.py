"""ROS 2 robot interface for the 0603 no-RCM real experiment."""

import time
from typing import Optional

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .pinocchio_model import PinocchioModelHelper


INIT_JOINTS = (
    -0.03572926, -0.71236292, -0.05355629,
    -2.31286173, 0.04212054, 1.5332542, 0.71300622,
)


class Ros2FrankaArm(Node):
    """Minimal PandaArm replacement backed by ROS 2 joint states and effort commands."""

    def __init__(
        self,
        node_name="coop_gt_no_rcm_real",
        default_tool_frame="fr3_link11",
        default_flange_frame="fr3_link8",
    ):
        super().__init__(node_name)

        self.declare_parameter("cmd_topic", "/NS_1/joint_group_effort_controller/commands")
        self.declare_parameter("state_topic", "/NS_1/joint_states")
        self.declare_parameter("rsp_node", "/NS_1/robot_state_publisher")
        self.declare_parameter("reference_frame", "fr3_link8")
        self.declare_parameter("tool_frame", default_tool_frame)
        self.declare_parameter("flange_frame", default_flange_frame)
        self.declare_parameter("watchdog_sec", 0.2)
        self.declare_parameter("add_gravity_compensation", True)
        self.declare_parameter("gravity_compensation_scale", 1.0)
        self.declare_parameter(
            "controlled_joints",
            [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
        )

        # Replacement for ROS 1 PandaArm position moves while the effort controller is active.
        self.declare_parameter("joint_move_rate_hz", 200.0)
        self.declare_parameter("joint_move_kp", [32.0, 32.0, 32.0, 32.0, 18.0, 12.0, 10.0])
        self.declare_parameter("joint_move_kd", [12.0, 12.0, 12.0, 12.0, 8.0, 6.0, 5.0])
        self.declare_parameter("joint_move_max_tau_abs", 10.0)
        self.declare_parameter("joint_move_max_tau_rate", 55.0)
        self.declare_parameter("joint_move_max_speed", 0.12)
        self.declare_parameter("joint_move_min_duration", 4.0)
        self.declare_parameter("joint_move_tolerance", 0.030)
        self.declare_parameter("joint_move_velocity_tolerance", 0.060)

        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.rsp_node = self.get_parameter("rsp_node").value
        self.reference_frame = self.get_parameter("reference_frame").value
        self.tool_frame = self.get_parameter("tool_frame").value
        self.flange_frame = self.get_parameter("flange_frame").value
        self.watchdog_sec = float(self.get_parameter("watchdog_sec").value)
        self.add_gravity_compensation = bool(
            self.get_parameter("add_gravity_compensation").value
        )
        self.gravity_compensation_scale = float(
            self.get_parameter("gravity_compensation_scale").value
        )
        self.ctrl_joints = list(self.get_parameter("controlled_joints").value)

        self.joint_move_rate_hz = float(self.get_parameter("joint_move_rate_hz").value)
        self.joint_move_kp = np.asarray(self.get_parameter("joint_move_kp").value, dtype=float)
        self.joint_move_kd = np.asarray(self.get_parameter("joint_move_kd").value, dtype=float)
        self.joint_move_max_tau_abs = float(self.get_parameter("joint_move_max_tau_abs").value)
        self.joint_move_max_tau_rate = float(self.get_parameter("joint_move_max_tau_rate").value)
        self.joint_move_max_speed = float(self.get_parameter("joint_move_max_speed").value)
        self.joint_move_min_duration = float(self.get_parameter("joint_move_min_duration").value)
        self.joint_move_tolerance = float(self.get_parameter("joint_move_tolerance").value)
        self.joint_move_velocity_tolerance = float(
            self.get_parameter("joint_move_velocity_tolerance").value
        )

        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.sub = self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)

        n = len(self.ctrl_joints)
        self._q_meas = np.zeros(n)
        self._qd_meas = np.zeros(n)
        self._last_state_time = None
        self._tau_prev = np.zeros(n)

        self.pm = PinocchioModelHelper(
            node=self,
            rsp_node=self.rsp_node,
            controlled_joints=self.ctrl_joints,
            reference_frame=self.reference_frame,
            tip_frame=self.tool_frame,
        )
        if not self.pm.load():
            self.get_logger().error("Pinocchio model not loaded.")

        self.flange_frame_id: Optional[int] = None
        if self.pm.model is not None:
            self.flange_frame_id = self.pm.model.getFrameId(self.flange_frame)
            if self.flange_frame_id >= len(self.pm.model.frames):
                self.get_logger().error(f"Flange frame '{self.flange_frame}' not found.")
                self.flange_frame_id = None

        self.get_logger().info(f"Publishing effort commands: {self.cmd_topic}")
        self.get_logger().info(f"Subscribing joint states: {self.state_topic}")
        self.get_logger().info(
            f"Frames: reference={self.reference_frame}, tool={self.tool_frame}, "
            f"flange={self.flange_frame}"
        )
        if self.add_gravity_compensation:
            self.get_logger().info(
                "Pinocchio gravity compensation enabled "
                f"(scale={self.gravity_compensation_scale:.3f})."
            )

    def _on_joint_state(self, msg: JointState):
        name_to_i = {name: i for i, name in enumerate(msg.name)}
        for k, joint_name in enumerate(self.ctrl_joints):
            i = name_to_i.get(joint_name)
            if i is None:
                self.get_logger().error(f"missing joint {joint_name} in joint_states")
                return
            if i < len(msg.position):
                self._q_meas[k] = float(msg.position[i])
            if i < len(msg.velocity):
                self._qd_meas[k] = float(msg.velocity[i])
        self._last_state_time = self.get_clock().now()

    def spin_once(self, timeout_sec=0.0):
        if rclpy.ok():
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def state_valid(self):
        if self.pm.model is None or self.pm.data is None:
            return False
        if self._last_state_time is None:
            return False
        age = (self.get_clock().now() - self._last_state_time).nanoseconds * 1e-9
        return age <= self.watchdog_sec

    def wait_until_ready(self, timeout=5.0):
        deadline = time.time() + float(timeout)
        while rclpy.ok() and time.time() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self.state_valid() and self.pm.tip_frame_id is not None:
                return True
        return self.state_valid()

    def angles(self):
        self.spin_once(timeout_sec=0.0)
        return self._q_meas.copy()

    def exec_torque_cmd(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(-1)
        tau_control = tau.copy()
        tau_cmd = tau_control + self._gravity_compensation_torque()
        msg = Float64MultiArray()
        msg.data = tau_cmd.tolist()
        self.pub.publish(msg)
        if tau_control.shape == self._tau_prev.shape:
            self._tau_prev = tau_control.copy()

    def _gravity_compensation_torque(self):
        if (
            not self.add_gravity_compensation
            or self.pm.model is None
            or self.pm.data is None
            or self.gravity_compensation_scale == 0.0
        ):
            return np.zeros(len(self.ctrl_joints), dtype=float)
        try:
            q, _ = self.pm.build_full_state(self._q_meas, self._qd_meas)
            tau_g = pin.computeGeneralizedGravity(self.pm.model, self.pm.data, q)
            tau_g = self.pm.extract_controlled_effort(np.asarray(tau_g, dtype=float))
            return self.gravity_compensation_scale * tau_g.reshape(len(self.ctrl_joints))
        except Exception as exc:
            self.get_logger().warn(f"Gravity compensation failed once: {exc}")
            return np.zeros(len(self.ctrl_joints), dtype=float)

    def exec_position_cmd(self, joints):
        """ROS 1 PandaArm-compatible one-step joint position command.

        The ROS 2 setup only exposes an effort controller here, so a position
        command is implemented as one PD effort update toward the requested
        joint target. This keeps the original streaming fallback code usable.
        """
        self.spin_once(timeout_sec=0.0)
        target = np.asarray(joints, dtype=float).reshape(len(self.ctrl_joints))
        q = self._q_meas.copy()
        qd = self._qd_meas.copy()
        period = 1.0 / max(self.joint_move_rate_hz, 1.0)
        tau = self.joint_move_kp * (target - q) - self.joint_move_kd * qd
        tau = self._rate_and_clip_joint_move(tau, period)
        self.exec_torque_cmd(tau)

    def _rate_and_clip_joint_move(self, tau, dt):
        tau = np.clip(tau, -self.joint_move_max_tau_abs, self.joint_move_max_tau_abs)
        max_delta = self.joint_move_max_tau_rate * max(float(dt), 1e-6)
        delta = np.clip(tau - self._tau_prev, -max_delta, max_delta)
        return self._tau_prev + delta

    @staticmethod
    def _smoothstep5(s):
        s = float(np.clip(s, 0.0, 1.0))
        return 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5

    @staticmethod
    def _smoothstep5_dot(s):
        s = float(np.clip(s, 0.0, 1.0))
        return 30.0 * s ** 2 - 60.0 * s ** 3 + 30.0 * s ** 4

    def move_to_joint_position(self, joints, timeout=10.0, use_moveit=False):
        del use_moveit
        target = np.asarray(joints, dtype=float).reshape(len(self.ctrl_joints))
        period = 1.0 / max(self.joint_move_rate_hz, 1.0)
        last_log = 0.0

        if not self.wait_until_ready(timeout=2.0):
            self.get_logger().error("No fresh joint state; cannot move to joint target.")
            return False

        start_q = self._q_meas.copy()
        delta_q = target - start_q
        max_delta_q = float(np.max(np.abs(delta_q)))
        duration = max(
            self.joint_move_min_duration,
            max_delta_q / max(self.joint_move_max_speed, 1e-6),
        )
        deadline = time.time() + max(float(timeout), duration + 3.0)
        t0 = time.time()
        self.get_logger().info(
            f"  joint move: smooth trajectory to target, max_delta={max_delta_q:.3f}rad, "
            f"duration={duration:.1f}s, max_ref_speed={self.joint_move_max_speed:.3f}rad/s, "
            f"tau_limit={self.joint_move_max_tau_abs:.1f}Nm, "
            f"tau_rate={self.joint_move_max_tau_rate:.1f}Nm/s"
        )

        while rclpy.ok() and time.time() < deadline:
            self.spin_once(timeout_sec=0.0)
            q = self._q_meas.copy()
            qd = self._qd_meas.copy()
            elapsed = time.time() - t0
            s = min(1.0, max(0.0, elapsed / max(duration, 1e-6)))
            blend = self._smoothstep5(s)
            blend_dot = self._smoothstep5_dot(s) / max(duration, 1e-6)
            q_ref = start_q + blend * delta_q
            qd_ref = blend_dot * delta_q
            err = target - q
            err_ref = q_ref - q
            err_inf = float(np.max(np.abs(err)))
            err_norm = float(np.linalg.norm(err))
            err_ref_norm = float(np.linalg.norm(err_ref))
            qd_norm = float(np.linalg.norm(qd))

            tau = self.joint_move_kp * err_ref - self.joint_move_kd * (qd - qd_ref)
            tau = self._rate_and_clip_joint_move(tau, period)
            self.exec_torque_cmd(tau)

            now = time.time()
            if now - last_log >= 1.0:
                last_log = now
                self.get_logger().info(
                    f"  joint move: t={elapsed:.2f}s/{duration:.2f}s, "
                    f"err_inf={err_inf:.4f}rad, err_norm={err_norm:.4f}rad, "
                    f"q_ref_err={err_ref_norm:.4f}rad, qd_norm={qd_norm:.4f}rad/s"
                )

            if (
                s >= 1.0
                and err_inf <= self.joint_move_tolerance
                and qd_norm <= self.joint_move_velocity_tolerance
            ):
                self.get_logger().info(
                    f"  joint move: reached target "
                    f"(err_inf={err_inf:.4f}rad, err_norm={err_norm:.4f}rad)."
                )
                return True

            time.sleep(period)

        self.get_logger().warn(f"  joint move: timeout after {timeout:.1f}s.")
        return False

    def kinematics(self, frame_name):
        frame_name = self._normalize_frame_name(frame_name)
        frame_id = None
        if frame_name == self.tool_frame:
            frame_id = self.pm.tip_frame_id
        elif frame_name == self.flange_frame:
            frame_id = self.flange_frame_id
        elif self.pm.model is not None:
            frame_id = self.pm.model.getFrameId(frame_name)

        if frame_id is None or self.pm.model is None or frame_id >= len(self.pm.model.frames):
            raise ValueError(f"Frame '{frame_name}' not available in Pinocchio model.")
        return FrameKinematics(self, frame_id)

    @staticmethod
    def _normalize_frame_name(frame_name):
        frame_name = str(frame_name)
        if frame_name.startswith("panda_"):
            return "fr3_" + frame_name[len("panda_"):]
        return frame_name


class FrameKinematics:
    """PandaKinematics-shaped wrapper around Pinocchio frame kinematics."""

    def __init__(self, robot: Ros2FrankaArm, frame_id: int):
        self.robot = robot
        self.frame_id = int(frame_id)

    def _state(self):
        self.robot.spin_once(timeout_sec=0.0)
        q, qd = self.robot.pm.build_full_state(self.robot._q_meas, self.robot._qd_meas)
        return self.robot.pm.get_frame_state(q, qd, self.frame_id)

    def forward_position_kinematics(self):
        p, R, _, _, _, _ = self._state()
        quat = pin.Quaternion(R)
        quat.normalize()
        return [
            float(p[0]), float(p[1]), float(p[2]),
            float(quat.x), float(quat.y), float(quat.z), float(quat.w),
        ]

    def forward_velocity_kinematics(self):
        _, _, J6, _, _, _ = self._state()
        qd = self.robot._qd_meas.copy()
        return (J6 @ qd).copy()

    def jacobian(self):
        _, _, J6, _, _, _ = self._state()
        return J6.copy()


def update_robot_state(kin_tool, kin_flange):
    """Read tool and flange pose/velocity in the original ROS 1 dict shape."""
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


def safe_move_to_joint_position(robot, joints, timeout=20.0,
                                prefer_streaming=False, log_prefix="safe move"):
    del prefer_streaming
    robot.get_logger().info(f"  {log_prefix}: moving with ROS 2 effort joint PD.")
    return robot.move_to_joint_position(joints, timeout=timeout, use_moveit=False)


def compute_position_rcm(tool_pos, flange_pos, trocar_pos):
    """Point on the instrument axis closest to the trocar."""
    v = tool_pos - flange_pos
    w = trocar_pos - flange_pos
    v_sq = np.dot(v, v)
    if v_sq < 1e-10:
        return flange_pos.copy()
    return flange_pos + (np.dot(v, w) / v_sq) * v


def tool_to_flange_full_ref(x_tool_ref, xdot_tool_ref, trocar_pos, length):
    """Map a tool-tip reference to flange position, velocity, and orientation."""
    d = trocar_pos - x_tool_ref
    r = np.linalg.norm(d)
    if r < 1e-6:
        return x_tool_ref.copy(), np.zeros(3), np.zeros(3)

    n_hat = d / r
    x_flange_ref = x_tool_ref + length * n_hat

    P_perp = np.eye(3) - np.outer(n_hat, n_hat)
    M_vel = np.eye(3) - (length / r) * P_perp
    xdot_flange_ref = M_vel @ xdot_tool_ref

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
    del y_ref
    return Rotation.from_euler(
        "xyz", np.deg2rad([-90.0, 0.0, -45.0])
    ).as_euler("xyz", degrees=False)


def tool_visual_front_axis_local():
    return np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=float)


def euler_angle_diff(a, b):
    diff = b - a
    wrapped = diff % (2 * np.pi)
    wrapped[wrapped > np.pi] -= 2 * np.pi
    return wrapped


def compute_u_rotation(robot_state_euler, robot_state_euler_vel,
                       ref_euler, integ_euler, dt,
                       P_ori=20.0, D_ori=1.0, I_ori=30.0,
                       integ_limit=0.08, omega_limit=None):
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
    pos_weights = np.asarray(
        getattr(ctrl, "no_rcm_null_position_weights", np.ones(n)),
        dtype=float,
    ).reshape(-1)
    vel_weights = np.asarray(
        getattr(ctrl, "no_rcm_null_velocity_weights", pos_weights),
        dtype=float,
    ).reshape(-1)
    pos_weights = np.pad(pos_weights[:n], (0, max(0, n - pos_weights.size)), constant_values=1.0)[:n]
    vel_weights = np.pad(vel_weights[:n], (0, max(0, n - vel_weights.size)), constant_values=1.0)[:n]
    tau_raw = kp * pos_weights * (q_nom - q) - kd * vel_weights * qd
    Jn = np.asarray(J[:, :n], dtype=float)
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


def compute_torque_with_rcm(ctrl, robot_state, kin_flange,
                            x_tool_ref, xdot_tool_ref,
                            e_f, sigma_f,
                            alpha, K_e_hat,
                            trocar_pos, length,
                            integ_euler, dt):
    """Assemble RCM joint torque using the original flange-space control law."""
    tool_pos = robot_state["tool_position"]
    flange_pos = robot_state["flange_position"]
    flange_vel = robot_state["flange_position_velocity"]

    x_flange_ref, xdot_flange_ref, euler_ref = tool_to_flange_full_ref(
        x_tool_ref, xdot_tool_ref, trocar_pos, length
    )

    e_r1_flange = flange_pos - x_flange_ref
    e_r2_flange = flange_vel - xdot_flange_ref

    u_flange, K_eff = ctrl.compute_control(
        e_r1_flange, e_r2_flange, e_f, sigma_f, alpha, K_e_hat
    )

    u_rot, integ_euler_new = compute_u_rotation(
        robot_state["flange_rotation_euler"],
        robot_state["flange_rotation_euler_velocity"],
        euler_ref, integ_euler, dt,
        P_ori=ctrl.P_ori, D_ori=ctrl.D_ori, I_ori=ctrl.I_ori,
    )

    u_cart = np.hstack([u_flange, u_rot])
    cart_norm = np.linalg.norm(u_cart)
    if cart_norm > ctrl.u_threshold:
        u_cart = u_cart * ctrl.u_threshold / cart_norm

    J = np.array(kin_flange.jacobian())
    j_torque = np.clip(
        (J.T @ u_cart).flatten(),
        -ctrl.tau_max, ctrl.tau_max,
    )

    p_rcm = compute_position_rcm(tool_pos, flange_pos, trocar_pos)
    error_rcm = np.linalg.norm(trocar_pos - p_rcm)
    error_track = np.linalg.norm(tool_pos - x_tool_ref)

    return (j_torque, u_flange, K_eff, integ_euler_new,
            x_flange_ref, np.array([error_rcm, error_track]))


def compute_torque_no_rcm(ctrl, robot_state, kin_tool,
                          e_r1, e_r2, e_f, sigma_f,
                          alpha, K_e_hat,
                          ref_euler_fixed, integ_euler, dt):
    """Assemble no-RCM joint torque from tool-space force-position control."""
    if hasattr(ctrl, "compute_control_axis_alpha"):
        alpha_xyz = np.array([1.0, 1.0, float(alpha)], dtype=float)
        u_tool, K_axes = ctrl.compute_control_axis_alpha(
            e_r1, e_r2, e_f, sigma_f, alpha_xyz, K_e_hat
        )
        K_eff = K_axes[2]
    else:
        u_tool, K_eff = ctrl.compute_control(e_r1, e_r2, e_f, sigma_f, alpha, K_e_hat)
    xy_kp = float(getattr(ctrl, "no_rcm_xy_stiffness_boost", 0.0))
    xy_kd = float(getattr(ctrl, "no_rcm_xy_damping_boost", 0.0))
    if xy_kp > 0.0 or xy_kd > 0.0:
        e_r1_arr = np.asarray(e_r1, dtype=float).reshape(3)
        e_r2_arr = np.asarray(e_r2, dtype=float).reshape(3)
        u_tool[:2] += -(xy_kp * e_r1_arr[:2] + xy_kd * e_r2_arr[:2])
    if hasattr(ctrl, "no_rcm_u_tool_limits"):
        limits = np.asarray(ctrl.no_rcm_u_tool_limits, dtype=float)
        if limits.shape == (3,):
            u_tool = np.clip(u_tool, -limits, limits)

    omega_for_damping = _filter_vector_state(
        ctrl,
        "_no_rcm_omega_filtered",
        robot_state["tool_rotation_euler_velocity"],
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

    u_cart = np.hstack([u_tool, u_rot])
    cart_norm = np.linalg.norm(u_cart)
    if cart_norm > ctrl.u_threshold:
        u_cart = u_cart * ctrl.u_threshold / cart_norm

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
