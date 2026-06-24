"""Startup gravity-compensated joint hold for Gazebo effort control."""

import time

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .pinocchio_model import PinocchioModelHelper
from .robot_interface_ros2 import INIT_JOINTS


class GravityHoldNode(Node):
    def __init__(self):
        super().__init__("gravity_hold")

        self.declare_parameter("cmd_topic", "/no_rcm_effort_controller/commands")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("rsp_node", "/robot_state_publisher")
        self.declare_parameter("reference_frame", "fr3_link8")
        self.declare_parameter("tip_frame", "fr3_link11")
        self.declare_parameter("rate_hz", 500.0)
        self.declare_parameter("gravity_compensation_scale", 1.0)
        self.declare_parameter("target_mode", "initial")
        self.declare_parameter("target_joints", list(INIT_JOINTS))
        self.declare_parameter("kp", [10.0, 10.0, 10.0, 10.0, 4.0, 3.0, 2.0])
        self.declare_parameter("kd", [3.0, 3.0, 3.0, 3.0, 1.2, 1.0, 0.8])
        self.declare_parameter("max_tau_abs", 18.0)
        self.declare_parameter("max_tau_rate", 120.0)
        self.declare_parameter("handoff_after_sec", 0.0)
        self.declare_parameter("log_interval_sec", 1.0)
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

        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.rsp_node = self.get_parameter("rsp_node").value
        self.reference_frame = self.get_parameter("reference_frame").value
        self.tip_frame = self.get_parameter("tip_frame").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.gravity_scale = float(self.get_parameter("gravity_compensation_scale").value)
        self.target_mode = str(self.get_parameter("target_mode").value)
        self.target_joints = np.asarray(self.get_parameter("target_joints").value, dtype=float)
        self.kp = np.asarray(self.get_parameter("kp").value, dtype=float)
        self.kd = np.asarray(self.get_parameter("kd").value, dtype=float)
        self.max_tau_abs = float(self.get_parameter("max_tau_abs").value)
        self.max_tau_rate = float(self.get_parameter("max_tau_rate").value)
        self.handoff_after_sec = float(self.get_parameter("handoff_after_sec").value)
        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.ctrl_joints = list(self.get_parameter("controlled_joints").value)

        n = len(self.ctrl_joints)
        self._q = np.zeros(n)
        self._qd = np.zeros(n)
        self._have_state = False
        self._target_locked = False
        self._tau_prev = np.zeros(n)
        self._started_at = None
        self._last_log = 0.0
        self._last_model_load_attempt = 0.0
        self.done = False

        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.sub = self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)

        self.pm = PinocchioModelHelper(
            node=self,
            rsp_node=self.rsp_node,
            controlled_joints=self.ctrl_joints,
            reference_frame=self.reference_frame,
            tip_frame=self.tip_frame,
        )
        self._try_load_model()

        period = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"gravity_hold publishing {self.cmd_topic} at {self.rate_hz:.1f}Hz, "
            f"target_mode={self.target_mode}, gravity_scale={self.gravity_scale:.3f}, "
            f"handoff_after_sec={self.handoff_after_sec:.1f}"
        )

    def _on_joint_state(self, msg: JointState):
        name_to_i = {name: i for i, name in enumerate(msg.name)}
        for k, joint_name in enumerate(self.ctrl_joints):
            i = name_to_i.get(joint_name)
            if i is None:
                return
            if i < len(msg.position):
                self._q[k] = float(msg.position[i])
            if i < len(msg.velocity):
                self._qd[k] = float(msg.velocity[i])
        self._have_state = True

    def _try_load_model(self):
        if self.pm.model is not None and self.pm.data is not None:
            return True
        now = time.time()
        if now - self._last_model_load_attempt < 1.0:
            return False
        self._last_model_load_attempt = now
        if self.pm.load():
            return True
        self.get_logger().warn("Waiting for robot_description before gravity hold can run.")
        return False

    def _lock_target_if_needed(self):
        if self._target_locked:
            return
        if self.target_mode == "current":
            self.target_joints = self._q.copy()
        else:
            self.target_joints = self.target_joints.reshape(len(self.ctrl_joints))
        self._target_locked = True
        self._started_at = time.time()
        self.get_logger().info(f"gravity_hold target joints: {self.target_joints}")

    def _gravity_torque(self):
        if self.gravity_scale == 0.0 or self.pm.model is None or self.pm.data is None:
            return np.zeros(len(self.ctrl_joints), dtype=float)
        q, _ = self.pm.build_full_state(self._q, self._qd)
        tau_g = pin.computeGeneralizedGravity(self.pm.model, self.pm.data, q)
        tau_g = self.pm.extract_controlled_effort(np.asarray(tau_g, dtype=float))
        return self.gravity_scale * tau_g.reshape(len(self.ctrl_joints))

    def _rate_and_clip(self, tau, dt):
        tau = np.clip(tau, -self.max_tau_abs, self.max_tau_abs)
        max_delta = self.max_tau_rate * max(float(dt), 1e-6)
        delta = np.clip(tau - self._tau_prev, -max_delta, max_delta)
        tau_limited = self._tau_prev + delta
        self._tau_prev = tau_limited.copy()
        return tau_limited

    def _on_timer(self):
        if not self._try_load_model():
            return
        if not self._have_state:
            return
        self._lock_target_if_needed()

        period = 1.0 / max(self.rate_hz, 1.0)
        err = self.target_joints - self._q
        tau_pd = self.kp * err - self.kd * self._qd
        tau = self._gravity_torque() + tau_pd
        tau = self._rate_and_clip(tau, period)

        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.pub.publish(msg)

        now = time.time()
        if now - self._last_log >= self.log_interval_sec:
            self._last_log = now
            self.get_logger().info(
                f"hold err_inf={np.max(np.abs(err)):.4f}rad, "
                f"qd_norm={np.linalg.norm(self._qd):.4f}rad/s, "
                f"tau_max={np.max(np.abs(tau)):.2f}Nm"
            )

        if self.handoff_after_sec > 0.0 and self._started_at is not None:
            if now - self._started_at >= self.handoff_after_sec:
                self.get_logger().info("gravity_hold handoff timeout reached; exiting.")
                self.done = True


def main():
    rclpy.init()
    node = GravityHoldNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
