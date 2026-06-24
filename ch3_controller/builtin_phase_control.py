"""Helpers for using ros2_control built-in trajectory phases in Gazebo."""

import subprocess
import time
import re
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class BuiltinPhaseController:
    """Switch to ros2_control trajectory controller for non-scan phases."""

    def __init__(
        self,
        node,
        robot,
        trajectory_controller="no_rcm_position_trajectory_controller",
        effort_controller="no_rcm_effort_controller",
        controller_manager="/controller_manager",
    ):
        self.node = node
        self.robot = robot
        self.trajectory_controller = trajectory_controller
        self.effort_controller = effort_controller
        self.controller_manager = controller_manager
        self.action_name = f"/{trajectory_controller}/follow_joint_trajectory"
        self._action_client = ActionClient(node, FollowJointTrajectory, self.action_name)
        self.param_file = str(
            Path(get_package_share_directory("franka_gazebo_bringup"))
            / "config"
            / "franka_gazebo_controllers.yaml"
        )

    def _run(self, cmd, timeout=20.0):
        self.node.get_logger().info("builtin phase command: " + " ".join(cmd))
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    def _controller_state(self, name):
        proc = self._run(
            ["ros2", "control", "list_controllers", "-c", self.controller_manager],
            timeout=10.0,
        )
        if proc.returncode != 0:
            self.node.get_logger().warn(proc.stdout[-500:])
            return None
        for line in proc.stdout.splitlines():
            line = _ANSI_RE.sub("", line)
            parts = line.split()
            if len(parts) >= 3 and parts[0] == name:
                return parts[-1]
        return None

    def ensure_trajectory_controller_loaded(self):
        state = self._controller_state(self.trajectory_controller)
        if state is not None:
            return True
        proc = self._run(
            [
                "ros2",
                "run",
                "controller_manager",
                "spawner",
                self.trajectory_controller,
                "--controller-manager",
                self.controller_manager,
                "--param-file",
                self.param_file,
                "--inactive",
                "--controller-manager-timeout",
                "30",
            ],
            timeout=35.0,
        )
        if proc.returncode != 0:
            self.node.get_logger().error(proc.stdout[-1000:])
            return False
        return self._controller_state(self.trajectory_controller) is not None

    def switch_to_trajectory(self):
        if not self.ensure_trajectory_controller_loaded():
            return False
        if self._controller_state(self.trajectory_controller) == "active":
            return True
        proc = self._run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--deactivate",
                self.effort_controller,
                "--activate",
                self.trajectory_controller,
                "--strict",
                "--activate-asap",
                "-c",
                self.controller_manager,
            ],
            timeout=15.0,
        )
        if proc.returncode != 0:
            self.node.get_logger().error(proc.stdout[-1000:])
            return False
        return True

    def switch_to_effort(self):
        if self._controller_state(self.effort_controller) == "active":
            return True
        proc = self._run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--deactivate",
                self.trajectory_controller,
                "--activate",
                self.effort_controller,
                "--strict",
                "--activate-asap",
                "-c",
                self.controller_manager,
            ],
            timeout=15.0,
        )
        if proc.returncode != 0:
            self.node.get_logger().error(proc.stdout[-1000:])
            return False
        return True

    def move_joints(self, target, duration=5.0, timeout=None):
        target = np.asarray(target, dtype=float).reshape(len(JOINT_NAMES))
        if not self._action_client.wait_for_server(timeout_sec=8.0):
            self.node.get_logger().error(f"Action server unavailable: {self.action_name}")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((float(duration) - int(duration)) * 1e9)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance.sec = 3
        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=timeout or 10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error("Built-in trajectory goal rejected.")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self.node,
            result_future,
            timeout_sec=timeout or max(10.0, duration + 5.0),
        )
        result = result_future.result()
        if result is None:
            self.node.get_logger().error("Built-in trajectory goal timed out.")
            return False
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().warn(
                "Built-in trajectory finished with error "
                f"{result.result.error_code}: {result.result.error_string}"
            )
            return False
        return True

    def solve_tool_ik(
        self,
        target_pos,
        target_euler,
        q_seed=None,
        max_iters=120,
        pos_tol=0.002,
        ori_tol_deg=3.0,
    ):
        pm = self.robot.pm
        if pm.model is None or pm.data is None or pm.tip_frame_id is None:
            return None
        q = np.asarray(q_seed if q_seed is not None else self.robot.angles(), dtype=float).copy()
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_R = Rotation.from_euler("xyz", target_euler).as_matrix()
        lower = np.asarray(pm.model.lowerPositionLimit, dtype=float)
        upper = np.asarray(pm.model.upperPositionLimit, dtype=float)
        finite = np.isfinite(lower) & np.isfinite(upper) & (lower < upper)
        damping = 1e-4
        ori_weight = 0.35
        for _ in range(int(max_iters)):
            q_full, qd_full = pm.build_full_state(q, np.zeros_like(q))
            p, R, J6, Jv, Jw, _ = pm.get_frame_state(q_full, qd_full, pm.tip_frame_id)
            pos_err = target_pos - p
            rot_err = Rotation.from_matrix(target_R @ R.T).as_rotvec()
            if (
                np.linalg.norm(pos_err) <= pos_tol
                and np.degrees(np.linalg.norm(rot_err)) <= ori_tol_deg
            ):
                return q.copy()
            err = np.hstack([pos_err, ori_weight * rot_err])
            J = np.vstack([Jv, ori_weight * Jw])
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -0.08, 0.08)
            q = pin.integrate(pm.model, q, dq)
            q[finite] = np.clip(q[finite], lower[finite] + 1e-3, upper[finite] - 1e-3)
        return q.copy()

    def move_tool_coarse(self, target_pos, target_euler, duration=6.0, timeout=None):
        q_target = self.solve_tool_ik(target_pos, target_euler)
        if q_target is None:
            self.node.get_logger().error("Coarse descent IK failed.")
            return False
        self.node.get_logger().info(
            "Built-in coarse target joints: "
            + np.array2string(q_target, precision=4, suppress_small=True)
        )
        return self.move_joints(q_target, duration=duration, timeout=timeout)

    def wait_after_switch(self, seconds=0.2):
        deadline = time.time() + float(seconds)
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.01)
