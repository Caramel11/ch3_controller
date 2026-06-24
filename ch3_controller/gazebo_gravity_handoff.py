"""Restore Gazebo gravity after startup gravity compensation is active."""

import re
import subprocess
import time

import rclpy
from rclpy.node import Node


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


class GazeboGravityHandoff(Node):
    def __init__(self):
        super().__init__("gazebo_gravity_handoff")
        self.declare_parameter("world_name", "empty_no_gravity")
        self.declare_parameter("controller_name", "gravity_compensation_example_controller")
        self.declare_parameter("gravity_z", -9.8)
        self.declare_parameter("timeout_sec", 30.0)
        self.declare_parameter("settle_sec", 0.5)
        self.declare_parameter("physics_profile", "1ms")
        self.declare_parameter("max_step_size", 0.001)
        self.declare_parameter("real_time_factor", 1.0)

        self.world_name = str(self.get_parameter("world_name").value)
        self.controller_name = str(self.get_parameter("controller_name").value)
        self.gravity_z = float(self.get_parameter("gravity_z").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.physics_profile = str(self.get_parameter("physics_profile").value)
        self.max_step_size = float(self.get_parameter("max_step_size").value)
        self.real_time_factor = float(self.get_parameter("real_time_factor").value)

    def _controller_active(self):
        try:
            proc = subprocess.run(
                ["ros2", "control", "list_controllers"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2.0,
            )
        except Exception:
            return False
        clean = _ANSI_RE.sub("", proc.stdout)
        for line in clean.splitlines():
            parts = line.split()
            if parts and parts[0] == self.controller_name and "active" in parts:
                return True
        return False

    def _restore_gravity(self):
        req = (
            f'profile_name: "{self.physics_profile}" '
            f"gravity: {{x: 0 y: 0 z: {self.gravity_z}}} "
            "enable_physics: true "
            f"max_step_size: {self.max_step_size} "
            f"real_time_factor: {self.real_time_factor}"
        )
        service = f"/world/{self.world_name}/set_physics"
        cmd = [
            "ign",
            "service",
            "-s",
            service,
            "--reqtype",
            "ignition.msgs.Physics",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "3000",
            "--req",
            req,
        ]
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )

    def run(self):
        self.get_logger().info(
            f"Waiting for {self.controller_name} before restoring gravity "
            f"in world '{self.world_name}'."
        )
        deadline = time.time() + self.timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if self._controller_active():
                self.get_logger().info(
                    f"{self.controller_name} is active; settling for "
                    f"{self.settle_sec:.2f}s before gravity restore."
                )
                time.sleep(max(self.settle_sec, 0.0))
                proc = self._restore_gravity()
                if proc.returncode == 0 and "true" in proc.stdout.lower():
                    self.get_logger().info(
                        f"Gazebo gravity restored to z={self.gravity_z:.3f}."
                    )
                    return 0
                self.get_logger().error(
                    "Failed to restore Gazebo gravity: "
                    f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
                )
                return 1
            time.sleep(0.1)
        self.get_logger().error(
            f"Timed out waiting for active controller '{self.controller_name}'."
        )
        return 1


def main():
    rclpy.init()
    node = GazeboGravityHandoff()
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
