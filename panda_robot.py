"""Minimal ROS 2-backed replacement for the ROS 1 panda_robot API."""

import rclpy

import rospy
from ch3_controller.robot_interface_ros2 import Ros2FrankaArm


class PandaArm(Ros2FrankaArm):
    """PandaArm-compatible facade used by the original 0618 scripts."""

    def __init__(self, *args, **kwargs):
        del args
        node_name = kwargs.pop("node_name", rospy.get_node_name())
        default_tool_frame = kwargs.pop("default_tool_frame", "fr3_link11")
        default_flange_frame = kwargs.pop("default_flange_frame", "fr3_link8")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported PandaArm arguments in ROS 2 shim: {unknown}")
        if not rclpy.ok():
            rospy.init_node(node_name)
        super().__init__(
            node_name=node_name,
            default_tool_frame=default_tool_frame,
            default_flange_frame=default_flange_frame,
        )
        rospy.set_node(self)


class PandaKinematics:
    """PandaKinematics-compatible facade over Pinocchio frame kinematics."""

    def __init__(self, robot, frame_name):
        self.robot = robot
        self.frame_name = frame_name
        self._impl = robot.kinematics(frame_name)

    def forward_position_kinematics(self):
        return self._impl.forward_position_kinematics()

    def forward_velocity_kinematics(self):
        return self._impl.forward_velocity_kinematics()

    def jacobian(self):
        return self._impl.jacobian()
