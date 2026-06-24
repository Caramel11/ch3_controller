"""ROS 2 console entry points for the original 0618 scripts."""

import importlib
import sys

from rclpy.utilities import remove_ros_args


def _run(module_name):
    ros_args = list(sys.argv)
    sys.argv = remove_ros_args(args=ros_args)
    from ch3_controller import ros_compat

    ros_compat.set_rclpy_args(ros_args)
    module = importlib.import_module(module_name)
    try:
        return module.main()
    finally:
        try:
            import rospy

            rospy.shutdown()
        except Exception:
            pass


def run_no_rcm():
    return _run("run_no_rcm")


def run_no_rcm_builtin_phases():
    return _run("run_no_rcm_builtin_phases")


def run_with_rcm():
    return _run("run_with_rcm")


def run_no_rcm_real():
    return _run("run_no_rcm_real")


def run_with_rcm_real():
    return _run("run_with_rcm_real")


def run_no_rcm_force_margin_challenge():
    return _run("run_no_rcm_force_margin_challenge")


def run_with_rcm_force_margin_challenge():
    return _run("run_with_rcm_force_margin_challenge")


def test_real_mock_sensor_gazebo():
    return _run("test_real_mock_sensor_gazebo")
