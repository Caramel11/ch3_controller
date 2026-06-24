"""Tiny rospy-shaped compatibility layer for the 0618 ch3 ROS 2 port."""

import logging
import sys
import time

import rclpy


_node = None
_node_name = "ch3_controller"
_rclpy_args = None
_throttle_state = {}
_fallback_logger = logging.getLogger("ch3_controller")
_fallback_logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def set_rclpy_args(args):
    global _rclpy_args
    _rclpy_args = list(args)


def init_node(name, anonymous=False, **kwargs):
    del kwargs
    global _node_name
    _node_name = str(name)
    if anonymous:
        _node_name = f"{_node_name}_{int(time.time() * 1000) % 100000}"
    if not rclpy.ok():
        rclpy.init(args=_rclpy_args or sys.argv)


def set_node(node):
    global _node
    _node = node


def get_node():
    return _node


def get_node_name():
    return _node_name


def is_shutdown():
    return not rclpy.ok()


def get_time():
    if _node is not None:
        return _node.get_clock().now().nanoseconds * 1e-9
    return time.time()


def spin_once(timeout_sec=0.0):
    if _node is not None and rclpy.ok():
        rclpy.spin_once(_node, timeout_sec=timeout_sec)


def sleep(duration):
    deadline = time.time() + max(0.0, float(duration))
    while rclpy.ok() and time.time() < deadline:
        spin_once(timeout_sec=0.0)
        time.sleep(min(0.01, max(0.0, deadline - time.time())))


class Rate:
    def __init__(self, hz):
        self.period = 1.0 / max(float(hz), 1e-9)
        self._next = time.time() + self.period

    def sleep(self):
        spin_once(timeout_sec=0.0)
        now = time.time()
        remaining = self._next - now
        if remaining > 0.0:
            deadline = now + remaining
            while rclpy.ok() and time.time() < deadline:
                spin_once(timeout_sec=0.0)
                time.sleep(min(0.001, max(0.0, deadline - time.time())))
        else:
            self._next = now
        self._next += self.period


class _TimeValue:
    def __init__(self, seconds):
        self._seconds = float(seconds)

    def to_sec(self):
        return self._seconds

    def __sub__(self, other):
        return Duration(self._seconds - other._seconds)

    def __add__(self, other):
        return _TimeValue(self._seconds + other.to_sec())

    def __lt__(self, other):
        return self._seconds < other._seconds

    def __le__(self, other):
        return self._seconds <= other._seconds

    def __gt__(self, other):
        return self._seconds > other._seconds

    def __ge__(self, other):
        return self._seconds >= other._seconds


class Time:
    @staticmethod
    def now():
        return _TimeValue(get_time())


class Duration:
    def __init__(self, seconds):
        self._seconds = float(seconds)

    def to_sec(self):
        return self._seconds


def _logger():
    if _node is not None:
        return _node.get_logger()
    return _fallback_logger


def loginfo(message, *args, **kwargs):
    del args, kwargs
    _logger().info(str(message))


def logwarn(message, *args, **kwargs):
    del args, kwargs
    _logger().warn(str(message))


def logerr(message, *args, **kwargs):
    del args, kwargs
    _logger().error(str(message))


def logdebug(message, *args, **kwargs):
    del args, kwargs
    _logger().debug(str(message))


def logwarn_throttle(period, message):
    now = time.time()
    key = str(message)
    last = _throttle_state.get(key)
    if last is None or now - last >= float(period):
        _throttle_state[key] = now
        logwarn(message)


def Subscriber(topic, msg_type, callback, queue_size=1):
    if _node is None:
        raise RuntimeError("rospy compatibility node is not set")
    return _node.create_subscription(msg_type, topic, callback, queue_size)


def signal_shutdown(reason=""):
    if reason:
        loginfo(f"shutdown requested: {reason}")
    shutdown()


def shutdown():
    global _node
    if _node is not None:
        try:
            _node.destroy_node()
        except Exception:
            pass
        _node = None
    if rclpy.ok():
        rclpy.shutdown()
