"""
Direct MIOS force sensor reader for real experiments.

This module mirrors the protocol logic from tests/force_sensor_ros_node_test.py,
but it does not publish or subscribe to ROS topics.  It owns the serial port,
starts the sensor stream, reads frames in a background thread, and exposes the
latest wrench through the same small API used by the cooperative GT scripts.
"""
import struct
import threading
import time
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import rospy
import serial


MANUAL_HEAD = b"\xAA"
MANUAL_TAIL_TX = b"\x0B\x0C"
MANUAL_TAIL_RX = b"\xFE"
MANUAL_FRAME_LEN_RX = 1 + 1 + 24 + 1

VENDOR_HEAD = b"\xAA\x55"
VENDOR_TAIL = b"\x0D\x0A"
VENDOR_FRAME_LEN_RX = 2 + 1 + 24 + 2

BINARY_28B_FRAME_LEN = 1 + 1 + 24 + 2

CMD_STOP = 0x01
CMD_STREAM_1KHZ = 0x02
CMD_SINGLE_SHOT = 0x03
CMD_TARE = 0x30
CMD_DEBUG_EXIT = 0x31
CMD_MATRIX_FILTERED = 0x33
CMD_RAW_DATA = 0x34

DATA_CMDS = frozenset({
    CMD_STREAM_1KHZ,
    CMD_SINGLE_SHOT,
    CMD_MATRIX_FILTERED,
    CMD_RAW_DATA,
})


def _unpack_floats(data24: bytes) -> Optional[np.ndarray]:
    if len(data24) != 24:
        return None
    try:
        return np.array(struct.unpack("<6f", data24), dtype=float)
    except struct.error:
        return None


def _periodic_pattern(positions, period: int, min_hits: int = 2) -> bool:
    if len(positions) < min_hits + 1:
        return False
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    return Counter(diffs).get(period, 0) >= min_hits


class DirectForceSensorInput:
    """
    Direct serial replacement for ForceSensorInput.

    The public methods intentionally match the ROS-topic ForceSensorInput used
    by run_with_rcm.py and run_no_rcm.py: available(), contact_force(),
    wrench_vector(), wait_for_data(), age(), seq(), and close().
    """

    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=460800,
        serial_timeout=0.05,
        freshness_timeout=0.02,
        force_axis=2,
        force_sign=1.0,
        command_format="both",
        data_source_cmd=CMD_MATRIX_FILTERED,
        use_streaming=True,
        output_units="N",
        gravity=9.80665,
        tare_on_start=True,
        tare_settle_s=1.0,
        exit_debug_on_start=True,
        stop_on_start=True,
        poll_hz=100.0,
    ):
        if force_axis not in (0, 1, 2):
            raise ValueError("force_axis must be 0, 1, or 2")
        if command_format not in ("manual", "vendor", "both"):
            raise ValueError("command_format must be manual, vendor, or both")
        if output_units not in ("N", "kgf"):
            raise ValueError("output_units must be N or kgf")

        self.port = port
        self.baudrate = int(baudrate)
        self.serial_timeout = float(serial_timeout)
        self.timeout = float(freshness_timeout)
        self.force_axis = int(force_axis)
        self.force_sign = float(force_sign)
        self.command_format = command_format
        self.data_source_cmd = int(data_source_cmd)
        self.use_streaming = bool(use_streaming)
        self.output_units = output_units
        self.gravity = float(gravity)
        self.tare_on_start = bool(tare_on_start)
        self.tare_settle_s = float(tare_settle_s)
        self.exit_debug_on_start = bool(exit_debug_on_start)
        self.stop_on_start = bool(stop_on_start)
        self.poll_hz = float(poll_hz)

        self._force = np.zeros(3)
        self._torque = np.zeros(3)
        self._stamp = None
        self._seq = 0
        self._format = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.serial_timeout,
        )
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self._startup_sequence()
        self._start_data_stream()
        time.sleep(0.2)
        self._format = self._detect_format_with_tare_recovery()
        if self._format not in ("binary_28b", "binary_manual", "binary_vendor"):
            raise RuntimeError(f"force sensor format detection failed: {self._format}")

        self.ser.reset_input_buffer()
        if self.use_streaming:
            self._cmd_stream_1khz()

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    @property
    def detected_format(self):
        return self._format

    def _build_manual(self, cmd: int, payload: bytes = b"") -> bytes:
        return MANUAL_HEAD + bytes([cmd]) + payload + MANUAL_TAIL_TX

    def _build_vendor(self, cmd: int, payload: bytes = b"") -> bytes:
        return VENDOR_HEAD + bytes([cmd]) + payload + VENDOR_TAIL

    def _send_command(self, cmd: int, payload: bytes = b"", delay_s: float = 0.03):
        if self.command_format == "manual":
            self.ser.write(self._build_manual(cmd, payload))
        elif self.command_format == "vendor":
            self.ser.write(self._build_vendor(cmd, payload))
        else:
            self.ser.write(self._build_manual(cmd, payload))
            time.sleep(delay_s)
            self.ser.write(self._build_vendor(cmd, payload))
        self.ser.flush()

    def _cmd_stop(self):
        self._send_command(CMD_STOP)
        time.sleep(0.1)

    def _cmd_stream_1khz(self):
        self._send_command(CMD_STREAM_1KHZ)
        time.sleep(0.05)

    def _cmd_single_shot(self):
        self._send_command(CMD_SINGLE_SHOT)

    def _drain_input_until_quiet(self, quiet_s=0.25, max_s=2.0, label=""):
        deadline = time.time() + float(max_s)
        quiet_deadline = time.time() + float(quiet_s)
        drained = bytearray()
        while time.time() < deadline and not self._stop_event.is_set():
            waiting = self.ser.in_waiting
            if waiting:
                drained.extend(self.ser.read(waiting))
                quiet_deadline = time.time() + float(quiet_s)
                continue
            if time.time() >= quiet_deadline:
                break
            time.sleep(0.01)
        self.ser.reset_input_buffer()
        if drained:
            prefix = f"{label}: " if label else ""
            rospy.loginfo(f"{prefix}drained {len(drained)} bytes of sensor echo/residual data")
        return bytes(drained)

    def _startup_sequence(self):
        if self.stop_on_start:
            rospy.loginfo("Direct force sensor: stop residual stream (0x01)")
            self._cmd_stop()
            self.ser.reset_input_buffer()
        if self.exit_debug_on_start:
            rospy.loginfo("Direct force sensor: exit debug mode (0x31)")
            self._send_command(CMD_DEBUG_EXIT)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
        if self.tare_on_start:
            rospy.loginfo("Direct force sensor: tare/zero on start (0x30)")
            self._send_command(CMD_TARE)
            time.sleep(0.2)
            self._drain_input_until_quiet(
                quiet_s=0.25, max_s=self.tare_settle_s, label="after tare"
            )

    def _start_data_stream(self):
        self.ser.reset_input_buffer()
        if self.data_source_cmd in (CMD_MATRIX_FILTERED, CMD_RAW_DATA):
            rospy.loginfo(f"Direct force sensor: data source 0x{self.data_source_cmd:02X}")
            self._send_command(self.data_source_cmd)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        if self.use_streaming:
            rospy.loginfo("Direct force sensor: start 1kHz stream (0x02)")
            self._cmd_stream_1khz()

    def _detect_format_sample(self, duration_s: float = 0.8) -> Tuple[str, bytes]:
        deadline = time.time() + duration_s
        buf = bytearray()
        while time.time() < deadline and len(buf) < 2000:
            if not self.use_streaming:
                self._cmd_single_shot()
            chunk = self.ser.read(128)
            if chunk:
                buf.extend(chunk)
        captured = bytes(buf)
        if not captured:
            return "none", captured

        pos_28b = [
            i for i in range(len(captured) - BINARY_28B_FRAME_LEN + 1)
            if captured[i] == 0xAA
            and captured[i + 1] in DATA_CMDS
            and captured[i + BINARY_28B_FRAME_LEN - 2:i + BINARY_28B_FRAME_LEN] == VENDOR_TAIL
        ]
        if _periodic_pattern(pos_28b, BINARY_28B_FRAME_LEN):
            return "binary_28b", captured

        pos_vendor = [
            i for i in range(len(captured) - VENDOR_FRAME_LEN_RX + 1)
            if captured[i] == 0xAA
            and captured[i + 1] == 0x55
            and captured[i + 2] in DATA_CMDS
            and captured[i + VENDOR_FRAME_LEN_RX - 2:i + VENDOR_FRAME_LEN_RX] == VENDOR_TAIL
        ]
        if _periodic_pattern(pos_vendor, VENDOR_FRAME_LEN_RX):
            return "binary_vendor", captured

        pos_manual = [
            i for i in range(len(captured) - MANUAL_FRAME_LEN_RX + 1)
            if captured[i] == 0xAA
            and captured[i + 1] in DATA_CMDS
            and captured[i + MANUAL_FRAME_LEN_RX - 1] == MANUAL_TAIL_RX[0]
        ]
        if _periodic_pattern(pos_manual, MANUAL_FRAME_LEN_RX):
            return "binary_manual", captured

        return "unknown", captured

    def _detect_format(self, duration_s: float = 0.8) -> str:
        fmt, _ = self._detect_format_sample(duration_s=duration_s)
        return fmt

    def _detect_format_with_tare_recovery(self) -> str:
        fmt, sample = self._detect_format_sample(duration_s=0.8)
        rospy.loginfo(f"Direct force sensor: detected_format={fmt}, captured={len(sample)} bytes")
        if fmt == "unknown" and b"zero[" in sample:
            rospy.logwarn(
                "Direct force sensor: tare zero[] echo polluted format detection; "
                "draining, restarting 0x33/0x02 stream, and retrying"
            )
            self._drain_input_until_quiet(quiet_s=0.25, max_s=1.0, label="tare recovery")
            self._start_data_stream()
            time.sleep(0.2)
            fmt, sample = self._detect_format_sample(duration_s=0.8)
            rospy.loginfo(
                f"Direct force sensor: retry detected_format={fmt}, captured={len(sample)} bytes"
            )
        return fmt

    def _read_binary_28b(self) -> Optional[Tuple[int, np.ndarray]]:
        deadline = time.time() + self.serial_timeout
        while time.time() < deadline and not self._stop_event.is_set():
            b = self.ser.read(1)
            if not b or b[0] != 0xAA:
                continue
            cmd_byte = self.ser.read(1)
            if len(cmd_byte) != 1 or cmd_byte[0] not in DATA_CMDS:
                continue
            rest = self.ser.read(BINARY_28B_FRAME_LEN - 2)
            if len(rest) != BINARY_28B_FRAME_LEN - 2:
                return None
            if rest[24:26] != VENDOR_TAIL:
                continue
            vals = _unpack_floats(rest[0:24])
            return (cmd_byte[0], vals) if vals is not None else None
        return None

    def _read_binary_manual(self) -> Optional[Tuple[int, np.ndarray]]:
        deadline = time.time() + self.serial_timeout
        while time.time() < deadline and not self._stop_event.is_set():
            b = self.ser.read(1)
            if not b or b[0] != 0xAA:
                continue
            cmd_byte = self.ser.read(1)
            if len(cmd_byte) != 1 or cmd_byte[0] not in DATA_CMDS:
                continue
            rest = self.ser.read(MANUAL_FRAME_LEN_RX - 2)
            if len(rest) != MANUAL_FRAME_LEN_RX - 2:
                return None
            if rest[24] != MANUAL_TAIL_RX[0]:
                continue
            vals = _unpack_floats(rest[0:24])
            return (cmd_byte[0], vals) if vals is not None else None
        return None

    def _read_binary_vendor(self) -> Optional[Tuple[int, np.ndarray]]:
        deadline = time.time() + self.serial_timeout
        while time.time() < deadline and not self._stop_event.is_set():
            b1 = self.ser.read(1)
            if not b1 or b1 != b"\xAA":
                continue
            b2 = self.ser.read(1)
            if b2 != b"\x55":
                continue
            rest = self.ser.read(VENDOR_FRAME_LEN_RX - 2)
            if len(rest) != VENDOR_FRAME_LEN_RX - 2:
                return None
            cmd = rest[0]
            if cmd not in DATA_CMDS or rest[25:27] != VENDOR_TAIL:
                continue
            vals = _unpack_floats(rest[1:25])
            return (cmd, vals) if vals is not None else None
        return None

    def _read_frame(self):
        if self._format == "binary_28b":
            return self._read_binary_28b()
        if self._format == "binary_manual":
            return self._read_binary_manual()
        if self._format == "binary_vendor":
            return self._read_binary_vendor()
        return None

    def _convert_units(self, vals_kg: np.ndarray) -> np.ndarray:
        return vals_kg * self.gravity if self.output_units == "N" else vals_kg

    def _read_loop(self):
        rate = rospy.Rate(self.poll_hz) if not self.use_streaming and self.poll_hz > 0 else None
        while not rospy.is_shutdown() and not self._stop_event.is_set():
            if not self.use_streaming:
                self.ser.reset_input_buffer()
                self._cmd_single_shot()

            result = self._read_frame()
            if result is not None:
                _, vals_kg = result
                vals = self._convert_units(vals_kg)
                with self._lock:
                    self._force = vals[:3].copy()
                    self._torque = vals[3:].copy()
                    self._stamp = time.time()
                    self._seq += 1

            if rate is not None:
                rate.sleep()

    def available(self):
        with self._lock:
            stamp = self._stamp
        return stamp is not None and (time.time() - stamp) <= self.timeout

    def age(self):
        with self._lock:
            stamp = self._stamp
        return float("inf") if stamp is None else time.time() - stamp

    def seq(self):
        with self._lock:
            return self._seq

    def wait_for_data(self, timeout=2.0):
        deadline = time.time() + float(timeout)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.available():
                return True
            time.sleep(0.01)
        return self.available()

    def force_vector(self):
        with self._lock:
            return self._force.copy()

    def wrench_vector(self):
        with self._lock:
            return np.hstack([self._force, self._torque])

    def signed_axis_force(self):
        with self._lock:
            return self.force_sign * self._force[self.force_axis]

    def contact_force(self):
        return abs(self.signed_axis_force())

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._cmd_stop()
        except Exception:
            pass
        if self.ser and self.ser.is_open:
            self.ser.close()
