"""
通用工具: 虚拟接触环境、力传感器输入、滤波限幅和数据记录。

当前 no-RCM 推荐命令默认使用 VirtualStiffnessSurface 生成接触力；
ForceSensorInput 保留为传感器接入的兼容工具；DataLogger 负责把每个控制周期
的关键物理量保存为 npz，便于离线画图和统计指标。
"""
import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped


class VirtualStiffnessSurface:
    """
    分段 Kelvin-Voigt 虚拟接触面
    zones: [(x_start, x_end, K_e, B_e), ...]

    力模型:
      F = K_e(x) * delta + B_e(x) * delta_dot

    其中 K_e/B_e 可随 x 分段变化，transition_width 用于在相邻刚度区间边界
    做平滑过渡，避免扫描跨区时力突变。
    """

    def __init__(self, zones, transition_width=0.02):
        self.zones = zones
        self.transition_width = float(max(transition_width, 0.0))

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _blend(self, left, right, x, boundary):
        if self.transition_width <= 0.0:
            return right if x >= boundary else left
        half = 0.5 * self.transition_width
        s = (x - (boundary - half)) / max(self.transition_width, 1e-12)
        w = self._smoothstep(s)
        K = (1.0 - w) * left[2] + w * right[2]
        B = (1.0 - w) * left[3] + w * right[3]
        return K, B

    def get_stiffness(self, x):
        """根据当前 x 位置返回插值后的 (K_e, B_e)。"""
        if x < self.zones[0][0]:
            return self.zones[0][2], self.zones[0][3]
        for idx, (x0, x1, K, B) in enumerate(self.zones):
            if x0 <= x < x1:
                if idx + 1 < len(self.zones):
                    nxt = self.zones[idx + 1]
                    if abs(x1 - nxt[0]) < 1e-9 and x >= x1 - 0.5 * self.transition_width:
                        return self._blend((x0, x1, K, B), nxt, x, x1)
                if idx > 0:
                    prev = self.zones[idx - 1]
                    if abs(prev[1] - x0) < 1e-9 and x <= x0 + 0.5 * self.transition_width:
                        return self._blend(prev, (x0, x1, K, B), x, x0)
                return K, B
        return self.zones[-1][2], self.zones[-1][3]

    def compute_force(self, x, delta, delta_dot):
        """根据 Kelvin-Voigt 模型计算单向接触力。

        delta <= 0 表示未压入表面，此时接触力为 0；只有压入表面时才输出正力。
        """
        if delta <= 0:
            return 0.0
        K, B = self.get_stiffness(x)
        return K * delta + B * delta_dot


class ForceSensorInput:
    """
    六维力传感器 ROS 输入。

    force_sensor_ros_node.py 发布 geometry_msgs/WrenchStamped，本类只负责订阅、
    新鲜度判定和按指定轴提取接触力。若 timeout 内没有新数据，控制层会回退
    到 Kelvin-Voigt 虚拟环境。
    """

    def __init__(self, topic="/force_sensor/wrench", timeout=0.2,
                 force_axis=2, force_sign=1.0):
        if force_axis not in (0, 1, 2):
            raise ValueError("force_axis must be 0, 1, or 2")
        self.topic = topic
        self.timeout = float(timeout)
        self.force_axis = int(force_axis)
        self.force_sign = float(force_sign)
        self._force = np.zeros(3)
        self._torque = np.zeros(3)
        self._stamp = None
        self._seq = 0
        self._sub = rospy.Subscriber(
            self.topic, WrenchStamped, self._callback, queue_size=1
        )

    def _callback(self, msg):
        force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ], dtype=float)
        torque = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ], dtype=float)
        if not (np.all(np.isfinite(force)) and np.all(np.isfinite(torque))):
            return
        self._force = force
        self._torque = torque
        self._stamp = rospy.Time.now()
        self._seq += 1

    def available(self):
        """判断最近一次传感器消息是否仍在 timeout 时间窗内。"""
        if self._stamp is None:
            return False
        age = (rospy.Time.now() - self._stamp).to_sec()
        return age <= self.timeout

    def age(self):
        if self._stamp is None:
            return float("inf")
        return (rospy.Time.now() - self._stamp).to_sec()

    def seq(self):
        return self._seq

    def wait_for_data(self, timeout=2.0):
        deadline = rospy.Time.now() + rospy.Duration(float(timeout))
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.available():
                return True
            rate.sleep()
        return self.available()

    def force_vector(self):
        return self._force.copy()

    def wrench_vector(self):
        return np.hstack([self._force, self._torque])

    def signed_axis_force(self):
        return self.force_sign * self._force[self.force_axis]

    def contact_force(self):
        """返回按本文 +z 接触反力约定使用的非负标量力。"""
        return abs(self.signed_axis_force())


class FirstOrderLowPass:
    """一阶低通滤波器，用于抑制力/刚度等测量噪声。"""

    def __init__(self, tau, initial=None):
        self.tau = float(tau)
        self._value = None if initial is None else np.asarray(initial, dtype=float)

    def reset(self, value=None):
        self._value = None if value is None else np.asarray(value, dtype=float)

    def update(self, value, dt):
        value = np.asarray(value, dtype=float)
        if self._value is None:
            self._value = value.copy()
            return self._value.copy()
        beta = float(np.clip(dt / max(self.tau, dt), 0.0, 1.0))
        self._value = self._value + beta * (value - self._value)
        return self._value.copy()


class VectorRateLimiter:
    """按元素限制向量每秒变化率，避免力矩命令跳变激发高频抖动。"""

    def __init__(self, max_rate, initial=None):
        self.max_rate = float(max_rate)
        self._value = None if initial is None else np.asarray(initial, dtype=float)

    def reset(self, value=None):
        self._value = None if value is None else np.asarray(value, dtype=float)

    def update(self, value, dt):
        value = np.asarray(value, dtype=float)
        if self._value is None:
            self._value = value.copy()
            return self._value.copy()
        max_step = self.max_rate * max(float(dt), 1e-6)
        step = np.clip(value - self._value, -max_step, max_step)
        self._value = self._value + step
        return self._value.copy()


class AlphaCommandLimiter:
    """平滑并限速 alpha 命令，避免仲裁权重逐采样跳变。"""

    def __init__(self, tau=0.20, max_rate=1.0, initial=None,
                 alpha_min=0.01, alpha_max=0.99):
        self.tau = float(max(tau, 0.0))
        self.max_rate = float(max(max_rate, 0.0))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self._filtered = None
        self._value = None
        if initial is not None:
            value = float(np.clip(initial, self.alpha_min, self.alpha_max))
            self._filtered = value
            self._value = value

    def reset(self, value=None):
        if value is None:
            self._filtered = None
            self._value = None
            return
        value = float(np.clip(value, self.alpha_min, self.alpha_max))
        self._filtered = value
        self._value = value

    def update(self, command, dt):
        command = float(np.clip(command, self.alpha_min, self.alpha_max))
        dt = max(float(dt), 1e-6)
        if self._filtered is None:
            self._filtered = command
        else:
            beta = float(np.clip(dt / max(self.tau, dt), 0.0, 1.0))
            self._filtered += beta * (command - self._filtered)

        if self._value is None:
            self._value = self._filtered
        elif self.max_rate > 0.0:
            max_step = self.max_rate * dt
            self._value += float(np.clip(
                self._filtered - self._value, -max_step, max_step
            ))
        else:
            self._value = self._filtered
        return float(np.clip(self._value, self.alpha_min, self.alpha_max))


class ContactDeltaDotEstimator:
    """由压入量差分估计接触速度，避免机器人状态速度尖峰污染虚拟力。

    `tool_position_velocity` 在 Gazebo 中偶发单帧符号翻转。虚拟接触力的
    Kelvin-Voigt 阻尼项直接使用该速度会产生假的力尖峰。这里以连续的
    `delta` 差分为主，经过一阶低通和限幅后作为控制/虚拟环境使用的
    `delta_dot`，同时返回 raw/fd 便于日志诊断。
    """

    def __init__(self, tau=0.05, limit=0.02, initial=0.0):
        self.tau = float(tau)
        self.limit = float(abs(limit))
        self._filter = FirstOrderLowPass(self.tau, initial=float(initial))
        self._prev_delta = None

    def reset(self, delta=None, value=0.0):
        self._prev_delta = None if delta is None else float(delta)
        self._filter.reset(float(value))

    def update(self, delta, dt, raw_delta_dot=None):
        delta = float(delta)
        dt = max(float(dt), 1e-6)
        raw = 0.0 if raw_delta_dot is None else float(raw_delta_dot)
        if self._prev_delta is None:
            fd = raw if np.isfinite(raw) else 0.0
        else:
            fd = (delta - self._prev_delta) / dt
        self._prev_delta = delta

        fd = 0.0 if not np.isfinite(fd) else fd
        filtered = float(self._filter.update(fd, dt))
        if self.limit > 0.0:
            filtered = float(np.clip(filtered, -self.limit, self.limit))
        return filtered, raw, fd


class DataLogger:
    """时序数据记录器。

    log(...) 接收 run_no_rcm 中的命名字段，并拆分向量字段为标量数组；
    save(...) 统一保存为 npz。字段命名尽量兼容旧实验脚本，便于复用画图代码。
    """

    def __init__(self):
        self._d = {
            't': [],
            # 当前 tool 位置 (3 轴)
            'pos_x': [], 'pos_y': [], 'pos_z': [],
            # 期望 tool 位置 (3 轴)
            'pos_des_x': [], 'pos_des_y': [], 'pos_des_z': [],
            # tool 位置误差 (3 轴 + 范数)
            'pos_err_x': [], 'pos_err_y': [], 'pos_err_z': [],
            'pos_err_norm': [],
            # 力 (标量)
            'F_measured': [], 'F_desired': [], 'F_err': [],
            'F_raw': [],
            # 六维力传感器/力源状态
            'Fx': [], 'Fy': [], 'Fz': [], 'Mx': [], 'My': [], 'Mz': [],
            'force_source': [], 'sensor_available': [],
            # 兼容旧字段
            'e_r': [], 'e_f': [],
            'sigma_f_norm': [], 'e_r1_norm': [],
            'alpha': [], 'K_hat': [],
            'K_total': [], 'K_r2': [], 'K_ef': [], 'K_sf': [],
            'x_desired': [],
            'error_rcm': [], 'error_track': [],
            'arbitration_strategy': [],
            'u_norm': [], 'phase': [],
            'K_hat_raw': [], 'force_blend': [],
            'z_min_safe': [],
            'K_hat_ctrl': [],
            'delta': [], 'delta_dot': [],
            'delta_dot_raw': [], 'delta_dot_fd': [],
            'K_env_true': [], 'B_env_true': [], 'B_hat': [],
            'contact_plane_z': [], 'scan_z_ref': [],
            # 关节姿态和末端姿态诊断
            'q1': [], 'q2': [], 'q3': [], 'q4': [], 'q5': [], 'q6': [], 'q7': [],
            'qd1': [], 'qd2': [], 'qd3': [], 'qd4': [], 'qd5': [], 'qd6': [], 'qd7': [],
            'q_err1': [], 'q_err2': [], 'q_err3': [], 'q_err4': [],
            'q_err5': [], 'q_err6': [], 'q_err7': [],
            'q_err_norm': [], 'q_err_inf': [], 'qd_norm': [],
            'tool_roll': [], 'tool_pitch': [], 'tool_yaw': [],
            'tool_roll_ref': [], 'tool_pitch_ref': [], 'tool_yaw_ref': [],
            'ori_err_x': [], 'ori_err_y': [], 'ori_err_z': [],
            'ori_err_norm': [], 'ori_err_deg': [],
            'tool_omega_x': [], 'tool_omega_y': [], 'tool_omega_z': [],
            'tool_omega_norm': [],
            'front_axis_err_deg': [],
        }

    def log(self, **kw):
        """记录一个控制周期的数据。

        未出现在 self._d 中的字段会被忽略，避免调用端新增临时字段时破坏保存。
        """
        for k, v in kw.items():
            if k == 'pos':
                self._d['pos_x'].append(v[0])
                self._d['pos_y'].append(v[1])
                self._d['pos_z'].append(v[2])
            elif k == 'pos_des':
                self._d['pos_des_x'].append(v[0])
                self._d['pos_des_y'].append(v[1])
                self._d['pos_des_z'].append(v[2])
            elif k == 'pos_err':
                self._d['pos_err_x'].append(v[0])
                self._d['pos_err_y'].append(v[1])
                self._d['pos_err_z'].append(v[2])
                self._d['pos_err_norm'].append(float(np.linalg.norm(v)))
            elif k == 'K_eff':
                self._d['K_total'].append(v[0])
                self._d['K_r2'].append(v[1])
                self._d['K_ef'].append(v[2])
                self._d['K_sf'].append(v[3])
            elif k == 'wrench':
                self._d['Fx'].append(v[0])
                self._d['Fy'].append(v[1])
                self._d['Fz'].append(v[2])
                self._d['Mx'].append(v[3])
                self._d['My'].append(v[4])
                self._d['Mz'].append(v[5])
            elif k == 'q':
                for i, value in enumerate(np.asarray(v).reshape(-1)[:7], start=1):
                    self._d[f'q{i}'].append(float(value))
            elif k == 'qd':
                for i, value in enumerate(np.asarray(v).reshape(-1)[:7], start=1):
                    self._d[f'qd{i}'].append(float(value))
            elif k == 'q_err':
                q_err = np.asarray(v, dtype=float).reshape(-1)[:7]
                for i, value in enumerate(q_err, start=1):
                    self._d[f'q_err{i}'].append(float(value))
                if q_err.size:
                    self._d['q_err_norm'].append(float(np.linalg.norm(q_err)))
                    self._d['q_err_inf'].append(float(np.max(np.abs(q_err))))
            elif k == 'tool_euler':
                self._d['tool_roll'].append(v[0])
                self._d['tool_pitch'].append(v[1])
                self._d['tool_yaw'].append(v[2])
            elif k == 'tool_euler_ref':
                self._d['tool_roll_ref'].append(v[0])
                self._d['tool_pitch_ref'].append(v[1])
                self._d['tool_yaw_ref'].append(v[2])
            elif k == 'ori_err_rotvec':
                self._d['ori_err_x'].append(v[0])
                self._d['ori_err_y'].append(v[1])
                self._d['ori_err_z'].append(v[2])
                norm = float(np.linalg.norm(v))
                self._d['ori_err_norm'].append(norm)
                self._d['ori_err_deg'].append(float(np.degrees(norm)))
            elif k == 'tool_omega':
                self._d['tool_omega_x'].append(v[0])
                self._d['tool_omega_y'].append(v[1])
                self._d['tool_omega_z'].append(v[2])
                self._d['tool_omega_norm'].append(float(np.linalg.norm(v)))
            elif k in self._d:
                self._d[k].append(v)

    @property
    def count(self):
        return len(self._d['t'])

    def save(self, filepath):
        np.savez(filepath, **{k: np.array(v) for k, v in self._d.items() if v})

    @staticmethod
    def load(filepath):
        return dict(np.load(filepath))
