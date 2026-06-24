"""
模糊仲裁用的一阶增量积分器与 Kalman 融合器。

当前推荐的 `online_priority` 不依赖 KalmanFilterFusion；本文件保留是为了
支持旧的 `coop_fuzzy`、`force_margin`、`continuous_force_margin` 策略。
这些策略会先通过模糊逻辑得到 lambda，再用本文件中的滤波器抑制跳变。
"""
import numpy as np


class DeltaLambdaUpdater:
    """将模糊规则输出的 Δλ 积分为 λ。

    这是一个很轻量的 Euler 积分器，主要用于旧策略的对照实验。
    """

    def __init__(self, lambda0=0.5, dt=0.01):
        self.lambda_current = lambda0  # 初始λ
        self.dt = dt  # 采样时间

    def update(self, delta_lambda):
        """欧拉法积分Δλ，更新λ并截断到[0,1]"""
        self.lambda_current += delta_lambda * self.dt
        self.lambda_current = max(0.0, min(1.0, self.lambda_current))
        return self.lambda_current


class KalmanFilterFusion:
    """二状态 Kalman 融合器。

    状态向量为 [λ, Δλ]^T。观测来自两路模糊输出:
      - z_k: lambda_based 规则给出的 λ；
      - tau_k: [由 λ 差分得到的 Δλ, delta_lambda_based 规则输出的 Δλ]。

    该滤波器不是当前 CAC2026 online_priority 的核心，只用于保留原策略。
    """

    def __init__(self, dt=0.01, epsilon=0.01, alpha=0.9):
        self.y_hat = np.array([[0.5], [0.0]])  # 状态向量[λ_w, Δλ_w]^T
        self.P = np.diag([0.001, 0.001])  # 状态协方差矩阵
        self.A = np.array([[1.0, dt], [0.0, 1.0]])  # 状态转移矩阵
        self.B = np.array(
            [[epsilon * dt, -epsilon * dt], [epsilon, -epsilon]]
        )  # 控制矩阵
        self.H = np.eye(2)  # 观测矩阵
        self.Q = 0.0004 * np.eye(2)  # 过程噪声协方差
        self.R = np.eye(2)  # 观测噪声协方差
        self.alpha = alpha  # 平滑系数
        self.lambda_prev = 0.5  # 上一时刻λ（用于平滑）
        # 用于更新R的历史数据
        self.history_z = []
        self.history_y_hat_prior = []
        self.N = 20  # 历史数据长度

    def predict(self, tau_k):
        """KF预测步骤"""
        self.y_hat_prior = self.A @ self.y_hat + self.B @ tau_k
        self.P_prior = self.A @ self.P @ self.A.T + self.Q
        return self.y_hat_prior

    def update_R(self, z_k):
        """最小二乘更新观测噪声R"""
        self.history_z.append(z_k)
        self.history_y_hat_prior.append(self.y_hat_prior[0, 0])
        if len(self.history_z) >= self.N:
            z_window = np.array(self.history_z[-self.N :])
            y_prior_window = np.array(self.history_y_hat_prior[-self.N :])
            residuals = z_window - y_prior_window
            self.R = np.array([[np.mean(residuals**2)]])

    def update(self, z_k, tau_k):
        """KF更新步骤，输出最终λ"""
        # 预测
        self.predict(tau_k)
        # 更新R
        self.update_R(z_k)
        # 计算卡尔曼增益
        S = self.H @ self.P_prior @ self.H.T + self.R
        self.K = self.P_prior @ self.H.T @ np.linalg.inv(S)
        # 后验估计
        innovation = z_k - self.H @ self.y_hat_prior
        self.y_hat = self.y_hat_prior + self.K @ innovation
        self.P = (np.eye(2) - self.K @ self.H) @ self.P_prior
        # 截断+平滑
        lambda_k = max(0.0, min(1.0, self.y_hat[0, 0]))
        lambda_smoothed = self.alpha * lambda_k + (1 - self.alpha) * self.lambda_prev
        self.lambda_prev = lambda_smoothed
        return lambda_smoothed
