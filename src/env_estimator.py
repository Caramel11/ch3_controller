"""
环境参数 RLS 在线估计器 (Kelvin-Voigt 模型)
===========================================

模型: F = K_e·δ + B_e·δ̇ = φᵀθ
      φ = [δ, δ̇]ᵀ,  θ = [K_e, B_e]ᵀ

带遗忘因子 λ_RLS 的递归最小二乘:
  k = Pφ / (λ_RLS + φᵀ P φ)
  θ ← θ + k·(F − φᵀθ)
  P ← (P − k·φᵀP) / λ_RLS

输出 EMA 低通滤波: K̂_filt ← K̂_filt + α_lp·(θ[0] − K̂_filt)
"""
import numpy as np


class EnvironmentEstimator:
    """鲁棒 Kelvin-Voigt 环境估计器。

    在扫描阶段，控制器主要需要可靠的 `K_hat` 做增益查表。Gazebo/真机
    恒力扫描时压入量通常只有数毫米，同时 `delta_dot` 来自机器人速度估计，
    二参数 RLS 很容易把速度噪声误解释成刚度变化。这里保留 bounded RLS
    作为阻尼估计辅助，但控制输出的 K_hat 使用准静态表观刚度 F/delta
    的 EMA 低通值。
    """

    def __init__(self, forgetting_factor=0.995, P0=1e4,
                 theta_init=None, alpha_lp=0.05,
                 min_delta=2e-4, K_bounds=(50.0, 5000.0),
                 B_bounds=(0.1, 100.0), delta_dot_limit=0.02,
                 min_delta_dot_for_B=5e-4):
        self.lam = float(forgetting_factor)
        self.P0 = float(P0)
        self.alpha_lp = float(alpha_lp)
        self.min_delta = float(min_delta)
        self.K_min, self.K_max = map(float, K_bounds)
        self.B_min, self.B_max = map(float, B_bounds)
        self.delta_dot_limit = float(delta_dot_limit)
        self.min_delta_dot_for_B = float(min_delta_dot_for_B)

        # theta = [K_e, B_e]，默认从中等偏软环境开始估计。
        if theta_init is None:
            theta_init = [200.0, 5.0]
        self._theta_init = np.asarray(theta_init, dtype=float)

        self.theta = self._theta_init.copy()
        self.P = self.P0 * np.eye(2)
        self._K_filt = float(self._theta_init[0])
        self._prev_K = self._K_filt
        self._K_observed = self._K_filt

    def update(self, F_meas, delta, delta_dot):
        """
        更新估计

        非接触保护: F_meas < 0.3N 或 δ 太小时跳过更新。

        Returns
        -------
        K_e_filt : scalar   低通滤波后的表观刚度估计
        B_e      : scalar   有界粘滞阻尼估计
        """
        self._prev_K = self._K_filt

        # 非接触保护: 未接触时 phi 信息不足，强行更新会污染刚度估计。
        if abs(F_meas) < 0.3 or abs(delta) < self.min_delta:
            return self._K_filt, float(np.clip(self.theta[1], self.B_min, self.B_max))

        # 控制器真正敏感的是 K_hat。恒力扫描中的 delta_dot 噪声会让二参数
        # RLS 病态，因此 K_hat 采用准静态表观刚度作为主估计。
        self._K_observed = float(np.clip(abs(F_meas) / abs(delta),
                                         self.K_min, self.K_max))
        self._K_filt += self.alpha_lp * (self._K_observed - self._K_filt)
        self._K_filt = float(np.clip(self._K_filt, self.K_min, self.K_max))

        # B_hat 仍用有界 RLS 辅助估计，但速度回归量先限幅，避免单个
        # 速度尖峰把 theta 拉飞。K 输出不直接采用 RLS 的 theta[0]。
        delta_dot_rls = float(np.clip(delta_dot,
                                      -self.delta_dot_limit,
                                      self.delta_dot_limit))
        if abs(delta_dot_rls) < self.min_delta_dot_for_B:
            return self._K_filt, float(np.clip(self.theta[1],
                                               self.B_min, self.B_max))
        phi = np.array([delta, delta_dot_rls])
        Pp = self.P @ phi
        gain = Pp / (self.lam + phi @ Pp)
        err = F_meas - phi @ self.theta
        self.theta = self.theta + gain * err
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.lam

        self.theta[0] = np.clip(self.theta[0], self.K_min, self.K_max)
        self.theta[1] = np.clip(self.theta[1], self.B_min, self.B_max)

        return self._K_filt, self.theta[1]

    @property
    def K_e(self):
        return self._K_filt

    @property
    def B_e(self):
        return float(np.clip(self.theta[1], self.B_min, self.B_max))

    @property
    def K_observed(self):
        """最近一次 F/delta 表观刚度观测。"""
        return self._K_observed

    @property
    def dK_e(self):
        """刚度变化率 (per step)"""
        return self._K_filt - self._prev_K

    def reset(self):
        self.theta = self._theta_init.copy()
        self.P = self.P0 * np.eye(2)
        self._K_filt = float(self._theta_init[0])
        self._prev_K = self._K_filt
        self._K_observed = self._K_filt
