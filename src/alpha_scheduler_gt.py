"""
力-位仲裁 α 调度器 — 阶段感知 + 模糊逻辑 + KF 融合
=====================================================

三层框架:
  α = w_φ · α_φ + (1 − w_φ) · α_fuzzy

  - 阶段先验 (α_φ, w_φ): 处理模糊逻辑无法覆盖的边界情况
  - 模糊推理 + KF: 在线自适应调节 α_fuzzy
  - 输入映射: 力-位物理量 → 模糊逻辑原始论域

输入映射:
  |e_f|  (力误差)      → F_h (0-2,    比例 0.4)
  K̂_e   (环境刚度)    → T_h (0-6,    比例 0.6/K_ref)
  |e_r|  (位置误差)    → D_r (0-0.1,  反向映射)  [关键]
"""
import numpy as np
from enum import Enum

from fuzzy_logic import FuzzyLogicTool
from kalman_filter import KalmanFilterFusion, DeltaLambdaUpdater


# ================================================================
# 任务阶段
# ================================================================
class TaskPhase(Enum):
    FREE_SPACE = 0
    APPROACHING = 1
    CONTACT_TRANSIENT = 2
    CONTACT_STEADY = 3
    RETREAT = 4


# 阶段参数: (α_φ, w_φ)
PHASE_PARAMS = {
    TaskPhase.FREE_SPACE:        (1.0,  1.0),
    TaskPhase.APPROACHING:       (0.85, 0.6),
    TaskPhase.CONTACT_TRANSIENT: (0.25, 0.5),
    TaskPhase.CONTACT_STEADY:    (0.50, 0.0),
    TaskPhase.RETREAT:           (1.0,  1.0),
}


FORCE_MARGIN_PHASE_PARAMS = {
    TaskPhase.FREE_SPACE:        (1.0,  1.0),
    TaskPhase.APPROACHING:       (0.85, 0.6),
    TaskPhase.CONTACT_TRANSIENT: (0.65, 0.25),
    TaskPhase.CONTACT_STEADY:    (0.50, 0.0),
    TaskPhase.RETREAT:           (1.0,  1.0),
}


class PhaseDetector:
    """
    任务阶段自动检测

    规则:
      |F| < F_thresh AND v_z ≥ 0              → FREE_SPACE
      |F| < F_thresh AND v_z < 0              → APPROACHING
      |F| ≥ F_thresh AND t_contact < T_trans  → CONTACT_TRANSIENT
      |F| ≥ F_thresh AND t_contact ≥ T_trans  → CONTACT_STEADY
      外部调用 set_retreat(True)               → RETREAT
    """

    def __init__(self, F_thresh=0.3, T_transient=1.0, dt=0.01):
        self.F_thresh = F_thresh
        self.T_transient = T_transient
        self.dt = dt
        self._phase = TaskPhase.FREE_SPACE
        self._t_contact = 0.0
        self._is_retreat = False

    def update(self, F_norm, z_vel):
        if self._is_retreat:
            self._phase = TaskPhase.RETREAT
            return self._phase

        if F_norm >= self.F_thresh:
            self._t_contact += self.dt
            if self._t_contact < self.T_transient:
                self._phase = TaskPhase.CONTACT_TRANSIENT
            else:
                self._phase = TaskPhase.CONTACT_STEADY
        else:
            self._t_contact = 0.0
            if z_vel < -0.001:
                self._phase = TaskPhase.APPROACHING
            else:
                self._phase = TaskPhase.FREE_SPACE
        return self._phase

    def set_retreat(self, val=True):
        self._is_retreat = val

    def reset(self):
        self._phase = TaskPhase.FREE_SPACE
        self._t_contact = 0.0
        self._is_retreat = False

    @property
    def phase(self):
        return self._phase


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class ContinuousPhasePrior:
    """
    连续阶段先验:
      (F_norm, z_vel, t_contact) -> (alpha_phi, w_phi)

    保留 FREE/APPROACH/TRANSIENT/STEADY 的语义，但用连续门控
    替代阶段表查表，从而减少阶段切换造成的 alpha 跳变。
    """

    def __init__(self, F_thresh=0.3, F_band=0.05,
                 T_transient=1.0, vz_thresh=0.001, vz_band=0.001,
                 alpha_free=1.0, alpha_approach=0.85,
                 alpha_transient=0.75, alpha_steady=0.50,
                 w_free=1.0, w_approach=0.60,
                 w_transient=0.35, w_steady=0.0):
        self.F_thresh = float(F_thresh)
        self.F_band = float(max(F_band, 1e-6))
        self.T_transient = float(max(T_transient, 1e-6))
        self.vz_thresh = float(vz_thresh)
        self.vz_band = float(max(vz_band, 1e-6))
        self.alpha_free = float(alpha_free)
        self.alpha_approach = float(alpha_approach)
        self.alpha_transient = float(alpha_transient)
        self.alpha_steady = float(alpha_steady)
        self.w_free = float(w_free)
        self.w_approach = float(w_approach)
        self.w_transient = float(w_transient)
        self.w_steady = float(w_steady)
        self.t_contact = 0.0
        self._is_retreat = False

    def update(self, F_norm, z_vel, dt):
        if self._is_retreat:
            return 1.0, 1.0

        if F_norm >= self.F_thresh:
            self.t_contact += float(dt)
        else:
            self.t_contact = 0.0

        c_F = float(_sigmoid((F_norm - self.F_thresh) / self.F_band))
        c_t = float(_smoothstep(self.t_contact / self.T_transient))
        a_v = float(_sigmoid((-z_vel - self.vz_thresh) / self.vz_band))

        alpha_pre = (
            (1.0 - a_v) * self.alpha_free
            + a_v * self.alpha_approach
        )
        w_pre = (
            (1.0 - a_v) * self.w_free
            + a_v * self.w_approach
        )
        alpha_contact = (
            (1.0 - c_t) * self.alpha_transient
            + c_t * self.alpha_steady
        )
        w_contact = (
            (1.0 - c_t) * self.w_transient
            + c_t * self.w_steady
        )

        alpha_phi = (1.0 - c_F) * alpha_pre + c_F * alpha_contact
        w_phi = (1.0 - c_F) * w_pre + c_F * w_contact
        return (
            float(np.clip(alpha_phi, 0.0, 1.0)),
            float(np.clip(w_phi, 0.0, 1.0)),
        )

    def set_retreat(self, val=True):
        self._is_retreat = bool(val)

    def reset(self):
        self.t_contact = 0.0
        self._is_retreat = False


# ================================================================
# α 调度器
# ================================================================
class PhaseAwareFuzzyAlphaScheduler:
    """
    完整的 α 计算管线 (阶段感知 + 模糊 + KF)
    """

    def __init__(self, dt=0.01, K_ref=1000.0, phase_params=None):
        self.dt = dt
        self.K_ref = K_ref
        self.name = "phase_fuzzy_KF"

        self.phase_detector = PhaseDetector(dt=dt)
        self.phase_params = phase_params or PHASE_PARAMS

        self.lambda_fuzzy = FuzzyLogicTool(type="lambda_based")
        self.delta_lambda_fuzzy = FuzzyLogicTool(type="delta_lambda_based")

        self.kf_fusion = KalmanFilterFusion(dt=dt, epsilon=0.01)
        self.delta_updater = DeltaLambdaUpdater(lambda0=0.5, dt=dt)

        self.lambda_list = [0.5]
        self.alpha_history = []
        self.phase_history = []

        # 输入映射比例
        self.e_f_scale = 2.0 / 5.0          # |e_f| [0,5N] → F_h [0,2]
        self.K_scale = 6.0 / 10.0           # K̂/K_ref [0,10] → T_h [0,6]
        self.e_r_scale = 0.1 / 0.005        # e_r [0,5mm] → D_r [0,0.1]

    def _map_inputs(self, e_f, K_hat, e_r):
        F_h = np.clip(abs(e_f) * self.e_f_scale, 0.0, 2.0)
        T_h = np.clip(K_hat / self.K_ref * self.K_scale, 0.0, 6.0)
        # 反向映射: e_r 大 → D_r 小 → λ 小 → α 大 (位控接管)
        D_r = np.clip(0.1 - abs(e_r) * self.e_r_scale, 0.0, 0.1)
        return F_h, T_h, D_r

    def _map_delta_inputs(self, de_f, dK, de_r):
        dF = np.clip(de_f * 0.5, -3.0, 3.0)
        dT = np.clip(dK / self.K_ref + 1.0, 0.0, 2.0)
        dD = np.clip(-de_r * 20.0, -0.08, 0.08)
        return dF, dT, dD

    def compute(self, F_norm, e_f, K_hat, e_r, z_vel,
                de_f=0.0, dK=0.0, de_r=0.0, **unused):
        """
        计算当前 α

        Parameters
        ----------
        F_norm : float   接触力范数 (N), 用于阶段检测
        e_f    : float   力误差 (N)
        K_hat  : float   估计环境刚度 (N/m)
        e_r    : float   位置误差范数 (m)
        z_vel  : float   z 方向速度 (m/s)
        de_f, dK, de_r : 各量导数

        Returns
        -------
        alpha : float ∈ [0, 1]
        """
        # 1. 阶段检测
        phase = self.phase_detector.update(F_norm, z_vel)
        alpha_phi, w_phi = self.phase_params[phase]

        # 2. 模糊 → KF → α_fuzzy
        F_h, T_h, D_r = self._map_inputs(e_f, K_hat, e_r)
        dF, dT, dD = self._map_delta_inputs(de_f, dK, de_r)

        lambda_w = self.lambda_fuzzy.compute([F_h, T_h, D_r], "lambda_based")
        dlam_w = self.delta_lambda_fuzzy.compute([dF, dT, dD], "delta_lambda_based")
        lambda_delta = self.delta_updater.update(dlam_w)

        dl_w = (lambda_w - self.lambda_list[-1]) / self.dt
        tau_k = np.array([[dl_w], [lambda_delta]])
        lambda_kf = self.kf_fusion.update(lambda_w, tau_k)

        self.lambda_list.append(lambda_w)

        alpha_fuzzy = 1.0 - np.clip(lambda_kf, 0.0, 1.0)

        # 3. 阶段融合
        alpha = w_phi * alpha_phi + (1 - w_phi) * alpha_fuzzy
        alpha = float(np.clip(alpha, 0.01, 0.99))

        self.alpha_history.append(alpha)
        self.phase_history.append(phase.value)
        return alpha

    def set_retreat(self, val=True):
        self.phase_detector.set_retreat(val)

    def reset(self):
        self.lambda_list = [0.5]
        self.alpha_history = []
        self.phase_history = []
        self.phase_detector.reset()
        self.kf_fusion = KalmanFilterFusion(dt=self.dt, epsilon=0.01)
        self.delta_updater = DeltaLambdaUpdater(lambda0=0.5, dt=self.dt)


class ForceMarginFuzzyAlphaScheduler:
    """
    基于交互力上下界安全裕度的 α 仲裁器。

    输入保持与旧调度器兼容，额外接收 F_desired/F_min/F_max。核心变量:
      F_h   = 2|F-Fd|/(Fmax-Fmin)
      S_h   = 6ρ_F, ρ_F 为到上下界的归一化安全裕度
      D_r   = 0.1 - |e_r| * 0.1/0.005
      s_F   = (F-Fd)/((Fmax-Fmin)/2)

    模糊表直接输出 alpha_fuzzy，再按 s_F 做上下界方向安全修正。
    """

    def __init__(self, dt=0.01, F_min=0.2, F_max=1.0,
                 F_desired=0.5, phase_params=None,
                 k_upper=0.45, k_lower=0.25,
                 alpha_min=0.05, alpha_max=0.95,
                 upper_guard_alpha=0.85, lower_guard_alpha=0.25,
                 smooth_tau=0.04):
        self.dt = dt
        self.F_min = float(F_min)
        self.F_max = float(F_max)
        self.F_desired = float(F_desired)
        self.phase_params = phase_params or FORCE_MARGIN_PHASE_PARAMS
        self.phase_detector = PhaseDetector(dt=dt)
        self.name = "force_margin_alpha"

        self.k_upper = float(k_upper)
        self.k_lower = float(k_lower)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.upper_guard_alpha = float(upper_guard_alpha)
        self.lower_guard_alpha = float(lower_guard_alpha)
        self.smooth_beta = float(np.clip(dt / max(smooth_tau, dt), 0.0, 1.0))

        self.alpha_history = []
        self.phase_history = []
        self.rho_history = []
        self.margin_history = []
        self._alpha_filt = 0.5

        self.input_sets = [
            {"PS": (0.0, 0.4, 0.8), "PM": (0.4, 1.0, 1.5), "PL": (1.2, 1.5, 2.0)},
            {"PS": (0.0, 1.0, 2.0), "PM": (1.0, 3.0, 5.0), "PL": (3.0, 5.0, 6.0)},
            {"PS": (0.0, 0.02, 0.04), "PM": (0.03, 0.05, 0.07), "PL": (0.06, 0.08, 0.1)},
        ]
        self.output_sets = {
            "Z": (0.0, 0.05, 0.1),
            "PS": (0.05, 0.25, 0.5),
            "PM": (0.25, 0.5, 0.7),
            "P": (0.5, 0.7, 0.95),
            "PL": (0.9, 0.95, 1.0),
        }
        self.rule_dict = {
            ("PS", "PS", "PS"): "PL", ("PS", "PS", "PM"): "P",  ("PS", "PS", "PL"): "PM",
            ("PS", "PM", "PS"): "PL", ("PS", "PM", "PM"): "P",  ("PS", "PM", "PL"): "PM",
            ("PS", "PL", "PS"): "P",  ("PS", "PL", "PM"): "PM", ("PS", "PL", "PL"): "PM",
            ("PM", "PS", "PS"): "PL", ("PM", "PS", "PM"): "P",  ("PM", "PS", "PL"): "PS",
            ("PM", "PM", "PS"): "P",  ("PM", "PM", "PM"): "PM", ("PM", "PM", "PL"): "PS",
            ("PM", "PL", "PS"): "P",  ("PM", "PL", "PM"): "PM", ("PM", "PL", "PL"): "PS",
            ("PL", "PS", "PS"): "PL", ("PL", "PS", "PM"): "PM", ("PL", "PS", "PL"): "Z",
            ("PL", "PM", "PS"): "P",  ("PL", "PM", "PM"): "PS", ("PL", "PM", "PL"): "Z",
            ("PL", "PL", "PS"): "P",  ("PL", "PL", "PM"): "PS", ("PL", "PL", "PL"): "Z",
        }

    @staticmethod
    def _triangular_mf(x, params):
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return (x - a) / max(b - a, 1e-12)
        return (c - x) / max(c - b, 1e-12)

    @staticmethod
    def _left_shoulder_mf(x, params):
        a, b, c = params
        if x <= b:
            return 1.0 if x >= a else 0.0
        if x >= c:
            return 0.0
        return (c - x) / max(c - b, 1e-12)

    @staticmethod
    def _right_shoulder_mf(x, params):
        a, b, c = params
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0 if x <= c else 0.0
        return (x - a) / max(b - a, 1e-12)

    def _membership(self, x, label, params):
        if label in ("PS", "Z"):
            return self._left_shoulder_mf(x, params)
        if label == "PL":
            return self._right_shoulder_mf(x, params)
        return self._triangular_mf(x, params)

    def _fuzzify(self, F_h, S_h, D_r):
        inputs = [
            np.clip(F_h, 0.0, 2.0),
            np.clip(S_h, 0.0, 6.0),
            np.clip(D_r, 0.0, 0.1),
        ]
        fuzzified = []
        for value, sets in zip(inputs, self.input_sets):
            fuzzified.append({
                label: self._membership(value, label, params)
                for label, params in sets.items()
            })
        return fuzzified

    def _infer(self, fuzzified):
        output_membership = {}
        for (f_label, s_label, d_label), out_label in self.rule_dict.items():
            activation = min(
                fuzzified[0][f_label],
                fuzzified[1][s_label],
                fuzzified[2][d_label],
            )
            output_membership[out_label] = max(
                activation, output_membership.get(out_label, 0.0)
            )
        return output_membership

    def _defuzzify_alpha(self, output_membership):
        z_range = np.linspace(0.0, 1.0, 101)
        numerator = 0.0
        denominator = 0.0
        for z in z_range:
            mu_total = 0.0
            for label, mu in output_membership.items():
                mu_z = self._membership(z, label, self.output_sets[label])
                mu_total = max(mu_total, min(mu, mu_z))
            numerator += mu_total * z
            denominator += mu_total
        if denominator < 1e-9:
            return 0.5
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def _force_margin_inputs(self, F_norm, F_desired, F_min, F_max, e_r):
        width = max(F_max - F_min, 1e-6)
        half_width = max(0.5 * width, 1e-6)
        F_err = F_norm - F_desired
        d_lower = F_norm - F_min
        d_upper = F_max - F_norm
        rho_F = float(np.clip(min(d_lower, d_upper) / half_width, 0.0, 1.0))
        s_F = float(np.clip(F_err / half_width, -1.0, 1.0))
        F_h = float(np.clip(abs(F_err) / width * 2.0, 0.0, 2.0))
        S_h = 6.0 * rho_F
        D_r = float(np.clip(0.1 - abs(e_r) * (0.1 / 0.005), 0.0, 0.1))
        return F_h, S_h, D_r, rho_F, s_F, F_err

    def _compute_phase_prior(self, phase, F_norm, z_vel):
        return self.phase_params[phase]

    def compute(self, F_norm, e_f=0.0, K_hat=None, e_r=0.0, z_vel=0.0,
                de_f=0.0, dK=0.0, de_r=0.0, F_desired=None,
                F_min=None, F_max=None, tracking_boost_enabled=True, **unused):
        F_desired = self.F_desired if F_desired is None else float(F_desired)
        F_min = self.F_min if F_min is None else float(F_min)
        F_max = self.F_max if F_max is None else float(F_max)
        if F_max <= F_min:
            F_max = F_min + 1e-6

        phase = self.phase_detector.update(F_norm, z_vel)
        alpha_phi, w_phi = self._compute_phase_prior(phase, F_norm, z_vel)

        F_h, S_h, D_r, rho_F, s_F, F_err = self._force_margin_inputs(
            F_norm, F_desired, F_min, F_max, e_r
        )
        alpha_0 = self._defuzzify_alpha(self._infer(self._fuzzify(F_h, S_h, D_r)))

        r_F = 1.0 - rho_F
        alpha_safe = (
            alpha_0
            + self.k_upper * r_F * max(s_F, 0.0)
            - self.k_lower * r_F * max(-s_F, 0.0)
        )
        if F_norm >= F_max:
            alpha_safe = max(alpha_safe, self.upper_guard_alpha)
        elif F_norm <= F_min:
            alpha_safe = min(alpha_safe, self.lower_guard_alpha)
        alpha_safe = float(np.clip(alpha_safe, self.alpha_min, self.alpha_max))

        alpha_raw = w_phi * alpha_phi + (1.0 - w_phi) * alpha_safe
        alpha_raw = float(np.clip(alpha_raw, 0.01, 0.99))
        alpha = self._alpha_filt + self.smooth_beta * (alpha_raw - self._alpha_filt)
        if F_norm >= F_max:
            alpha = max(alpha, self.upper_guard_alpha)
        elif F_norm <= F_min:
            alpha = min(alpha, self.lower_guard_alpha)
        alpha = float(np.clip(alpha, 0.01, 0.99))
        self._alpha_filt = alpha

        self.alpha_history.append(alpha)
        self.phase_history.append(phase.value)
        self.rho_history.append(rho_F)
        self.margin_history.append((F_h, S_h, D_r, s_F, F_err))
        return alpha

    def set_retreat(self, val=True):
        self.phase_detector.set_retreat(val)

    def reset(self):
        self.alpha_history = []
        self.phase_history = []
        self.rho_history = []
        self.margin_history = []
        self._alpha_filt = 0.5
        self.phase_detector.reset()


class ContinuousForceMarginFuzzyAlphaScheduler(ForceMarginFuzzyAlphaScheduler):
    """
    Force-margin 模糊安全仲裁 + 连续阶段先验。

    继承 ForceMarginFuzzyAlphaScheduler 的模糊推理、上下界安全修正和
    一阶平滑，并将离散 phase_params 查表替换为 ContinuousPhasePrior。
    对 no-RCM 扫描而言，力误差和边界裕度安全时应优先修正几毫米量级的
    横向轨迹滞后，因此额外加入 safe-tracking boost:
      - 力处于安全裕度内且 |F-Fd| 很小时，提高 alpha 下限；
      - 高力风险提高位置权重以抑制振荡，低力风险才小幅降低 alpha。
    """

    def __init__(self, dt=0.01, F_min=0.2, F_max=1.0,
                 F_desired=0.5, phase_prior=None,
                 safe_tracking_alpha=0.84,
                 safe_tracking_extra=0.05,
                 safe_margin_start=0.45,
                 safe_margin_full=0.75,
                 force_error_start=0.04,
                 force_error_full=0.12,
                 force_error_hard=0.35,
                 track_error_start=0.001,
                 track_error_full=0.003,
                 force_balance_alpha=0.84,
                 force_guard_alpha=0.92,
                 low_force_guard_alpha=0.72,
                 risk_margin_start=0.35,
                 risk_margin_full=0.10,
                 phase_contact_threshold=0.3,
                 stiffness_alpha_enabled=False,
                 stiffness_low_threshold=250.0,
                 stiffness_high_threshold=1000.0,
                 stiffness_low_alpha=0.45,
                 stiffness_high_alpha=0.90,
                 stiffness_blend=0.0,
                 **kwargs):
        kwargs.setdefault("smooth_tau", 0.12)
        super().__init__(
            dt=dt,
            F_min=F_min,
            F_max=F_max,
            F_desired=F_desired,
            **kwargs
        )
        self.name = "continuous_force_margin_alpha"
        self.phase_prior = phase_prior or ContinuousPhasePrior(
            F_thresh=min(F_min, phase_contact_threshold),
            T_transient=1.0,
        )
        self.phase_prior_history = []
        self.safe_tracking_alpha = float(safe_tracking_alpha)
        self.safe_tracking_extra = float(safe_tracking_extra)
        self.safe_margin_start = float(safe_margin_start)
        self.safe_margin_full = float(max(safe_margin_full, safe_margin_start + 1e-6))
        self.force_error_start = float(force_error_start)
        self.force_error_full = float(max(force_error_full, force_error_start + 1e-6))
        self.force_error_hard = float(max(force_error_hard, self.force_error_full + 1e-6))
        self.track_error_start = float(track_error_start)
        self.track_error_full = float(max(track_error_full, track_error_start + 1e-6))
        self.force_balance_alpha = float(force_balance_alpha)
        self.force_guard_alpha = float(force_guard_alpha)
        self.low_force_guard_alpha = float(low_force_guard_alpha)
        self.risk_margin_start = float(risk_margin_start)
        self.risk_margin_full = float(min(risk_margin_full, risk_margin_start - 1e-6))
        self.phase_contact_threshold = float(phase_contact_threshold)
        self.stiffness_alpha_enabled = bool(stiffness_alpha_enabled)
        self.stiffness_low_threshold = float(stiffness_low_threshold)
        self.stiffness_high_threshold = float(max(
            stiffness_high_threshold,
            stiffness_low_threshold + 1e-6,
        ))
        self.stiffness_low_alpha = float(stiffness_low_alpha)
        self.stiffness_high_alpha = float(stiffness_high_alpha)
        self.stiffness_blend = float(np.clip(stiffness_blend, 0.0, 1.0))
        self._alpha_filt = self.safe_tracking_alpha

    def _compute_phase_prior(self, phase, F_norm, z_vel):
        alpha_phi, w_phi = self.phase_prior.update(F_norm, z_vel, self.dt)
        self.phase_prior_history.append((alpha_phi, w_phi))
        return alpha_phi, w_phi

    def _safe_tracking_boost(self, alpha_safe, rho_F, F_err, e_r):
        margin_gate = _smoothstep(
            (rho_F - self.safe_margin_start)
            / (self.safe_margin_full - self.safe_margin_start)
        )
        force_gate = 1.0 - _smoothstep(
            (abs(F_err) - self.force_error_start)
            / (self.force_error_full - self.force_error_start)
        )
        track_gate = _smoothstep(
            (abs(e_r) - self.track_error_start)
            / (self.track_error_full - self.track_error_start)
        )

        gate = margin_gate * force_gate * track_gate
        if gate <= 1e-9:
            return alpha_safe

        alpha_track = self.safe_tracking_alpha + self.safe_tracking_extra * track_gate
        alpha_track = float(np.clip(alpha_track, self.alpha_min, self.alpha_max))
        return float(alpha_safe + gate * max(0.0, alpha_track - alpha_safe))

    def _force_risk_rebalance(self, alpha_safe, rho_F, F_err):
        """按力边界方向做非对称 alpha 修正。

        no-RCM 固定 z 扫描中，高力峰值通常来自 tool 偏离参考平面和横向振荡。
        因此上边界风险升高时不降低 alpha，而是平滑提高位置权重；低力风险时
        才小幅降低 alpha，给力控留出补偿空间。
        """
        margin_risk = 1.0 - _smoothstep(
            (rho_F - self.risk_margin_full)
            / (self.risk_margin_start - self.risk_margin_full)
        )
        pos_err_gate = _smoothstep(
            (F_err - self.force_error_start)
            / (self.force_error_hard - self.force_error_start)
        )
        neg_err_gate = _smoothstep(
            (-F_err - self.force_error_start)
            / (self.force_error_hard - self.force_error_start)
        )
        high_gate = max(pos_err_gate, margin_risk if F_err > 0.0 else 0.0)
        low_gate = max(neg_err_gate, margin_risk if F_err < 0.0 else 0.0)

        alpha = float(alpha_safe)
        if high_gate > 1e-9:
            alpha_high = (
                (1.0 - high_gate) * self.force_balance_alpha
                + high_gate * self.force_guard_alpha
            )
            alpha_target = max(alpha, alpha_high)
            alpha = (1.0 - high_gate) * alpha + high_gate * alpha_target
        if low_gate > 1e-9:
            alpha_low = (
                (1.0 - low_gate) * self.force_balance_alpha
                + low_gate * self.low_force_guard_alpha
            )
            alpha = (1.0 - low_gate) * alpha + low_gate * alpha_low
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    def _stiffness_rebalance(self, alpha_safe, K_hat, F_err=0.0):
        """根据估计刚度和力误差方向平滑调节 z 向 alpha 目标。

        alpha 只作用在 z/压入方向时，高刚度区不能无条件提高 alpha。
        若高刚度区实际力低于期望，继续提高 z 位置权重会把压入深度锁在
        参考附近，反而扩大力误差；此时应降低 alpha 给力通道补偿空间。
        只有高刚度且力偏高时，才提高 alpha 抑制过压和振荡。
        """
        if (
            not self.stiffness_alpha_enabled
            or self.stiffness_blend <= 1e-9
            or K_hat is None
            or not np.isfinite(K_hat)
        ):
            return alpha_safe

        k_gate = _smoothstep(
            (float(K_hat) - self.stiffness_low_threshold)
            / (self.stiffness_high_threshold - self.stiffness_low_threshold)
        )
        if k_gate <= 1e-9:
            return alpha_safe

        low_force_gate = _smoothstep(
            (-float(F_err) - self.force_error_start)
            / (self.force_error_full - self.force_error_start)
        )
        high_force_gate = _smoothstep(
            (float(F_err) - self.force_error_start)
            / (self.force_error_full - self.force_error_start)
        )

        alpha = float(alpha_safe)
        low_blend = self.stiffness_blend * k_gate * low_force_gate
        if low_blend > 1e-9:
            alpha = (1.0 - low_blend) * alpha + low_blend * self.stiffness_low_alpha

        high_blend = self.stiffness_blend * k_gate * high_force_gate
        if high_blend > 1e-9:
            alpha = (1.0 - high_blend) * alpha + high_blend * self.stiffness_high_alpha

        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    def compute(self, F_norm, e_f=0.0, K_hat=None, e_r=0.0, z_vel=0.0,
                de_f=0.0, dK=0.0, de_r=0.0, F_desired=None,
                F_min=None, F_max=None, tracking_boost_enabled=True, **unused):
        F_desired = self.F_desired if F_desired is None else float(F_desired)
        F_min = self.F_min if F_min is None else float(F_min)
        F_max = self.F_max if F_max is None else float(F_max)
        if F_max <= F_min:
            F_max = F_min + 1e-6

        phase = self.phase_detector.update(F_norm, z_vel)
        alpha_phi, w_phi = self._compute_phase_prior(phase, F_norm, z_vel)

        F_h, S_h, D_r, rho_F, s_F, F_err = self._force_margin_inputs(
            F_norm, F_desired, F_min, F_max, e_r
        )
        alpha_0 = self._defuzzify_alpha(self._infer(self._fuzzify(F_h, S_h, D_r)))

        r_F = 1.0 - rho_F
        alpha_safe = (
            alpha_0
            + self.k_upper * r_F * max(s_F, 0.0)
            - self.k_lower * r_F * max(-s_F, 0.0)
        )
        if F_norm >= F_max or F_norm <= F_min:
            alpha_safe = self._force_risk_rebalance(
                alpha_safe, rho_F=rho_F, F_err=F_err
            )
        elif tracking_boost_enabled:
            alpha_safe = self._safe_tracking_boost(
                alpha_safe, rho_F=rho_F, F_err=F_err, e_r=e_r
            )
            alpha_safe = self._force_risk_rebalance(
                alpha_safe, rho_F=rho_F, F_err=F_err
            )
        alpha_safe = self._stiffness_rebalance(alpha_safe, K_hat, F_err=F_err)
        alpha_safe = float(np.clip(alpha_safe, self.alpha_min, self.alpha_max))

        alpha_raw = w_phi * alpha_phi + (1.0 - w_phi) * alpha_safe
        alpha_raw = float(np.clip(alpha_raw, 0.01, 0.99))
        alpha = self._alpha_filt + self.smooth_beta * (alpha_raw - self._alpha_filt)
        alpha = float(np.clip(alpha, 0.01, 0.99))
        self._alpha_filt = alpha

        self.alpha_history.append(alpha)
        self.phase_history.append(phase.value)
        self.rho_history.append(rho_F)
        self.margin_history.append((F_h, S_h, D_r, s_F, F_err))
        return alpha

    def set_retreat(self, val=True):
        super().set_retreat(val)
        self.phase_prior.set_retreat(val)

    def reset(self):
        super().reset()
        self.phase_prior.reset()
        self.phase_prior_history = []
        self._alpha_filt = self.safe_tracking_alpha


class OnlinePriorityAdaptationAlphaScheduler:
    """
    CAC2026 Sec. III-C 在线力/位置优先级自适应仲裁器。

    论文约定与本代码一致:
      alpha 越大 → 越偏位置跟踪；
      alpha 越小 → 越偏力调节。

    r_e = ||S_f e_f|| / eps_f
    r_b = max(0, (y_f-f_U)/(f_max-f_U), (f_L-y_f)/(f_L-f_min))
    rho_f = max(r_e, r_b)
    alpha_r = alpha_min + (alpha_max-alpha_min) exp(-kappa rho_f^2)
    alpha_dot = lambda_alpha (alpha_r-alpha)

    与旧模糊调度器相比，本类不依赖 fuzzy table，而是直接把“力误差”
    和“越界风险”合成为 rho_f。这样更贴合论文 C 节的在线优先级更新，
    同时保留 compute/reset/set_retreat/phase_detector 接口，方便 run_no_rcm
    以最小改动切换策略。
    """

    def __init__(self, dt=0.01, F_desired=1.0, F_min=0.3, F_max=2.0,
                 F_lower=None, F_upper=None, eps_f=0.25,
                 alpha_min=0.05, alpha_max=0.95,
                 kappa=2.5, lambda_alpha=8.0):
        # 基本物理参数与安全边界。F_lower/F_upper 是期望力附近的舒适带，
        # F_min/F_max 是绝对安全边界。
        self.dt = float(dt)
        self.F_desired = float(F_desired)
        self.F_min = float(F_min)
        self.F_max = float(F_max)
        # 未显式给出舒适带时，默认取目标力 ±20%。
        band_half_width = 0.2 * max(self.F_desired, 1e-6)
        self.F_lower = float(F_lower) if F_lower is not None else max(
            self.F_min + 1e-6, self.F_desired - band_half_width
        )
        self.F_upper = float(F_upper) if F_upper is not None else min(
            self.F_max - 1e-6, self.F_desired + band_half_width
        )
        # eps_f 控制力误差归一化尺度；越小表示对力误差越敏感。
        self.eps_f = float(max(eps_f, 1e-9))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.kappa = float(kappa)
        self.lambda_alpha = float(lambda_alpha)
        self.name = "online_priority_alpha"

        # 阶段检测器仍使用原项目接口。自由空间/接近/退回阶段强制位置优先，
        # 防止未接触时为了“追踪力目标”而向表面猛压。
        self.phase_detector = PhaseDetector(
            F_thresh=self.F_min,
            T_transient=1.0,
            dt=self.dt,
        )
        # 初值取 alpha_min 与 alpha_max 的中点，随后由一阶滤波逐步收敛。
        self.alpha = float(np.clip(
            0.5 * (self.alpha_min + self.alpha_max),
            self.alpha_min,
            self.alpha_max,
        ))
        self.alpha_history = []
        self.phase_history = []
        self.risk_history = []

    def compute(self, F_norm, e_f=0.0, K_hat=None, e_r=0.0, z_vel=0.0,
                de_f=0.0, dK=0.0, de_r=0.0, F_desired=None,
                F_min=None, F_max=None, **unused):
        """计算当前周期 alpha。

        兼容原调度器签名，因此保留 K_hat/e_r/de_f/dK/de_r 等参数。
        本类实际只使用 F_norm、e_f、z_vel 和力边界参数，其余输入用于
        与 run_no_rcm 的统一调用接口对齐。
        """
        F_desired = self.F_desired if F_desired is None else float(F_desired)
        F_min = self.F_min if F_min is None else float(F_min)
        F_max = self.F_max if F_max is None else float(F_max)
        F_lower = float(np.clip(self.F_lower, F_min + 1e-6, F_max - 1e-6))
        F_upper = float(np.clip(self.F_upper, F_lower + 1e-6, F_max - 1e-6))

        # 先根据接触力和 z 向速度更新任务阶段。
        phase = self.phase_detector.update(F_norm, z_vel)
        if phase in (TaskPhase.FREE_SPACE, TaskPhase.APPROACHING,
                     TaskPhase.RETREAT):
            # 未进入有效接触调节前，alpha 目标值固定为位置优先。
            alpha_r = self.alpha_max
            rho_f = 0.0
            r_e = 0.0
            r_b = 0.0
        else:
            # r_e: 归一化力误差风险；r_b: 离开舒适力带/安全带的边界风险。
            r_e = abs(float(e_f)) / self.eps_f
            r_b = max(
                0.0,
                (float(F_norm) - F_upper) / max(F_max - F_upper, 1e-9),
                (F_lower - float(F_norm)) / max(F_lower - F_min, 1e-9),
            )
            # 论文中取较大的风险作为当前力调节优先级依据。
            rho_f = max(r_e, r_b)
            alpha_r = self.alpha_min + (
                self.alpha_max - self.alpha_min
            ) * np.exp(-self.kappa * rho_f * rho_f)

        # 离散化一阶动态 alpha_dot=lambda(alpha_r-alpha)。
        beta = float(np.clip(self.lambda_alpha * self.dt, 0.0, 1.0))
        self.alpha = self.alpha + beta * (float(alpha_r) - self.alpha)
        self.alpha = float(np.clip(
            self.alpha,
            self.alpha_min,
            self.alpha_max,
        ))

        # 历史量用于离线分析 alpha 曲线和风险项。
        self.alpha_history.append(self.alpha)
        self.phase_history.append(phase.value)
        self.risk_history.append((rho_f, r_e, r_b, alpha_r))
        return self.alpha

    def set_retreat(self, val=True):
        """通知阶段检测器进入/退出退回阶段。"""
        self.phase_detector.set_retreat(val)

    def reset(self):
        """每次 trial 前恢复初始 alpha 和历史记录。"""
        self.alpha = float(np.clip(
            0.5 * (self.alpha_min + self.alpha_max),
            self.alpha_min,
            self.alpha_max,
        ))
        self.alpha_history = []
        self.phase_history = []
        self.risk_history = []
        self.phase_detector.reset()


class FixedAlphaScheduler:
    """固定 α 对照组"""

    def __init__(self, val=0.5):
        self.alpha = val
        self.name = f"fixed_{val:.1f}"
        self.alpha_history = []
        self.phase_history = []

    def compute(self, **kw):
        self.alpha_history.append(self.alpha)
        self.phase_history.append(-1)
        return self.alpha

    def set_retreat(self, val=True):
        pass

    def reset(self):
        self.alpha_history = []
        self.phase_history = []

    @property
    def phase_detector(self):
        class _D:
            phase = type('P', (), {'value': -1})()
        return _D()
