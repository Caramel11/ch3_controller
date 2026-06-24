"""
合作博弈力-位控制器 — 4D 逐轴 ARE + 基线虚拟刚度 K_v
=====================================================

数学模型 (逐轴标量, 三轴相同由 Kronecker 扩展)
----------------------------------------------

1. 修正的阻抗动力学:
     m ẍ + c ẋ + K_v (x − x_r) = u + f_ext        (m=10, c=300, K_v=100)

   关键: 新增 K_v(x−x_r) 基线虚拟弹簧, 消除原方案 Ā 矩阵 A[1,0]=0 的
         结构奇异性。

2. 增广状态 (每轴 4D):
     z_axis = [e_r1, e_r2, e_f, σ_f]ᵀ ∈ ℝ⁴

   其中 e_r1 和 σ_f 是**泄漏积分器**的状态:
     ė_r1 = e_r2 − ε_r · e_r1                (ε_r = 1 s⁻¹)
     σ̇_f  = e_f  − ε_f · σ_f                 (ε_f = 2 s⁻¹)
     ė_r2 = -K_v/m · e_r1 − c/m · e_r2 + 1/m · e_f + 1/m · u
     ė_f  = -(K_e − B_e·c/m) · e_r2 − B_e/m · e_f − B_e/m · u

3. 每轴 4×4 系统矩阵:

     Ā_axis = ┌ −ε_r     1         0        0   ┐
              │ −K_v/m  −c/m       1/m      0   │
              │  0      −κ        −β        0   │
              └  0       0         1       −ε_f ┘

     b̄_axis = [0, 1/m, −β, 0]ᵀ

     其中 κ = K_e − B_e·c/m,  β = B_e/m

4. 三轴合并 (Kronecker 积):
     z = [e_r1 (3), e_r2 (3), e_f (3), σ_f (3)]ᵀ ∈ ℝ¹²
     Ā_12 = Ā_axis ⊗ I_3    ∈ ℝ¹²ˣ¹²
     B̄_12 = b̄_axis ⊗ I_3    ∈ ℝ¹²ˣ³

5. 归一化代价矩阵 (每轴 4×4):
     Q_1 = diag(q_r1, q_r2, 0, 0) = diag(40000, 400, 0, 0)
     Q_2 = diag(0, 0, q_f, q_sf) = diag(0, 0, 1.0, 0.25)
     Q_α = α·Q_1 + (1−α)·Q_2
     R_α = α·R_1 + (1−α)·R_2 = 1.0

   归一化原理: q_i · e_i² 在典型误差下均为 O(1):
     q_r1 · (5mm)²  = 40000 × 0.025×10⁻³ = 1.0
     q_r2 · (50mm/s)² = 400 × 2.5×10⁻³ = 1.0
     q_f · (1N)²    = 1.0
     q_sf · (2Ns)²  = 0.25 × 4 = 1.0

6. ARE 解与增益:
     Ā_axisᵀ P + P Ā_axis − P b̄ R_α⁻¹ b̄ᵀ P + Q_α_axis = 0
     K_raw_axis = R_α⁻¹ b̄ᵀ P    ∈ ℝ¹ˣ⁴
     K_raw_axis = [K_r1, K_r2, K_ef, K_sf]

7. 等效增益 (折入 K_v 基线):
     K_eff_axis = [K_v + K_r1, K_r2, K_ef, K_sf]
     K_total = K_v + K_r1    (总等效位置刚度)

   应用控制律 (每轴):
     u_axis = −K_eff_axis · [e_r1, e_r2, e_f, σ_f]ᵀ

   三轴合并:
     u = −(K_eff_axis ⊗ I_3) · z     ∈ ℝ³

8. 离线预计算:
     α_grid  = 0.0, 0.05, ..., 1.0          (21 点)
     K_e_grid = [50, 80, 100, ..., 5000]     (13 点)
     对每个 (α, K_e) 求 ARE → 增益表

9. 在线查表 + 双线性插值: gain_table[α, K_e] → K_eff_axis
"""
import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov


# ================================================================
# 系统矩阵构造 (逐轴 4D)
# ================================================================
def build_axis_system(M, C, Kv, Ke, Be, eps_r, eps_f):
    """
    构造每轴 4×4 A 矩阵和 4×1 b 向量

    Parameters
    ----------
    M, C  : scalar   期望惯性, 阻尼
    Kv    : scalar   基线虚拟刚度 (N/m)
    Ke, Be: scalar   环境刚度, 阻尼
    eps_r : scalar   位置泄漏因子 (s⁻¹)
    eps_f : scalar   力积分泄漏因子 (s⁻¹)

    Returns
    -------
    A : (4, 4)
    b : (4, 1)
    """
    kappa = Ke - Be * C / M        # K_e − B_e·c/m
    beta  = Be / M                 # B_e / m

    A = np.array([
        [-eps_r,   1.0,      0.0,     0.0   ],
        [-Kv/M,   -C/M,      1/M,     0.0   ],
        [ 0.0,    -kappa,   -beta,    0.0   ],
        [ 0.0,     0.0,      1.0,    -eps_f ],
    ])

    b = np.array([
        [0.0],
        [1/M],
        [-beta],
        [0.0],
    ])

    return A, b


# ================================================================
# 单点 ARE 求解 (Pareto 加权合并)
# ================================================================
def solve_axis_ARE(A, b, alpha, q_r1, q_r2, q_f, q_sf, R1, R2):
    """
    求解给定 (α) 下的每轴 4×4 合作博弈 ARE

    Q_α = α·diag(q_r1, q_r2, 0, 0) + (1−α)·diag(0, 0, q_f, q_sf)
    R_α = α·R_1 + (1−α)·R_2

    Āᵀ P + P Ā − P b R_α⁻¹ bᵀ P + Q_α = 0
    K_raw = R_α⁻¹ bᵀ P                      ∈ ℝ¹ˣ⁴

    Returns
    -------
    K_raw : (4,)   原始 ARE 增益 [K_r1, K_r2, K_ef, K_sf]
    P     : (4, 4) Riccati 解
    max_re: scalar 闭环最大特征值实部 (必须 < 0)
    """
    Q_alpha = np.diag([
        alpha * q_r1,
        alpha * q_r2,
        (1 - alpha) * q_f,
        (1 - alpha) * q_sf,
    ]) + 1e-8 * np.eye(4)   # 正则化防止端点 (α=0 或 α=1) 的可检测性问题

    R_alpha = alpha * R1 + (1 - alpha) * R2
    R_mat = np.array([[R_alpha]])

    P = solve_continuous_are(A, b, Q_alpha, R_mat)
    K_raw = (1.0 / R_alpha) * b.T @ P        # (1, 4)

    A_cl = A - b @ K_raw
    max_re = max(e.real for e in np.linalg.eigvals(A_cl))

    return K_raw.flatten(), P, max_re


def solve_axis_pareto_iteration(
        A, b, alpha0, q_r1, q_r2, q_f, q_sf, R1, R2,
        init_K=None, max_iter=80, tol=1e-7):
    """
    Algorithm 2 风格的 Contact-Aware Multi Objective Pareto 迭代.

    本代码沿用 no-RCM 控制器的符号约定: u = -K z, 因此闭环矩阵为
    A_cl = A - bK。参考 0926_circle_200hz.py 中的迭代片段, 在给定公共
    策略 K 下分别求两个目标的 Lyapunov/Riccati 评价矩阵 P1/P2, 再用
    Pareto 加权组合更新公共策略。
    """
    # Q1 表示位置/速度目标，Q2 表示力误差/力积分目标。
    # 端点正则项用于避免 alpha=0 或 alpha=1 时出现可检测性数值问题。
    Q1 = np.diag([q_r1, q_r2, 0.0, 0.0]) + 1e-8 * np.eye(4)
    Q2 = np.diag([0.0, 0.0, q_f, q_sf]) + 1e-8 * np.eye(4)

    # alpha0 是当前网格点的 Pareto 权重；本版本按每个 alpha 网格点
    # 单独迭代，避免用单一 alpha0 外推造成接近阶段混入力目标。
    alpha0 = float(np.clip(alpha0, 0.0, 1.0))
    R_alpha = alpha0 * R1 + (1.0 - alpha0) * R2

    if init_K is None:
        # 若外部没有给初始策略，则用标量化 ARE 得到一个稳定初值。
        K, _, _ = solve_axis_ARE(
            A, b, alpha0, q_r1, q_r2, q_f, q_sf, R1, R2
        )
        K = K.reshape(1, 4)
    else:
        K = np.asarray(init_K, dtype=float).reshape(1, 4)

    P1 = Q1.copy()
    P2 = Q2.copy()
    converged = False
    max_re = np.inf

    for it in range(max_iter):
        # 固定当前公共策略 K，闭环矩阵为 A-bK，因为控制律是 u=-Kz。
        A_cl = A - b @ K
        max_re = max(e.real for e in np.linalg.eigvals(A_cl))
        if max_re >= -1e-8:
            # 闭环不稳定时停止该点迭代，由外层 fallback 处理。
            break

        # 策略评估: 分别求两个目标在当前策略下的 Lyapunov 方程。
        # scipy 的 solve_continuous_lyapunov 求 AX + XA^T = Q，
        # 因此这里加负号以匹配 A_cl^T P + P A_cl + cost = 0。
        P1 = -solve_continuous_lyapunov(A_cl.T, Q1 + K.T @ (R1 * K))
        P2 = -solve_continuous_lyapunov(A_cl.T, Q2 + K.T @ (R2 * K))
        P1 = 0.5 * (P1 + P1.T)
        P2 = 0.5 * (P2 + P2.T)

        # 策略改进: 用 Pareto 加权后的 P 更新公共控制增益。
        K_new = (1.0 / R_alpha) * b.T @ (
            alpha0 * P1 + (1.0 - alpha0) * P2
        )

        # 相对步长比绝对步长更稳健，避免不同刚度点的增益量级影响收敛判据。
        rel_step = np.linalg.norm(K_new - K) / max(1.0, np.linalg.norm(K))
        K = K_new
        if rel_step < tol:
            converged = True
            break

    return {
        'P1': P1,
        'P2': P2,
        'K0': K.flatten(),
        'max_re': max_re,
        'iterations': it + 1,
        'converged': converged,
    }


# ================================================================
# 控制器类
# ================================================================
class CooperativeGameController:
    """
    合作博弈力-位控制器 (4D 每轴 ARE + K_v 基线)

    使用方法:
      ctrl = CooperativeGameController()
      ctrl.precompute_gains()
      ctrl.save_gains('gains.npy')    # 保存以免重复计算
      # 在线:
      u_tool = ctrl.compute_control(e_r1, e_r2, e_f, sigma_f, alpha, K_e_hat)
    """

    def __init__(self, control_mode='are'):
        """初始化控制器参数。

        control_mode:
          - 'are': 原项目的加权 ARE 预计算；
          - 'pareto_iter': Algorithm 2 的迭代式 Pareto 增益预计算。
        两种模式最终都生成同形状的 gains_db，因此在线 compute_control 不变。
        """
        # ---- 阻抗模型参数 ----
        self.M = 10.0         # kg    期望惯性
        self.C = 300.0        # Ns/m  阻尼
        self.Kv = 100.0       # N/m   基线虚拟刚度 [新增]

        # ---- 泄漏积分器参数 ----
        self.eps_r = 1.0      # s⁻¹   位置泄漏因子
        self.eps_f = 2.0      # s⁻¹   力积分泄漏因子

        # ---- 归一化代价矩阵参数 [新增] ----
        self.q_r1 = 40000.0   # 1/(5mm)²
        self.q_r2 = 400.0     # 1/(50mm/s)²
        self.q_f  = 100.0       # 1/(1N)²
        self.q_sf = 0.25      # 1/(2N·s)²
        self.R1 = 1.0
        self.R2 = 1.0

        # ---- 姿态 PD+I 增益 ----
        self.P_ori = 20.0
        self.D_ori = 1.0
        self.I_ori = 30.0

        # ---- 增益表 ----
        # 键: (alpha_round, Ke_round)
        # 值: np.array shape (4,) = [K_v + K_r1, K_r2, K_ef, K_sf]  (即 K_eff)
        # gains_db 是在线真正使用的增益表。pareto_db 额外保存 P1/P2 和迭代信息，
        # 便于离线检查 Algorithm 2 的收敛状态。
        self.gains_db = {}
        self.pareto_db = {}
        self.alpha_grid = None
        self.Ke_grid = None
        self.control_mode = control_mode
        self.pareto_alpha0 = 0.5
        self.pareto_max_iter = 80
        self.pareto_tol = 1e-7

        # ---- 安全限幅 ----
        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        self.u_threshold = 30.0

    # ----------------------------------------------------------------
    # 预计算
    # ----------------------------------------------------------------
    def precompute_gains(self, alpha_grid=None, Ke_grid=None, Be_default=5.0):
        """
        在 (α, K_e) 网格上预计算增益表

        每个网格点求一次 4D ARE (标量系统), 典型耗时 < 1ms/点。
        273 个点总计约 0.5 秒完成。
        """
        import rospy

        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 1.0, 21)
        if Ke_grid is None:
            Ke_grid = [50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500,
                       2000, 3000, 5000]

        self.alpha_grid = np.array(alpha_grid)
        self.Ke_grid = np.array(Ke_grid)

        n_alpha = len(alpha_grid)
        n_Ke = len(Ke_grid)

        # 通过同一个入口切换求解模式，run_no_rcm 无需改 compute_control 调用。
        if self.control_mode == 'pareto_iter':
            return self.precompute_pareto_gains(
                alpha_grid=alpha_grid,
                Ke_grid=Ke_grid,
                Be_default=Be_default,
            )

        rospy.loginfo(f"Precomputing cooperative ARE gains: "
                      f"{n_alpha} α × {n_Ke} K_e = {n_alpha * n_Ke} points")

        count = 0
        fail = 0
        for alpha in alpha_grid:
            a_key = round(float(alpha), 3)
            for Ke in Ke_grid:
                Ke_key = int(Ke)
                A, b = build_axis_system(
                    self.M, self.C, self.Kv, Ke, Be_default,
                    self.eps_r, self.eps_f
                )
                try:
                    K_raw, P, max_re = solve_axis_ARE(
                        A, b, a_key,
                        self.q_r1, self.q_r2, self.q_f, self.q_sf,
                        self.R1, self.R2
                    )
                    if max_re < -1e-8:
                        # 折入 K_v: K_eff[0] = K_v + K_r1
                        K_eff = K_raw.copy()
                        K_eff[0] = self.Kv + K_raw[0]
                        self.gains_db[(a_key, Ke_key)] = K_eff
                        count += 1
                    else:
                        fail += 1
                        self._fallback(a_key, Ke_key)
                except Exception:
                    fail += 1
                    self._fallback(a_key, Ke_key)

        rospy.loginfo(f"  Done. {count} stable, {fail} failures.")

    def precompute_pareto_gains(self, alpha_grid=None, Ke_grid=None,
                                Be_default=5.0):
        """在 (alpha, K_e) 网格上用 Algorithm 2 迭代预计算增益。"""
        import rospy

        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 1.0, 21)
        if Ke_grid is None:
            Ke_grid = [50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500,
                       2000, 3000, 5000]

        self.alpha_grid = np.array(alpha_grid)
        self.Ke_grid = np.array(Ke_grid)
        self.pareto_db = {}
        self.gains_db = {}

        rospy.loginfo(
            f"Precomputing Pareto iteration gains: "
            f"{len(alpha_grid)} α × {len(Ke_grid)} K_e = "
            f"{len(alpha_grid) * len(Ke_grid)} points"
        )

        count = 0
        fail = 0
        total_iter = 0
        for alpha in alpha_grid:
            a_key = round(float(alpha), 3)
            for Ke in Ke_grid:
                Ke_key = int(Ke)
                A, b = build_axis_system(
                    self.M, self.C, self.Kv, Ke, Be_default,
                    self.eps_r, self.eps_f
                )
                try:
                    # 使用同一网格点的 ARE 解初始化策略，保证迭代从稳定策略出发。
                    K_init, _, _ = solve_axis_ARE(
                        A, b, a_key,
                        self.q_r1, self.q_r2, self.q_f, self.q_sf,
                        self.R1, self.R2
                    )
                    sol = solve_axis_pareto_iteration(
                        A, b, a_key,
                        self.q_r1, self.q_r2, self.q_f, self.q_sf,
                        self.R1, self.R2,
                        init_K=K_init,
                        max_iter=self.pareto_max_iter,
                        tol=self.pareto_tol,
                    )
                    if np.isfinite(sol['max_re']) and sol['max_re'] < -1e-8:
                        # K0 是未折入基线刚度的增益；实际控制中 K_eff[0]
                        # 需要加上 Kv，与原 no-RCM 控制器保持一致。
                        K_eff = sol['K0'].copy()
                        K_eff[0] = self.Kv + K_eff[0]
                        self.gains_db[(a_key, Ke_key)] = K_eff
                        self.pareto_db[(a_key, Ke_key)] = sol
                        total_iter += sol['iterations']
                        count += 1
                    else:
                        fail += 1
                        self._fallback(a_key, Ke_key)
                except Exception:
                    fail += 1
                    self._fallback(a_key, Ke_key)

        avg_iter = total_iter / max(1, count)
        rospy.loginfo(
            f"  Done. {count} stable, {fail} failures, "
            f"avg_iter={avg_iter:.1f}."
        )

    def _fallback(self, a_key, Ke_key):
        """ARE 失败时用最近的成功增益填充"""
        if not self.gains_db:
            return
        nearest = min(
            self.gains_db.keys(),
            key=lambda k: abs(k[0] - a_key) + abs(k[1] - Ke_key) / 1000.0
        )
        self.gains_db[(a_key, Ke_key)] = self.gains_db[nearest]

    def has_precomputed_gains(self):
        return bool(self.gains_db)

    def save_gains(self, path):
        """保存增益表。

        注意: results/ 下的 .npy 是可再生成文件，不建议提交到仓库。
        """
        np.save(path, {
            'gains_db': self.gains_db,
            'pareto_db': self.pareto_db,
            'alpha_grid': self.alpha_grid,
            'Ke_grid': self.Ke_grid,
            'control_mode': self.control_mode,
            'pareto_alpha0': self.pareto_alpha0,
        })

    def load_gains(self, path):
        """加载已预计算增益表，减少重复启动仿真的等待时间。"""
        data = np.load(path, allow_pickle=True).item()
        self.gains_db = data.get('gains_db', {})
        self.pareto_db = data.get('pareto_db', {})
        self.alpha_grid = data['alpha_grid']
        self.Ke_grid = data['Ke_grid']
        if 'pareto_alpha0' in data:
            self.pareto_alpha0 = float(data['pareto_alpha0'])

    # ----------------------------------------------------------------
    # 在线查表 (双线性插值)
    # ----------------------------------------------------------------
    def _bracket(self, grid, value):
        """在有序 grid 中找到 value 的左右相邻点"""
        if value <= grid[0]:
            return grid[0], grid[0], 0.0
        if value >= grid[-1]:
            return grid[-1], grid[-1], 0.0
        for i in range(len(grid) - 1):
            if grid[i] <= value <= grid[i + 1]:
                t = (value - grid[i]) / (grid[i + 1] - grid[i])
                return grid[i], grid[i + 1], t
        return grid[0], grid[0], 0.0

    def get_gain(self, alpha, Ke):
        """
        双线性插值查增益表

        Returns
        -------
        K_eff_axis : (4,)  [K_v+K_r1, K_r2, K_ef, K_sf]
        """
        if not self.gains_db:
            # 未预计算时的兜底: 返回纯 K_v + 少量阻尼
            return np.array([self.Kv, 20.0, 0.0, 0.0])

        # 找 alpha 和 K_e 的相邻网格点，然后做双线性插值。
        a_lo, a_hi, t_a = self._bracket(self.alpha_grid, alpha)
        Ke_lo, Ke_hi, t_Ke = self._bracket(self.Ke_grid, Ke)

        a_lo = round(float(a_lo), 3)
        a_hi = round(float(a_hi), 3)
        Ke_lo = int(Ke_lo)
        Ke_hi = int(Ke_hi)

        # 四角增益。如果某个角点缺失，退回到距离最近的成功点。
        def _get(ak, Kk):
            if (ak, Kk) in self.gains_db:
                return self.gains_db[(ak, Kk)]
            # fallback to nearest
            nearest = min(
                self.gains_db.keys(),
                key=lambda k: abs(k[0] - ak) + abs(k[1] - Kk) / 1000.0
            )
            return self.gains_db[nearest]

        K_00 = _get(a_lo, Ke_lo)
        K_01 = _get(a_lo, Ke_hi)
        K_10 = _get(a_hi, Ke_lo)
        K_11 = _get(a_hi, Ke_hi)

        # 双线性插值保证 alpha/K_hat 连续变化时 K_eff 也连续变化。
        K = ((1 - t_a) * (1 - t_Ke) * K_00
             + (1 - t_a) * t_Ke * K_01
             + t_a * (1 - t_Ke) * K_10
             + t_a * t_Ke * K_11)

        return K

    # ----------------------------------------------------------------
    # 控制律
    # ----------------------------------------------------------------
    def compute_control(self, e_r1, e_r2, e_f, sigma_f, alpha, Ke):
        """
        计算工具尖端笛卡尔控制力

        每个误差量为 3D 向量 (x, y, z)。因为三轴动力学解耦且各向同性,
        增益各向同性: K_eff_12 = K_eff_axis ⊗ I_3 ∈ ℝ³ˣ¹²

        统一矩阵乘法:
          u = -K_eff_12 · z
            = -K_eff[0]·e_r1 - K_eff[1]·e_r2 - K_eff[2]·e_f - K_eff[3]·σ_f

        注: K_eff[0] = K_v + K_r1 (已包含基线刚度 K_v, 阻抗模型的
            K_v(x−x_r) 项由此项在高层显式实现)。

        Parameters
        ----------
        e_r1   : (3,)   泄漏积分的位置误差状态
        e_r2   : (3,)   速度误差 ẋ − ẋ_r
        e_f    : (3,)   力误差 F_ext − F_d
        sigma_f: (3,)   泄漏积分的力误差状态
        alpha  : float  仲裁参数
        Ke     : float  RLS 估计的环境刚度

        Returns
        -------
        u_tool : (3,)   笛卡尔控制力
        K_eff  : (4,)   当前激活的增益 (用于日志)
        """
        K = self.get_gain(alpha, Ke)

        u = -(K[0] * np.asarray(e_r1)
              + K[1] * np.asarray(e_r2)
              + K[2] * np.asarray(e_f)
              + K[3] * np.asarray(sigma_f))

        return u, K

    def compute_control_axis_alpha(self, e_r1, e_r2, e_f, sigma_f,
                                   alpha_xyz, Ke):
        """
        逐轴 alpha 的笛卡尔控制力。

        原 `compute_control(...)` 使用同一个 alpha 查一组各向同性增益，
        因而即使 x/y 没有力误差，较低的 z 向力控权重也会同步降低 x/y
        的位置刚度。对扫描实验而言，力位仲裁只应作用于压入深度 z；
        x/y 是给定轨迹，应严格位置跟踪。因此本函数允许
        `alpha_xyz=[1, 1, alpha_z]`，分别查每个轴的 4D 增益。

        Parameters
        ----------
        e_r1, e_r2, e_f, sigma_f : (3,)  三轴误差量
        alpha_xyz : (3,)                 每轴仲裁参数
        Ke        : float                RLS 估计环境刚度

        Returns
        -------
        u_tool : (3,)     笛卡尔控制力
        K_axes : (3, 4)   每轴激活增益；行顺序为 x/y/z
        """
        e_r1 = np.asarray(e_r1, dtype=float)
        e_r2 = np.asarray(e_r2, dtype=float)
        e_f = np.asarray(e_f, dtype=float)
        sigma_f = np.asarray(sigma_f, dtype=float)
        alpha_xyz = np.asarray(alpha_xyz, dtype=float)
        if alpha_xyz.shape != (3,):
            raise ValueError("alpha_xyz must have shape (3,)")

        K_axes = np.vstack([
            self.get_gain(float(np.clip(a, 0.0, 1.0)), Ke)
            for a in alpha_xyz
        ])
        u = -(
            K_axes[:, 0] * e_r1
            + K_axes[:, 1] * e_r2
            + K_axes[:, 2] * e_f
            + K_axes[:, 3] * sigma_f
        )
        return u, K_axes
