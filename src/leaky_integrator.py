"""
泄漏积分器 — 用于 e_r1 和 σ_f 状态更新
========================================

连续时间定义:
  ẋ = input − ε·x

离散化 (一阶 Euler, 周期 T_s):
  x[k+1] = (1 − ε·T_s)·x[k] + T_s·input[k]

物理意义:
  ε = 0 时等价于纯积分器 x = ∫input dt
  ε > 0 时相当于时间常数 τ = 1/ε 的低通滤波
  稳态值 x_ss = input_ss / ε
  离散稳定性要求: ε·T_s < 1 (实际中 T_s = 1ms 时 ε < 1000)
"""
import numpy as np


class LeakyIntegrator:
    """
    向量化泄漏积分器 (支持 3D 或 1D)

    用法:
        integ = LeakyIntegrator(eps=1.0, dt=0.001, dim=3)
        integ.reset()
        # 每个控制周期:
        state = integ.update(input_vec)   # 返回当前积分状态
    """

    def __init__(self, eps, dt, dim=3, initial=None):
        # eps 越大，历史误差衰减越快；dt 必须与控制周期一致。
        self.eps = float(eps)
        self.dt = float(dt)
        self.dim = dim
        # decay 是离散泄漏系数。若 decay<=0，说明积分器离散化不稳定。
        self.decay = 1.0 - self.eps * self.dt   # (1 − ε·T_s)
        assert self.decay > 0, f"ε·T_s = {self.eps*self.dt} ≥ 1, unstable"

        self._initial = np.zeros(dim) if initial is None else np.asarray(initial, dtype=float)
        self.state = self._initial.copy()

    def reset(self, initial=None):
        """重置状态, 可选指定初始值"""
        if initial is None:
            self.state = self._initial.copy()
        else:
            self.state = np.asarray(initial, dtype=float).copy()

    def update(self, input_vec):
        """
        状态更新一步

        Parameters
        ----------
        input_vec : np.ndarray
            当前输入 (e_r2 或 e_f)

        Returns
        -------
        state : np.ndarray
            更新后的积分状态
        """
        # 向量化实现，run_no_rcm 中 dim=3，对 x/y/z 三轴同时更新。
        inp = np.asarray(input_vec, dtype=float)
        self.state = self.decay * self.state + self.dt * inp
        return self.state.copy()

    def get(self):
        """获取当前状态 (不更新)"""
        return self.state.copy()
