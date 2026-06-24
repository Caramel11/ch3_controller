# Current Stiffness Estimation Method

本文档描述 `tests/0603/src/env_estimator.py` 中当前使用的环境刚度估计方法，
以及 `run_no_rcm.py`、`run_with_rcm.py`、`run_no_rcm_real.py`、
`run_with_rcm_real.py` 对该估计器的调用方式。

## 1. 接触模型与符号

扫描方向为工具端沿 \(x\) 轴运动，接触法向近似取工具端 \(z\) 方向。
环境局部采用 Kelvin-Voigt 模型：

\[
F(t) = K_e(t)\,\delta(t) + B_e(t)\,\dot{\delta}(t) + \eta(t),
\]

其中：

- \(F(t) \ge 0\)：接触法向力标量，单位 N。
- \(\delta(t) \ge 0\)：压入量，单位 m。
- \(\dot{\delta}(t)\)：压入速度，单位 m/s。
- \(K_e(t)\)：等效环境刚度，单位 N/m。
- \(B_e(t)\)：等效环境阻尼，单位 N·s/m。
- \(\eta(t)\)：测量噪声、模型误差和速度估计误差导致的残差。

no-RCM 普通 Gazebo 入口中：

\[
\delta(t) = \max(0, z_s - z(t)), \qquad \dot{\delta}(t) = -\dot z(t),
\]

其中 \(z_s=\texttt{cfg.approach\_z}\) 是虚拟接触面高度。

no-RCM real/mock 入口中，接触后使用检测到的接触平面：

\[
\delta(t) = \max(0, z_c - z(t)), \qquad \dot{\delta}(t) = -\dot z(t),
\]

其中 \(z_c\) 来自接触检测阶段的 `contact_plane_z`。

RCM 入口中，为了维持轻微预载，扫描阶段采用：

\[
\delta(t)=\max(0, z_c-z(t)+\delta_b), \qquad \dot{\delta}(t)=-\dot z(t),
\]

其中 \(\delta_b=\texttt{cfg.scan\_contact\_bias}\)。

## 2. 为什么不直接使用二参数 RLS 的 \(K_e\)

经典二参数 RLS 可写为：

\[
F(t)=\phi(t)^\top\theta(t), \qquad
\phi(t)=
\begin{bmatrix}
\delta(t)\\
\dot{\delta}(t)
\end{bmatrix}, \qquad
\theta(t)=
\begin{bmatrix}
K_e(t)\\
B_e(t)
\end{bmatrix}.
\]

递推形式为：

\[
g_k = \frac{P_{k-1}\phi_k}
{\lambda + \phi_k^\top P_{k-1}\phi_k},
\]

\[
\theta_k = \theta_{k-1}
       + g_k\left(F_k-\phi_k^\top\theta_{k-1}\right),
\]

\[
P_k = \frac{P_{k-1}-g_k\phi_k^\top P_{k-1}}{\lambda}.
\]

在当前 Gazebo/真机恒力扫描中，控制器会主动维持接触力接近常数，因此
\(\delta\) 往往只有数毫米且变化很小；同时 \(\dot{\delta}\) 来自机器人速度估计，
容易含有高频噪声。此时 \(\delta\) 与 \(\dot{\delta}\) 的可辨识性很差，
二参数 RLS 容易把速度噪声或瞬时阻尼项误归入刚度项，导致
\(\hat K_e\) 反向漂移。

因此当前实现不把 RLS 的第一维 \(\theta_0\) 作为控制用刚度，而是使用
准静态表观刚度估计。

## 3. 控制用刚度估计

估计器每个控制周期接收：

\[
(F_k,\delta_k,\dot{\delta}_k).
\]

### 3.1 非接触保护

若

\[
|F_k| < F_{\min}^{est}
\quad\text{or}\quad
|\delta_k| < \delta_{\min},
\]

则认为当前观测不足以更新刚度估计，直接保持上一时刻输出：

\[
\hat K_k = \hat K_{k-1}.
\]

当前默认参数为：

\[
F_{\min}^{est}=0.3\ \mathrm{N}, \qquad
\delta_{\min}=2\times 10^{-4}\ \mathrm{m}.
\]

### 3.2 表观刚度观测

有效接触时，先计算表观刚度：

\[
K_k^{obs} = \frac{|F_k|}{|\delta_k|}.
\]

然后做边界投影：

\[
\bar K_k^{obs}
= \Pi_{[K_{\min},K_{\max}]}\left(K_k^{obs}\right),
\]

其中：

\[
K_{\min}=50\ \mathrm{N/m}, \qquad
K_{\max}=5000\ \mathrm{N/m}.
\]

这里 \(\Pi_{[a,b]}(x)=\min(\max(x,a),b)\)。

### 3.3 EMA 低通

控制器实际使用的刚度估计为：

\[
\hat K_k =
\Pi_{[K_{\min},K_{\max}]}
\left(
\hat K_{k-1}
+ \alpha_K(\bar K_k^{obs}-\hat K_{k-1})
\right).
\]

默认：

\[
\alpha_K = 0.05.
\]

因此当前的 \(\hat K_k\) 是低通后的表观刚度，而不是 RLS 参数向量中的
第一维。代码中 `K_observed` 对应 \(\bar K_k^{obs}\)，`K_e`/返回值中的
`K_hat` 对应 \(\hat K_k\)。

## 4. 阻尼辅助估计

虽然控制用刚度不再直接取 RLS 的 \(K_e\)，估计器仍保留有界 RLS 来给出
辅助阻尼估计 \(\hat B_k\)。

速度回归量先限幅：

\[
\bar{\dot{\delta}}_k
= \Pi_{[-v_{\max},v_{\max}]}\left(\dot{\delta}_k\right),
\]

默认：

\[
v_{\max}=0.02\ \mathrm{m/s}.
\]

RLS 使用：

\[
\phi_k =
\begin{bmatrix}
\delta_k\\
\bar{\dot{\delta}}_k
\end{bmatrix}.
\]

参数递推仍为：

\[
g_k = \frac{P_{k-1}\phi_k}
{\lambda+\phi_k^\top P_{k-1}\phi_k},
\]

\[
\theta_k^-=\theta_{k-1}
+g_k(F_k-\phi_k^\top\theta_{k-1}),
\]

\[
P_k=\frac{P_{k-1}-g_k\phi_k^\top P_{k-1}}{\lambda}.
\]

随后对参数做投影：

\[
\theta_k =
\begin{bmatrix}
\Pi_{[K_{\min},K_{\max}]}(\theta_{k,0}^-)\\
\Pi_{[B_{\min},B_{\max}]}(\theta_{k,1}^-)
\end{bmatrix}.
\]

默认：

\[
\lambda=0.995,\qquad
P_0=10^4 I_2,\qquad
B_{\min}=0.1,\qquad
B_{\max}=100.
\]

输出的阻尼估计为：

\[
\hat B_k = \theta_{k,1}.
\]

注意：\(\theta_{k,0}\) 仅作为 RLS 内部状态保留，不作为控制器增益查表的
刚度输入。

## 5. 初始化与重置

默认初始参数为：

\[
\theta_0 =
\begin{bmatrix}
200\\
5
\end{bmatrix},
\qquad
\hat K_0 = 200,\qquad
K_0^{obs}=200.
\]

若脚本显式传入 `theta_init`，则使用脚本配置。0603 real/mock 入口通常使用：

\[
\theta_0 =
\begin{bmatrix}
\texttt{cfg.estimator\_initial\_K}\\
\texttt{cfg.estimator\_initial\_B}
\end{bmatrix}.
\]

每次 trial 开始时调用 `reset()`，重置：

\[
\theta\leftarrow\theta_0,\qquad
P\leftarrow P_0 I_2,\qquad
\hat K\leftarrow \theta_{0,0},\qquad
K^{obs}\leftarrow \theta_{0,0}.
\]

## 6. 脚本中的输出关系

### 6.1 普通 Gazebo no-RCM 和 with-RCM

`run_no_rcm.py` 与 `run_with_rcm.py` 直接使用估计器输出：

\[
K_{\text{ctrl},k} = \hat K_k.
\]

日志字段含义：

- `K_hat`：\(\hat K_k\)，控制器使用的低通表观刚度。
- `K_hat_raw`：\(\bar K_k^{obs}\)，当前周期表观刚度观测。
- `B_hat`：\(\hat B_k\)，bounded RLS 阻尼估计。
- `delta`：\(\delta_k\)。
- `delta_dot`：\(\dot{\delta}_k\)。
- `K_env_true`：Gazebo 虚拟环境返回的真实/插值刚度，仅用于诊断。
- `B_env_true`：Gazebo 虚拟环境返回的真实/插值阻尼，仅用于诊断。

### 6.2 no-RCM real/mock

`run_no_rcm_real.py` 中估计器返回值先记为：

\[
K_k^{raw}=\hat K_k.
\]

随后额外通过一阶时间常数低通：

\[
K_k^{log}
= K_{k-1}^{log}
+ \beta_k(K_k^{raw}-K_{k-1}^{log}),
\]

\[
\beta_k =
\operatorname{clip}\left(
\frac{\Delta t_k}{\max(\tau_K,\Delta t_k)},0,1
\right),
\]

其中：

\[
\tau_K=\texttt{cfg.stiffness\_filter\_tau}.
\]

控制器查表前还会做上限裁剪：

\[
K_{\text{ctrl},k}
= \min(K_k^{log}, K_{\max}^{ctrl}),
\]

其中：

\[
K_{\max}^{ctrl}=\texttt{cfg.scan\_z\_gain\_khat\_max}.
\]

对应日志字段：

- `K_hat_raw`：估计器直接输出 \(K_k^{raw}\)。
- `K_hat`：二次低通后的 \(K_k^{log}\)。
- `K_hat_ctrl`：送入控制器的 \(K_{\text{ctrl},k}\)。

### 6.3 with-RCM real/mock

`run_with_rcm_real.py` 当前直接使用估计器输出：

\[
K_{\text{ctrl},k} = \hat K_k.
\]

日志字段与普通 with-RCM 一致。

## 7. 与控制器的接口

合作博弈控制器在线调用形式为：

\[
u_k = \mathcal{G}
\left(
e_{r1,k}, e_{r2,k}, e_{f,k}, \sigma_{f,k},
\alpha_k, K_{\text{ctrl},k}
\right).
\]

其中 \(K_{\text{ctrl},k}\) 用于在预计算增益表中按环境刚度插值。
因此当前方法的设计目标不是精确分离 \(K_e\) 与 \(B_e\)，而是在恒力扫描、
小压入量和速度噪声存在时，为控制器提供单调、稳定、物理量级合理的
刚度调度量。

## 8. 适用性与局限

当前估计量 \(\hat K_k\) 是表观刚度：

\[
\hat K_k \approx \frac{F_k}{\delta_k}.
\]

在 \(\dot{\delta}\neq 0\) 时，由于真实力包含阻尼项，

\[
\frac{F_k}{\delta_k}
= K_e + B_e\frac{\dot{\delta}_k}{\delta_k}
+ \frac{\eta_k}{\delta_k}.
\]

因此 \(\hat K_k\) 不应被理解为严格的静态材料刚度，而应理解为
控制调度使用的等效接触刚度。对于需要精确辨识 \(K_e,B_e\) 的离线系统辨识，
应使用包含充分激励的独立辨识实验，而不是恒力扫描段本身。
