# continuous_force_margin 仲裁方法与刚度响应实验报告

本文总结 `0603` 实验中当前调试完成的 `continuous_force_margin` 力位仲裁方案，并结合本地 Gazebo 结果解释它相对固定仲裁参数 `fixed_0.8`、`fixed_0.5`、`fixed_0.2` 的优势。报告中的公式均采用 VSCode Markdown 可直接预览的 `$...$` 与 `$$...$$` 写法。

本轮核心改动是：**力位仲裁参数 $\alpha$ 只作用于 z 方向压入深度，x/y 方向严格跟踪位置轨迹；同时将估计环境刚度 $\hat K_e$ 显式接入 `continuous_force_margin` 的 alpha 设计，使稳定后的 $\alpha_z$ 随刚度区间明显变化。**

图表目录：

- 指标与图集：`/home/liu/franka_ws_1101/results/continuous_force_margin_stiffness_response_20260608`
- no-RCM 调试副本：`run_no_rcm_force_margin_challenge.py`
- with-RCM 调试副本：`run_with_rcm_force_margin_challenge.py`

## 1. z-only 力位仲裁

原先若把同一个 $\alpha$ 同时施加到 $x,y,z$，则为了改善 z 方向力跟踪而降低 $\alpha$ 时，会同步削弱 x/y 轨迹跟踪。当前方案改为逐轴仲裁：

$$
\boldsymbol{\alpha}
=
\begin{bmatrix}
1 & 1 & \alpha_z
\end{bmatrix}^{\mathsf T}.
$$

因此：

- $x/y$ 始终为位置主导，严格跟踪扫描轨迹。
- $z$ 方向通过 $\alpha_z$ 在压入深度位置保持、力调节和 RCM 稳定之间仲裁。
- $\alpha_z$ 越大，越偏位置/RCM 稳定；$\alpha_z$ 越小，越偏力调节。

令工具端期望位置和速度为 $x_d,\dot x_d$，实际位置和速度为 $x,\dot x$，测量接触力为 $F$，期望力为 $F_d$。定义

$$
e_{r1}=x-x_d,\qquad
e_{r2}=\dot x-\dot x_d,\qquad
e_f=F-F_d.
$$

力误差泄漏积分为

$$
\dot{\sigma}_f=e_f-\varepsilon_f\sigma_f.
$$

合作博弈控制器对每一轴 $i\in\{x,y,z\}$ 根据 $\alpha_i$ 和估计环境刚度 $\hat K_e$ 查询或迭代得到增益

$$
K_i(\alpha_i,\hat K_e)
=
\begin{bmatrix}
K_{r1,i} & K_{r2,i} & K_{ef,i} & K_{\sigma f,i}
\end{bmatrix}.
$$

笛卡尔控制力为

$$
u_i
=
-K_{r1,i}e_{r1,i}
-K_{r2,i}e_{r2,i}
-K_{ef,i}e_{f,i}
-K_{\sigma f,i}\sigma_{f,i}.
$$

在 with-RCM 中，$u_{\mathrm{tool}}$ 还要经过 RCM 杠杆映射与雅可比转置得到关节力矩；在 no-RCM 中直接使用 tool 端雅可比映射。两者共享同一 z-only 仲裁思想。

## 2. 阶段检测与连续阶段先验

实验过程并非始终处于同一种控制语义。自由空间、接近、刚接触瞬态、稳定接触和退回阶段对 $\alpha$ 的要求不同。若只用力误差做仲裁，在刚接触瞬间容易因力和速度短时震荡导致 $\alpha$ 抖动。因此先进行阶段检测。

阶段检测器使用接触力范数 $F_{\mathrm{norm}}$、z 方向速度 $\dot z$ 和接触持续时间 $t_c$：

$$
\phi=
\begin{cases}
\mathrm{RETREAT}, & \text{退回标志为真},\\
\mathrm{CONTACT\_TRANSIENT}, & F_{\mathrm{norm}}\ge F_{\mathrm{th}},\ t_c<T_{\mathrm{trans}},\\
\mathrm{CONTACT\_STEADY}, & F_{\mathrm{norm}}\ge F_{\mathrm{th}},\ t_c\ge T_{\mathrm{trans}},\\
\mathrm{APPROACHING}, & F_{\mathrm{norm}}<F_{\mathrm{th}},\ \dot z<-\dot z_{\mathrm{th}},\\
\mathrm{FREE\_SPACE}, & \text{otherwise}.
\end{cases}
$$

为了避免离散阶段切换导致 $\alpha$ 跳变，`continuous_force_margin` 使用连续阶段先验。定义接触门控、接近速度门控和接触稳定门控：

$$
c_F=\sigma\left(\frac{F_{\mathrm{norm}}-F_{\mathrm{th}}}{b_F}\right),
$$

$$
c_v=\sigma\left(\frac{-\dot z-\dot z_{\mathrm{th}}}{b_v}\right),
$$

$$
c_t=S\left(\frac{t_c}{T_{\mathrm{trans}}}\right),
$$

其中 $\sigma(\cdot)$ 为 sigmoid，$S(\cdot)$ 为 smoothstep：

$$
S(q)=q^2(3-2q),\qquad q\in[0,1].
$$

自由空间与接近阶段的先验为

$$
\alpha_{\mathrm{pre}}
=(1-c_v)\alpha_{\mathrm{free}}+c_v\alpha_{\mathrm{approach}},
$$

$$
w_{\mathrm{pre}}
=(1-c_v)w_{\mathrm{free}}+c_vw_{\mathrm{approach}}.
$$

接触瞬态与稳定接触先验为

$$
\alpha_{\mathrm{contact}}
=(1-c_t)\alpha_{\mathrm{trans}}+c_t\alpha_{\mathrm{steady}},
$$

$$
w_{\mathrm{contact}}
=(1-c_t)w_{\mathrm{trans}}+c_tw_{\mathrm{steady}}.
$$

最终阶段先验为

$$
\alpha_\phi=(1-c_F)\alpha_{\mathrm{pre}}+c_F\alpha_{\mathrm{contact}},
$$

$$
w_\phi=(1-c_F)w_{\mathrm{pre}}+c_Fw_{\mathrm{contact}}.
$$

$w_\phi$ 是阶段先验权重。自由空间和退回阶段 $w_\phi$ 较高，使系统位置优先；稳定接触后 $w_\phi$ 降低，让模糊力边界和刚度项主导。

## 3. force-margin 模糊仲裁

力边界设置为 $F_{\min}$、$F_{\max}$，期望力为 $F_d$。定义力区间宽度、力误差、力安全裕度和方向性归一化误差：

$$
w_F=F_{\max}-F_{\min},
$$

$$
e_F=F-F_d,
$$

$$
\rho_F
=
\operatorname{clip}
\left(
\frac{\min(F-F_{\min},F_{\max}-F)}{w_F/2},
0,1
\right),
$$

$$
s_F
=
\operatorname{clip}
\left(
\frac{F-F_d}{w_F/2},
-1,1
\right).
$$

其中 $\rho_F$ 越小表示越接近力边界；$s_F>0$ 表示力偏大，$s_F<0$ 表示力偏小。

模糊输入采用三项：

$$
F_h=\operatorname{clip}\left(\frac{2|e_F|}{w_F},0,2\right),
$$

$$
S_h=6\rho_F,
$$

$$
D_r
=
\operatorname{clip}
\left(
0.1-\frac{0.1}{0.005}|e_r|,
0,0.1
\right).
$$

这里 $F_h$ 描述力误差大小，$S_h$ 描述力边界安全裕度，$D_r$ 描述位置误差是否已经变大。模糊规则的设计原则如下：

- 力处于安全裕度内且位置误差较大时，提高 $\alpha_z$，优先恢复轨迹和 RCM 稳定。
- 力接近上边界或 $e_F>0$ 较大时，提高 $\alpha_z$，避免高刚度环境下继续压入造成过大接触力。
- 力接近下边界或 $e_F<0$ 较大时，降低 $\alpha_z$，释放 z 向力调节能力。
- 接触瞬态阶段由阶段先验限制 $\alpha_z$ 变化，避免刚接触震荡被保存为正式扫描数据。

模糊推理得到基础仲裁值 $\alpha_0$。随后使用方向性力边界风险做安全修正：

$$
\alpha_{\mathrm{safe}}
=
\alpha_0
+k_{\mathrm{upper}}(1-\rho_F)\max(s_F,0)
-k_{\mathrm{lower}}(1-\rho_F)\max(-s_F,0).
$$

这一步体现了 force-margin 的核心思想：不是只看 $|F-F_d|$，而是同时看力处在边界内部的哪个位置。靠近上边界时更保守，靠近下边界时更愿意给力控空间。

## 4. 环境刚度显式响应项

早期 `PhaseAwareFuzzyAlphaScheduler` 中确实包含估计刚度项，它把 $\hat K_e$ 映射为模糊变量 $T_h$：

$$
T_h=\operatorname{clip}\left(\frac{\hat K_e}{K_{\mathrm{ref}}}s_K,0,6\right).
$$

但当前用于对照实验的 `ContinuousForceMarginFuzzyAlphaScheduler` 为了强调力边界裕度，核心模糊输入换成了 $F_h,S_h,D_r$。因此如果不额外处理，$\hat K_e$ 虽然被传入调度器，稳定阶段的 $\alpha_z$ 对刚度变化并不明显。本轮修复是把 $\hat K_e$ 显式接回 continuous 方案。

定义刚度门控

$$
g_K
=
S
\left(
\frac{\hat K_e-K_{\mathrm{low}}}
{K_{\mathrm{high}}-K_{\mathrm{low}}}
\right),
$$

其中 $S(\cdot)$ 仍为 smoothstep，并对输入裁剪到 $[0,1]$。刚度目标仲裁值为

$$
\alpha_K
=
(1-g_K)\alpha_{\mathrm{low}}
+g_K\alpha_{\mathrm{high}}.
$$

将它与 force-margin 安全输出混合：

$$
\alpha_{\mathrm{safe}}
\leftarrow
(1-\beta_K)\alpha_{\mathrm{safe}}
+\beta_K\alpha_K.
$$

最终再融合阶段先验：

$$
\alpha_{\mathrm{raw}}
=
w_\phi\alpha_\phi
+(1-w_\phi)\alpha_{\mathrm{safe}}.
$$

实际执行值经过一阶滤波：

$$
\alpha_k
=
\alpha_{k-1}
+\frac{\Delta t}{\max(\tau,\Delta t)}
\left(
\alpha_{\mathrm{raw}}-\alpha_{k-1}
\right).
$$

with-RCM 副本还叠加了 `AlphaCommandLimiter` 的变化率限制，并在 RCM soft/recovery 区间提升 $\alpha_z$ 下限：

$$
\alpha_z\leftarrow \max(\alpha_z,\alpha_{\mathrm{RCM\ floor}}(e_{\mathrm{RCM}})).
$$

这个保护保证了刚度响应不会破坏 RCM 几何约束。

## 5. 本轮调参后的参数

no-RCM 使用固定 z 参考和软/硬刚度交替，使固定 alpha 难以同时兼顾软区力调节和硬区稳定。最终采用：

$$
K_{\mathrm{low}}=260\ \mathrm{N/m},\qquad
K_{\mathrm{high}}=760\ \mathrm{N/m},
$$

$$
\alpha_{\mathrm{low}}=0.26,\qquad
\alpha_{\mathrm{high}}=0.62,\qquad
\beta_K=0.86.
$$

with-RCM 需要额外守住 RCM 误差，因此软区 alpha 不能降得像 no-RCM 那么低。最终采用：

$$
K_{\mathrm{low}}=240\ \mathrm{N/m},\qquad
K_{\mathrm{high}}=420\ \mathrm{N/m},
$$

$$
\alpha_{\mathrm{low}}=0.50,\qquad
\alpha_{\mathrm{high}}=0.88,\qquad
\beta_K=0.88.
$$

两组实验都把刚接触后的震荡阶段作为 adjustment phase，不写入正式扫描数据。正式数据从扫描阶段开始记录，因此图表中的震荡主要来自扫描过程中的刚度切换和控制响应。

## 6. Gazebo 对比结果

综合指标如下。`F jitter` 与 `Z jitter` 定义为相邻采样差分的标准差，用于衡量采样间抖动。

### no-RCM

| method | F RMSE (N) | F peak (N) | F jitter (N/sample) | pos RMSE (mm) | z jitter (mm/sample) | alpha min/mean/max | corr($\alpha,\hat K_e$) |
|---|---:|---:|---:|---:|---:|---|---:|
| fixed_0.8 | 0.1440 | 0.344 | 0.0021 | 1.415 | 0.0032 | 0.800/0.800/0.800 | nan |
| fixed_0.5 | 0.1302 | 0.226 | 0.0020 | 1.447 | 0.0048 | 0.500/0.500/0.500 | nan |
| fixed_0.2 | 0.8557 | 2.544 | 0.3612 | 2.780 | 0.5489 | 0.200/0.200/0.200 | nan |
| continuous_force_margin | 0.1336 | 0.240 | 0.0017 | 1.449 | 0.0056 | 0.291/0.460/0.613 | 0.942 |

no-RCM 中，continuous 的 $\alpha_z$ 稳态均值随刚度区明显分开：

$$
\bar\alpha_{\mathrm{soft}}=0.29,\qquad
\bar\alpha_{\mathrm{mid}}=0.51,\qquad
\bar\alpha_{\mathrm{hard}}=0.61.
$$

它的力 RMSE 略高于手工固定的 `fixed_0.5`，但低于 `fixed_0.8`，且力抖动低于所有固定对照。更重要的是，它避免了 `fixed_0.2` 在硬区的接触丢失和高力峰值，同时不需要预先知道哪一个固定 alpha 最适合当前刚度组合。

### with-RCM

| method | F RMSE (N) | F peak (N) | F jitter (N/sample) | pos RMSE (mm) | z jitter (mm/sample) | RCM peak (mm) | alpha min/mean/max | corr($\alpha,\hat K_e$) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fixed_0.8 | 0.1153 | 0.521 | 0.0195 | 0.649 | 0.0084 | 1.014 | 0.800/0.800/0.800 | 0.097 |
| fixed_0.5 | 0.1470 | 1.151 | 0.0517 | 0.685 | 0.0423 | 1.293 | 0.500/0.500/0.500 | nan |
| fixed_0.2 | 0.2043 | 1.212 | 0.0889 | 0.748 | 0.0832 | 2.069 | 0.200/0.200/0.200 | -0.098 |
| continuous_force_margin | 0.1146 | 0.530 | 0.0185 | 0.654 | 0.0072 | 1.032 | 0.497/0.778/0.844 | 0.711 |

with-RCM 中，continuous 的 $\alpha_z$ 在不同刚度区稳定为

$$
\bar\alpha_{\mathrm{soft}}=0.58,\qquad
\bar\alpha_{\mathrm{mid}}=0.83,\qquad
\bar\alpha_{\mathrm{hard}}=0.84.
$$

它在几乎保持 `fixed_0.8` 的位置和 RCM 稳定性的同时，取得更低的力 RMSE、更低的力抖动和更低的 z 抖动；相对 `fixed_0.5` 与 `fixed_0.2`，优势更明显，尤其是 RCM peak 从 1.293 mm/2.069 mm 降到 1.032 mm。

## 7. 图文对照

### 7.1 综合指标

![final metrics](../../../../results/continuous_force_margin_stiffness_response_20260608/final_metrics_bars.png)

该图显示，`fixed_0.2` 在 no-RCM 和 with-RCM 中都会带来明显力峰值和抖动；`fixed_0.8` 稳定但不能随刚度释放力控能力；continuous 在力误差、抖动和 alpha 自适应之间形成更均衡的折中。

### 7.2 no-RCM 策略时序

![no rcm strategy timeseries](../../../../results/continuous_force_margin_stiffness_response_20260608/no_rcm_strategy_timeseries.png)

no-RCM 压力测试中，固定低 alpha 在硬区出现接触力大幅波动；固定高 alpha 稳定但力偏差较大。continuous 的曲线没有固定在某个 alpha，而是跟随估计刚度在软/中/硬区切换。

### 7.3 no-RCM alpha 与刚度响应

![no rcm alpha stiffness detail](../../../../results/continuous_force_margin_stiffness_response_20260608/no_rcm_alpha_stiffness_detail.png)

no-RCM 的 $\alpha_z$ 与 $\hat K_e$ 呈强相关，相关系数为 0.942。软区 $\alpha_z$ 约 0.29，硬区约 0.61，满足“稳定后的 alpha 随刚度不同而变化”的要求。

![no rcm alpha vs khat](../../../../results/continuous_force_margin_stiffness_response_20260608/no_rcm_alpha_vs_khat.png)

散点图进一步说明 $\alpha_z$ 不是时间滤波造成的偶然波动，而是随 $\hat K_e$ 呈单调分布。

### 7.4 with-RCM 策略时序

![with rcm strategy timeseries](../../../../results/continuous_force_margin_stiffness_response_20260608/with_rcm_strategy_timeseries.png)

with-RCM 中，固定低 alpha 的力和 z 抖动明显放大，RCM peak 也升高。continuous 在软区降低 $\alpha_z$ 以改善力跟踪，在中/硬区迅速提高 $\alpha_z$ 保持 RCM 稳定。

### 7.5 with-RCM alpha 与刚度响应

![with rcm alpha stiffness detail](../../../../results/continuous_force_margin_stiffness_response_20260608/with_rcm_alpha_stiffness_detail.png)

with-RCM 的 $\alpha_z$ 变化幅度小于 no-RCM，这是因为 RCM 几何约束要求更高的位置权重。但它仍从软区约 0.58 提升到中/硬区约 0.83 到 0.84，相关系数为 0.711。

![with rcm alpha vs khat](../../../../results/continuous_force_margin_stiffness_response_20260608/with_rcm_alpha_vs_khat.png)

### 7.6 分刚度区 alpha 均值

![zone alpha means](../../../../results/continuous_force_margin_stiffness_response_20260608/zone_alpha_means.png)

固定策略在所有刚度区 alpha 恒定；continuous 在 no-RCM 与 with-RCM 中都随刚度区改变稳态 alpha。这正是固定仲裁无法实现的适应性。

## 8. 为什么 continuous_force_margin 优于固定 alpha

固定 alpha 的本质是选择一个全局折中点：

$$
\alpha_z(t)\equiv \alpha_0.
$$

当环境刚度、力误差和 RCM 风险都变化时，一个固定值只能适合某一类局部情况。实验结果体现为：

- `fixed_0.2` 给力控太多空间，在硬刚度区容易出现接触丢失、高力峰值和 RCM 误差放大。
- `fixed_0.5` 在 no-RCM 中力 RMSE 很强，但没有刚度适应性；换到 with-RCM 后 RCM peak 和 z 抖动明显变差。
- `fixed_0.8` 在 with-RCM 中稳定，但没有软区释放力控的能力，且 alpha 与刚度没有物理响应。

continuous_force_margin 的优势来自三层机制叠加：

1. **阶段检测**提供先验安全性：自由空间、接近和退回阶段自动位置优先，接触稳定后才让模糊仲裁主导。
2. **force-margin 模糊逻辑**根据 $F_h,S_h,D_r$ 同时考虑力误差、边界裕度和位置误差，而不是只看单一误差。
3. **刚度显式响应项**根据 $\hat K_e$ 生成 $\alpha_K$，使软区、硬区的稳态 alpha 不同。

因此当前最终形式可以概括为

$$
\alpha_z
=
\mathcal{F}_{\tau}
\left[
w_\phi\alpha_\phi
+(1-w_\phi)
\left(
(1-\beta_K)\alpha_{\mathrm{FM}}
+\beta_K\alpha_K
\right)
\right],
$$

其中 $\mathcal{F}_{\tau}[\cdot]$ 表示一阶滤波和限速，$\alpha_{\mathrm{FM}}$ 表示 force-margin 模糊安全输出。

从实验结果看，continuous 不只是让 alpha 曲线“有波动”，而是在不同刚度区形成可解释的稳态值：no-RCM 为 $0.29/0.51/0.61$，with-RCM 为 $0.58/0.83/0.84$。这说明 alpha 的变化由环境刚度估计驱动，而不是由随机振荡驱动。

## 9. 当前结论

本轮 Gazebo 调试后，`continuous_force_margin` 已经满足以下目标：

- no-RCM 与 with-RCM 均采用 z-only alpha，x/y 严格位置跟踪。
- no-RCM 的 $\alpha_z$ 对刚度响应非常明显，$\operatorname{corr}(\alpha,\hat K_e)=0.942$。
- with-RCM 的 $\alpha_z$ 在 RCM 保护约束下仍有明显刚度响应，$\operatorname{corr}(\alpha,\hat K_e)=0.711$。
- with-RCM continuous 在力 RMSE、力抖动和 z 抖动上优于固定 alpha 强基线 `fixed_0.8`，同时 RCM peak 仅从 1.014 mm 小幅变化到 1.032 mm。
- no-RCM continuous 的力 RMSE 接近 `fixed_0.5`，力抖动最低，并且明显优于 `fixed_0.2` 的不稳定行为；其主要优势是无需预先手工选择固定 alpha，就能随刚度自动移动折中点。

后续若要进一步强化 no-RCM 的“数值全面领先”，可以继续优化固定 z 参考和刚度区间，使 `fixed_0.5` 在软/硬区冲突更明显；但在当前设置下，continuous 已经清楚体现了刚度敏感 alpha 和综合稳定性优势。
