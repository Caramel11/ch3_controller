# Alpha stiffness adaptation debug report - 2026-06-24

## Scope

本轮目标是在有重力补偿的 Gazebo 可视化仿真中，分析最近一次实验后半段高刚度区域的力和位置误差波动，并从 `continuous_force_margin` 的 alpha 调节律角度优化 no-RCM 控制器。

使用的启动方式为已验证的同步重力补偿 handoff：

```bash
source /opt/ros/humble/setup.bash
source /home/liu/franka_ros2_ws/install/setup.bash
ros2 launch franka_gazebo_bringup gazebo_franka_arm_startup_gravity_comp.launch.py \
  robot_type:=fr3 \
  load_gripper:=false \
  rviz:=false
```

随后切换控制器：

```bash
ros2 run controller_manager spawner no_rcm_effort_controller \
  --controller-manager /controller_manager \
  --param-file /home/liu/franka_ros2_ws/install/franka_gazebo_bringup/share/franka_gazebo_bringup/config/franka_gazebo_controllers.yaml \
  --inactive \
  --controller-manager-timeout 30 || true

ros2 control switch_controllers \
  --deactivate gravity_compensation_example_controller \
  --activate no_rcm_effort_controller \
  --strict \
  --activate-asap
```

实验命令：

```bash
ros2 run ch3_controller run_no_rcm \
  --strategy continuous_force_margin \
  --controller-mode are \
  --trials 1 \
  --output-dir /home/liu/franka_ros2_ws/results/ch3_controller_alpha_adapt_debug \
  --no-auto-plot \
  --plot-no-show \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=0.0
```

## Root Cause

原实现中，z-only alpha 实际只作用于压入方向，因为 `compute_torque_no_rcm()` 内部使用 `alpha_xyz=[1.0, 1.0, alpha_z]`，x/y 始终由笛卡尔位置控制保持高权重。

但调度器输入 `e_r` 曾使用 xy 跟踪误差：

```python
e_r_scalar = np.linalg.norm(tp[:2] - np.array([x_cur, cfg.scan_y]))
```

这会把约 2.5 mm 的横向滞后误差错误喂给 z 向 alpha。结果是在高刚度区域中，alpha 长时间被推到上限 0.46，z 向力控无法根据欠力状态降低 alpha，导致压入深度被位置参考锁住，形成持续负力误差。

高刚度区域中，极小的 z 位置/速度扰动会被 500 N/m 环境刚度和 Kelvin-Voigt 阻尼放大。若 alpha 饱和，调度器只能维持位置优先，不能适应低力/高力方向，因此后半段会同时看到力误差和位置误差波动。

## Adopted Changes

1. `run_no_rcm.py`
   - alpha 输入改为 z 轴位置误差：

```python
e_r_alpha = abs(float(e_r1[cfg.force_axis]))
```

   - xy 误差仍保留在日志中用于位置跟踪诊断，但不再参与 z 向 alpha 决策。
   - 保留可选的控制用力误差死区/限幅/积分限幅代码路径，但默认关闭，因为本轮实验验证它会造成欠力补偿不足。

2. `src/alpha_scheduler_gt.py`
   - `ContinuousForceMarginFuzzyAlphaScheduler` 新增刚度-力误差方向重平衡：
     - 高刚度且实际力低于期望时，降低 z alpha，给力通道更多权重。
     - 高刚度且实际力高于期望时，提高 z alpha，抑制过压。
   - 当前主程序采用保守参数：

```python
continuous_smooth_tau = 0.80
continuous_stiffness_alpha_enabled = True
continuous_stiffness_low_threshold = 320.0
continuous_stiffness_high_threshold = 500.0
continuous_stiffness_low_alpha = 0.22
continuous_stiffness_high_alpha = 0.44
continuous_stiffness_blend = 0.45
```

## Gazebo Results

完整指标表：

- `alpha_stiffness_adaptation_metrics_20260624.csv`

关键结果如下：

| Run | Zone | Force RMSE (N) | Force std (N) | Peak force err (N) | Position RMSE (mm) | Mean alpha |
|---|---|---:|---:|---:|---:|---:|
| baseline fixed high alpha | all scan | 0.08737 | 0.02569 | 0.16942 | 2.6442 | 0.4592 |
| baseline fixed high alpha | high | 0.07657 | 0.03429 | 0.12785 | 2.5683 | 0.4600 |
| baseline fixed high alpha | tail high | 0.07350 | 0.03594 | 0.12061 | 2.5449 | 0.4600 |
| strong stiffness alpha | all scan | 0.08512 | 0.02860 | 0.16392 | 2.6337 | 0.4215 |
| strong stiffness alpha | high | 0.07226 | 0.03671 | 0.12337 | 2.5471 | 0.3849 |
| strong stiffness alpha | tail high | 0.06907 | 0.03663 | 0.11829 | 2.5277 | 0.3907 |
| selected conservative alpha | all scan | 0.08560 | 0.02677 | 0.16324 | 2.6400 | 0.4350 |
| selected conservative alpha | high | 0.07374 | 0.03551 | 0.12448 | 2.5603 | 0.4148 |
| selected conservative alpha | tail high | 0.07033 | 0.03549 | 0.11651 | 2.5400 | 0.4184 |
| rejected force deadband/limit | all scan | 0.10027 | 0.03021 | 0.24763 | 2.6491 | 0.4193 |
| rejected force deadband/limit | high | 0.08626 | 0.03607 | 0.13554 | 2.5635 | 0.4011 |
| rejected force deadband/limit | tail high | 0.08283 | 0.03621 | 0.13115 | 2.5421 | 0.4046 |

## Figures

绘图文件均由 `plot_latest_result.py` 和 `analyze_pose_tracking_result.py` 生成，保存在各实验结果目录的 `analysis/` 子目录。

### Selected conservative alpha

结果目录：

`/home/liu/franka_ros2_ws/results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis`

![Selected conservative alpha overview](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/continuous_force_margin_alpha_t00_overview.png)

![Selected conservative alpha force tracking](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/pose_force_tracking.png)

![Selected conservative alpha position tracking](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/pose_position_tracking.png)

![Selected conservative alpha end-effector orientation](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/end_effector_orientation.png)

![Selected conservative alpha jitter components](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/tracking_jitter_components.png)

![Selected conservative alpha joint posture drift](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045051/analysis/joint_posture_drift.png)

### Rejected force deadband/limit experiment

该实验用于验证力误差死区/限幅是否能压低高刚度区抖动。结果显示 force RMSE 和峰值误差变差，因此没有作为主程序默认参数。

结果目录：

`/home/liu/franka_ros2_ws/results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045700/analysis`

![Rejected force deadband overview](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045700/analysis/continuous_force_margin_alpha_t00_overview.png)

![Rejected force deadband force tracking](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045700/analysis/pose_force_tracking.png)

![Rejected force deadband jitter components](../../results/ch3_controller_alpha_adapt_debug/no_rcm_20260624_045700/analysis/tracking_jitter_components.png)

## Selected Version

主程序采用 `selected_conservative_alpha` 对应的控制律。

选择理由：

- 相比 baseline，全扫描力 RMSE 从 0.08737 N 降到 0.08560 N。
- 高刚度尾段峰值力误差从 0.12061 N 降到 0.11651 N。
- 强适应版虽然均值误差更低，但力方差更大；保守版在误差和抖动之间更均衡。
- 力误差死区/限幅方案被拒绝，因为它让欠力状态补偿不足，全扫描 force RMSE 升至 0.10027 N。
- 复位阶段仍稳定，约 7 s 回到末端初始笛卡尔位置附近，位置误差约 3.5-3.9 mm，末端正面朝前误差约 2 deg 内。

## Remaining Risk

高刚度段实际力仍存在约 0.03-0.13 N 的采样级波动。当前主要来源不是 alpha 跳变，而是虚拟接触力中的 `delta_dot` 阻尼项和高刚度下的微小 z 位置扰动放大。后续若要进一步降低力抖动，应优先单独评估：

- 只对 `delta_dot` 或虚拟力反馈做很小时间常数的控制用低通，同时保留 raw force 作为评价指标。
- 在高刚度区根据 `K_hat` 降低扫描速度 `scan_vx`，而不是继续降低 alpha。
- 将 `B_hat` 估计只用于诊断，避免阻尼估计瞬态影响增益或调度。
