# Gazebo no-RCM 控制器可视化调试报告

生成时间: 2026-06-24

## 1. 实验目的

本轮调试目标是在 Gazebo 可视化仿真中验证 `run_no_rcm` 当前控制器版本，重点检查:

- Gazebo 启动时机械臂是否在有重力环境下保持初始姿态，不发生启动后下坠。
- 接近、接触调整、扫描、复位阶段是否能连续稳定运行。
- 复位阶段是否回到末端初始笛卡尔位置与姿态，而不是抬高到不安全位置。
- 绘制位置、力、末端姿态、关节姿态和抖动指标，并基于数据分析误差来源。

## 2. 本地 Gazebo 调试流程

启动 Gazebo 和同步重力补偿:

```bash
source /opt/ros/humble/setup.bash
source /home/liu/franka_ros2_ws/install/setup.bash
ros2 launch franka_gazebo_bringup gazebo_franka_arm_startup_gravity_comp.launch.py \
  robot_type:=fr3 \
  load_gripper:=false \
  rviz:=false
```

该启动流程使用 `empty_no_gravity.sdf`、`gravity_compensation_example_controller` 和 `gazebo_gravity_handoff` 节点。handoff 节点等待重力补偿控制器进入 `active` 后，再通过 `/world/empty_no_gravity/set_physics` 恢复正常重力 `z=-9.8`，避免 Gazebo 已经运行但重力补偿尚未接管的空窗期。

切换到 no-RCM effort 控制器:

```bash
source /opt/ros/humble/setup.bash
source /home/liu/franka_ros2_ws/install/setup.bash
ros2 run controller_manager spawner no_rcm_effort_controller \
  --controller-manager /controller_manager \
  --param-file /home/liu/franka_ros2_ws/install/franka_gazebo_bringup/share/franka_gazebo_bringup/config/franka_gazebo_controllers.yaml \
  --inactive \
  --controller-manager-timeout 30

ros2 control switch_controllers \
  --deactivate gravity_compensation_example_controller \
  --activate no_rcm_effort_controller \
  --strict \
  --activate-asap
```

运行控制器:

```bash
source /opt/ros/humble/setup.bash
source /home/liu/franka_ros2_ws/install/setup.bash
ros2 run ch3_controller run_no_rcm \
  --strategy continuous_force_margin \
  --controller-mode are \
  --trials 1 \
  --output-dir /home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug \
  --no-auto-plot \
  --plot-no-show \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=0.0
```

注意: 在当前 Gazebo `no_rcm_effort_controller` 环境中，`run_no_rcm` 使用 `gravity_compensation_scale:=0.0`。Gazebo 侧已由启动和控制器链路处理重力保持；若在该仿真中再次叠加 Pinocchio 重力力矩，容易造成额外力矩注入和姿态/位置振荡。真实机器人或不同控制接口下需要重新确认该参数。

绘图与分析:

```bash
python3 /home/liu/franka_ros2_ws/src/ch3_controller/plot_latest_result.py \
  --input /home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610 \
  --no-show

python3 /home/liu/franka_ros2_ws/src/ch3_controller/analyze_pose_tracking_result.py \
  --input /home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610
```

## 3. 数据与图表路径

结果目录:

```text
/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610
```

原始数据:

```text
/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/continuous_force_margin_alpha_t00.npz
```

运行日志:

```text
/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/run_no_rcm_gazebo_debug.log
```

生成图表:

- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/continuous_force_margin_alpha_t00_overview.png`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/continuous_force_margin_alpha_t00_overview.pdf`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/pose_position_tracking.png`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/pose_force_tracking.png`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/end_effector_orientation.png`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/joint_posture_drift.png`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/tracking_jitter_components.png`

分析文件:

- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/pose_tracking_metrics.csv`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/continuous_force_margin_alpha_t00_summary.csv`
- `/home/liu/franka_ros2_ws/results/ch3_controller_final_gazebo_debug/no_rcm_20260624_042610/analysis/pose_tracking_analysis_report.md`

## 4. 关键实验结果

本次 Gazebo 运行完整完成，采样数 4992，持续时间 49.91 s。

整体指标:

| 指标 | 数值 |
| --- | ---: |
| 位置 RMSE | 2.644 mm |
| 位置峰值误差 | 2.973 mm |
| x 轴 RMSE | 2.223 mm |
| y 轴 RMSE | 1.413 mm |
| z 轴 RMSE | 0.230 mm |
| 力 RMSE | 0.087 N |
| 力平均误差 | -0.084 N |
| 力峰值误差 | 0.169 N |
| SO(3) 姿态误差 RMSE | 1.174 deg |
| SO(3) 姿态误差峰值 | 1.750 deg |
| 正面朝前轴误差 RMSE | 1.015 deg |
| 正面朝前轴误差峰值 | 1.683 deg |
| 姿态抖动 RMS | 0.045 deg |
| 位置抖动 RMS | 0.071 mm |
| 力抖动 RMS | 0.017 N |
| 关节速度 RMS | 0.0075 rad/s |
| 关节速度峰值 | 0.0179 rad/s |

扫描阶段 `continuous_force_margin` 统计:

| 指标 | 数值 |
| --- | ---: |
| alpha 均值 | 0.459 |
| alpha 最小值 | 0.407 |
| alpha 最大值 | 0.460 |
| 控制输入 RMS | 0.472 |
| RCM RMSE | 0.000 mm |
| RCM 峰值 | 0.000 mm |

复位阶段日志显示:

- 扫描结束位置: `[0.47773676, -0.00124173, 0.29806031]`
- 复位目标，即末端初始位置: `[0.31583767, -0.0209762, 0.48984559]`
- 分阶段复位参数: `clear_z=0.3481 m`, `z_speed=0.0300 m/s`, `return_speed=0.0450 m/s`, `timeout=28 s`
- 复位阶段依次经过 `clear_contact`、`return_home`、`settle`
- 约 7.24 s 到达目标附近
- 到达判据触发时: `pos_err=3.66 mm`, `ori=2.00 deg`, `front=1.98 deg`, `qd=0.0106 rad/s`

## 5. 结果分析

### 5.1 启动重力补偿

本次启动阶段没有观察到机械臂因重力立刻下坠。关键原因是先在零重力 world 中启动 Gazebo 和重力补偿控制器，等待控制器 active 后再恢复正常重力。这解决了此前 Gazebo 已开始积分动力学、但重力补偿尚未接管导致的初始下坠问题。

### 5.2 接近和接触阶段

接近阶段采用接近 ROS1 no-RCM Phase 切换思想的实际跟随参考，不直接使用过快的开环目标跳变。接触切换同时参考高度、平面误差和接触力，减少了在 z 轴 0.4 m 以上提前卡住的问题。接触 handoff 时日志显示 `z=0.2992 m`，接近预期 0.3 m 接触高度。

### 5.3 扫描阶段

扫描阶段位置 RMSE 为 2.64 mm，y 轴 RMSE 为 1.41 mm。此前肉眼可见的末端左右自转抖动已显著降低，姿态抖动 RMS 为 0.045 deg，正面朝前轴峰值误差 1.68 deg。当前剩余位置误差主要来自:

- 扫描方向和末端姿态约束同时作用时，冗余自由度仍会产生轻微任务竞争。
- 力跟踪平均偏差为 -0.084 N，说明实际接触力略低于期望，接触法向阻抗仍偏保守。
- 为避免腕部自转和末端抖动，当前姿态环和阻尼取值较稳健，会牺牲一部分平面跟踪速度。

### 5.4 复位阶段

此前复位不稳定的根因是复位阶段与扫描/接近阶段控制律差异过大，且存在对接触力或不合适高度目标的残留依赖，导致末端不是回到初始笛卡尔位置，而是抬高到错误位置。

当前版本复位阶段采用自写控制器的位置优先控制，不再调用官方归位程序，也不再跟踪接触力。复位逻辑分为:

1. `clear_contact`: 从接触面小幅抬起，先解除接触力。
2. `return_home`: 按限速笛卡尔路径返回初始末端位置。
3. `settle`: 在初始末端位置附近提高收敛精度，降低速度。

复位过程保持期望姿态 `[-90, 0, -45] deg`，即末端竖直向下且正面朝前。最终位置误差 3.66 mm，姿态误差约 2 deg，关节速度约 0.0106 rad/s，说明复位阶段已从“不稳定抬高”修正为“可控返回初始末端位姿”。

## 6. 当前结论

当前版本在 Gazebo 有重力仿真中可以完成:

- 重力补偿启动，无启动后下坠。
- 接近阶段稳定下探到目标接触高度。
- 扫描阶段位置、力、姿态三类指标稳定。
- 复位阶段使用自写位置控制器返回初始末端位置和姿态。
- 末端正面朝前约束有效，没有发生明显不定向或末端自转振荡。

当前仍可继续优化的方向:

- 力跟踪平均误差为 -0.084 N，可适度提高接触法向补偿或调整连续力裕度参数。
- 最大关节偏移约 0.67 rad，说明冗余自由度姿态仍有可优化空间，但不应恢复早期造成振荡的强零空间姿态保持控制器。
- 复位终点姿态误差约 2 deg，若需要更严格终点姿态，可在 `settle` 阶段增加短时姿态终端增益，但要保留角速度阻尼和力矩限幅。

