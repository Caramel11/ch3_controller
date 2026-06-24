# 0603 RCM / no-RCM controller interface alignment

本目录用于保留 2026-06-03 的 RCM 与 no-RCM 对照版本，所有文件均独立放在
`tests/0603/` 下，不修改原始 `0526_controller` 或 `cooperative_gt_0428`
目录。

## ROS 2 移植包入口

本包保留原脚本的 `import rospy`、`from panda_robot import PandaArm` 和
`from src...` 写法；ROS 2 相关逻辑集中在 `rospy.py`、`panda_robot.py` 和
`ch3_controller/` 适配层中。

```bash
cd /home/liu/franka_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ch3_controller
source install/setup.bash
ros2 run ch3_controller run_no_rcm_real --controller-mode pareto_iter
ros2 run ch3_controller run_with_rcm_real --controller-mode pareto_iter
```

默认 ROS 2 接口参数:

- effort command: `/NS_1/joint_group_effort_controller/commands`
- joint state: `/NS_1/joint_states`
- robot description: `/NS_1/robot_state_publisher`
- frame alias: `panda_link8/10/11` 会映射为 `fr3_link8/10/11`

## 给全新官方 franka_ros2 环境

如果师姐使用的是同一台机械臂和同一套末端工具，可以把本工作区里已经适配过的
`franka_description` 一起给她覆盖。覆盖前先让她备份官方原包，便于回退：

```bash
cd <her_franka_ros2_ws>/src
cp -a franka_description franka_description.official_backup
```

本控制器依赖 `robot_state_publisher` 发布的 `robot_description` 中存在
`fr3_link8`、`fr3_link10`、`fr3_link11`。本工作区的
`franka_description/robots/fr3/fr3.urdf.xacro` 已经加入了 `fr3_link10` 和
`fr3_link11` 这两个固定工具 frame；如果她覆盖了这份 `franka_description`，
就不需要另外改 URDF。

需要给她的最小文件:

- `ch3_controller/`: 本移植包，包含原 0618 代码和 ROS 2 适配层。
- `franka_description/`: 同一台机械臂/同一套工具时建议一起给，用于提供
  `fr3_link10`、`fr3_link11` 和 Gazebo effort 接口相关 xacro。
- `franka_bringup/config/controllers.yaml`: 真机运行时建议给，里面包含
  `joint_group_effort_controller` 配置。
- `franka_bringup/config/franka.config.yaml`: 如果她也使用 `/NS_1` 命名空间，可以给；
  否则让她保留自己的命名空间并在运行命令中覆盖 topic 参数。
- `rcm_bringup/`: 可选，但建议给；用于复现本机 Gazebo effort-controller 测试流程。

如果不用 `rcm_bringup`，她自己的 bringup/launch 至少需要加载一个
`JointGroupEffortController`，名字可以不同，但运行 `ch3_controller` 时要把
`cmd_topic` 参数指向它的 command topic。典型 controller YAML:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    joint_group_effort_controller:
      type: effort_controllers/JointGroupEffortController

joint_group_effort_controller:
  ros__parameters:
    joints:
      - fr3_joint1
      - fr3_joint2
      - fr3_joint3
      - fr3_joint4
      - fr3_joint5
      - fr3_joint6
      - fr3_joint7
```

新环境检查流程:

```bash
source install/setup.bash
ros2 control list_controllers
ros2 topic list | grep -E 'joint_states|joint_group_effort_controller'
ros2 param get /robot_state_publisher robot_description | grep fr3_link11
```

如果使用 `/NS_1` 命名空间，最后一条改为：

```bash
ros2 param get /NS_1/robot_state_publisher robot_description | grep fr3_link11
```

如果 Gazebo/官方 launch 没有命名空间，运行测试时覆盖为根命名空间:

```bash
ros2 run ch3_controller test_real_mock_sensor_gazebo \
  --mode no_rcm \
  --trials 1 \
  --quick-scan-length 0.006 \
  --controller-mode are \
  --ros-args \
  -p cmd_topic:=/joint_group_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher
```

本机官方 `franka_gazebo_bringup` 使用的 effort controller 名称是
`no_rcm_effort_controller`。no-RCM 调试建议使用无重力 world；普通
`empty.sdf` 下 effort controller 在收到命令前会让机械臂受重力下落，导致
启动姿态偏离初始位。启动 Gazebo 可视化仿真:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
  robot_type:=fr3 \
  load_gripper:=false \
  rviz:=false \
  gz_args:="-r /home/liu/franka_ros2_ws/install/franka_gazebo_bringup/share/franka_gazebo_bringup/worlds/empty_no_gravity.sdf" \
  controller:=no_rcm_effort_controller
```

如果只做自动化测试、不需要可视化窗口，把 `gz_args` 改成
`"-r -s /home/liu/franka_ros2_ws/install/franka_gazebo_bringup/share/franka_gazebo_bringup/worlds/empty_no_gravity.sdf"`。
另开终端检查:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 control list_controllers
ros2 topic list | grep -E 'joint_states|no_rcm_effort_controller'
ros2 param get /robot_state_publisher robot_description | grep fr3_link11
```

在本机 Gazebo 中运行 mock 传感器测试:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 run ch3_controller test_real_mock_sensor_gazebo \
  --mode no_rcm \
  --trials 1 \
  --quick-scan-length 0.004 \
  --controller-mode are \
  --output-dir /home/liu/franka_ros2_ws/results/ch3_controller_debug \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=0.0
```

在本机 Gazebo 中运行 no-RCM 控制器:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 run ch3_controller run_no_rcm \
  --strategy continuous_force_margin \
  --controller-mode are \
  --trials 1 \
  --output-dir /home/liu/franka_ros2_ws/results/ch3_controller_debug \
  --no-auto-plot \
  --plot-no-show \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=0.0
```

ROS 2 适配层默认在发布 effort 前使用 Pinocchio 叠加重力补偿；本机无重力
Gazebo world 中应使用 `gravity_compensation_scale:=0.0`，避免额外重力项影响
仿真。如果切到已经内部补重力的硬件接口，可在运行命令后追加
`-p add_gravity_compensation:=false`。

### 有重力 Gazebo 启动

`no_rcm_effort_controller` 是 effort forward controller，本身只把
`/no_rcm_effort_controller/commands` 的力矩原样送到 Gazebo；有重力
`empty.sdf` 中，如果没有命令发布器持续发力矩，机械臂会下坠。调试有重力补偿
时，先启动启动持位节点，让它等待 Gazebo 中的 `robot_description` 和
`/joint_states`。它会用 Pinocchio 计算当前关节重力矩，并加小的关节 PD，把
机械臂保持在 ROS1 no-RCM 初始位附近:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 run ch3_controller gravity_hold \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=1.0 \
  -p target_mode:=initial \
  -p max_tau_abs:=45.0 \
  -p max_tau_rate:=300.0
```

另开终端，启动普通有重力 Gazebo:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 launch franka_gazebo_bringup gazebo_franka_arm_example_controller.launch.py \
  robot_type:=fr3 \
  load_gripper:=false \
  rviz:=false \
  gz_args:="-r empty.sdf" \
  controller:=no_rcm_effort_controller
```

确认 Gazebo 里机械臂已经稳定后，停止 `gravity_hold`，立即启动 no-RCM。no-RCM
启动后会先发布约 1 秒零控制力矩，由 ROS 2 适配层叠加重力补偿，避免交接瞬间
掉落。此时不要关闭重力补偿:

```bash
source /opt/ros/humble/setup.bash
cd /home/liu/franka_ros2_ws
source install/setup.bash
ros2 run ch3_controller run_no_rcm \
  --strategy continuous_force_margin \
  --controller-mode are \
  --trials 1 \
  --output-dir /home/liu/franka_ros2_ws/results/ch3_controller_gravity_debug \
  --no-auto-plot \
  --plot-no-show \
  --ros-args \
  -p cmd_topic:=/no_rcm_effort_controller/commands \
  -p state_topic:=/joint_states \
  -p rsp_node:=/robot_state_publisher \
  -p gravity_compensation_scale:=1.0
```

no-RCM 脚本结束后不再持续发布 effort。若要继续保持可视化窗口中的姿态，重新
启动 `gravity_hold` 接管；如果实验已结束，直接停止 Gazebo 即可。

如果真机 bringup 使用 `/NS_1` 命名空间，则可以使用默认值，或显式写:

```bash
ros2 run ch3_controller run_no_rcm_real --controller-mode pareto_iter \
  --ros-args \
  -p cmd_topic:=/NS_1/joint_group_effort_controller/commands \
  -p state_topic:=/NS_1/joint_states \
  -p rsp_node:=/NS_1/robot_state_publisher
```

## 文件说明

- `run_no_rcm.py`: 基于 `tests/0526_controller/run_no_rcm.py` 复制生成，默认
  `--strategy continuous_force_margin`，保留 0526 的 `--controller-mode`
  接口。
- `run_with_rcm.py`: 基于 `tests/cooperative_gt_0428/run_with_rcm.py` 复制生成，
  控制器接口参考 0526 更新为 `CooperativeGameController(control_mode=...)`，
  并新增 `--controller-mode {are,pareto_iter}`。RCM 几何、trocar 约束、
  flange-space 力矩装配、`error_rcm` 日志均保留。接近阶段采用限速参考:
  先慢速对齐扫描起点 x/y，再慢速下降，以降低起点接近过程中的速度峰值和
  RCM 误差。如果初始 tool z 已低于配置的 `scan_z`，脚本会自动使用当前
  高度下方的小距离作为局部接触阈值，避免一开始就误判进入扫描。接近段调试
  数据会保存到结果目录的 `approach_debug/` 子目录。
- `run_no_rcm_0526_reference.py`: 原始 0526 no-RCM 参考文件。
- `run_with_rcm_0428_reference.py`: 原始 cooperative_gt_0428 RCM 参考文件。
- `src/`: 从 0526 版本复制的控制器、调度器、估计器、机器人接口和数据记录依赖，
  包含 `pareto_iter` 和 `ContinuousForceMarginFuzzyAlphaScheduler`。
- `fuzzy_logic.py`, `kalman_filter.py`: `src/alpha_scheduler_gt.py` 运行所需的顶层
  依赖文件。
- `plot_latest_result.py`: 默认绘制最近一次实验 `.npz`，保存 overview 图和
  summary CSV。
- `plot_arbitration_compare.py`: 分析对照多组仲裁方法结果，按策略汇总 trial
  指标，保存对照图、时序叠加图和 CSV。

## 推荐命令

```bash
cd /home/liu/franka_ws_1101/src/panda_robot/tests/0603
python run_no_rcm.py --controller-mode pareto_iter
python run_with_rcm.py --controller-mode pareto_iter
```

如需关闭力传感器，仅使用虚拟环境回退运行 RCM 版本:

```bash
python run_with_rcm.py --controller-mode pareto_iter --no-force-sensor
```

默认策略均为 `continuous_force_margin`。也可以显式指定:

```bash
python run_with_rcm.py --strategy continuous_force_margin --controller-mode pareto_iter
python run_no_rcm.py --strategy continuous_force_margin --controller-mode pareto_iter
```

实验脚本默认会在结束后自动调用 `plot_latest_result.py`；当结果目录下存在多个
`.npz` 文件时，还会调用 `plot_arbitration_compare.py`。如只想保存图片不弹窗:

```bash
python run_with_rcm.py --controller-mode pareto_iter --plot-no-show
python run_no_rcm.py --controller-mode pareto_iter --plot-no-show
```

如需关闭自动绘图:

```bash
python run_with_rcm.py --controller-mode pareto_iter --no-auto-plot
```

单独绘制最近一次实验:

```bash
python plot_latest_result.py
python plot_latest_result.py --input /path/to/result_dir --no-show
```

多组仲裁方法对照:

```bash
python run_with_rcm.py --strategy all --trials 3 --controller-mode pareto_iter --plot-no-show
python plot_arbitration_compare.py --input /path/to/result_dir --no-show
```
