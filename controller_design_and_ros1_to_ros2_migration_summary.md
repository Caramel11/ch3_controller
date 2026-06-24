# no-RCM 控制器设计与 ROS1 到 ROS2 迁移总结

生成时间: 2026-06-24

## 1. 对话工作主线总结

本轮长对话从本地工作站配置开始，逐步转入 Franka FR3/Panda 控制器仿真与 ROS2 迁移。主要内容包括:

- 本地环境配置: 安装微信、配置 MX Anywhere2S 鼠标侧键复制/粘贴、蓝牙开机自启动和自动连接。
- 从源码安装 `franka_ros2`，工作区放在 `/home/liu/franka_ros2_ws`。
- 安装和配置 MuJoCo/Gazebo 仿真环境，后续调试重点切换到 Gazebo，而不使用真实机器人。
- 参考 ROS1 仓库 `Caramel11/panda_robot` 的 `gt-controller-dev` 分支，尤其是 `tests/0526_controller` 和 `tests/0603` 中的 no-RCM Phase 切换与控制代码。
- 将 ROS1 no-RCM 控制逻辑迁移为 ROS2 下的 `ch3_controller` 包，并在 Gazebo 中通过 `ros2_control` effort controller 运行。
- 用 Pinocchio 替代原 ROS1 控制器中的运动学/动力学接口，提供 FK、Jacobian、姿态误差和必要的动力学计算。
- 解决 Gazebo 启动后机械臂因重力下坠的问题: 零重力 world 启动，重力补偿控制器 active 后再恢复正常重力。
- 反复调试 no-RCM 控制器的接近、接触、扫描、复位阶段，重点处理:
  - 初始位置不正确。
  - 冗余自由度导致机械臂姿态扭曲。
  - 末端需要竖直向下且正面朝前。
  - 末端关节自转抖动导致 y 轴位置误差变大。
  - z 轴下降阶段停在 0.4 m 以上，无法接近目标 0.3 m。
  - 扫描后复位阶段不稳定或抬高到错误位置。
- 最终采用自写控制器完成复位，不使用官方归位程序；复位阶段只跟踪位置和姿态，不跟踪力。
- 本轮 Gazebo 可视化验证结果显示: 控制器可完成启动、接近、扫描、复位，复位终点回到初始末端笛卡尔位置附近。

## 2. 当前控制器结构

核心文件:

- `/home/liu/franka_ros2_ws/src/ch3_controller/run_no_rcm.py`
- `/home/liu/franka_ros2_ws/src/ch3_controller/panda_robot.py`
- `/home/liu/franka_ros2_ws/src/ch3_controller/ch3_controller/robot_interface_ros2.py`
- `/home/liu/franka_ros2_ws/src/ch3_controller/ch3_controller/pinocchio_model.py`
- `/home/liu/franka_ros2_ws/src/ch3_controller/ch3_controller/gazebo_gravity_handoff.py`
- `/home/liu/franka_ros2_ws/src/franka_ros2/franka_gazebo_bringup/launch/gazebo_franka_arm_startup_gravity_comp.launch.py`

当前控制器以 ROS2 Python 包 `ch3_controller` 形式组织，保留部分 ROS1 风格调用习惯，同时在底层替换为 ROS2 topic、parameter、time、logging、`ros2_control` 和 Pinocchio 模型。

## 3. 当前 no-RCM 控制器设计关键点

### 3.1 Gazebo 启动与重力补偿 handoff

Gazebo 仿真启动阶段先使用零重力 world，启动并激活 `gravity_compensation_example_controller`，再由 `gazebo_gravity_handoff` 节点恢复正常重力。这样避免启动初期机械臂还没有控制力矩时被重力拉下。

该方案的关键点:

- world 初始重力为 0。
- 先加载 robot、controller manager 和重力补偿控制器。
- 轮询确认 `gravity_compensation_example_controller` 为 `active`。
- 调用 `/world/empty_no_gravity/set_physics` 恢复 `z=-9.8`。
- 后续切换到 `no_rcm_effort_controller`。

### 3.2 控制器接口

Gazebo 中当前使用 effort command topic:

```text
/no_rcm_effort_controller/commands
```

状态来自:

```text
/joint_states
```

运行 `run_no_rcm` 时需要显式指定:

```bash
-p cmd_topic:=/no_rcm_effort_controller/commands
-p state_topic:=/joint_states
-p rsp_node:=/robot_state_publisher
-p gravity_compensation_scale:=0.0
```

`gravity_compensation_scale:=0.0` 是当前 Gazebo 控制链路下的经验结论，不代表真实机器人也使用该值。

### 3.3 期望末端姿态

当前期望姿态取整数角度:

```text
roll = -90 deg
pitch = 0 deg
yaw = -45 deg
```

该姿态对应末端竖直向下，并使正面朝向视觉上接近正前方。控制律中应优先使用 SO(3) 旋转误差，不应直接用欧拉角差作为控制量。欧拉角主要用于配置、日志和绘图。

### 3.4 去除早期强零空间姿态保持

早期尝试过用零空间姿态保持抑制冗余自由度漂移，但该项与笛卡尔位置、力、姿态任务竞争后会放大振荡，尤其会诱发末端左右自转和 y 轴误差。因此当前主控制策略不再使用强零空间姿态保持控制器。

当前原则:

- 末端姿态通过笛卡尔空间姿态任务控制。
- 冗余自由度只做温和限幅和速度阻尼。
- 避免在扫描阶段对腕部关节施加强而突变的直接正则项。
- 所有关节/力矩修正都必须经过限幅和滤波。

### 3.5 Phase 结构

当前 no-RCM 逻辑继承 ROS1 版本的 Phase 思想，但按 ROS2/Gazebo 的实际动力学做了稳健化:

1. 初始安全移动与姿态预对齐。
2. 粗下降接近阶段，使用实际跟随参考，避免目标跳变。
3. 接触调整阶段，等待接触力、高度和平面误差满足条件。
4. 扫描阶段，执行连续力裕度仲裁 `continuous_force_margin`，同时保持位置、力和姿态稳定。
5. 复位阶段，使用自写位置控制器，不跟踪力，保持末端竖直向下且正面朝前，返回初始末端位置。

### 3.6 接近阶段

接近阶段要解决两个问题:

- 不能因为过早的切换条件卡在 z=0.4 m 以上。
- 不能以过快速度冲向接触面。

当前采用:

- 由当前实际位置生成限速参考。
- xy 平面误差、z 高度和接触力共同决定是否进入接触调整。
- 姿态控制保持 `[-90, 0, -45] deg`。
- 力矩和速度均限幅。

### 3.7 扫描阶段

扫描阶段使用 `continuous_force_margin`:

- 位置任务负责 xy 路径跟踪。
- z 方向接触由力误差和虚拟阻抗共同调节。
- `alpha` 连续变化，避免硬切换造成瞬时力矩跳变。
- 记录 force、position、orientation、front-axis、joint posture、jitter 等指标。

本轮结果:

- 位置 RMSE: 2.644 mm
- 力 RMSE: 0.087 N
- 正面朝前轴 RMSE: 1.015 deg
- 姿态抖动 RMS: 0.045 deg

### 3.8 复位阶段

复位阶段不再使用官方归位程序，也不跟踪力。原因是官方控制器与当前 no-RCM effort 控制器的切换、目标语义、姿态约束和仿真接触状态并不完全一致，反而容易引入不稳定。

当前复位控制律:

1. 从扫描终点开始记录当前位姿。
2. `clear_contact`: 小幅抬升到 `clear_z`，解除接触。
3. `return_home`: 以限速笛卡尔路径回到初始末端位置。
4. `settle`: 在终点附近只做位置和姿态收敛。

复位阶段继续保持末端姿态 `[-90, 0, -45] deg`。最新验证中约 7.24 s 完成复位，到达时位置误差 3.66 mm，姿态误差约 2 deg。

## 4. ROS1 控制器迁移到 ROS2 的详细方案

### 4.1 先做代码盘点

迁移前应从 ROS1 仓库中整理:

- 主入口脚本，例如 `run_no_rcm.py`。
- 控制律模块，例如力控制、阻抗控制、模糊控制、卡尔曼滤波、Phase 切换。
- 机器人接口，例如 `PandaArm`、`PandaKinematics`、真实机器人/仿真机器人适配层。
- 外部依赖，例如动力学库、TF、消息类型、服务、launch 文件。
- ROS1 topic、service、action、parameter、frame 名称。
- 关节顺序、末端 frame、工具坐标系和接触坐标系定义。

不要一开始就逐行翻译。应先确认 ROS1 控制器的数学结构和状态机，再按 ROS2 的运行环境重建接口。

### 4.2 创建 ROS2 功能包

建议使用 `ament_python`:

```bash
cd /home/liu/franka_ros2_ws/src
ros2 pkg create ch3_controller --build-type ament_python --dependencies rclpy sensor_msgs std_msgs geometry_msgs
```

实际包内应包含:

- `package.xml`: 声明 ROS2 依赖。
- `setup.py`: 注册 console scripts。
- `ch3_controller/`: Python 模块。
- `run_no_rcm.py`: 可执行入口。
- 分析脚本: 绘图、实验指标统计。
- launch 或说明文档: Gazebo 启动、控制器切换、实验复现。

### 4.3 替换 rospy 运行时

ROS1 中常见接口和 ROS2 替换关系:

| ROS1 | ROS2 |
| --- | --- |
| `rospy.init_node` | `rclpy.init`, `Node(...)` |
| `rospy.Publisher` | `node.create_publisher` |
| `rospy.Subscriber` | `node.create_subscription` |
| `rospy.Rate` | `node.create_rate` 或基于时钟循环 |
| `rospy.Time.now()` | `node.get_clock().now()` |
| `rospy.get_param` | `declare_parameter`, `get_parameter` |
| `rospy.loginfo` | `node.get_logger().info` |
| `rospy.is_shutdown()` | `rclpy.ok()` |

如果 ROS1 控制器代码较大，可以先写薄兼容层，例如 `rospy.py` 或 `ros_compat.py`，但长期建议核心控制器只依赖明确的 ROS2 adapter，而不是到处混用兼容接口。

### 4.4 重建机器人接口

ROS1 代码通常依赖 `panda_robot` 这类高级 API。ROS2 迁移时建议保留上层控制器调用习惯，但重写底层:

- 从 `/joint_states` 订阅关节位置、速度、力矩。
- 通过 `ros2_control` 的 effort controller 发布 `Float64MultiArray` 力矩命令。
- 用 Pinocchio 从 URDF 构建模型，提供 FK、Jacobian、姿态误差和重力项。
- 对外提供与旧代码类似的方法，例如:
  - `angles()`
  - `joint_velocities()`
  - `ee_pose()`
  - `jacobian()`
  - `exec_torque_cmd()`

这样可以减少控制律本身的改动，把迁移风险集中在 adapter 层。

### 4.5 关节顺序必须按名字映射

Gazebo `/joint_states` 中关节顺序不一定等于 URDF 或控制器期望顺序。迁移时必须用 joint name 映射到 canonical order，例如:

```text
fr3_joint1 ... fr3_joint7
```

不要假设 `/joint_states.position[0:7]` 就是正确顺序。关节顺序错误会直接导致:

- FK/Jacobian 计算错误。
- 力矩命令施加到错误关节。
- 姿态误差看似发散。
- 末端左右自转、y 轴误差和复位失败。

### 4.6 动力学/运动学迁移到 Pinocchio

Pinocchio 在 ROS2 迁移中的职责:

- 从 `robot_description` 或 URDF 文件加载模型。
- 计算末端位姿。
- 计算几何 Jacobian。
- 计算 SO(3) 姿态误差。
- 可选计算重力项、质量矩阵和非线性项。

注意事项:

- 明确 Pinocchio model frame 和 Gazebo/ROS TF frame 的对应关系。
- 确认末端 frame 是 `fr3_hand_tcp`、`fr3_link8` 还是其他工具 frame。
- 姿态控制使用旋转矩阵/四元数误差，避免欧拉角奇异和跳变。
- 仿真中是否叠加 Pinocchio 重力项取决于 `ros2_control` 和 Gazebo 控制器语义，不能照搬真实机器人配置。

### 4.7 ros2_control 与控制器切换

Gazebo 中推荐显式使用 `ros2 control` 管理控制器:

```bash
ros2 control list_controllers
ros2 control switch_controllers --deactivate A --activate B --strict --activate-asap
```

迁移注意事项:

- 同一组关节同一时间通常只能由一个 command controller 控制。
- 从重力补偿切到 no-RCM controller 时，要确认目标 controller 已 loaded 或 inactive。
- 切换后立即检查 `list_controllers`。
- 控制器 topic、接口类型、关节名必须与 YAML 配置一致。

### 4.8 Phase 切换迁移

ROS1 no-RCM 代码中的 Phase 切换不能只按时间迁移，应加入 ROS2/Gazebo 的实际观测条件:

- 接近阶段: 高度、xy 误差、速度和接触力共同判断。
- 接触调整: 先稳定接触，再进入扫描。
- 扫描阶段: 力位混合控制必须连续切换，避免 alpha 硬跳。
- 复位阶段: 先解除接触，再回初始位姿，最后 settle。

Phase 切换时必须保证参考轨迹连续，包括位置参考、姿态参考、速度参考和力参考。硬切目标是大部分振荡的直接原因。

### 4.9 姿态控制迁移

本项目中最关键的经验是: 末端朝向问题应作为笛卡尔空间姿态任务处理，而不是依赖强零空间关节姿态保持。

推荐做法:

- 固定期望姿态 `[-90, 0, -45] deg`。
- 用 SO(3) 误差计算姿态控制量。
- 控制量通过 `J_ori.T` 映射到关节力矩。
- 对角速度和姿态误差进行阻尼。
- 记录 front-axis error，用于判断是否正面朝前。
- 终端 settle 阶段可以适度提高姿态增益，但必须保留阻尼和限幅。

不推荐:

- 直接用欧拉角差作为力矩控制量。
- 用强零空间项硬拉某个关节回初始角。
- 在接触扫描阶段对腕部关节施加突变正则。

### 4.10 力控制迁移

ROS1 到 ROS2 后，力信号来源、滤波和接触模型可能变化。迁移时应:

- 明确仿真力来自接触模型、估计器还是外部传感器。
- 保留 Kalman/低通滤波，但检查滤波延迟。
- 接触前不要让力误差项主导控制。
- 接触后再启用力位仲裁。
- 力控制输出必须限幅，避免穿透或弹跳。
- force target、stiffness、damping 和 alpha 变化必须连续。

### 4.11 复位阶段迁移

复位阶段应与接近阶段一样认真设计，而不是简单切换到另一个控制器。推荐:

- 保存初始末端笛卡尔位置和期望姿态。
- 扫描结束后先 `clear_contact`，解除接触。
- 然后按限速 3D 轨迹回到初始末端位置。
- 复位阶段不跟踪力。
- 复位阶段仍保持末端竖直向下且正面朝前。
- 到达条件同时检查位置误差、姿态误差和关节速度。

若采用官方控制器归位，应先验证控制器切换过程是否平滑、目标关节姿态是否等价于目标末端位姿、以及切换时是否会丢失姿态约束。本项目当前结论是不使用官方归位程序。

## 5. 常见问题与排查

### 5.1 Gazebo 启动后机械臂下坠

原因通常是 Gazebo 动力学已经开始积分，但重力补偿或 effort controller 尚未 active。解决方案是零重力启动，控制器 active 后恢复重力。

### 5.2 z 轴目标 0.3 m，但下降卡在 0.4 m 以上

常见原因:

- 接触切换条件过早触发。
- 高度阈值和力阈值逻辑冲突。
- 接近阶段速度太低或参考没有继续更新。
- 姿态/冗余约束与 z 下降任务竞争。

解决方案:

- 接近阶段用实际跟随参考。
- 接触切换同时检查 z、xy、force。
- 接触前弱化力项和冗余正则。
- 保证下降参考能继续推进到目标附近。

### 5.3 y 轴误差大且末端左右抖动

常见原因:

- 腕部/末端自转没有被笛卡尔姿态任务约束。
- 强零空间关节姿态项和末端任务竞争。
- 关节顺序映射错误。
- 姿态控制缺少角速度阻尼。

解决方案:

- 用 SO(3) 笛卡尔姿态控制保持末端正面朝前。
- 去掉强零空间姿态保持。
- 检查 `/joint_states` 名称映射。
- 增加姿态阻尼和力矩限幅。

### 5.4 扫描后复位不稳定或抬到错误高度

常见原因:

- 复位目标不是初始末端笛卡尔位置。
- 复位阶段仍在跟踪接触力。
- 复位阶段直接跳目标，参考不连续。
- 官方控制器切换后目标语义不一致。

解决方案:

- 保存初始末端位姿作为唯一复位目标。
- 复位阶段只跟踪位置和姿态，不跟踪力。
- 采用 `clear_contact -> return_home -> settle` 三段式。
- 保留姿态控制，禁止末端复位时自转。

## 6. 后续优化建议

- 在不重新引入强零空间振荡的前提下，加入温和关节姿态代价，降低最大关节漂移。
- 对扫描阶段力误差的负偏差进行小幅补偿，使平均力更接近 1.0 N。
- 在复位 `settle` 阶段增加短时终端姿态增益，将终点姿态误差从约 2 deg 进一步压低。
- 将 Gazebo 调试命令整理成 launch 或脚本，自动完成启动、controller switch、运行、绘图和报告生成。
- 将分析脚本输出的 CSV 指标作为回归测试门槛，避免后续改控制律时重新引入末端抖动或复位失败。

