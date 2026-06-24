"""0526 no-RCM 控制器辅助模块包。

本包由 run_no_rcm.py 引用，包含:
  - alpha_scheduler_gt: alpha 仲裁策略；
  - gt_controller: ARE/Pareto 迭代控制器；
  - robot_interface: 机器人状态与力矩映射；
  - env_estimator/leaky_integrator/utils: 估计、积分、日志等工具。
"""
