from setuptools import find_packages, setup


package_name = "ch3_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    py_modules=[
        "rospy",
        "panda_robot",
        "fuzzy_logic",
        "kalman_filter",
        "analyze_pose_tracking_result",
        "plot_arbitration_compare",
        "plot_latest_result",
        "run_no_rcm",
        "run_no_rcm_builtin_phases",
        "run_no_rcm_0526_reference",
        "run_no_rcm_force_margin_challenge",
        "run_no_rcm_real",
        "run_with_rcm",
        "run_with_rcm_0428_reference",
        "run_with_rcm_force_margin_challenge",
        "run_with_rcm_real",
        "test_real_mock_sensor_gazebo",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name,
            [
                "README.md",
                "STIFFNESS_ESTIMATION_METHOD.md",
                "continuous_force_margin_arbitration_report.md",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="done",
    maintainer_email="2796708472@qq.com",
    description="ROS 2 compatibility package for the 0618 ch3 controller scripts.",
    license="TODO",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "run_no_rcm = ch3_controller.entrypoints:run_no_rcm",
            "run_no_rcm_builtin_phases = ch3_controller.entrypoints:run_no_rcm_builtin_phases",
            "run_with_rcm = ch3_controller.entrypoints:run_with_rcm",
            "run_no_rcm_real = ch3_controller.entrypoints:run_no_rcm_real",
            "run_with_rcm_real = ch3_controller.entrypoints:run_with_rcm_real",
            (
                "run_no_rcm_force_margin_challenge = "
                "ch3_controller.entrypoints:run_no_rcm_force_margin_challenge"
            ),
            (
                "run_with_rcm_force_margin_challenge = "
                "ch3_controller.entrypoints:run_with_rcm_force_margin_challenge"
            ),
            (
                "test_real_mock_sensor_gazebo = "
                "ch3_controller.entrypoints:test_real_mock_sensor_gazebo"
            ),
            "gravity_hold = ch3_controller.gravity_hold:main",
            "gazebo_gravity_handoff = ch3_controller.gazebo_gravity_handoff:main",
        ],
    },
)
