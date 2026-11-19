import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fast_lio_pkg = get_package_share_directory("fast_lio")
    description_pkg = get_package_share_directory("go2_description")

    urdf_path = os.path.join(
        description_pkg,
        "urdf",
        "go2_description.urdf",
    )
    rviz_config_path = os.path.join(
        fast_lio_pkg,
        "rviz",
        "fastlio.rviz",
    )
    fast_lio_config_path = os.path.join(
        fast_lio_pkg,
        "config",
        "hesaixt16.yaml",
    )
    persistent_map_config_path = os.path.join(
        fast_lio_pkg,
        "config",
        "persistent_global_map.yaml",
    )

    with open(urdf_path, "r", encoding="utf-8") as file:
        robot_description = file.read()

    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Start RViz together with long-run mapping.",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": robot_description,
            }],
            output="screen",
        ),

        ExecuteProcess(
            cmd=[
                "python3",
                os.path.expanduser(
                    "~/go2_ws/src/FAST_LIO_ROS2/scripts/imu_correct.py"
                ),
            ],
            output="screen",
        ),

        Node(
            package="fast_lio",
            executable="fastlio_mapping",
            name="laser_mapping",
            parameters=[fast_lio_config_path],
            output="screen",
        ),

        Node(
            package="fast_lio",
            executable="persistent_global_map_node",
            name="persistent_global_map",
            parameters=[persistent_map_config_path],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_init_to_map_viz",
            arguments=[
                "0", "0", "0",
                "1", "0", "0", "0",
                "camera_init", "map_viz",
            ],
            output="screen",
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config_path],
            condition=IfCondition(use_rviz),
            output="screen",
        ),
    ])
