# mapping_go2_flio01.launch.py (PHASE FLIO-01-R1 test launch)
# 원본 mapping.launch.py 는 유지. 이 test launch 는 config 를 인자로 받아
# src/FAST_LIO_ROS2/config/<config> 를 절대경로로 로드(재빌드 불필요).
# 사용:
#   ros2 launch fast_lio mapping_go2_flio01.launch.py config:=go2_hesai_xt16_flio01_scan10.yaml
#   ros2 launch fast_lio mapping_go2_flio01.launch.py config:=go2_hesai_xt16_flio01_baseline.yaml rviz:=true
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SRC_CONFIG_DIR = os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2/config')

def _setup(context, *args, **kwargs):
    config_name = LaunchConfiguration('config').perform(context)
    rviz_on = LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes')
    yaml_config_path = config_name if os.path.isabs(config_name) else os.path.join(SRC_CONFIG_DIR, config_name)

    fast_lio_pkg = get_package_share_directory('fast_lio')
    description_pkg = get_package_share_directory('go2_description')
    urdf_path = os.path.join(description_pkg, 'urdf', 'go2_description.urdf')
    rviz_config_path = os.path.join(fast_lio_pkg, 'rviz', 'fastlio.rviz')
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc}], output='screen'),
        ExecuteProcess(cmd=['python3', os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2/scripts/imu_correct.py')],
                       output='screen'),
        Node(package='fast_lio', executable='fastlio_mapping', name='laser_mapping',
             parameters=[yaml_config_path], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_init_to_map_viz',
             arguments=['0', '0', '0', '1', '0', '0', '0', 'camera_init', 'map_viz'], output='screen'),
    ]
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio01_scan10.yaml',
                              description='config filename under src/FAST_LIO_ROS2/config (or absolute path)'),
        DeclareLaunchArgument('rviz', default_value='false', description='launch rviz2 (true/false)'),
        OpaqueFunction(function=_setup),
    ])
