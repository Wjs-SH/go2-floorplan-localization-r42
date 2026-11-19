# mapping_go2_flio02_rviz.launch.py  (PHASE FLIO-02-R1)
# 목적: scan10 config 기본 + RViz 자동 실행(기존 mapping.launch.py 구조 재사용).
# 사용:
#   ros2 launch fast_lio mapping_go2_flio02_rviz.launch.py
#   ros2 launch fast_lio mapping_go2_flio02_rviz.launch.py config:=go2_hesai_xt16_flio01_baseline.yaml rviz:=true
#   ros2 launch fast_lio mapping_go2_flio02_rviz.launch.py rviz:=false           # headless
#   ros2 launch fast_lio mapping_go2_flio02_rviz.launch.py rviz_config:=/abs/x.rviz
# 원본 mapping.launch.py / config / source 는 수정하지 않음.
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _resolve_config(name, share_cfg_dir):
    if os.path.isabs(name):
        return name
    # 1) 설치 share/config  2) 소스 config  3) share fallback
    cand = [os.path.join(share_cfg_dir, name),
            os.path.expanduser(os.path.join('~/go2_ws/src/FAST_LIO_ROS2/config', name))]
    for c in cand:
        if os.path.isfile(c):
            return c
    return os.path.join(share_cfg_dir, name)

def _resolve_rviz(name, share_rviz_dir):
    if not name:
        return os.path.join(share_rviz_dir, 'fastlio.rviz')
    if os.path.isabs(name):
        return name
    cand = [os.path.join(share_rviz_dir, name),
            os.path.expanduser(os.path.join('~/go2_ws/src/FAST_LIO_ROS2/rviz', name))]
    for c in cand:
        if os.path.isfile(c):
            return c
    return os.path.join(share_rviz_dir, 'fastlio.rviz')

def _setup(context, *args, **kwargs):
    fast_lio_pkg = get_package_share_directory('fast_lio')
    description_pkg = get_package_share_directory('go2_description')
    share_cfg_dir = os.path.join(fast_lio_pkg, 'config')
    share_rviz_dir = os.path.join(fast_lio_pkg, 'rviz')

    config_name = LaunchConfiguration('config').perform(context)
    rviz_on = LaunchConfiguration('rviz').perform(context).lower() in ('1', 'true', 'yes')
    rviz_config_arg = LaunchConfiguration('rviz_config').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in ('1', 'true', 'yes')

    yaml_config_path = _resolve_config(config_name, share_cfg_dir)
    rviz_config_path = _resolve_rviz(rviz_config_arg, share_rviz_dir)

    urdf_path = os.path.join(description_pkg, 'urdf', 'go2_description.urdf')
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    print('[flio02] config  = %s' % yaml_config_path)
    print('[flio02] rviz    = %s (%s)' % (rviz_config_path, 'on' if rviz_on else 'off'))

    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc, 'use_sim_time': use_sim_time}], output='screen'),
        ExecuteProcess(cmd=['python3', os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2/scripts/imu_correct.py')],
                       output='screen'),
        Node(package='fast_lio', executable='fastlio_mapping', name='laser_mapping',
             parameters=[yaml_config_path, {'use_sim_time': use_sim_time}], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_init_to_map_viz',
             arguments=['0', '0', '0', '1', '0', '0', '0', 'camera_init', 'map_viz'], output='screen'),
    ]
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=['-d', rviz_config_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio01_scan10.yaml',
                              description='config filename (share/src config dir) or absolute path'),
        DeclareLaunchArgument('rviz', default_value='true', description='launch rviz2 (true/false)'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='rviz config filename or abs path (default fastlio.rviz)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        OpaqueFunction(function=_setup),
    ])
