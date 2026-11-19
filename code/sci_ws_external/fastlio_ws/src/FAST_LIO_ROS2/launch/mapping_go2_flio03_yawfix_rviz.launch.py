# mapping_go2_flio03_yawfix_rviz.launch.py (PHASE FLIO-03-R1)
# flio02(scan10+RViz) + 시각화 전용 odom yaw 보정(/Odometry_yawfix) + yawfix RViz.
# 원본 /Odometry, FAST-LIO 코어, flio02 launch 는 미변경.
# 사용:
#   ros2 launch fast_lio mapping_go2_flio03_yawfix_rviz.launch.py
#   ros2 launch fast_lio mapping_go2_flio03_yawfix_rviz.launch.py odom_yaw_order:=post
#   ros2 launch fast_lio mapping_go2_flio03_yawfix_rviz.launch.py odom_yaw_offset_deg:=90.0
#   ros2 launch fast_lio mapping_go2_flio03_yawfix_rviz.launch.py odom_yaw_fix:=false
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SRC = os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2')

def _resolve(name, share_dir, sub, default):
    if not name:
        return os.path.join(share_dir, default)
    if os.path.isabs(name):
        return name
    for c in [os.path.join(share_dir, name), os.path.join(SRC, sub, name)]:
        if os.path.isfile(c):
            return c
    return os.path.join(share_dir, default)

def _setup(context, *a, **k):
    fast_lio_pkg = get_package_share_directory('fast_lio')
    description_pkg = get_package_share_directory('go2_description')
    cfg = LaunchConfiguration('config').perform(context)
    rviz_on = LaunchConfiguration('rviz').perform(context).lower() in ('1','true','yes')
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)
    yawfix_on = LaunchConfiguration('odom_yaw_fix').perform(context).lower() in ('1','true','yes')
    yaw_deg = LaunchConfiguration('odom_yaw_offset_deg').perform(context)
    yaw_order = LaunchConfiguration('odom_yaw_order').perform(context)
    use_sim = LaunchConfiguration('use_sim_time').perform(context).lower() in ('1','true','yes')

    yaml_config_path = _resolve(cfg, os.path.join(fast_lio_pkg,'config'), 'config', 'hesaixt16.yaml')
    rviz_config_path = _resolve(rviz_cfg, os.path.join(fast_lio_pkg,'rviz'), 'rviz', 'go2_fast_lio_yawfix.rviz')

    urdf_path = os.path.join(description_pkg,'urdf','go2_description.urdf')
    with open(urdf_path,'r') as f: robot_desc=f.read()

    print('[flio03] config=%s' % yaml_config_path)
    print('[flio03] rviz=%s (%s)' % (rviz_config_path, 'on' if rviz_on else 'off'))
    print('[flio03] odom_yaw_fix=%s offset=%sdeg order=%s -> /Odometry_yawfix (viz-only)' % (yawfix_on, yaw_deg, yaw_order))

    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc, 'use_sim_time': use_sim}], output='screen'),
        ExecuteProcess(cmd=['python3', os.path.join(SRC,'scripts','imu_correct.py')], output='screen'),
        Node(package='fast_lio', executable='fastlio_mapping', name='laser_mapping',
             parameters=[yaml_config_path, {'use_sim_time': use_sim}], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_init_to_map_viz',
             arguments=['0','0','0','1','0','0','0','camera_init','map_viz'], output='screen'),
    ]
    if yawfix_on:
        nodes.append(ExecuteProcess(cmd=['python3', os.path.join(SRC,'scripts','odom_yaw_visual_fix.py'),
                                         '--input','/Odometry','--output','/Odometry_yawfix',
                                         '--yaw-offset-deg', str(yaw_deg), '--order', str(yaw_order)],
                                    output='screen'))
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=['-d', rviz_config_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio01_scan10.yaml'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),   # 기본 go2_fast_lio_yawfix.rviz
        DeclareLaunchArgument('odom_yaw_fix', default_value='true'),
        DeclareLaunchArgument('odom_yaw_offset_deg', default_value='90.0'),   # FLIO-03-R2: 사용자 라이브 확인 확정값(+90)
        DeclareLaunchArgument('odom_yaw_order', default_value='pre'),  # pre=q_off*q_orig, post=q_orig*q_off
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('log_level', default_value='info'),
        OpaqueFunction(function=_setup),
    ])
