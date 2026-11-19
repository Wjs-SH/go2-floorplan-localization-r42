# mapping_go2_flio18_high_density_save.launch.py (PHASE FLIO-18-R1)
# = flio14(flio03 화면 + 내장 pcd_save) 와 동일한 front-end/RViz/yawfix
#   + filter_size_surf/map 을 낮춘 density profile config 로 저장맵 밀도 상향.
#   기존 flio03/flio14 launch·config 는 미변경. 새 파일.
# profile: moderate(0.2)/high(0.1)/ultra(0.05). config 인자로 직접 지정도 가능.
# 사용:
#   ros2 launch fast_lio mapping_go2_flio18_high_density_save.launch.py                      # moderate
#   ros2 launch fast_lio mapping_go2_flio18_high_density_save.launch.py profile:=high
#   ros2 launch fast_lio mapping_go2_flio18_high_density_save.launch.py config:=go2_hesai_xt16_flio18_save_density_ultra.yaml
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SRC = os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2')
PROFILE_CFG = {
    'moderate': 'go2_hesai_xt16_flio18_save_density_moderate.yaml',
    'high':     'go2_hesai_xt16_flio18_save_density_high.yaml',
    'ultra':    'go2_hesai_xt16_flio18_save_density_ultra.yaml',
}

def _resolve(name, share_dir, sub, default):
    if not name: return os.path.join(share_dir, default)
    if os.path.isabs(name): return name
    for c in [os.path.join(share_dir, name), os.path.join(SRC, sub, name)]:
        if os.path.isfile(c): return c
    return os.path.join(share_dir, default)

def _setup(context, *a, **k):
    fast_lio_pkg = get_package_share_directory('fast_lio')
    description_pkg = get_package_share_directory('go2_description')
    profile = LaunchConfiguration('profile').perform(context)
    cfg = LaunchConfiguration('config').perform(context)
    # config 인자가 명시되면 그것을 우선, 아니면 profile 로 매핑
    if not cfg:
        cfg = PROFILE_CFG.get(profile, PROFILE_CFG['moderate'])
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

    print('[flio18] profile=%s config=%s (filter_size 낮춰 저장맵 밀도 상향)' % (profile, yaml_config_path))
    print('[flio18] rviz=%s (%s) — flio03 화면 동일' % (rviz_config_path, 'on' if rviz_on else 'off'))
    print('[flio18] 종료(Ctrl+C) 시 map_file_path 에 dense ikd-tree map 저장. 화면 흔들리면 profile 낮추기.')

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
        DeclareLaunchArgument('profile', default_value='moderate'),  # moderate|high|ultra
        DeclareLaunchArgument('config', default_value=''),           # 지정 시 profile 무시
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),      # 기본 go2_fast_lio_yawfix.rviz(flio03)
        DeclareLaunchArgument('odom_yaw_fix', default_value='true'),
        DeclareLaunchArgument('odom_yaw_offset_deg', default_value='90.0'),
        DeclareLaunchArgument('odom_yaw_order', default_value='pre'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
