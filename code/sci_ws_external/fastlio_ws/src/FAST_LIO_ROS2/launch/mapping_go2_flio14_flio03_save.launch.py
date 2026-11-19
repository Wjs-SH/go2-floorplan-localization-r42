# mapping_go2_flio14_flio03_save.launch.py (PHASE FLIO-14-R1)
# = flio03(yawfix+RViz) 와 100% 동일한 front-end/RViz + FAST-LIO 내장 pcd_save 활성화 config 만 사용.
#   추가 노드/토픽/sidecar/accumulated map 없음. 종료(Ctrl+C) 시 save_to_pcd() 가 ikd-tree map 을 저장.
#   기존 mapping_go2_flio03_yawfix_rviz.launch.py 는 삭제/수정하지 않음(별도 파일).
# 사용:
#   ros2 launch fast_lio mapping_go2_flio14_flio03_save.launch.py
#   (다른 config 로 저장) ros2 launch fast_lio mapping_go2_flio14_flio03_save.launch.py config:=<name>.yaml
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

    # flio03 와 동일한 RViz(yawfix) 그대로 사용 → 화면 100% 동일
    yaml_config_path = _resolve(cfg, os.path.join(fast_lio_pkg,'config'), 'config', 'hesaixt16.yaml')
    rviz_config_path = _resolve(rviz_cfg, os.path.join(fast_lio_pkg,'rviz'), 'rviz', 'go2_fast_lio_yawfix.rviz')

    urdf_path = os.path.join(description_pkg,'urdf','go2_description.urdf')
    with open(urdf_path,'r') as f: robot_desc=f.read()

    print('[flio14] config=%s (FAST-LIO 내장 pcd_save 활성)' % yaml_config_path)
    print('[flio14] rviz=%s (%s) — flio03 화면 100%% 동일' % (rviz_config_path, 'on' if rviz_on else 'off'))
    print('[flio14] 종료(Ctrl+C) 시 map_file_path 에 ikd-tree map 저장. 저장 디렉터리 사전 생성 필요.')

    # --- flio03 와 완전히 동일한 노드 구성 (front-end/RViz 미변경, 추가 노드 없음) ---
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
        # 유일한 차이: 기본 config = 저장 활성 config (front-end 파라미터는 scan10 과 동일)
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio03_save.yaml'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),   # 기본 go2_fast_lio_yawfix.rviz (flio03)
        DeclareLaunchArgument('odom_yaw_fix', default_value='true'),
        DeclareLaunchArgument('odom_yaw_offset_deg', default_value='90.0'),
        DeclareLaunchArgument('odom_yaw_order', default_value='pre'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
