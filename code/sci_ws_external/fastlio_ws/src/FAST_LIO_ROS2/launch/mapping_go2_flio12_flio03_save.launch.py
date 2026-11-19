# mapping_go2_flio12_flio03_save.launch.py (PHASE FLIO-12-R1)
# = flio03(yawfix+RViz) 를 그대로 유지 + map 저장 sidecar 만 추가.
#   flio03 front-end/RViz/config 미변경. /go2_accumulated_map 미사용(화면 보존).
#   기존 mapping_go2_flio03_yawfix_rviz.launch.py 는 삭제/수정하지 않음(별도 파일).
# 사용:
#   ros2 launch fast_lio mapping_go2_flio12_flio03_save.launch.py
#   ros2 launch fast_lio mapping_go2_flio12_flio03_save.launch.py config:=go2_hesai_xt16_flio07_imu_cov_hi.yaml
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
    save_on = LaunchConfiguration('save_map').perform(context).lower() in ('1','true','yes')
    voxel = LaunchConfiguration('voxel_size').perform(context)
    min_r = LaunchConfiguration('min_range').perform(context)
    max_r = LaunchConfiguration('max_range').perform(context)
    min_z = LaunchConfiguration('min_z').perform(context)
    max_z = LaunchConfiguration('max_z').perform(context)
    save_path = LaunchConfiguration('save_path').perform(context)
    save_fmt = LaunchConfiguration('save_format').perform(context)
    save_shutdown = LaunchConfiguration('save_on_shutdown').perform(context)
    autosave = LaunchConfiguration('autosave_sec').perform(context)
    pub_preview = LaunchConfiguration('publish_preview').perform(context)
    mode = LaunchConfiguration('save_mode').perform(context)
    confirmed_hits = LaunchConfiguration('confirmed_min_hits').perform(context)

    # flio03 기본 RViz(yawfix) 그대로 사용 → 화면 보존
    yaml_config_path = _resolve(cfg, os.path.join(fast_lio_pkg,'config'), 'config', 'hesaixt16.yaml')
    rviz_config_path = _resolve(rviz_cfg, os.path.join(fast_lio_pkg,'rviz'), 'rviz', 'go2_fast_lio_yawfix.rviz')

    urdf_path = os.path.join(description_pkg,'urdf','go2_description.urdf')
    with open(urdf_path,'r') as f: robot_desc=f.read()

    print('[flio12] config=%s' % yaml_config_path)
    print('[flio12] rviz=%s (%s) — flio03 화면 보존' % (rviz_config_path, 'on' if rviz_on else 'off'))
    print('[flio12] map save sidecar=%s mode=%s voxel=%s range=[%s,%s] z=[%s,%s] -> %s (%s)'
          % (save_on, mode, voxel, min_r, max_r, min_z, max_z, save_path, save_fmt))
    print('[flio12] publish_preview=%s (기본 OFF: /go2_accumulated_map 미사용, 화면 미변경)' % pub_preview)

    # --- flio03 와 동일한 노드 구성 (front-end/RViz 미변경) ---
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
    # --- map 저장 sidecar (구독+저장만, 화면 미변경) ---
    if save_on:
        # FLIO-10-R3 교훈: Foxy 는 빈 '-p name:=' 를 파싱 못함 → 모든 파라미터 값이 있어야 함(빈값 없음).
        nodes.append(ExecuteProcess(cmd=[
            'python3', os.path.join(SRC,'scripts','flio03_map_save_sidecar.py'),
            '--ros-args',
            '-p', 'input_topic:=/cloud_registered',
            '-p', 'voxel_size:=%s' % voxel,
            '-p', 'min_range:=%s' % min_r,
            '-p', 'max_range:=%s' % max_r,
            '-p', 'min_z:=%s' % min_z,
            '-p', 'max_z:=%s' % max_z,
            '-p', 'save_path:=%s' % save_path,
            '-p', 'save_format:=%s' % save_fmt,
            '-p', 'save_on_shutdown:=%s' % save_shutdown,
            '-p', 'autosave_sec:=%s' % autosave,
            '-p', 'publish_preview:=%s' % pub_preview,
            '-p', 'mode:=%s' % mode,
            '-p', 'confirmed_min_hits:=%s' % confirmed_hits,
        ], output='screen'))
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=['-d', rviz_config_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        # flio03 와 동일 기본값
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio01_scan10.yaml'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),   # 기본 go2_fast_lio_yawfix.rviz (flio03)
        DeclareLaunchArgument('odom_yaw_fix', default_value='true'),
        DeclareLaunchArgument('odom_yaw_offset_deg', default_value='90.0'),
        DeclareLaunchArgument('odom_yaw_order', default_value='pre'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # map save sidecar
        DeclareLaunchArgument('save_map', default_value='true'),
        DeclareLaunchArgument('save_mode', default_value='raw_save'),   # raw_save | confirmed_save
        DeclareLaunchArgument('confirmed_min_hits', default_value='2'),
        DeclareLaunchArgument('voxel_size', default_value='0.08'),
        DeclareLaunchArgument('min_range', default_value='0.8'),
        DeclareLaunchArgument('max_range', default_value='50.0'),
        DeclareLaunchArgument('min_z', default_value='-3.0'),
        DeclareLaunchArgument('max_z', default_value='3.0'),
        DeclareLaunchArgument('save_path', default_value='/home/unitree/go2_ws/maps'),
        DeclareLaunchArgument('save_format', default_value='ply'),
        DeclareLaunchArgument('save_on_shutdown', default_value='true'),
        DeclareLaunchArgument('autosave_sec', default_value='0.0'),
        DeclareLaunchArgument('publish_preview', default_value='false'),  # 화면 보존: 기본 OFF
        OpaqueFunction(function=_setup),
    ])
