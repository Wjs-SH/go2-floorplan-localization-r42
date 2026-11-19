# mapping_go2_flio10_accumulated_map.launch.py (PHASE FLIO-10-R1)
# flio03(scan10+yawfix+RViz) 구조 재사용 + /go2_accumulated_map 누적 map publisher + 전용 RViz.
# 기존 flio03 launch/코어/config 는 미변경. 새 파일로 추가.
# 사용:
#   ros2 launch fast_lio mapping_go2_flio10_accumulated_map.launch.py
#   ros2 launch fast_lio mapping_go2_flio10_accumulated_map.launch.py config:=go2_hesai_xt16_flio08_time_m5ms.yaml accumulate_voxel_size:=0.05
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
    acc_on = LaunchConfiguration('accumulate_map').perform(context).lower() in ('1','true','yes')
    voxel = LaunchConfiguration('accumulate_voxel_size').perform(context)
    min_r = LaunchConfiguration('accumulate_min_range').perform(context)
    max_r = LaunchConfiguration('accumulate_max_range').perform(context)
    pub_rate = LaunchConfiguration('accumulate_publish_rate').perform(context)
    target_frame = LaunchConfiguration('accumulate_target_frame').perform(context)
    save_on = LaunchConfiguration('save_on_shutdown').perform(context)
    save_path = LaunchConfiguration('save_path').perform(context)
    save_fmt = LaunchConfiguration('save_format').perform(context)

    yaml_config_path = _resolve(cfg, os.path.join(fast_lio_pkg,'config'), 'config', 'hesaixt16.yaml')
    rviz_config_path = _resolve(rviz_cfg, os.path.join(fast_lio_pkg,'rviz'), 'rviz', 'go2_fast_lio_accumulated_map.rviz')

    urdf_path = os.path.join(description_pkg,'urdf','go2_description.urdf')
    with open(urdf_path,'r') as f: robot_desc=f.read()

    print('[flio10] config=%s' % yaml_config_path)
    print('[flio10] rviz=%s (%s)' % (rviz_config_path, 'on' if rviz_on else 'off'))
    print('[flio10] accumulate_map=%s voxel=%s range=[%s,%s] rate=%s target_frame="%s"'
          % (acc_on, voxel, min_r, max_r, pub_rate, target_frame))
    print('[flio10] save_on_shutdown=%s path=%s fmt=%s' % (save_on, save_path, save_fmt))

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
    if acc_on:
        # FLIO-10-R3: Foxy rcl 는 빈 값 '-p name:=' 를 파싱하지 못해 노드가 rclpy.init 에서 죽는다.
        # 따라서 값이 있는 파라미터만 --ros-args 로 넘긴다(빈 target_frame 은 아예 생략 → 노드 기본값 사용).
        acc_args = [
            'python3', os.path.join(SRC,'scripts','go2_accumulated_map_publisher.py'),
            '--ros-args',
            '-p', 'input_topic:=/cloud_registered',
            '-p', 'output_topic:=/go2_accumulated_map',
            '-p', 'voxel_size:=%s' % voxel,
            '-p', 'min_range:=%s' % min_r,
            '-p', 'max_range:=%s' % max_r,
            '-p', 'publish_rate:=%s' % pub_rate,
            '-p', 'save_on_shutdown:=%s' % save_on,
            '-p', 'save_path:=%s' % save_path,
            '-p', 'save_format:=%s' % save_fmt,
        ]
        tf = (target_frame or '').strip()
        if tf and tf.lower() not in ('none', 'null', ''):
            acc_args += ['-p', 'target_frame:=%s' % tf]
        nodes.append(ExecuteProcess(cmd=acc_args, output='screen'))
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=['-d', rviz_config_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value='go2_hesai_xt16_flio01_scan10.yaml'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),   # 기본 go2_fast_lio_accumulated_map.rviz
        DeclareLaunchArgument('odom_yaw_fix', default_value='true'),
        DeclareLaunchArgument('odom_yaw_offset_deg', default_value='90.0'),
        DeclareLaunchArgument('odom_yaw_order', default_value='pre'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # accumulated map
        DeclareLaunchArgument('accumulate_map', default_value='true'),
        DeclareLaunchArgument('accumulate_voxel_size', default_value='0.05'),
        DeclareLaunchArgument('accumulate_min_range', default_value='0.5'),
        DeclareLaunchArgument('accumulate_max_range', default_value='80.0'),
        DeclareLaunchArgument('accumulate_publish_rate', default_value='1.0'),
        DeclareLaunchArgument('accumulate_target_frame', default_value='camera_init',
                              description='Accumulated map frame. camera_init(기본, /cloud_registered 와 동일). '
                                          '빈값/none 이면 입력 cloud frame_id 사용.'),
        DeclareLaunchArgument('save_on_shutdown', default_value='true'),
        DeclareLaunchArgument('save_path', default_value=os.path.expanduser('~/go2_ws/maps')),
        DeclareLaunchArgument('save_format', default_value='ply'),
        OpaqueFunction(function=_setup),
    ])
