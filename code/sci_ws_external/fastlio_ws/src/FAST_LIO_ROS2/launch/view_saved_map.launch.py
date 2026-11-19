# view_saved_map.launch.py (PHASE FLIO-17-R1)
# 저장된 PCD/PLY 맵만 RViz 로 본다. FAST-LIO/go2_bringup 실행 안 함, /cloud_registered 구독 안 함.
#   - publish_saved_map.py 로 저장맵을 /saved_raw_map(또는 지정 topic)으로 발행(시각화 전용)
#   - camera_init->map_viz static tf 를 여기서 직접 발행(FAST-LIO 없이도 map_viz Fixed Frame 동작)
#   - rviz2 를 saved_map_viewer.rviz(=/cloud_registered 없음)로 실행
# 사용:
#   ros2 launch fast_lio view_saved_map.launch.py \
#     input:=/home/unitree/go2_ws/maps/fast_lio_pcd/flio03_map.pcd topic:=/saved_raw_map frame:=camera_init
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SRC = os.path.expanduser('~/go2_ws/src/FAST_LIO_ROS2')

def _resolve_rviz(name, share_dir, default):
    if not name: return os.path.join(share_dir, 'rviz', default)
    if os.path.isabs(name): return name
    for c in [os.path.join(share_dir,'rviz',name), os.path.join(SRC,'rviz',name)]:
        if os.path.isfile(c): return c
    return os.path.join(share_dir,'rviz',default)

def _setup(context, *a, **k):
    fast_lio_pkg = get_package_share_directory('fast_lio')
    inp = LaunchConfiguration('input').perform(context)
    topic = LaunchConfiguration('topic').perform(context)
    frame = LaunchConfiguration('frame').perform(context)
    rate = LaunchConfiguration('rate').perform(context)
    rviz_on = LaunchConfiguration('rviz').perform(context).lower() in ('1','true','yes')
    rviz_cfg = LaunchConfiguration('rviz_config').perform(context)
    rviz_path = _resolve_rviz(rviz_cfg, fast_lio_pkg, 'saved_map_viewer.rviz')

    print('[view_saved_map] input=%s topic=%s frame=%s' % (inp, topic, frame))
    print('[view_saved_map] rviz=%s (%s)  — FAST-LIO/bringup 실행 안 함, /cloud_registered 미사용' % (rviz_path, 'on' if rviz_on else 'off'))
    if not os.path.isfile(os.path.expanduser(inp)):
        print('[view_saved_map] WARNING: 입력 파일 없음 → %s (경로 확인)' % inp)

    nodes = [
        # FAST-LIO 없이도 map_viz Fixed Frame 동작하도록 static tf 직접 발행(flio03 와 동일 변환)
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_init_to_map_viz',
             arguments=['0','0','0','1','0','0','0','camera_init','map_viz'], output='screen'),
        ExecuteProcess(cmd=['python3', os.path.join(SRC,'scripts','publish_saved_map.py'),
                            '--input', inp, '--topic', topic, '--frame', frame, '--rate', str(rate)],
                       output='screen'),
    ]
    if rviz_on:
        nodes.append(Node(package='rviz2', executable='rviz2', name='rviz2',
                          arguments=['-d', rviz_path], output='screen'))
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input', default_value=os.path.expanduser('~/go2_ws/maps/fast_lio_pcd/flio03_map.pcd')),
        DeclareLaunchArgument('topic', default_value='/saved_raw_map'),
        DeclareLaunchArgument('frame', default_value='camera_init'),
        DeclareLaunchArgument('rate', default_value='0.5'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=''),  # 기본 saved_map_viewer.rviz
        OpaqueFunction(function=_setup),
    ])
