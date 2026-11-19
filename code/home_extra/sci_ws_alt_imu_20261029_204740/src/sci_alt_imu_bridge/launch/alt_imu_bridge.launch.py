"""Launch only the alt IMU bridge. Usage:
  ros2 launch sci_alt_imu_bridge alt_imu_bridge.launch.py log_dir:=/path/to/run_dir
Writes health JSONL + bounded timing CSV into log_dir (must be a new, run-specific directory)."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _nodes(context):
    log_dir = LaunchConfiguration("log_dir").perform(context)
    if not log_dir:
        raise RuntimeError("log_dir:=<new run directory> is required")
    os.makedirs(log_dir, exist_ok=True)
    cfg = os.path.join(get_package_share_directory("sci_alt_imu_bridge"), "config", "bridge.yaml")
    return [Node(package="sci_alt_imu_bridge", executable="bridge_node", name="sci_alt_imu_bridge", output="screen",
                 parameters=[cfg, {"health_jsonl": os.path.join(log_dir, "bridge_health.jsonl"),
                                   "timing_csv": os.path.join(log_dir, "bridge_timing.csv")}])]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("log_dir", default_value=""), OpaqueFunction(function=_nodes)])
