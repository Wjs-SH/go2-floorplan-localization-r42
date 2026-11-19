# source this for any offline/replay command (domain 42, loopback-only Cyclone profile)
set +u
source /opt/ros/foxy/setup.bash >/dev/null 2>&1
source $HOME/go2_ws/install/setup.bash >/dev/null 2>&1
source /home/unitree/sci_ws_alt_imu_20261029_204740/install/local_setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/sci_ws_alt_imu_20261029_204740/config/cyclonedds_offline_loopback.xml
