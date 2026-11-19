# live domain 0, same DDS profile as the operational FAST-LIO (eth0); only new /sci_alt_imu/* topics are created
set +u
source /opt/ros/foxy/setup.bash >/dev/null 2>&1
[ -f $HOME/unitree_ros2/setup.sh ] && source $HOME/unitree_ros2/setup.sh >/dev/null 2>&1
source $HOME/go2_ws/install/setup.bash >/dev/null 2>&1
source /home/unitree/sci_ws_alt_imu_20261029_204740/install/local_setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unitree/sci_ws/config/fastlio/cyclonedds_live.xml
