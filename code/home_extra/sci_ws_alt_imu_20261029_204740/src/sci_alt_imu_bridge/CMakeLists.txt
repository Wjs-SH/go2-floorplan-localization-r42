cmake_minimum_required(VERSION 3.5)
project(sci_alt_imu_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
add_compile_options(-Wall -Wextra -O2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(nav_msgs REQUIRED)

add_library(sci_alt_imu_core STATIC src/core.cpp)
target_include_directories(sci_alt_imu_core PUBLIC include)
set_target_properties(sci_alt_imu_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(bridge_node src/bridge_node.cpp)
target_link_libraries(bridge_node sci_alt_imu_core)
ament_target_dependencies(bridge_node rclcpp sensor_msgs std_msgs unitree_go)

add_executable(offline_stamp src/offline_stamp.cpp)
target_link_libraries(offline_stamp sci_alt_imu_core)

add_executable(calib_monitor src/calib_monitor.cpp)
ament_target_dependencies(calib_monitor rclcpp unitree_go)

add_executable(l1_adapter_node src/l1_adapter_node.cpp)
ament_target_dependencies(l1_adapter_node rclcpp sensor_msgs std_msgs nav_msgs)

install(TARGETS bridge_node offline_stamp calib_monitor l1_adapter_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_core test/test_core.cpp)
  target_link_libraries(test_core sci_alt_imu_core)
endif()

ament_package()
