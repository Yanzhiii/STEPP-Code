cmake_minimum_required(VERSION 3.8)
project(stepp_ros2_humble)

# Default to C++17 (required by ROS 2 toolchain)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# ------------------------------------------------------------
# Custom message generation
# ------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Float32Stamped.msg"
  DEPENDENCIES std_msgs
)

# ------------------------------------------------------------
# Python runtime dependencies (for install target discovery)
# ------------------------------------------------------------
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ------------------------------------------------------------
# Install Python executables
# ------------------------------------------------------------
install(PROGRAMS
  scripts/inference_node.py
  scripts/depth_projection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ------------------------------------------------------------
# Install launch files and config
# ------------------------------------------------------------
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# ------------------------------------------------------------
ament_package()
