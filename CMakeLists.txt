cmake_minimum_required(VERSION 3.8)
project(point_cloud_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})




add_executable(point_cloud_processor src/point_cloud_processor.cpp)

ament_target_dependencies(point_cloud_processor
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

# Add Ground Plane Processor Node
add_executable(ground_plane_processor src/ground_plane_processor.cpp)

ament_target_dependencies(ground_plane_processor
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

add_executable(joined_processor src/joined_processor.cpp)

ament_target_dependencies(joined_processor
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

add_executable(obstacle_detector src/obstacle_detector.cpp)

# Link OpenCV libraries to the obstacle_detector executable
target_link_libraries(obstacle_detector ${OpenCV_LIBS})


ament_target_dependencies(obstacle_detector
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

install(TARGETS
  point_cloud_processor
  ground_plane_processor
  joined_processor
  obstacle_detector
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
