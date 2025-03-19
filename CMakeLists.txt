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
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
set(Ceres_DIR "/usr/lib/cmake/Ceres")
find_package(Ceres REQUIRED)




include_directories(
  ${OpenCV_INCLUDE_DIRS}
  /opt/ros/humble/include  # Ensures ROS2 headers are included
)




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

add_executable(rel_localization src/rel_localization_processor.cpp)

ament_target_dependencies(rel_localization
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  
)

target_link_libraries(rel_localization
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
      # Ensure this is linked correctly
)

add_executable(duarte_init src/duarte_init.cpp)

ament_target_dependencies(duarte_init rclcpp visualization_msgs)

add_executable(kalman_rel src/kalman_rel_localization.cpp)

ament_target_dependencies(kalman_rel
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  
)

target_link_libraries(kalman_rel
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
      # Ensure this is linked correctly
)

add_executable(circle_kalman src/kalman_circle_localization.cpp)

ament_target_dependencies(circle_kalman
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  
)

target_link_libraries(circle_kalman
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
      # Ensure this is linked correctly
)

add_executable(test_cluster src/cluster_test.cpp)

ament_target_dependencies(test_cluster
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  
)

target_link_libraries(test_cluster
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
      # Ensure this is linked correctly
)

add_executable(same_center src/same_center.cpp)

ament_target_dependencies(same_center
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs

)

target_link_libraries(same_center
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
    Ceres::ceres
      # Ensure this is linked correctly
)



install(TARGETS
  point_cloud_processor
  ground_plane_processor
  joined_processor
  obstacle_detector
  rel_localization
  duarte_init
  kalman_rel
  circle_kalman
  test_cluster
  same_center
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
