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
find_package(GSL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(draco REQUIRED)
find_package(point_cloud_interfaces REQUIRED)

find_package(nav2_msgs REQUIRED)
find_package(std_srvs REQUIRED)   # Add this



find_package(PCL REQUIRED COMPONENTS common io surface)
find_package(PCL 1.12 REQUIRED COMPONENTS common segmentation filters)



include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  /usr/include/pcl-1.12
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${nav2_msgs_INCLUDE_DIRS}
)


link_directories(
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})



include_directories(
  ${OpenCV_INCLUDE_DIRS}
  /opt/ros/humble/include  # Ensures ROS2 headers are included
)



add_executable(kalman_rel_localization
  src/kalman_rel_localization.cpp
  src/utilities.cpp
  src/visualization.cpp
)


ament_target_dependencies(kalman_rel_localization
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)

target_link_libraries(kalman_rel_localization
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  tf2::tf2
  tf2_ros::tf2_ros
)



add_executable(global_localization
  src/global_localization.cpp
  src/utilities.cpp
  src/visualization.cpp
)


ament_target_dependencies(global_localization
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  nav2_msgs
  std_srvs
)

target_link_libraries(global_localization
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  tf2::tf2
  tf2_ros::tf2_ros
)


add_executable(map_transformer src/map_transformer.cpp)


ament_target_dependencies(map_transformer
  rclcpp
  tf2_ros
  geometry_msgs
  tf2_geometry_msgs
)


add_executable(map_republisher src/map_republisher.cpp)
ament_target_dependencies(map_republisher
  rclcpp
  nav_msgs
  tf2_ros
  geometry_msgs
  tf2_geometry_msgs
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
#add_executable(ground_plane_processor src/ground_plane_processor.cpp)

#ament_target_dependencies(ground_plane_processor
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#)

#add_executable(joined_processor src/joined_processor.cpp)

#ament_target_dependencies(joined_processor
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#)

#add_executable(obstacle_detector src/obstacle_detector.cpp)

# Link OpenCV libraries to the obstacle_detector executable
#target_link_libraries(obstacle_detector ${OpenCV_LIBS})


#ament_target_dependencies(obstacle_detector
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#)#

#add_executable(rel_localization src/rel_localization_processor.cpp)

#ament_target_dependencies(rel_localization
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_sensor_msgs
  
#)

#target_link_libraries(rel_localization
#    ${OpenCV_LIBS}
#    tf2::tf2
#    tf2_ros::tf2_ros
      # Ensure this is linked correctly
#)

#add_executable(duarte_init src/duarte_init.cpp)

#ament_target_dependencies(duarte_init rclcpp visualization_msgs)


#add_executable(circle_kalman src/kalman_circle_localization.cpp)

#ament_target_dependencies(circle_kalman
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_sensor_msgs
  
#)

#target_link_libraries(circle_kalman
#    ${OpenCV_LIBS}
#    tf2::tf2
#    tf2_ros::tf2_ros
      # Ensure this is linked correctly
#)

#add_executable(test_cluster src/cluster_test.cpp)

#ament_target_dependencies(test_cluster
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_sensor_msgs
  
#)

#target_link_libraries(test_cluster
#    ${OpenCV_LIBS}
#    tf2::tf2
#    tf2_ros::tf2_ros
      # Ensure this is linked correctly
#)

#add_executable(same_center src/same_center.cpp)

#ament_target_dependencies(same_center
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_sensor_msgs

#)

#target_link_libraries(same_center
#    ${OpenCV_LIBS}
#    tf2::tf2
#    tf2_ros::tf2_ros

      # Ensure this is linked correctly
#)

#add_executable(skeleton_spline src/skeleton_spline.cpp)

#ament_target_dependencies(skeleton_spline
#  rclcpp
#  std_msgs
#  sensor_msgs
#  pcl_ros
#  pcl_conversions
#  visualization_msgs
#  tf2
#  tf2_ros
#  tf2_sensor_msgs
  
#)



add_executable(frenet_motion src/frenet_motion.cpp)

ament_target_dependencies(frenet_motion
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  
)

target_link_libraries(frenet_motion
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
    ${PCL_LIBRARIES}
)

add_executable(test_compression src/test_compression.cpp)

ament_target_dependencies(test_compression
  rclcpp
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  point_cloud_interfaces
  
)

target_link_libraries(test_compression
    ${OpenCV_LIBS}
    tf2::tf2
    tf2_ros::tf2_ros
    ${PCL_LIBRARIES}
    draco::draco
)

add_executable(dynamic_tf_publisher src/dynamic_tf_publisher.cpp)
ament_target_dependencies(dynamic_tf_publisher
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
)

add_executable(pose_waiter src/pose_waiter.cpp)
ament_target_dependencies(pose_waiter
  rclcpp
  geometry_msgs
  std_srvs
  tf2 
  tf2_ros
  tf2_geometry_msgs
)





install(TARGETS
  point_cloud_processor
  frenet_motion
  kalman_rel_localization
  global_localization
  test_compression
  map_transformer
  map_republisher
  dynamic_tf_publisher
  pose_waiter 
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch maps config urdf
  DESTINATION share/${PROJECT_NAME}
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
