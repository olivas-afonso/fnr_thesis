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
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc dnn)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(GSL REQUIRED)

find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_eigen REQUIRED) 
find_package(draco REQUIRED)
find_package(point_cloud_interfaces REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(yaml-cpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(nav_msgs REQUIRED)  
find_package(rclcpp REQUIRED)
find_package(zed_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)



find_package(nav2_msgs REQUIRED)
find_package(std_srvs REQUIRED)   # Add this



find_package(PCL REQUIRED COMPONENTS common io surface)
find_package(PCL 1.12 REQUIRED COMPONENTS common segmentation filters)

find_package(PCL REQUIRED COMPONENTS common segmentation kdtree filters)

find_package(image_transport REQUIRED)



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
  nav_msgs
)

target_link_libraries(kalman_rel_localization
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  tf2::tf2
  tf2_ros::tf2_ros
  ${tf2_geometry_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
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

add_executable(map_creator_pc src/map_creator_pc.cpp)
ament_target_dependencies(map_creator_pc
  rclcpp
  sensor_msgs
  nav_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  yaml-cpp
  rcpputils
)

add_executable(zed_image_saver src/zed_image_saver.cpp)
ament_target_dependencies(zed_image_saver
  rclcpp
  sensor_msgs
  cv_bridge
)

add_executable(yolo_visualize src/yolo_visualize.cpp)
ament_target_dependencies(yolo_visualize
  rclcpp
  zed_msgs
  visualization_msgs
  geometry_msgs
  cv_bridge
  image_transport
  sensor_msgs
)

target_link_libraries(yolo_visualize
  ${OpenCV_LIBS}
  opencv_dnn  # Add explicit dnn library
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
)

target_include_directories(yolo_visualize PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(mobile_net src/mobile_net.cpp)
ament_target_dependencies(mobile_net
  rclcpp
  cv_bridge
  sensor_msgs
  zed_msgs
)

add_executable(object_detector
  src/object_detector.cpp
)

# Link dependencies
ament_target_dependencies(object_detector
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  std_msgs
  pcl_conversions
  tf2
)
add_executable(amcl_path_recorder src/amcl_path_recorder.cpp)
ament_target_dependencies(amcl_path_recorder rclcpp geometry_msgs)

# Add PCL linking for object_detector
target_link_libraries(object_detector
  ${PCL_LIBRARIES}
)

# Add OpenCV linking specifically for mobile_net
target_link_libraries(mobile_net
  ${OpenCV_LIBS}
)

# Add include directories for OpenCV
target_include_directories(mobile_net PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

#add_executable(map_creator_ls src/map_creator_ls.cpp)
#ament_target_dependencies(map_creator_ls
#  rclcpp
#  sensor_msgs
#  nav_msgs
#  pcl_conversions
#  tf2
#  tf2_ros
#  tf2_geometry_msgs
#  tf2_eigen
#  yaml-cpp
#  rcpputils
#)



install(PROGRAMS
  scripts/visualize_imu.py
  DESTINATION lib/${PROJECT_NAME}
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
  map_creator_pc
  #map_creator_ls
  zed_image_saver
  yolo_visualize
  mobile_net
  object_detector
  amcl_path_recorder
  DESTINATION lib/${PROJECT_NAME}
  
)

install(
  DIRECTORY launch maps config urdf rviz2
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
