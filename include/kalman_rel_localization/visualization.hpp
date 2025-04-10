// visualization.hpp
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

namespace lane_detection_visualization {

    void visualizeKalmanState(
        const Eigen::VectorXf& state,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        bool left_detected,
        bool right_detected,
        const pcl::PointIndices& left_cluster_indices,
        const pcl::PointIndices& right_cluster_indices,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& white_cloud,
        const geometry_msgs::msg::Pose& current_pose,
        const rclcpp::Time& current_time);

    void publishClusterMarkers(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const std::vector<pcl::PointIndices>& clusters,
        const geometry_msgs::msg::Pose& camera_pose,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time);

    void testClusterMarkers(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const pcl::PointIndices& left_indices, 
        const pcl::PointIndices& right_indices,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time);

    void publishCameraAxes(
        const geometry_msgs::msg::PoseStamped& msg,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time);

} // namespace lane_detection_visualization