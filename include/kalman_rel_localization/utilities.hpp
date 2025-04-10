#pragma once

#include <deque>  // Add this for std::deque
#include <utility> // For std::pair
#include <visualization_msgs/msg/marker.hpp>
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h> 
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace lane_detection_utils {

    std::pair<float, float> findNearestControl(const std::deque<std::pair<rclcpp::Time, float>>& history, rclcpp::Time query_time) ;

    Eigen::Vector3f fitQuadraticCurve(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
                                    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub,
                                    const geometry_msgs::msg::Pose& camera_pose,
                                    const rclcpp::Time& current_time);

    Eigen::Vector3f estimateRightFromLeft(const Eigen::Vector3f& left_coeffs_cam, 
                                        const Eigen::Vector3f& lane_transform_cam);

    Eigen::Vector3f estimateLeftFromRight(const Eigen::Vector3f& right_coeffs_cam, 
                                        const Eigen::Vector3f& lane_transform_cam);

    Eigen::Vector3f calculateAndStoreTransform(const Eigen::Vector3f& left_coeffs_cam, 
                                            const Eigen::Vector3f& right_coeffs_cam);

    Eigen::Vector3f shiftToMiddle(const Eigen::Vector3f& coeffs, 
                                const Eigen::Vector3f& lane_transform, 
                                bool is_left);

    void calculateDistanceAndOrientation(
        const Eigen::Vector3f& middle_coeffs_cam,
        const Eigen::Vector3f& camera_pos,
        const tf2::Matrix3x3& rotation_matrix,
        float min_distance,
        float max_distance,
        float& distance,
        float& orientation_diff);

    std::vector<pcl::PointIndices> clusterWhitePoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const geometry_msgs::msg::Pose& current_pose);

    void RGBtoHSV(int r, int g, int b, float &h, float &s, float &v);   

} // namespace lane_detection_utils