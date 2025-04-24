#pragma once

#include <deque>
#include <utility>
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>  // Added this
#include <Eigen/Dense>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav2_msgs/msg/particle_cloud.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


class GlobalLocalization : public rclcpp::Node
{
public:
    GlobalLocalization();
    
private:
    // Member variables
    bool map_received_ = false;
    nav_msgs::msg::OccupancyGrid::SharedPtr current_map_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    geometry_msgs::msg::PoseStamped current_pose_;
    Eigen::Vector2f current_control_ = Eigen::Vector2f::Zero();
    bool first_update_;
    rclcpp::Time last_update_time_;
    rclcpp::Time last_speed_time_;
    rclcpp::Time last_servo_time_;

    std::deque<std::pair<rclcpp::Time, float>> speed_history_;
    std::deque<std::pair<rclcpp::Time, float>> servo_history_;

    float last_speed_value_ = 0.0f;
    bool fit_side_;
    double time_jump_threshold_;

    // Constants
    const float wheel_radius_ = 0.1f;
    const float wheelbase_L_ = 0.22f;
    const float rpm_to_ms_ = (2.0 * M_PI * wheel_radius_) / 60.0f;
    const float servo_to_rad_ = 0.6f;
    const float speed_timeout_ = 0.5f;
    const float servo_timeout_ = 0.5f;
    float last_valid_dt_ = 1.0/30.0;

    // Callbacks
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void particleCallback(const nav2_msgs::msg::ParticleCloud::SharedPtr msg);  // Added this
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void speedCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void servoCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    sensor_msgs::msg::LaserScan clustersToLaserScan(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<pcl::PointIndices>& clusters,
        const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Helper methods
    std::pair<float, float> findNearestControl(const std::deque<std::pair<rclcpp::Time, float>>& history, 
                                             rclcpp::Time query_time);

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_left_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_right_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr kalman_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr middle_lane_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr test_cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr orientation_marker_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr distance_orientation_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr particle_viz_pub_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr servo_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav2_msgs::msg::ParticleCloud>::SharedPtr particle_sub_;
};