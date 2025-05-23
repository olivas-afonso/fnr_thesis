#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor(const rclcpp::NodeOptions& options)
    : Node("point_cloud_processor", options),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(
        this->get_clock(), 
        tf2::Duration(std::chrono::seconds(20))  // Increased from 10 seconds to 20
    )),
      tf_listener_(*tf_buffer_, this)
    {
        // Initialize without declaring use_sim_time again
        if (!this->has_parameter("use_sim_time")) {
            this->declare_parameter("use_sim_time", false);
        }

        // Create subscription
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered", 10,
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1));

        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);

        RCLCPP_INFO(this->get_logger(), "Node initialized");
    }

private:
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    rclcpp::Time last_processed_time_{0, 0, RCL_ROS_TIME};
    double min_processing_interval_ = 0.5; // seconds

    double estimated_ground_height_ = 0.0;
    bool has_ground_height_ = false;
    const double ground_height_alpha_ = 0.2;
    const double max_ground_height_change_ = 0.1;

    double last_known_camera_height_ = 0.5;
    bool has_camera_height_ = false;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Throttle processing
        auto now = this->now();
        if ((now - last_processed_time_).seconds() < min_processing_interval_) {
            return;
        }
        last_processed_time_ = now;
    
        processPointCloud(msg);
    }

    void processPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (msg->data.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received an empty point cloud. Waiting for valid data...");
            return;
        }
        
        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        //Downsampling
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.02f, 0.02f, 0.02f);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);

        // Apply height filter
        pcl::PassThrough<pcl::PointXYZRGB> z_filter;
        z_filter.setInputCloud(downsampled_cloud);
        z_filter.setFilterFieldName("z");
        z_filter.setFilterLimits(-1.0, 0.5); // Relative to camera
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr height_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        z_filter.filter(*height_filtered_cloud);

        // Segment ground plane
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.03);
        seg.setMaxIterations(500);
        seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
        seg.setEpsAngle(10.0 * M_PI / 180.0);
        seg.setInputCloud(height_filtered_cloud);
        
        pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr ground_coefficients(new pcl::ModelCoefficients());
        seg.segment(*ground_inliers, *ground_coefficients);

        if (ground_inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No ground plane found! Using last height estimate");
            if (!has_ground_height_) return;
        } else {
            std::vector<float> z_values;
            for (const auto& idx : ground_inliers->indices) {
                z_values.push_back(height_filtered_cloud->points[idx].z);
            }
            std::sort(z_values.begin(), z_values.end());
            double current_height = z_values[z_values.size() / 2];

            if (has_ground_height_) {
                if (fabs(current_height - estimated_ground_height_) < max_ground_height_change_) {
                    estimated_ground_height_ = ground_height_alpha_ * current_height + 
                                             (1.0 - ground_height_alpha_) * estimated_ground_height_;
                }
            } else {
                estimated_ground_height_ = current_height;
                has_ground_height_ = true;
            }
        }

        // Extract ground points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(height_filtered_cloud);
        extract.setIndices(ground_inliers);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        // Extract non-ground points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(true);
        extract.filter(*non_ground_cloud);

        /*
        // Get camera height with fallback
        double camera_height = 0.0;
        try {
            auto tf = tf_buffer_->lookupTransform("map", "zed_camera_link", tf2::TimePointZero);
            camera_height = tf.transform.translation.z;
            last_known_camera_height_ = camera_height;
            has_camera_height_ = true;
        } catch (...) {
            if (has_camera_height_) {
                camera_height = last_known_camera_height_;
                RCLCPP_WARN(this->get_logger(), "Using last known camera height: %.2f", camera_height);
            } else {
                camera_height = 0.5;
                RCLCPP_ERROR(this->get_logger(), "Using default camera height");
            }
        }
            

        // Normalize ground plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr normalized_ground(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto& pt : *ground_cloud) {
            pcl::PointXYZRGB new_pt = pt;
            new_pt.z = camera_height;
            normalized_ground->push_back(new_pt);
        }
            */

        // Publish results
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*ground_cloud, ground_msg);
        ground_msg.header = msg->header; // Keep original frame_id
        ground_publisher_->publish(ground_msg);
        
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground_cloud, non_ground_msg);
        non_ground_msg.header = msg->header; // Keep original frame_id
        nonground_publisher_->publish(non_ground_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true);
    options.automatically_declare_parameters_from_overrides(true);
    
    auto node = std::make_shared<PointCloudProcessor>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}