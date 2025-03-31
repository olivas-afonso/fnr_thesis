#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <thread>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>


class PointCloudProcessor : public rclcpp::Node
{
    
public:
    PointCloudProcessor() : Node("point_cloud_processor"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_, this) 
    {
        // Create a Reentrant callback group for parallel processing
        callback_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        // Subscription options to use the callback group
        rclcpp::SubscriptionOptions options;
        options.callback_group = callback_group_;

        // Use reliable QoS
        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered",
            qos_profile,
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1),
            options);

        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);

        RCLCPP_INFO(this->get_logger(), "Node initialized with Reentrant callback group");
    }

        
private:

private:
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Transform to map frame
        std::string target_frame = "map";
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        
        try {
            auto transform = tf_buffer_.lookupTransform(
                target_frame, msg->header.frame_id, msg->header.stamp);
            tf2::doTransform(*msg, transformed_cloud, transform);
            transformed_cloud.header.frame_id = target_frame;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF error: %s", ex.what());
            return;
        }

        // Process directly (executor handles threading)
        processPointCloud(std::make_shared<sensor_msgs::msg::PointCloud2>(transformed_cloud));
    }
    


    void processPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (msg->data.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received an empty point cloud. Waiting for valid data...");
            return; // Skip processing until we get valid data
        }
        
        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        //RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points", cloud->size());

        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.02f, 0.02f, 0.02f); // Larger voxel size = faster processing
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);


        // Step 1: Downsampling and Outlier Removal


        // Step 2: Apply a Height Filter (Keep points near the ground)
        pcl::PassThrough<pcl::PointXYZRGB> z_filter;
        z_filter.setInputCloud(downsampled_cloud);
        z_filter.setFilterFieldName("z");
        try {
            auto tf = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
            double robot_height = tf.transform.translation.z;
            z_filter.setFilterLimits(robot_height - 1.5, robot_height + 1.2);
        } catch (...) {
            z_filter.setFilterLimits(-1.0, 0.5);  // Fallback
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr height_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        z_filter.filter(*height_filtered_cloud);

        // Step 3: Segment the Ground using RANSAC Plane Detection
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.03);
        seg.setMaxIterations(500);
        seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));  // Z-axis = "up"
        seg.setEpsAngle(10.0 * M_PI / 180.0);         // Tighten angle tolerance (10°)
        seg.setInputCloud(height_filtered_cloud);
        
    
        pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr ground_coefficients(new pcl::ModelCoefficients());
        seg.segment(*ground_inliers, *ground_coefficients);

        if (ground_inliers->indices.empty()) {
            std::cout << "No ground plane found!" << std::endl;
            return;
        }


        // Step 4: Extract Ground Points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(height_filtered_cloud);
        extract.setIndices(ground_inliers);
        extract.setNegative(false);  // Extract the ground
        extract.filter(*ground_cloud);

        // Step 5: Extract Non-Ground Points (Objects + Walls)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(true);  // Extract everything except ground
        extract.filter(*non_ground_cloud);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr distance_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        *distance_filtered_cloud = *ground_cloud;


        // Convert and publish the ground plane
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*distance_filtered_cloud, ground_msg);
        ground_msg.header = msg->header;
        ground_msg.header.frame_id = "map"; 
        ground_publisher_->publish(ground_msg);

        // Convert and publish non-ground points
        
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground_cloud, non_ground_msg);
        non_ground_msg.header = msg->header;
        non_ground_msg.header.frame_id = "map";  // Add this line
        nonground_publisher_->publish(non_ground_msg);
        

        //RCLCPP_INFO(this->get_logger(), "Published ground and non-ground point clouds");
    }

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();

    // MultiThreadedExecutor with 2 threads (adjust as needed)
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 2 /* number of threads */);
    
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}

