#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "std_msgs/msg/float32_multi_array.hpp"

class PointCloudProcessor : public rclcpp::Node
{
public:
    PointCloudProcessor() : Node("point_cloud_processor")
    {
        // Subscribe to input point cloud topic
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered", rclcpp::SensorDataQoS(),
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1));

        // Publishers for downsampled and segmented point clouds
        //downsampled_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/downsampled_point_cloud", 10);
        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);

        // Publisher for the plane normal
        plane_normal_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("plane_normal", 10);
        

        RCLCPP_INFO(this->get_logger(), "Point Cloud Processor Node Initialized");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points", cloud->size());

        // Remove points that are far away
        double distance_threshold = 1.5; // Set your maximum distance threshold (in meters)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto& point : cloud->points)
        {
            double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (distance <= distance_threshold)
            {
                filtered_cloud->points.push_back(point);
            }
        }
        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = true;

        // Downsample the point cloud using voxel grid
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud);
        voxel_filter.setLeafSize(0.004f, 0.004f, 0.004f); // Set voxel size
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);

        RCLCPP_INFO(this->get_logger(), "Downsampled point cloud to %zu points", downsampled_cloud->size());

        // Publish downsampled point cloud
        /*
        sensor_msgs::msg::PointCloud2 downsampled_msg;
        pcl::toROSMsg(*downsampled_cloud, downsampled_msg);
        downsampled_msg.header = msg->header;
        downsampled_publisher_->publish(downsampled_msg);
        */

        // Perform RANSAC plane segmentation on the downsampled cloud
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02); // Adjust based on your data
        seg.setInputCloud(downsampled_cloud);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
        RCLCPP_WARN(this->get_logger(), "RANSAC failed to find a plane. Skipping processing.");
        return;
        }

        Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

        // Publish the plane normal
        publishPlaneNormal(plane_normal);


        RCLCPP_INFO(this->get_logger(), "Plane segmentation found %zu inliers", inliers->indices.size());

        // Extract ground plane points
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(downsampled_cloud);
        extract.setIndices(inliers);

        // Extract the ground plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(false);
        extract.filter(*ground_plane);

        // Extract non-ground points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_ground(new pcl::PointCloud<pcl::PointXYZRGB>());
        extract.setNegative(true);
        extract.filter(*non_ground);

        // Convert and publish the ground plane
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*ground_plane, ground_msg);
        ground_msg.header = msg->header;
        ground_publisher_->publish(ground_msg);

        // Convert and publish non-ground points
        
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground, non_ground_msg);
        non_ground_msg.header = msg->header;
        nonground_publisher_->publish(non_ground_msg);
        

        RCLCPP_INFO(this->get_logger(), "Published ground and non-ground point clouds");
    }

    void publishPlaneNormal(const Eigen::Vector3f& normal)
    {
        std_msgs::msg::Float32MultiArray plane_normal_msg;
        plane_normal_msg.data.push_back(normal[0]);
        plane_normal_msg.data.push_back(normal[1]);
        plane_normal_msg.data.push_back(normal[2]);
            
        plane_normal_publisher_->publish(plane_normal_msg);
    }

       
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    //rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr plane_normal_publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
