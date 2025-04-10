#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <chrono>
#include <iostream>



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
        //ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        //nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);

        // Publisher for the plane normal
        //plane_normal_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("plane_normal", 10);

        // Publisher for the filtered point cloud
        filtered_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_ground_plane", 10);

        // Publisher for the ground plane with only white points
        //transformed_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_plane", 10);

        // Publisher for the ground plane with only white points
        //white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_ground_plane", 10);

        // Publisher for laser scan
        laser_scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/laser_scan", 10);

        

        RCLCPP_INFO(this->get_logger(), "Point Cloud Processor Node Initialized");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {

        // Record the current time
        auto current_time = std::chrono::high_resolution_clock::now();

        // Calculate FPS if we have a previous timestamp
        if (last_time_)
        {
            std::chrono::duration<double> elapsed_time = current_time - *last_time_;
            double fps = 1.0 / elapsed_time.count();
            RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps);
        }

        // Update the last processed time
        last_time_ = current_time;

        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        //RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points", cloud->size());

        // Remove points that are far away
        double distance_threshold = 1.5; // Set your maximum distance threshold (in meters)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto& point : cloud->points)
        {
            double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (distance <= distance_threshold)
            {
                aux_cloud->points.push_back(point);
            }
        }
        aux_cloud->width = aux_cloud->points.size();
        aux_cloud->height = 1;
        aux_cloud->is_dense = true;

        // Downsample the point cloud using voxel grid
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(aux_cloud);
        voxel_filter.setLeafSize(0.008f, 0.008f, 0.008f); // Set voxel size
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);

        //RCLCPP_INFO(this->get_logger(), "Downsampled point cloud to %zu points", downsampled_cloud->size());

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


        //RCLCPP_INFO(this->get_logger(), "Plane segmentation found %zu inliers", inliers->indices.size());

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


        // Convert and publish non-ground points
        /*
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground, non_ground_msg);
        non_ground_msg.header = msg->header;
        nonground_publisher_->publish(non_ground_msg);
        */

        //RCLCPP_INFO(this->get_logger(), "Published ground and non-ground point clouds");

        // Define the maximum distance threshold (in meters)
        float max_distance = 2.0f;  // Change this value to whatever distance threshold you need

        // Reference point (for example, the origin, or a specific point)
        pcl::PointXYZRGB reference_point(0.0f, 0.0f, 0.0f); // Change this to your reference point

        // Transform the cloud to align with the ground plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // Compute the transformation matrix to align the plane normal with the z-axis
        Eigen::Vector3f vertical_axis(0.0f, 0.0f, 1.0f);
        Eigen::Vector3f rotation_axis = plane_normal.cross(vertical_axis).normalized();

        if (rotation_axis.isZero()) {
            RCLCPP_WARN(this->get_logger(), "Rotation axis is zero, skipping transformation.");
            return;
        }

        float rotation_angle = std::acos(plane_normal.dot(vertical_axis) / plane_normal.norm());

        if (!std::isfinite(rotation_angle)) {
            RCLCPP_WARN(this->get_logger(), "Invalid rotation angle, skipping transformation.");
            return;
        }
        Eigen::AngleAxisf rotation_vector(rotation_angle, rotation_axis);
        Eigen::Matrix3f rotation_matrix = rotation_vector.toRotationMatrix();
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(rotation_matrix);

        pcl::transformPointCloud(*ground_plane, *transformed_cloud, transform);

        // Publish the transformed point cloud for debugging
        /*
        sensor_msgs::msg::PointCloud2 transformed_msg;
        pcl::toROSMsg(*transformed_cloud, transformed_msg);
        transformed_msg.header = msg->header;
        transformed_publisher_->publish(transformed_msg);
        */


        // Step 1: Filter out non-white points based on RGB values
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto &point : transformed_cloud->points)
        {

            float max_rgb = std::max({point.r, point.g, point.b});
            float min_rgb = std::min({point.r, point.g, point.b});
            float value = max_rgb; // Value in HSV
            float saturation = (value == 0.0f) ? 0.0f : (value - min_rgb) / value;

            if (value > 200 && saturation < 0.2) { // Bright and low saturation
                white_cloud->points.push_back(point);
            }
        }
        white_cloud->width = white_cloud->points.size();
        white_cloud->height = 1;
        white_cloud->is_dense = true;

        //RCLCPP_INFO(this->get_logger(), "Filtered white points, remaining %zu points", white_cloud->size());


        // Step 2: Perform clustering to identify dashed lines
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        tree->setInputCloud(white_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(0.1); // Distance tolerance between points (adjust as needed)
        ec.setMinClusterSize(5);    // Minimum number of points for a cluster
        ec.setMaxClusterSize(1000);  // Maximum number of points for a cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(white_cloud);
        ec.extract(cluster_indices);


        

        // Debugging: Print the number of clusters created
        //RCLCPP_INFO(this->get_logger(), "Number of clusters found: %zu", cluster_indices.size());


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // Step 3: Remove small clusters (e.g., dashed lines)
        for (const auto &indices : cluster_indices)
        {
            // Calculate the bounding box or size of the cluster
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>());
            for (const auto &idx : indices.indices)
            {
                cluster->points.push_back(white_cloud->points[idx]);
            }
            // Print the size of each cluster
            //RCLCPP_INFO(this->get_logger(), "Cluster size: %zu points", cluster->size());


            // Heuristic: Retain larger clusters (assume dashed lines are small clusters)
            if (cluster->size() > 50) // Adjust this threshold based on your data
            {
                *filtered_cloud += *cluster;
            }
        }

        //RCLCPP_INFO(this->get_logger(), "Filtered out small clusters, remaining %zu points", filtered_cloud->size());

        // Step 4: Publish the filtered point cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*filtered_cloud, output_msg);
        output_msg.header = msg->header; // Keep the same frame and timestamp
        filtered_publisher_->publish(output_msg);

        //RCLCPP_INFO(this->get_logger(), "Published filtered ground plane point cloud");

        // Step 5: Publish the LaserScan
        publishLaserScan(filtered_cloud, msg->header, plane_normal);
    }


    void publishLaserScan(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std_msgs::msg::Header& header, const Eigen::Vector3f& plane_normal)
    {
        // Convert the point cloud to LaserScan
        sensor_msgs::msg::LaserScan laser_scan_msg;
        laser_scan_msg.header = header;

        // Set the properties for the laser scan (example configuration)
        laser_scan_msg.angle_min = -M_PI / 2.0;  // Min angle (-90 degrees)
        laser_scan_msg.angle_max = M_PI / 2.0;   // Max angle (90 degrees)
        laser_scan_msg.angle_increment = 0.0021816615  ; // 0.25 degree increments (0.004363323 radians)
        laser_scan_msg.range_min = 0.01;           // Minimum range (meters)
        laser_scan_msg.range_max = 2.0;          // Maximum range (meters)
        laser_scan_msg.time_increment = 0.0;     // Optional: Time increment between scans
        laser_scan_msg.scan_time = 1.0;           // Scan time (optional)

        // Initialize ranges with maximum range
        size_t num_ranges = (laser_scan_msg.angle_max - laser_scan_msg.angle_min) / laser_scan_msg.angle_increment;
        laser_scan_msg.ranges.assign(num_ranges, laser_scan_msg.range_max);

        size_t valid_rays_count = 0; // Counter for valid rays


        // Convert the transformed point cloud to LaserScan ranges
        for (const auto &point : cloud->points)
        {
            // Calculate the angle and distance for the point
            float angle = std::atan2(point.y, point.x);
            float range = std::sqrt(point.x * point.x + point.y * point.y);

            if (!std::isfinite(range)) {
                continue;  // Skip this point if range is not valid
            }


            if (angle >= laser_scan_msg.angle_min && angle <= laser_scan_msg.angle_max && range <= laser_scan_msg.range_max)
            {
                // Calculate the index in the LaserScan ranges
                size_t index = (size_t)((angle - laser_scan_msg.angle_min) / laser_scan_msg.angle_increment);

                // Update the range if it's smaller than the current value
                if (range < laser_scan_msg.ranges[index])
                {
                    laser_scan_msg.ranges[index] = range;
                }
            }
        }

        // Publish the LaserScan message
        laser_scan_publisher_->publish(laser_scan_msg);
        //RCLCPP_INFO(this->get_logger(), "Number of valid lidar rays calculated: %zu", valid_rays_count);
    }
    

    /*
    void publishPlaneNormal(const Eigen::Vector3f& normal)
    {
        std_msgs::msg::Float32MultiArray plane_normal_msg;
        plane_normal_msg.data.push_back(normal[0]);
        plane_normal_msg.data.push_back(normal[1]);
        plane_normal_msg.data.push_back(normal[2]);
            
        plane_normal_publisher_->publish(plane_normal_msg);
    }
    */

       
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_publisher_;
    std::optional<std::chrono::high_resolution_clock::time_point> last_time_; // For FPS calculation
    //rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_publisher_;
    //rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    //rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
    //rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr plane_normal_publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
