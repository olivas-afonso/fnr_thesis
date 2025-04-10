#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <pcl/common/transforms.h>

class GroundPlaneProcessor : public rclcpp::Node
{
public:
    GroundPlaneProcessor() : Node("ground_plane_processor")
    {
        // Subscribe to the input point cloud topic (ground plane points)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&GroundPlaneProcessor::processPointCloud, this, std::placeholders::_1));

        plane_normal_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/plane_normal", 10, std::bind(&GroundPlaneProcessor::planeNormalCallback, this, std::placeholders::_1));

        // Publisher for the filtered point cloud
        filtered_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_ground_plane", 10);

        // Publisher for the ground plane with only white points
        //transformed_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/transformed_plane", 10);

        // Publisher for the ground plane with only white points
        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_ground_plane", 10);

        // Publisher for laser scan
        laser_scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/laser_scan", 10);



        RCLCPP_INFO(this->get_logger(), "Ground Plane Processor Node Initialized");
    }

private:

    void planeNormalCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
            // Receive the plane normal
            plane_normal_ = Eigen::Vector3f(msg->data[0], msg->data[1], msg->data[2]);
            RCLCPP_INFO(this->get_logger(), "Received plane normal: (%f, %f, %f)", plane_normal_[0], plane_normal_[1], plane_normal_[2]);
    }
    void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Wait for the plane normal to be received
        if (plane_normal_.isZero()) {
            RCLCPP_WARN(this->get_logger(), "Plane normal not yet received, skipping processing.");
            return;
        }

        if (plane_normal_.isZero()) {
            RCLCPP_WARN(this->get_logger(), "Plane normal is zero, skipping transformation.");
            return;
        }
        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points", cloud->size());

        // Define the maximum distance threshold (in meters)
        float max_distance = 2.0f;  // Change this value to whatever distance threshold you need

        // Reference point (for example, the origin, or a specific point)
        pcl::PointXYZRGB reference_point(0.0f, 0.0f, 0.0f); // Change this to your reference point

        /*
        // Step 1: Filter out points based on distance from reference point
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr distance_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        for (const auto &point : cloud->points)
        {
            // Calculate Euclidean distance between point and reference point
            float distance = std::sqrt(
                std::pow(point.x - reference_point.x, 2) +
                std::pow(point.y - reference_point.y, 2) +
                std::pow(point.z - reference_point.z, 2)
            );

            // If distance is smaller than the maximum distance, keep the point
            if (distance <= max_distance)
            {
                distance_filtered_cloud->points.push_back(point);
            }
        }

                // Ensure the point cloud is non-empty before proceeding
        if (distance_filtered_cloud->points.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Filtered cloud is empty. Skipping processing.");
            return;
        }


        distance_filtered_cloud->width = distance_filtered_cloud->points.size();
        if (distance_filtered_cloud->width > 0) {
            distance_filtered_cloud->height = 1;
        } else {
            distance_filtered_cloud->height = 0;  // Safe fallback
        }

        distance_filtered_cloud->is_dense = true;
        */

        // Transform the cloud to align with the ground plane
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // Compute the transformation matrix to align the plane normal with the z-axis
        Eigen::Vector3f vertical_axis(0.0f, 0.0f, 1.0f);
        Eigen::Vector3f rotation_axis = plane_normal_.cross(vertical_axis).normalized();

        if (rotation_axis.isZero()) {
            RCLCPP_WARN(this->get_logger(), "Rotation axis is zero, skipping transformation.");
            return;
        }

        float rotation_angle = std::acos(plane_normal_.dot(vertical_axis) / plane_normal_.norm());

        if (!std::isfinite(rotation_angle)) {
            RCLCPP_WARN(this->get_logger(), "Invalid rotation angle, skipping transformation.");
            return;
        }
        Eigen::AngleAxisf rotation_vector(rotation_angle, rotation_axis);
        Eigen::Matrix3f rotation_matrix = rotation_vector.toRotationMatrix();
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(rotation_matrix);

        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

        // Flatten points to the ground plane
        for (auto &point : transformed_cloud->points) {
            point.z = 0.0f;
        }

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
            /*
            RCLCPP_DEBUG(this->get_logger(), "Point RGB: (%d, %d, %d)", point.r, point.g, point.b);

            if (point.r > 140 && point.g > 140 && point.b > 140) // Threshold for white color
            {
                white_cloud->points.push_back(point);
            }
             */
            
            /*
            float intensity = point.r + point.g + point.b; // Total brightness
            if (intensity > 0) {
                float r_norm = point.r / intensity;
                float g_norm = point.g / intensity;
                float b_norm = point.b / intensity;

                // Check if the point is white based on normalized values
                if (r_norm > 0.8f && g_norm > 0.8f && b_norm > 0.8f && intensity > 90.0f) {
                    white_cloud->points.push_back(point);
                }
            }
            */

            float max_rgb = std::max({point.r, point.g, point.b});
            float min_rgb = std::min({point.r, point.g, point.b});
            float value = max_rgb; // Value in HSV
            float saturation = (value == 0.0f) ? 0.0f : (value - min_rgb) / value;

            if (value > 250 && saturation < 0.03) { // Bright and low saturation
                white_cloud->points.push_back(point);
            }
        }
        white_cloud->width = white_cloud->points.size();
        white_cloud->height = 1;
        white_cloud->is_dense = true;

        RCLCPP_INFO(this->get_logger(), "Filtered white points, remaining %zu points", white_cloud->size());

        // Step 2: Publish the ground plane with only white points
        
        sensor_msgs::msg::PointCloud2 white_output_msg;
        pcl::toROSMsg(*white_cloud, white_output_msg);
        white_output_msg.header = msg->header; // Keep the same frame and timestamp
        white_publisher_->publish(white_output_msg);
        

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
        RCLCPP_INFO(this->get_logger(), "Number of clusters found: %zu", cluster_indices.size());


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
            RCLCPP_INFO(this->get_logger(), "Cluster size: %zu points", cluster->size());


            // Heuristic: Retain larger clusters (assume dashed lines are small clusters)
            if (cluster->size() > 45) // Adjust this threshold based on your data
            {
                *filtered_cloud += *cluster;
            }
        }

        RCLCPP_INFO(this->get_logger(), "Filtered out small clusters, remaining %zu points", filtered_cloud->size());

        // Step 4: Publish the filtered point cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*filtered_cloud, output_msg);
        output_msg.header = msg->header; // Keep the same frame and timestamp
        filtered_publisher_->publish(output_msg);

        RCLCPP_INFO(this->get_logger(), "Published filtered ground plane point cloud");

        // Step 5: Publish the LaserScan
        publishLaserScan(filtered_cloud, msg->header, plane_normal_);
    }


    void publishLaserScan(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const std_msgs::msg::Header& header, const Eigen::Vector3f& plane_normal)
    {
        // Convert the point cloud to LaserScan
        sensor_msgs::msg::LaserScan laser_scan_msg;
        laser_scan_msg.header = header;

        // Set the properties for the laser scan (example configuration)
        laser_scan_msg.angle_min = -M_PI / 2.0;  // Min angle (-90 degrees)
        laser_scan_msg.angle_max = M_PI / 2.0;   // Max angle (90 degrees)
        laser_scan_msg.angle_increment = 0.0174532925/2  ; // 0.25 degree increments (0.004363323 radians)
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
        RCLCPP_INFO(this->get_logger(), "Number of valid lidar rays calculated: %zu", valid_rays_count);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr plane_normal_subscription_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_publisher_;
    //rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr transformed_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laser_scan_publisher_;

    Eigen::Vector3f plane_normal_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GroundPlaneProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
