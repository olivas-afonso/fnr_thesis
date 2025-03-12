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
private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
public:
    PointCloudProcessor() : Node("point_cloud_processor"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_, this)
    {
        // Use reliable QoS
        rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered",
            qos_profile,
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1));

        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);
        plane_normal_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("plane_normal", 10);

        RCLCPP_INFO(this->get_logger(), "Point Cloud Processor Node Initialized");
    }

        
private:

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        //static rclcpp::Time last_time = this->now();
        //rclcpp::Time current_time = this->now();
        //double dt = (current_time - last_time).seconds();
        //last_time = current_time;
        //RCLCPP_INFO(this->get_logger(), "Received point cloud, dt=%.3f seconds", dt);

        // Transform to map frame before processing
        std::string target_frame = "map"; // Change this if you need "base_link" instead
        sensor_msgs::msg::PointCloud2 transformed_cloud;

        try {
            auto transform = tf_buffer_.lookupTransform(target_frame, msg->header.frame_id, msg->header.stamp);
            tf2::doTransform(*msg, transformed_cloud, transform);
            transformed_cloud.header.frame_id = target_frame;
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
            return;
        }

        // Process the transformed point cloud
        std::thread(&PointCloudProcessor::processPointCloud, this, std::make_shared<sensor_msgs::msg::PointCloud2>(transformed_cloud)).detach();
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
        voxel_filter.setLeafSize(0.03f, 0.03f, 0.03f); // Larger voxel size = faster processing
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);


        // Step 1: Downsampling and Outlier Removal


        // Step 2: Apply a Height Filter (Keep points near the ground)
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(downsampled_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.0, 0.5);  // Adjust limits if needed
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr height_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pass.filter(*height_filtered_cloud);

        // Step 3: Segment the Ground using RANSAC Plane Detection
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.03);
        seg.setMaxIterations(500);
        seg.setInputCloud(height_filtered_cloud);
        
        // Ensure it's a horizontal plane
        seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
        seg.setEpsAngle(5.0 * M_PI / 180.0);

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

        for (const auto& point : ground_cloud->points) {
            float distance = std::sqrt(point.x * point.x + point.z * point.z); // Distance in ground plane
            if (distance <= 3.0) {  // Keep points within 3 meters
                distance_filtered_cloud->points.push_back(point);
            }
        }
        distance_filtered_cloud->width = distance_filtered_cloud->points.size();
        distance_filtered_cloud->height = 1;
        distance_filtered_cloud->is_dense = true;
        
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::PassThrough<pcl::PointXYZRGB> pass;
        //pass.setInputCloud(cloud);
        //pass.setFilterFieldName("z");
        //pass.setFilterLimits(0.0, 2.0);
        //pass.filter(*filtered_cloud);

        // Downsample the point cloud using voxel grid
        //pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        //voxel_filter.setInputCloud(filtered_cloud);
        //voxel_filter.setLeafSize(0.006f, 0.006f, 0.006f); // Set voxel size
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        //voxel_filter.filter(*downsampled_cloud);

        //RCLCPP_INFO(this->get_logger(), "Downsampled point cloud to %zu points", downsampled_cloud->size());

        // Publish downsampled point cloud

        /*
        sensor_msgs::msg::PointCloud2 downsampled_msg;
        pcl::toROSMsg(*downsampled_cloud, downsampled_msg);
        downsampled_msg.header = msg->header;
        downsampled_publisher_->publish(downsampled_msg);
        */
        //////////////////////////////////


        //////////////////////////////////////


        // Perform RANSAC plane segmentation on the downsampled cloud
        /*
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02); // Adjust based on your data
        seg.setMaxIterations(1000); 
        seg.setInputCloud(downsampled_cloud);
        */
        /*
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
        */
        


        // Convert and publish the ground plane
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*distance_filtered_cloud, ground_msg);
        ground_msg.header = msg->header;
        ground_publisher_->publish(ground_msg);

        // Convert and publish non-ground points
        
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground_cloud, non_ground_msg);
        non_ground_msg.header = msg->header;
        nonground_publisher_->publish(non_ground_msg);
        

        //RCLCPP_INFO(this->get_logger(), "Published ground and non-ground point clouds");
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
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
}

