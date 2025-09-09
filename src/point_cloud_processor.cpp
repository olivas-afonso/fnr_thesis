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
        tf2::Duration(std::chrono::seconds(20))
    )),
    tf_listener_(*tf_buffer_, this)
    {
        if (!this->has_parameter("use_sim_time")) {
            this->declare_parameter("use_sim_time", false);
        }

        // Tunable parameters with default values
        this->declare_parameter("voxel_leaf_size", 0.02);
        this->declare_parameter("z_min", -1.0);
        this->declare_parameter("z_max", 0.5);
        this->declare_parameter("ransac_distance_threshold", 0.015);
        this->declare_parameter("ransac_max_iterations", 500);
        this->declare_parameter("ransac_eps_angle_deg", 10.0);
        this->declare_parameter("min_ground_points", 100);
        this->declare_parameter("ground_height_alpha", 0.2);
        this->declare_parameter("max_ground_height_change", 0.1);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered", 10,
            std::bind(&PointCloudProcessor::pointCloudCallback, this, std::placeholders::_1));

        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_plane", 10);
        nonground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);

        RCLCPP_INFO(this->get_logger(), "Node initialized with tunable parameters");
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

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        processPointCloud(msg);
    }

    void processPointCloud(sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (msg->data.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received an empty point cloud");
            return;
        }
        
        // Get current parameter values (can be changed at runtime)
        double voxel_leaf_size = this->get_parameter("voxel_leaf_size").as_double();
        double z_min = this->get_parameter("z_min").as_double();
        double z_max = this->get_parameter("z_max").as_double();
        double ransac_distance = this->get_parameter("ransac_distance_threshold").as_double();
        int ransac_iterations = this->get_parameter("ransac_max_iterations").as_int();
        double ransac_eps_angle_deg = this->get_parameter("ransac_eps_angle_deg").as_double();
        int min_ground_points = this->get_parameter("min_ground_points").as_int();
        double ground_height_alpha = this->get_parameter("ground_height_alpha").as_double();
        double max_ground_height_change = this->get_parameter("max_ground_height_change").as_double();
        
        // Convert to PCL (already in base_link frame now)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        // Downsampling
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_filter.filter(*downsampled_cloud);

        // Apply height filter (now relative to base_link)
        pcl::PassThrough<pcl::PointXYZRGB> z_filter;
        z_filter.setInputCloud(downsampled_cloud);
        z_filter.setFilterFieldName("z");
        z_filter.setFilterLimits(z_min, z_max);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr height_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        z_filter.filter(*height_filtered_cloud);

        // Segment ground plane with tunable parameters
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransac_distance);
        seg.setMaxIterations(ransac_iterations);
        seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
        seg.setEpsAngle(ransac_eps_angle_deg * M_PI / 180.0);
        seg.setInputCloud(height_filtered_cloud);
        
        pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr ground_coefficients(new pcl::ModelCoefficients());
        seg.segment(*ground_inliers, *ground_coefficients);

        if (!ground_inliers->indices.empty() && ground_inliers->indices.size() >= min_ground_points) {
            // Use a subset of points for faster median calculation
            std::vector<float> z_values;
            size_t step = std::max(1, static_cast<int>(ground_inliers->indices.size() / 100));
            for (size_t i = 0; i < ground_inliers->indices.size(); i += step) {
                z_values.push_back(height_filtered_cloud->points[ground_inliers->indices[i]].z);
            }
            
            // Partial sort instead of full sort for median
            auto mid = z_values.begin() + z_values.size() / 2;
            std::nth_element(z_values.begin(), mid, z_values.end());
            double current_height = *mid;

            if (has_ground_height_) {
                if (fabs(current_height - estimated_ground_height_) < max_ground_height_change) {
                    estimated_ground_height_ = ground_height_alpha * current_height + 
                                            (1.0 - ground_height_alpha) * estimated_ground_height_;
                }
            } else {
                estimated_ground_height_ = current_height;
                has_ground_height_ = true;
            }
        } else if (!has_ground_height_) {
            RCLCPP_WARN(this->get_logger(), "No ground plane found and no previous estimate available");
            return;
        }

        // Extract ground and non-ground points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(height_filtered_cloud);
        extract.setIndices(ground_inliers);
        
        // Ground points
        extract.setNegative(false);
        extract.filter(*ground_cloud);
        
        // Non-ground points
        extract.setNegative(true);
        extract.filter(*non_ground_cloud);

        // Publish results
        sensor_msgs::msg::PointCloud2 ground_msg;
        pcl::toROSMsg(*ground_cloud, ground_msg);
        ground_msg.header = msg->header;
        ground_publisher_->publish(ground_msg);
        
        sensor_msgs::msg::PointCloud2 non_ground_msg;
        pcl::toROSMsg(*non_ground_cloud, non_ground_msg);
        non_ground_msg.header = msg->header;
        nonground_publisher_->publish(non_ground_msg);
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "Processed: %zu ground points, %zu non-ground points, ground height: %.3f",
                    ground_cloud->size(), non_ground_cloud->size(), estimated_ground_height_);
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