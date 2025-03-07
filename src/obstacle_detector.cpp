#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>

class ColorFilteredClustering : public rclcpp::Node
{
public:
    ColorFilteredClustering() : Node("color_filtered_clustering")
    {
        // Subscribe to the input point cloud topic
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/non_ground_points", rclcpp::SensorDataQoS(),
            std::bind(&ColorFilteredClustering::processPointCloud, this, std::placeholders::_1));

        // Publishers for filtered points and final clusters
        filtered_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", 10);
        clustered_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_clusters", 10);

        RCLCPP_INFO(this->get_logger(), "Color Filtered Clustering Node Initialized");
    }

private:
    void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {

        // Convert ROS2 PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        RCLCPP_INFO(this->get_logger(), "Received point cloud with %zu points", cloud->size());

        // Filter points by color in HSV space
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_points(new pcl::PointCloud<pcl::PointXYZRGB>());

        for (const auto &point : cloud->points)
        {
            // Convert RGB to HSV
            cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(point.b, point.g, point.r));
            cv::Mat hsv;
            cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);

            cv::Vec3b hsv_color = hsv.at<cv::Vec3b>(0, 0);
            int h = hsv_color[0]; // Hue
            int s = hsv_color[1]; // Saturation
            int v = hsv_color[2]; // Value

            // Define pink range in HSV
            int target_hue = 160; // Adjust this value for the pink hue
            int hue_tolerance = 10; // Allowable range for hue
            int min_saturation = 150; // Minimum saturation for vivid pink
            int min_value = 100; // Minimum brightness

            if (std::abs(h - target_hue) <= hue_tolerance &&
                s >= min_saturation &&
                v >= min_value)
            {
                filtered_points->points.push_back(point);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Filtered cloud contains %zu points", filtered_points->size());

        // Publish the filtered points
        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*filtered_points, filtered_msg);
        filtered_msg.header = msg->header; // Retain the original header
        filtered_publisher_->publish(filtered_msg);

        // Skip clustering if there are not enough points
        if (filtered_points->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No points left after filtering. Skipping clustering.");
            return;
        }

        // Perform clustering on the filtered points
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
        tree->setInputCloud(filtered_points);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(0.1); // Distance tolerance between points (adjust as needed)
        ec.setMinClusterSize(10);    // Minimum number of points for a cluster
        ec.setMaxClusterSize(1000); // Maximum number of points for a cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(filtered_points);
        ec.extract(cluster_indices);

        RCLCPP_INFO(this->get_logger(), "Number of clusters found: %zu", cluster_indices.size());

        // Aggregate the clusters into one point cloud for publishing
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto &indices : cluster_indices)
        {
            for (const auto &idx : indices.indices)
            {
                clustered_cloud->points.push_back(filtered_points->points[idx]);
            }
        }

        // Publish the clustered points
        sensor_msgs::msg::PointCloud2 clustered_msg;
        pcl::toROSMsg(*clustered_cloud, clustered_msg);
        clustered_msg.header = msg->header; // Retain the original header
        clustered_publisher_->publish(clustered_msg);

        RCLCPP_INFO(this->get_logger(), "Published clustered points with %zu points", clustered_cloud->size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clustered_publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ColorFilteredClustering>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
