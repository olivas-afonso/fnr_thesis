#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>

class SimpleObjectDetector : public rclcpp::Node {
public:
    SimpleObjectDetector() : Node("simple_object_detector") {
        // Parameters
        this->declare_parameter("cluster_tolerance", 0.05);
        this->declare_parameter("min_cluster_size", 20);
        this->declare_parameter("max_cluster_size", 10000);
        this->declare_parameter("publish_markers", true);
        this->declare_parameter("min_object_size", 0.02);
        this->declare_parameter("max_object_size", 1.0);
        this->declare_parameter("max_split_depth", 3);
        this->declare_parameter("split_threshold", 0.7);
        
        // Subscriber to non-ground points
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/non_ground_points", 10,
            std::bind(&SimpleObjectDetector::pointCloudCallback, this, std::placeholders::_1));
        
        // Publisher for obstacle positions (x, y, width, height, orientation)
        obstacles_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/obstacles", 10);
        
        // Publisher for visualization markers
        markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacle_markers", 10);
        
        RCLCPP_INFO(this->get_logger(), "Object detector initialized");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obstacles_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_publisher_;
    
    struct Rectangle {
        float center_x;
        float center_y;
        float width;
        float height;  
        float orientation;
    };
    
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert to PCL
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);
        
        if (cloud->empty()) {
            RCLCPP_DEBUG(this->get_logger(), "Received empty point cloud");
            return;
        }
        
        // Get parameters
        double cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
        int min_cluster_size = this->get_parameter("min_cluster_size").as_int();
        int max_cluster_size = this->get_parameter("max_cluster_size").as_int();
        bool publish_markers = this->get_parameter("publish_markers").as_bool();
        double min_object_size = this->get_parameter("min_object_size").as_double();
        double max_object_size = this->get_parameter("max_object_size").as_double();
        int max_split_depth = this->get_parameter("max_split_depth").as_int();
        double split_threshold = this->get_parameter("split_threshold").as_double();
        
        // Create KD-tree for clustering
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(cloud);
        
        // Perform Euclidean clustering
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(cluster_tolerance);
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        
        // Prepare messages
        std_msgs::msg::Float32MultiArray obstacles_msg;
        visualization_msgs::msg::MarkerArray markers_msg;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        markers_msg.markers.push_back(clear_marker);
        
        // Process each cluster
        int marker_id = 0;
        int total_objects = 0;
        
        //RCLCPP_INFO(this->get_logger(), "Found %zu initial clusters", cluster_indices.size());
        
        for (const auto& cluster : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            // Extract the cluster from the cloud
            for (const auto& idx : cluster.indices) {
                object_cloud->points.push_back(cloud->points[idx]);
            }
            object_cloud->width = object_cloud->points.size();
            object_cloud->height = 1;
            object_cloud->is_dense = true;
            
            // Check if this cluster should be processed as a single object
            Rectangle rect = fitTightBoundingBox(object_cloud);
            
            // If the object is already within acceptable size limits, don't split it
            if (rect.width <= max_object_size && rect.height <= max_object_size &&
                std::max(rect.width, rect.height) / std::min(rect.width, rect.height) <= split_threshold) {
                
                if (rect.width >= min_object_size && rect.height >= min_object_size) {
                    processCluster(object_cloud, obstacles_msg, markers_msg, marker_id, 
                                 msg->header, publish_markers, rect);
                    total_objects++;
                }
                continue;
            }
            
            // Otherwise, recursively split large clusters
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> final_clusters;
            recursiveClusterSplit(object_cloud, final_clusters, max_object_size, 
                                min_cluster_size, max_split_depth, 0, split_threshold, min_object_size);
            
            // Process all resulting clusters
            for (const auto& final_cluster : final_clusters) {
                Rectangle final_rect = fitTightBoundingBox(final_cluster);
                if (final_rect.width >= min_object_size && final_rect.height >= min_object_size) {
                    processCluster(final_cluster, obstacles_msg, markers_msg, marker_id, 
                                 msg->header, publish_markers, final_rect);
                    total_objects++;
                }
            }
        }
        
        // Update the multiarray dimensions with actual object count
        obstacles_msg.layout.dim.clear();
        obstacles_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        obstacles_msg.layout.dim[0].label = "obstacles";
        obstacles_msg.layout.dim[0].size = total_objects;
        obstacles_msg.layout.dim[0].stride = total_objects * 5;
        
        // Publish results
        obstacles_publisher_->publish(obstacles_msg);
        
        if (publish_markers) {
            markers_publisher_->publish(markers_msg);
        }
        
        //RCLCPP_INFO(this->get_logger(), "Detected %d objects", total_objects);
    }
    
    void processCluster(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                      std_msgs::msg::Float32MultiArray& obstacles_msg,
                      visualization_msgs::msg::MarkerArray& markers_msg,
                      int& marker_id,
                      const std_msgs::msg::Header& header,
                      bool publish_markers,
                      const Rectangle& rect) {
        
        // Add to obstacles message
        obstacles_msg.data.push_back(rect.center_x);
        obstacles_msg.data.push_back(rect.center_y);
        obstacles_msg.data.push_back(rect.width);
        obstacles_msg.data.push_back(rect.height);
        obstacles_msg.data.push_back(rect.orientation);
        
        // Create visualization markers if enabled
        if (publish_markers) {
            auto marker = createRectangleMarker(rect, cloud, header, marker_id++);
            markers_msg.markers.push_back(marker);
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Object: %.2fx%.2f at (%.2f,%.2f) angle: %.2f", 
                  rect.width, rect.height, rect.center_x, rect.center_y, rect.orientation);
    }
    
    void recursiveClusterSplit(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                             std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& result_clusters,
                             double max_size,
                             int min_cluster_size,
                             int max_depth,
                             int current_depth,
                             double split_threshold,
                             double min_object_size) {
        
        // Check if we should stop splitting
        if (current_depth >= max_depth || cloud->size() < min_cluster_size * 2) {
            result_clusters.push_back(cloud);
            return;
        }
        
        // Fit a bounding box to check if splitting is needed
        Rectangle rect = fitTightBoundingBox(cloud);
        
        // Check if splitting would create objects that are too small
        // If the cluster is only slightly larger than max_size, don't split it
        bool would_create_too_small = (rect.width <= max_size * 1.2 && rect.height <= max_size * 1.2);
        
        // Check if the cluster should be split
        bool should_split = (rect.width > max_size || rect.height > max_size) &&
                          (std::max(rect.width, rect.height) / std::min(rect.width, rect.height) > split_threshold) &&
                          !would_create_too_small;
        
        if (!should_split) {
            result_clusters.push_back(cloud);
            return;
        }
        
        RCLCPP_DEBUG(this->get_logger(), "Splitting cluster: size=%zu, dim=%.2fx%.2f, depth=%d",
                   cloud->size(), rect.width, rect.height, current_depth);
        
        // Split the cluster along its major axis
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> sub_clusters = splitClusterAlongMajorAxis(cloud, rect);
        
        // Recursively process each sub-cluster
        for (const auto& sub_cluster : sub_clusters) {
            if (sub_cluster->size() >= min_cluster_size) {
                recursiveClusterSplit(sub_cluster, result_clusters, max_size, 
                                    min_cluster_size, max_depth, current_depth + 1, 
                                    split_threshold, min_object_size);
            } else if (sub_cluster->size() > 0) {
                result_clusters.push_back(sub_cluster);
            }
        }
    }
    
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> splitClusterAlongMajorAxis(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
        const Rectangle& rect) {
        
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> sub_clusters;
        
        if (cloud->size() < 2) {
            sub_clusters.push_back(cloud);
            return sub_clusters;
        }
        
        // Transform points to align with the rectangle's orientation
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-rect.orientation, Eigen::Vector3f::UnitZ()));
        transform.translate(Eigen::Vector3f(-rect.center_x, -rect.center_y, 0.0f));
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
        
        // Find median along the major axis (X-axis after transformation)
        std::vector<float> x_values;
        x_values.reserve(transformed_cloud->size());
        for (const auto& point : transformed_cloud->points) {
            x_values.push_back(point.x);
        }
        std::sort(x_values.begin(), x_values.end());
        float median_x = x_values[x_values.size() / 2];
        
        // Split based on median
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        for (const auto& point : cloud->points) {
            // Transform the point to check its position relative to median
            Eigen::Vector3f transformed_point = transform * Eigen::Vector3f(point.x, point.y, point.z);
            if (transformed_point.x() < median_x) {
                left_cluster->push_back(point);
            } else {
                right_cluster->push_back(point);
            }
        }
        
        if (!left_cluster->empty()) sub_clusters.push_back(left_cluster);
        if (!right_cluster->empty()) sub_clusters.push_back(right_cluster);
        
        return sub_clusters;
    }
    
    Rectangle fitTightBoundingBox(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
        Rectangle rect;
        
        if (cloud->empty()) {
            return rect;
        }
        
        // For oriented bounding box, use PCA
        pcl::PCA<pcl::PointXYZRGB> pca;
        pca.setInputCloud(cloud);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        
        // Get orientation from first eigenvector (major axis)
        rect.orientation = std::atan2(eigen_vectors(1, 0), eigen_vectors(0, 0));
        
        // Transform points to align with principal axes for tighter fit
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-rect.orientation, Eigen::Vector3f::UnitZ()));
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
        
        // Find bounding box in transformed coordinates
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*transformed_cloud, min_pt, max_pt);
        
        rect.width = max_pt.x - min_pt.x;
        rect.height = max_pt.y - min_pt.y;
        
        // Transform center back to original coordinate system
        Eigen::Vector3f center_transformed((min_pt.x + max_pt.x) / 2.0f, 
                                         (min_pt.y + max_pt.y) / 2.0f, 0.0f);
        Eigen::Vector3f center_original = transform.inverse() * center_transformed;
        rect.center_x = center_original.x();
        rect.center_y = center_original.y();
        
        return rect;
    }
    
    visualization_msgs::msg::Marker createRectangleMarker(
        const Rectangle& rect,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
        const std_msgs::msg::Header& header,
        int id) {
        
        // Calculate height from point cloud
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        for (const auto& point : cloud->points) {
            min_z = std::min(min_z, point.z);
            max_z = std::max(max_z, point.z);
        }
        float height = std::max(0.1f, max_z - min_z);
        
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "obstacles";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = rect.center_x;
        marker.pose.position.y = rect.center_y;
        marker.pose.position.z = min_z + height / 2.0f;
        
        // Set orientation
        Eigen::AngleAxisf rotation(rect.orientation, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf quat(rotation);
        
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
        
        marker.scale.x = rect.width;
        marker.scale.y = rect.height;
        marker.scale.z = height;
        
        // Different colors based on object size
        float size_ratio = std::max(rect.width, rect.height) / this->get_parameter("max_object_size").as_double();
        marker.color.r = std::min(1.0f, size_ratio);
        marker.color.g = 1.0f - std::min(1.0f, size_ratio);
        marker.color.b = 0.0;
        marker.color.a = 0.6;
        
        marker.lifetime = rclcpp::Duration::from_seconds(0.2);
        
        return marker;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleObjectDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}