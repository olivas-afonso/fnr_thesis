#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/multi_array_layout.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>

// Use Eigen for quaternion math instead
#include <Eigen/Geometry>

class ObjectDetector : public rclcpp::Node {
public:
    ObjectDetector() : Node("object_detector") {
        // Parameters
        this->declare_parameter("cluster_tolerance", 0.05);
        this->declare_parameter("min_cluster_size", 50);
        this->declare_parameter("max_cluster_size", 10000);
        this->declare_parameter("publish_markers", true);
        this->declare_parameter("max_aspect_ratio", 3.0);  // Max width/height ratio
        this->declare_parameter("min_rectangle_size", 0.1); // Minimum rectangle dimension
        this->declare_parameter("max_rectangle_size", 1.0); // Maximum rectangle dimension
        this->declare_parameter("split_large_clusters", true); // Enable cluster splitting
        this->declare_parameter("max_split_depth", 3); // Maximum recursion depth for splitting
        this->declare_parameter("min_coverage_ratio", 0.7); // Minimum points inside rectangle to be considered good fit
        this->declare_parameter("max_empty_area_ratio", 0.3); // Maximum empty area in rectangle
        
        // Subscriber to non-ground points
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/non_ground_points", 10,
            std::bind(&ObjectDetector::pointCloudCallback, this, std::placeholders::_1));
        
        // Publisher for obstacle positions (x, y, width, height, orientation)
        obstacles_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/obstacles", 10);
        
        // Publisher for visualization markers
        markers_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacle_markers", 10);
        
        //RCLCPP_INFO(this->get_logger(), "Rectangle-based Object detector initialized");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obstacles_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_publisher_;
    
    struct Rectangle {
        float center_x;
        float center_y;
        float width;     // Major dimension
        float height;    // Minor dimension  
        float orientation; // Angle in radians
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
        double max_aspect_ratio = this->get_parameter("max_aspect_ratio").as_double();
        double min_rectangle_size = this->get_parameter("min_rectangle_size").as_double();
        double max_rectangle_size = this->get_parameter("max_rectangle_size").as_double();
        bool split_large_clusters = this->get_parameter("split_large_clusters").as_bool();
        int max_split_depth = this->get_parameter("max_split_depth").as_int();
        double min_coverage_ratio = this->get_parameter("min_coverage_ratio").as_double();
        double max_empty_area_ratio = this->get_parameter("max_empty_area_ratio").as_double();
        
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
        
        //RCLCPP_INFO(this->get_logger(), "Found %zu clusters", cluster_indices.size());
        
        for (const auto& cluster : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            
            // Extract the cluster from the cloud
            for (const auto& idx : cluster.indices) {
                object_cloud->points.push_back(cloud->points[idx]);
            }
            object_cloud->width = object_cloud->points.size();
            object_cloud->height = 1;
            object_cloud->is_dense = true;
            
            //RCLCPP_INFO(this->get_logger(), "Cluster size: %zu", object_cloud->size());
            
            // Process cluster (with recursive splitting if needed)
            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> final_clusters;
            if (split_large_clusters) {
                recursiveSplitCluster(object_cloud, final_clusters, max_rectangle_size, 
                                    min_cluster_size, max_split_depth, 0,
                                    min_coverage_ratio, max_empty_area_ratio);
            } else {
                final_clusters.push_back(object_cloud);
            }
            
            // Process all resulting clusters
            for (const auto& final_cluster : final_clusters) {
                if (final_cluster->size() >= min_cluster_size) {
                    processSingleCluster(final_cluster, obstacles_msg, markers_msg, marker_id, 
                                       msg->header, publish_markers, max_aspect_ratio,
                                       min_rectangle_size, max_rectangle_size);
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
    
    void recursiveSplitCluster(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                              std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& result_clusters,
                              double max_size,
                              int min_cluster_size,
                              int max_depth,
                              int current_depth,
                              double min_coverage_ratio,
                              double max_empty_area_ratio) {
        
        // Check if we should stop splitting (max depth reached or cluster too small)
        if (current_depth >= max_depth || cloud->size() < min_cluster_size * 2) {
            result_clusters.push_back(cloud);
            //RCLCPP_INFO(this->get_logger(), "Stopping split: depth=%d, size=%zu", current_depth, cloud->size());
            return;
        }
        
        // Check if this cluster represents a good rectangle fit
        Rectangle test_rect = fitRectangleToObject(cloud);
        FitQuality fit_quality = evaluateRectangleFit(cloud, test_rect);
        
        // Debug print
        //RCLCPP_INFO(this->get_logger(), "Cluster %zu points, rect %.2fx%.2f, coverage=%.2f, empty=%.2f, depth=%d",
               //     cloud->size(), test_rect.width, test_rect.height, 
                //    fit_quality.coverage_ratio, fit_quality.empty_volume_ratio, current_depth);
        
        // Check if rectangle is a good fit with multiple criteria
        bool good_fit = (fit_quality.coverage_ratio >= min_coverage_ratio) &&
                    (fit_quality.empty_volume_ratio <= max_empty_area_ratio) &&
                    (fit_quality.point_density_ratio >= 0.3) && // At least 30% filled
                    (test_rect.width <= max_size) && 
                    (test_rect.height <= max_size);
        
        if (good_fit) {
            result_clusters.push_back(cloud);
            //RCLCPP_INFO(this->get_logger(), "Good fit, keeping cluster");
            return;
        }
        
        //RCLCPP_INFO(this->get_logger(), "Poor fit, splitting cluster");
        
        // If not a good fit, split the cluster
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> sub_clusters = splitCluster(cloud);
        
        // Recursively process each sub-cluster
        for (const auto& sub_cluster : sub_clusters) {
            if (sub_cluster->size() >= min_cluster_size) {
                recursiveSplitCluster(sub_cluster, result_clusters, max_size, 
                                    min_cluster_size, max_depth, current_depth + 1,
                                    min_coverage_ratio, max_empty_area_ratio);
            } else if (sub_cluster->size() > 0) {
                // Too small to split further, but still has points
                result_clusters.push_back(sub_cluster);
            }
        }
    }
    
    struct FitQuality {
        double coverage_ratio;    // Percentage of points inside the rectangle
        double empty_volume_ratio;  // Percentage of 3D box volume without points
        double point_density_ratio; // Ratio of actual vs expected point density
    };
    
    FitQuality evaluateRectangleFit(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                                const Rectangle& rect) {
        FitQuality quality{0.0, 1.0, 0.0};
        if (cloud->empty()) return quality;

        // Precompute rotation matrix for -orientation (rotate points into rect frame)
        Eigen::AngleAxisf rot(-rect.orientation, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f R = rot.toRotationMatrix();

        // Project points into rect frame
        std::vector<Eigen::Vector2f> pts;
        pts.reserve(cloud->size());
        for (const auto &p : cloud->points) {
            Eigen::Vector3f v(p.x - rect.center_x, p.y - rect.center_y, p.z);
            Eigen::Vector3f v_rot = R * v;
            pts.emplace_back(v_rot.x(), v_rot.y());
        }

        // Rectangle half extents
        float half_w = rect.width / 2.0f;
        float half_h = rect.height / 2.0f;

        // Count how many points fall inside the 2D rectangle
        int points_inside = 0;
        for (const auto &v : pts) {
            if (std::abs(v.x()) <= half_w && std::abs(v.y()) <= half_h) {
                points_inside++;
            }
        }
        quality.coverage_ratio = static_cast<double>(points_inside) / cloud->size();

        // 2D grid occupancy (instead of voxel hashing)
        const float CELL = 0.05f; // 5cm grid
        int nx = static_cast<int>(std::ceil(rect.width / CELL));
        int ny = static_cast<int>(std::ceil(rect.height / CELL));
        std::vector<uint8_t> grid(nx * ny, 0);

        for (const auto &v : pts) {
            if (std::abs(v.x()) <= half_w && std::abs(v.y()) <= half_h) {
                int ix = static_cast<int>((v.x() + half_w) / CELL);
                int iy = static_cast<int>((v.y() + half_h) / CELL);
                if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                    grid[iy * nx + ix] = 1;
                }
            }
        }

        int occupied = std::count(grid.begin(), grid.end(), 1);
        int total_cells = nx * ny;

        // Empty area ratio = fraction of rectangle cells with no points
        quality.empty_volume_ratio = 1.0 - static_cast<double>(occupied) / total_cells;

        // Density ratio = occupied cells vs total cells
        quality.point_density_ratio = static_cast<double>(occupied) / total_cells;

        return quality;
    }


        
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> splitCluster(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {

        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> sub_clusters;

        if (cloud->size() < 2) {
            sub_clusters.push_back(cloud);
            return sub_clusters;
        }

        // Convert to Eigen points (2D: x,y only)
        std::vector<Eigen::Vector2f> points;
        points.reserve(cloud->size());
        for (const auto& p : cloud->points) {
            points.emplace_back(p.x, p.y);
        }

        // Initialize centroids (pick two farthest points)
        Eigen::Vector2f c1 = points.front();
        Eigen::Vector2f c2 = points.back();

        for (int iter = 0; iter < 10; ++iter) {
            Eigen::Vector2f sum1(0,0), sum2(0,0);
            int count1=0, count2=0;
            for (const auto& pt : points) {
                float d1 = (pt - c1).squaredNorm();
                float d2 = (pt - c2).squaredNorm();
                if (d1 < d2) { sum1 += pt; count1++; }
                else { sum2 += pt; count2++; }
            }
            if (count1>0) c1 = sum1 / count1;
            if (count2>0) c2 = sum2 / count2;
        }

        // Assign points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster1(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster2(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto& p : cloud->points) {
            Eigen::Vector2f pt(p.x, p.y);
            float d1 = (pt - c1).squaredNorm();
            float d2 = (pt - c2).squaredNorm();
            if (d1 < d2) cluster1->push_back(p);
            else cluster2->push_back(p);
        }

        if (!cluster1->empty()) sub_clusters.push_back(cluster1);
        if (!cluster2->empty()) sub_clusters.push_back(cluster2);

        //RCLCPP_INFO(rclcpp::get_logger("object_detector"),
        //            "KMeans split into %zu and %zu points",
        //            cluster1->size(), cluster2->size());

        return sub_clusters;
    }

    
    void processSingleCluster(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& object_cloud,
                            std_msgs::msg::Float32MultiArray& obstacles_msg,
                            visualization_msgs::msg::MarkerArray& markers_msg,
                            int& marker_id,
                            const std_msgs::msg::Header& header,
                            bool publish_markers,
                            double max_aspect_ratio,
                            double min_rectangle_size,
                            double max_rectangle_size) {
        
        // Fit rectangle to the object
        Rectangle rect = fitRectangleToObject(object_cloud);
        
        // Apply size constraints
        if (rect.width < min_rectangle_size) rect.width = min_rectangle_size;
        if (rect.height < min_rectangle_size) rect.height = min_rectangle_size;
        if (rect.width > max_rectangle_size) rect.width = max_rectangle_size;
        if (rect.height > max_rectangle_size) rect.height = max_rectangle_size;
        
        // Apply aspect ratio constraint (for circular objects)
        float aspect_ratio = rect.width / rect.height;
        if (aspect_ratio > max_aspect_ratio) {
            // Make it more square-like
            float avg_size = (rect.width + rect.height) / 2.0f;
            rect.width = avg_size;
            rect.height = avg_size;
        } else if (aspect_ratio < 1.0f / max_aspect_ratio) {
            float avg_size = (rect.width + rect.height) / 2.0f;
            rect.width = avg_size;
            rect.height = avg_size;
        }
        
        // Add to obstacles message (x, y, width, height, orientation)
        obstacles_msg.data.push_back(rect.center_x);
        obstacles_msg.data.push_back(rect.center_y);
        obstacles_msg.data.push_back(rect.width);
        obstacles_msg.data.push_back(rect.height);
        obstacles_msg.data.push_back(rect.orientation);
        
        // Create visualization markers if enabled
        if (publish_markers) {
            auto marker = createRectangleMarker(rect, object_cloud, header, marker_id++);
            markers_msg.markers.push_back(marker);
        }
        
        //RCLCPP_INFO(this->get_logger(), "Final object: %.2fx%.2f at (%.2f,%.2f)\n", 
         //           rect.width, rect.height, rect.center_x, rect.center_y);
    }
    
    Rectangle fitRectangleToObject(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& object_cloud) {
        Rectangle rect;
        
        if (object_cloud->empty()) {
            return rect;
        }
        
        // Calculate centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*object_cloud, centroid);
        rect.center_x = centroid[0];
        rect.center_y = centroid[1];
        
        // Use PCA to find principal components (orientation)
        pcl::PCA<pcl::PointXYZRGB> pca;
        pca.setInputCloud(object_cloud);
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        
        // Get orientation from first eigenvector (major axis)
        rect.orientation = std::atan2(eigen_vectors(1, 0), eigen_vectors(0, 0));
        
        // Transform points to align with principal axes
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(-rect.orientation, Eigen::Vector3f::UnitZ()));
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*object_cloud, *transformed_cloud, transform);
        
        // Find bounding box in transformed coordinates
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*transformed_cloud, min_pt, max_pt);
        
        // Width and height are the dimensions along the principal axes
        rect.width = max_pt.x - min_pt.x;   // Major dimension
        rect.height = max_pt.y - min_pt.y;  // Minor dimension
        
        // Ensure width is always the larger dimension
        if (rect.width < rect.height) {
            std::swap(rect.width, rect.height);
            rect.orientation += M_PI_2; // Rotate by 90 degrees
        }
        
        // Normalize orientation to [-π/2, π/2]
        while (rect.orientation > M_PI_2) rect.orientation -= M_PI;
        while (rect.orientation < -M_PI_2) rect.orientation += M_PI;
        
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
        float height = std::max(0.1f, max_z - min_z); // Ensure minimum height
        
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "obstacles";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = rect.center_x;
        marker.pose.position.y = rect.center_y;
        marker.pose.position.z = min_z + height/2.0f; // Center in Z
        
        // Set orientation using Eigen instead of TF2
        Eigen::AngleAxisf rotation(rect.orientation, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf quat(rotation);
        
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
        
        marker.scale.x = rect.width;
        marker.scale.y = rect.height;
        marker.scale.z = height; // Use actual height
        
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.6;  // Semi-transparent green
        
        marker.lifetime = rclcpp::Duration::from_seconds(0.2);
        
        return marker;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}