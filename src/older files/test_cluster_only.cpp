#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <visualization_msgs/msg/marker.hpp>



#include <pcl/common/common.h>
#include <pcl/common/pca.h> 

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>



#include <vector>
#include <cmath>
#include <numeric>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>




class KalmanCircleLocalization : public rclcpp::Node
{
public:
    KalmanCircleLocalization() : Node("kalman_circ_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&KalmanCircleLocalization::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10,
            std::bind(&KalmanCircleLocalization::poseCallback, this, std::placeholders::_1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);
    
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);


        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);



        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);




    }

private:
    /////////////////////////////////////
    /*
    State Vecotr Initialize 12D state vector [a_L, b_L, c_L, ȧ_L, ḃ_L, ċ_L, a_R, b_R, c_R, ȧ_R, ḃ_R, ċ_R]

    */
    
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;
    Eigen::Vector3f lane_transform_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;


    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &point : cloud->points)
        {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
            if (v > 0.65f && s < 0.2f)
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }
        publishCameraAxes(current_pose_);
        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }

        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud, current_pose_.pose);


        publishClusterMarkers(white_cloud, selected_clusters, current_pose_.pose);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        Eigen::Vector2f left_observed, right_observed;
        bool left_detected = false, right_detected = false;
        float max_angle = -std::numeric_limits<float>::infinity();
        float min_angle = std::numeric_limits<float>::infinity();
            
        Eigen::Vector3f camera_position(0.0, 0.0, 0.0);  
        camera_position[0]=current_pose_.pose.position.x;
        camera_position[1]=current_pose_.pose.position.y;
        camera_position[2]=current_pose_.pose.position.z;
        
        if (selected_clusters.size() == 1)
        {
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            pcl::PointIndices cluster = selected_clusters[0]; 
            for (int idx : cluster.indices)
            {
                centroid[0] += white_cloud->points[idx].x;
                centroid[1] += white_cloud->points[idx].y;
                centroid[2] += white_cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();  // Compute centroid
    
            float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);
            
            if (angle >= 0)  // Cluster is on the left side
            {
                leftmost_cluster = cluster;
                left_observed = centroid.head<2>();
                left_detected = true;
                RCLCPP_INFO(this->get_logger(), "OI");
            }
            else  // Cluster is on the right side
            {
                rightmost_cluster = cluster;
                right_observed = centroid.head<2>();
                right_detected = true;
            }
        }
        else  // Standard case: multiple clusters
        {
            for (const auto& cluster : selected_clusters)
            {
                Eigen::Vector3f centroid(0.0, 0.0, 0.0);
                for (int idx : cluster.indices)
                {
                    centroid[0] += white_cloud->points[idx].x;
                    centroid[1] += white_cloud->points[idx].y;
                    centroid[2] += white_cloud->points[idx].z;
                }
                centroid /= cluster.indices.size();  // Compute centroid
            
                // Compute relative angle to the camera
                float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);
            
                if (angle > max_angle) 
                {
                    max_angle = angle;
                    leftmost_cluster = cluster;
                    left_observed = centroid.head<2>();
                    left_detected = true;
                }
            
                if (angle < min_angle) 
                {
                    min_angle = angle;
                    rightmost_cluster = cluster;
                    right_observed = centroid.head<2>();
                    right_detected = true;
                }
            }
        }

    }



    std::vector<pcl::PointIndices> clusterWhitePoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const geometry_msgs::msg::Pose &current_pose_)
    {
        // 1. Verify frame_id is map
        if (cloud->header.frame_id != "map") {
            RCLCPP_WARN(this->get_logger(), 
                "Expected point cloud in map frame, got %s. Proceeding anyway.",
                cloud->header.frame_id.c_str());
        }
    
        // 2. Perform Euclidean clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
        
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.07);  // 8cm
        ec.setMinClusterSize(200);     // Minimum points per cluster
        ec.setMaxClusterSize(5000);    // Maximum points per cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);
    
        // 3. Process clusters directly in map frame
        std::vector<pcl::PointIndices> filtered_clusters;
        const Eigen::Vector3f camera_pos(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);
    
        // Parameters for cluster selection
        const float max_distance_from_camera = 1.5f; // Maximum distance in meters
        const int max_clusters_to_select = 2;        // Maximum number of clusters to select
    
        // Vector to store clusters with their distances
        std::vector<std::pair<pcl::PointIndices, float>> cluster_distance_pairs;
    
        for (const auto &cluster : clusters) {
            // Calculate centroid
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            for (int idx : cluster.indices) {
                centroid[0] += cloud->points[idx].x;
                centroid[1] += cloud->points[idx].y;
                centroid[2] += cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();
    
            // Calculate distance from camera
            float distance = (centroid - camera_pos).norm();
    
            // Filter based on distance
            if (distance <= max_distance_from_camera) {
                cluster_distance_pairs.emplace_back(cluster, distance);
                RCLCPP_DEBUG(this->get_logger(),
                    "Cluster candidate: distance=%.2fm, points=%lu",
                    distance, cluster.indices.size());
            }
        }
    
        // Sort clusters by distance (closest first)
        std::sort(cluster_distance_pairs.begin(), cluster_distance_pairs.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
    
        // Select up to max_clusters_to_select closest clusters
        for (size_t i = 0; i < std::min(cluster_distance_pairs.size(), static_cast<size_t>(max_clusters_to_select)); ++i) {
            filtered_clusters.push_back(cluster_distance_pairs[i].first);
        }
    
        return filtered_clusters;
    }
    




    void RGBtoHSV(int r, int g, int b, float &h, float &s, float &v)
    {
        float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
        float maxC = std::max({rf, gf, bf});
        float minC = std::min({rf, gf, bf});
        v = maxC;

        float delta = maxC - minC;
        if (delta < 1e-5)
        {
            h = 0.0f;
            s = 0.0f;
            return;
        }

        s = (maxC > 0.0f) ? (delta / maxC) : 0.0f;

        if (maxC == rf)
            h = 60.0f * (fmod(((gf - bf) / delta), 6));
        else if (maxC == gf)
            h = 60.0f * (((bf - rf) / delta) + 2);
        else
            h = 60.0f * (((rf - gf) / delta) + 4);

        if (h < 0)
            h += 360.0f;
    }

    void publishClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
        const std::vector<pcl::PointIndices>& clusters,
        const geometry_msgs::msg::Pose& camera_pose)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;

        // Camera position and orientation in map frame
        Eigen::Vector3f camera_position(
        camera_pose.position.x,
        camera_pose.position.y,
        camera_pose.position.z);

        // Get camera orientation
        tf2::Quaternion q(
        camera_pose.orientation.x,
        camera_pose.orientation.y,
        camera_pose.orientation.z,
        camera_pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        const float cos_yaw = cos(yaw);
        const float sin_yaw = sin(yaw);

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            // Voting system for left/right determination
            int left_votes = 0;
            int right_votes = 0;
            float total_y_cam = 0.0f; // Accumulated y position in camera frame

            // First pass: analyze all points
            for (int idx : clusters[i].indices) {
                const auto& point = cloud->points[idx];
                Eigen::Vector3f relative_pos(point.x - camera_position[0],
                                    point.y - camera_position[1],
                                    point.z - camera_position[2]);

                // Transform to camera frame
                float y_cam = -relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw;
                total_y_cam += y_cam;

                // Vote based on point's position
                if (y_cam > 0) {
                    left_votes++;
                    
                } else {
                    right_votes++;
                }
            }

            // Determine cluster side based on voting
            bool is_left;
            if (left_votes == 0 && right_votes == 0) {
                is_left = true; // default if empty (shouldn't happen)

            } else {
                // Use both voting and mean y position for robustness
                float mean_y_cam = total_y_cam / clusters[i].indices.size();
                is_left = (left_votes > right_votes) || ((left_votes == right_votes) && (mean_y_cam > 0));
            }

            // Create marker
            visualization_msgs::msg::Marker cluster_marker;
            cluster_marker.header.frame_id = "map";
            cluster_marker.header.stamp = this->get_clock()->now();
            cluster_marker.ns = "clusters";
            cluster_marker.id = i;
            cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            cluster_marker.scale.x = 0.05;
            cluster_marker.scale.y = 0.05;
            cluster_marker.scale.z = 0.05;

            // Set color based on left/right
            if (is_left) {
                // Left cluster - green
                cluster_marker.color.r = 0.0;
                cluster_marker.color.g = 1.0;
                cluster_marker.color.b = 0.0;
            } else {
                // Right cluster - red
                cluster_marker.color.r = 1.0;
                cluster_marker.color.g = 0.0;
                cluster_marker.color.b = 0.0;
            }
            cluster_marker.color.a = 1.0;

            // Add points
            for (int idx : clusters[i].indices) {
                geometry_msgs::msg::Point p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                cluster_marker.points.push_back(p);
            }

            cluster_markers.markers.push_back(cluster_marker);
        }

        cluster_marker_pub_->publish(cluster_markers);
    }


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;

    //rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;


    rclcpp::Parameter fit_side_param_;
    bool fit_side_;

    void publishCameraAxes(const geometry_msgs::msg::PoseStamped &msg)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        // Extract camera position
        Eigen::Vector3f camera_position(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z);

        // Extract rotation matrix from quaternion
        tf2::Quaternion q(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        // Extract direction vectors
        Eigen::Vector3f x_axis(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]); // Right (Red)
        Eigen::Vector3f y_axis(rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]); // Up (Green)
        Eigen::Vector3f z_axis(rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]); // Forward (Blue)

        auto createArrowMarker = [&](int id, const Eigen::Vector3f &dir, const std_msgs::msg::ColorRGBA &color)
        {
            visualization_msgs::msg::Marker arrow;
            arrow.header.frame_id = "map"; // Change if needed
            arrow.header.stamp = this->now();
            arrow.ns = "camera_axes";
            arrow.id = id;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.scale.x = 0.05;  // Shaft diameter
            arrow.scale.y = 0.1;   // Head diameter
            arrow.scale.z = 0.1;   // Head length

            geometry_msgs::msg::Point start, end;
            start.x = camera_position.x();
            start.y = camera_position.y();
            start.z = camera_position.z();

            end.x = start.x + dir.x() * 0.5; // Scale for visibility
            end.y = start.y + dir.y() * 0.5;
            end.z = start.z + dir.z() * 0.5;

            arrow.points.push_back(start);
            arrow.points.push_back(end);

            arrow.color = color;
            arrow.lifetime = rclcpp::Duration::from_seconds(0.5); // Short lifespan to update
            arrow.frame_locked = true;

            return arrow;
        };

        // Define colors
        std_msgs::msg::ColorRGBA red, green, blue;
        red.r = 1.0; red.a = 1.0;
        green.g = 1.0; green.a = 1.0;
        blue.b = 1.0; blue.a = 1.0;

        marker_array.markers.push_back(createArrowMarker(0, x_axis, red));   // X-axis (Right)
        marker_array.markers.push_back(createArrowMarker(1, y_axis, green)); // Y-axis (Up)
        marker_array.markers.push_back(createArrowMarker(2, z_axis, blue));  // Z-axis (Forward)

        marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanCircleLocalization>());
    rclcpp::shutdown();
    return 0;
}
