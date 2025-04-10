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
#include <Eigen/Dense>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <visualization_msgs/msg/marker.hpp>



class RelLocalizationProcessor : public rclcpp::Node
{
public:
    RelLocalizationProcessor() : Node("rel_localization_processor")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&RelLocalizationProcessor::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10, 
            std::bind(&RelLocalizationProcessor::poseCallback, this, std::placeholders::_1));




        distance_orientation_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/distance_orientation_marker", 10);
        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_original", 10);
        marker_publisher2_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        marker_publisher3_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_shifted", 10);

        this->declare_parameter("fit_side", true); // Default to right fit
        this->get_parameter("fit_side", fit_side_);
    }

private:

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
    }
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        /*
        for (const auto& point : cloud->points)
        {
            int brightness = (point.r + point.g + point.b) / 3;
            if (brightness > 150)  
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }

        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }
            */

        for (const auto &point : cloud->points)
        {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
    
            if (v > 0.6f && s < 0.2f) // High brightness & low saturation
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }
    
        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
        }

        // Get the three closest clusters
        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud);

        if (selected_clusters.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return;
        }

        publishClusterMarkers(white_cloud, selected_clusters);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);



        // ------------------- UPDATED: Selecting Leftmost and Rightmost Clusters -------------------

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        float max_angle = -std::numeric_limits<float>::infinity();
        float min_angle = std::numeric_limits<float>::infinity();
        
        Eigen::Vector3f camera_position(0.0, 0.0, 0.0);  // Assuming camera at (0,0,0) in its own frame

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
            }

            if (angle < min_angle) 
            {
                min_angle = angle;
                rightmost_cluster = cluster;
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        
        if (!rightmost_cluster.indices.empty())
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *right_cluster);
        
        if (!leftmost_cluster.indices.empty())
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);

        if (fit_side_ ==true)
            publishCurveMarkers(right_cluster); // Green for right

        if (fit_side_ == false)
            publishCurveMarkers(left_cluster); // Red for left
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


    std::vector<pcl::PointIndices> clusterWhitePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.11);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(1500);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);

        if (clusters.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return {};
        }

        struct ClusterData
        {
            pcl::PointIndices indices;
            float min_distance;
        };
        
        std::vector<ClusterData> cluster_data;

        for (const auto& cluster : clusters)
        {
            float sum_x = 0.0, sum_z = 0.0;
            int num_points = cluster.indices.size();
        
            for (int idx : cluster.indices)
            {
                sum_x += cloud->points[idx].x;
                sum_z += cloud->points[idx].z;
            }
        
            float centroid_x = sum_x / num_points;
            float centroid_z = sum_z / num_points;
            float distance = std::sqrt(centroid_x * centroid_x + centroid_z * centroid_z);
        
            cluster_data.push_back({cluster, distance});
        }
        

        // Sort clusters by distance and keep the closest three
        std::sort(cluster_data.begin(), cluster_data.end(), [](const ClusterData& a, const ClusterData& b) {
            return a.min_distance < b.min_distance;
        });

        return {cluster_data[0].indices, cluster_data[1].indices};
    }

    void publishDistanceAndOrientation(float distance, float orientation)
    {
        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";  // Change to your relevant frame
        marker.header.stamp = this->now();
        marker.ns = "distance_orientation";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Position the text slightly above the camera position
        marker.pose.position.x = current_pose_.pose.position.x;
        marker.pose.position.y = current_pose_.pose.position.y;
        marker.pose.position.z = 2.0;  // Adjust for visibility

        marker.scale.z = 0.5;  // Adjust text size
        marker.color.a = 1.0;  // Fully visible
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;

        // Set text to display distance and orientation
        marker.text = "Dist: " + std::to_string(distance) + "m\nOrient: " + std::to_string(orientation) + " rad";

        distance_orientation_marker_pub_->publish(marker);
    }


    void computeFrenetFrame(const std::vector<geometry_msgs::msg::Point> &curve,
        float &distance, float &orientation)
    {
        if (curve.empty())
        {
        RCLCPP_WARN(this->get_logger(), "Shifted curve has no points!");
        distance = 0.0;
        orientation = 0.0;
        return;
        }

        // Convert camera position from PoseStamped
        Eigen::Vector2f camera_position(current_pose_.pose.position.x, current_pose_.pose.position.y);

        float min_dist = std::numeric_limits<float>::max();
        Eigen::Vector2f closest_point;
        size_t closest_index = 0;

        // Find the closest point on the curve
        for (size_t i = 0; i < curve.size(); ++i)
        {
        Eigen::Vector2f point(curve[i].x, curve[i].y);
        float dist = (camera_position - point).norm();

        if (dist < min_dist)
        {
        min_dist = dist;
        closest_point = point;
        closest_index = i;
        }
        }

        distance = min_dist; // The shortest distance to the curve

        // Compute tangent direction at the closest point
        Eigen::Vector2f tangent;

        if (closest_index == 0 && curve.size() > 1) // First point, use forward difference
        {
        Eigen::Vector2f next_point(curve[closest_index + 1].x, curve[closest_index + 1].y);
        tangent = (next_point - closest_point).normalized();
        }
        else if (closest_index == curve.size() - 1 && curve.size() > 1) // Last point, use backward difference
        {
        Eigen::Vector2f prev_point(curve[closest_index - 1].x, curve[closest_index - 1].y);
        tangent = (closest_point - prev_point).normalized();
        }
        else if (curve.size() > 2) // Middle points, use central difference
        {
        Eigen::Vector2f prev_point(curve[closest_index - 1].x, curve[closest_index - 1].y);
        Eigen::Vector2f next_point(curve[closest_index + 1].x, curve[closest_index + 1].y);
        tangent = (next_point - prev_point).normalized();
        }
        else
        {
        tangent = Eigen::Vector2f(1.0, 0.0); // Default to x-axis if only one point
        }

        // Convert camera orientation from quaternion to yaw angle
        tf2::Quaternion q(
        current_pose_.pose.orientation.x,
        current_pose_.pose.orientation.y,
        current_pose_.pose.orientation.z,
        current_pose_.pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // Compute the angle between camera's forward direction and tangent
        float curve_angle = atan2(tangent.y(), tangent.x());
        float angle_diff = curve_angle - yaw;

        // Compute the sign using the 2D cross product
        Eigen::Vector2f camera_forward(std::cos(yaw), std::sin(yaw));
        Eigen::Vector2f to_curve = closest_point - camera_position;
        float cross_product = camera_forward.x() * to_curve.y() - camera_forward.y() * to_curve.x();

        // Assign positive or negative sign based on the cross product
        if (cross_product < 0)
        {
            angle_diff = +std::abs(angle_diff); // Curve is to the right
        }
        else
        {
            angle_diff = -std::abs(angle_diff); // Curve is to the left
        }

        // Normalize orientation to [-π, π]
        orientation = fmod(angle_diff + M_PI, 2 * M_PI) - M_PI;

    }

    
    void publishCurveMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
    {
        std::vector<geometry_msgs::msg::Point> shifted_curve_points;
        
        if (cluster->size() < 3)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough points to fit a curve.");
            return;
        }

        // Retrieve the current setting: left fit or right fit
        this->get_parameter("fit_side", fit_side_); // Dynamic switching

        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points)
        {
            x_vals.push_back(point.x);
            y_vals.push_back(point.y);
        }

        // Fit a quadratic curve: y = ax^2 + bx + c
        Eigen::MatrixXf A(x_vals.size(), 3);
        Eigen::VectorXf Y(y_vals.size());

        for (size_t i = 0; i < x_vals.size(); i++)
        {
            A(i, 0) = x_vals[i] * x_vals[i]; // x^2
            A(i, 1) = x_vals[i];             // x
            A(i, 2) = 1;                     // constant
            Y(i) = y_vals[i];
        }

        Eigen::VectorXf coeffs = A.colPivHouseholderQr().solve(Y);
        float a = coeffs(0), b = coeffs(1), c = coeffs(2);

        // Compute transformation using camera pose
        Eigen::Matrix3f rotation_matrix;
        Eigen::Vector3f translation_vector;

        tf2::Quaternion q(
            current_pose_.pose.orientation.x,
            current_pose_.pose.orientation.y,
            current_pose_.pose.orientation.z,
            current_pose_.pose.orientation.w);
        tf2::Matrix3x3 tf_rotation(q);
        
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rotation_matrix(i, j) = tf_rotation[i][j];

        translation_vector << current_pose_.pose.position.x,
                            current_pose_.pose.position.y,
                            current_pose_.pose.position.z;

        // Decide the fit type dynamically
        int marker_id = fit_side_ ? 1 : 2; // 1 = Right Fit, 2 = Left Fit
        float shift_amount = fit_side_ ? 0.3 : -0.3; // Right = +0.3, Left = -0.3
        float r = fit_side_ ? 1.0f : 0.0f; // Right Fit = Red
        float g = fit_side_ ? 0.0f : 1.0f; // Left Fit = Green

        visualization_msgs::msg::Marker curve_marker;
        curve_marker.header.frame_id = "map";
        curve_marker.header.stamp = this->get_clock()->now();
        curve_marker.ns = "curve_fit";
        curve_marker.id = marker_id;
        curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        curve_marker.scale.x = 0.05; 
        curve_marker.color.r = r;
        curve_marker.color.g = g;
        curve_marker.color.b = 0.0;
        curve_marker.color.a = 1.0;

        for (float x = *std::min_element(x_vals.begin(), x_vals.end());
            x <= *std::max_element(x_vals.begin(), x_vals.end()); x += 0.05)
        {
            float y = a * x * x + b * x + c;
            Eigen::Vector3f point_in_map(x, y, 0.0);
            Eigen::Vector3f point_in_camera = rotation_matrix.inverse() * (point_in_map - translation_vector);

            // Apply lateral shift
            point_in_camera[1] += shift_amount;

            // Transform back to map frame
            Eigen::Vector3f translated_point = (rotation_matrix * point_in_camera) + translation_vector;

            geometry_msgs::msg::Point p;
            p.x = translated_point[0];
            p.y = translated_point[1];
            p.z = 0.0;

            curve_marker.points.push_back(p);
            shifted_curve_points.push_back(p);
        }

        float distance, delta_orientation;
        computeFrenetFrame(shifted_curve_points, distance, delta_orientation);
        publishDistanceAndOrientation(distance, delta_orientation);

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        marker_publisher_->publish(markers);
    }


    void publishClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::vector<pcl::PointIndices>& clusters)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;
        std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            visualization_msgs::msg::Marker cluster_marker;
            cluster_marker.header.frame_id = "map";
            cluster_marker.header.stamp = this->get_clock()->now();
            cluster_marker.ns = "clusters";
            cluster_marker.id = i;
            cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            cluster_marker.scale.x = 0.05;
            cluster_marker.scale.y = 0.05;
            cluster_marker.scale.z = 0.05;

            auto [r, g, b] = colors[i % colors.size()];
            cluster_marker.color.r = r;
            cluster_marker.color.g = g;
            cluster_marker.color.b = b;
            cluster_marker.color.a = 1.0;

            for (int idx : clusters[i].indices)
            {
                geometry_msgs::msg::Point p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                cluster_marker.points.push_back(p);
            }
            cluster_markers.markers.push_back(cluster_marker);
        }
        marker_publisher2_->publish(cluster_markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher3_;

    rclcpp::Parameter fit_side_param_;
    bool fit_side_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RelLocalizationProcessor>());
    rclcpp::shutdown();
    return 0;
}