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




        //distance_orientation_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/distance_orientation_marker", 10);
        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_original", 10);
        marker_publisher2_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("cluster_markers", 10);


        this->declare_parameter("fit_side", true); // Default to right fit
        this->get_parameter("fit_side", fit_side_);


        // Initialize state (arbitrary starting values)
        state << 0, 0, 1.50, 0;  // Assuming road width is 1.51m

        // Initialize covariance matrix P
        P = Eigen::Matrix4f::Identity() * 0.1;  // Small initial uncertainty

        // Process noise (how much we trust our motion model)
        Q = Eigen::Matrix4f::Identity() * 0.01;

        // Measurement noise (how much we trust sensor readings)
        R = Eigen::Matrix4f::Identity() * 0.5;

        // Identity matrix
        I = Eigen::Matrix4f::Identity();

        // State transition model (assuming road markings move smoothly)
        F = Eigen::Matrix4f::Identity();  // No movement model for now

        // Measurement model (we observe both left and right markings)
        H << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    }

private:
    Eigen::Vector4f state;  // [x_left, y_left, x_right, y_right]
    Eigen::Matrix4f P;       // Covariance matrix
    Eigen::Matrix4f Q;       // Process noise
    Eigen::Matrix4f R;       // Measurement noise
    Eigen::Matrix4f I;       // Identity matrix
    Eigen::Matrix4f F;       // State transition model
    Eigen::Matrix<float, 4, 4> H;  // Measurement model

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

        //publishClusterMarkers(white_cloud, selected_clusters);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);



        // ------------------- UPDATED: Selecting Leftmost and Rightmost Clusters -------------------

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        Eigen::Vector2f left_observed, right_observed;
        bool left_detected = false, right_detected = false;
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
            float angle = atan2(centroid[1], centroid[0]);
        
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

        if (left_detected && right_detected) {
            state << left_observed[0], left_observed[1], right_observed[0], right_observed[1];
        } else if (left_detected) {
            state << left_observed[0], left_observed[1], left_observed[0] + 1.5, left_observed[1];
        } else if (right_detected) {
            state << right_observed[0] - 1.5, right_observed[1], right_observed[0], right_observed[1];
        } else {
            state << 0, 0, 1.5, 0;  // Default fallback
        }
        
        // Kalman Filter Prediction Step
        state = F * state;
        P = F * P * F.transpose() + Q;

        if (left_detected && right_detected) {
            Eigen::Vector4f Z;
            Z << left_observed[0], left_observed[1], right_observed[0], right_observed[1];
        
            Eigen::Matrix4f H;
            H << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
        
            Eigen::Matrix4f S = H * P * H.transpose() + R;
            Eigen::Matrix4f K = P * H.transpose() * S.inverse();
        
            state = state + K * (Z - H * state);
            P = (I - K * H) * P;
        }

        // If only left marking detected, estimate right
        else if (left_detected)
        {
            state[0] = left_observed[0];
            state[1] = left_observed[1];
            state[2] = left_observed[0] + 1.51;  // Estimate right using known road width
            state[3] = left_observed[1];

            // Reduce covariance uncertainty
            P = P * 0.95 + Eigen::Matrix4f::Identity() * 0.01;  // Add small uncertainty

        }

        // If only right marking detected, estimate left
        else if (right_detected)
        {
            state[2] = right_observed[0];
            state[3] = right_observed[1];
            state[0] = right_observed[0] - 1.51;  // Estimate left using known road width
            state[1] = right_observed[1];

            // Update covariance (reduce uncertainty)
            P = P * 0.95 + Eigen::Matrix4f::Identity() * 0.01;  // Add small uncertainty

        }

        // Assign Kalman-filtered points
        left_observed[0] = state[0];
        left_observed[1] = state[1];
        right_observed[0] = state[2];
        right_observed[1] = state[3];

        pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());

        if (left_detected) {
            for (int idx : leftmost_cluster.indices) {
                left_cluster->push_back(white_cloud->points[idx]);
            }
        } else {
            // If no left cluster detected, add a single estimated point
            left_cluster->push_back(pcl::PointXYZ(state[0], state[1], 0));
        }

        if (right_detected) {
            for (int idx : rightmost_cluster.indices) {
                right_cluster->push_back(white_cloud->points[idx]);
            }
        } else {
            // If no right cluster detected, add a single estimated point
            right_cluster->push_back(pcl::PointXYZ(state[2], state[3], 0));
        }

        //RCLCPP_INFO(this->get_logger(), "Right Detected: %s", right_detected ? "Yes" : "No");
        //RCLCPP_INFO(this->get_logger(), "Right Observed: x=%.3f, y=%.3f", right_observed[0], right_observed[1]);

        // Create corrected clusters
        pcl::PointCloud<pcl::PointXYZ>::Ptr right_corrected_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr left_corrected_cluster(new pcl::PointCloud<pcl::PointXYZ>());

        // If right cluster is detected, apply Kalman correction to all points
        if (right_detected) {
            for (const auto &point : right_cluster->points) {
                pcl::PointXYZ corrected_point;
                corrected_point.x = (point.x + state[2]) / 2;  // Blend detected & Kalman estimate
                corrected_point.y = (point.y + state[3]) / 2;
                corrected_point.z = 0;
                right_corrected_cluster->push_back(corrected_point);
            }
        } else {
            // No detection: use only the Kalman estimate
            right_corrected_cluster->push_back(pcl::PointXYZ(state[2], state[3], 0));
        }

        // If left cluster is detected, apply Kalman correction to all points
        if (left_detected) {
            for (const auto &point : left_cluster->points) {
                pcl::PointXYZ corrected_point;
                corrected_point.x = (point.x + state[0]) / 2;
                corrected_point.y = (point.y + state[1]) / 2;
                corrected_point.z = 0;
                left_corrected_cluster->push_back(corrected_point);
            }
        } else {
            // No detection: use only the Kalman estimate
            left_corrected_cluster->push_back(pcl::PointXYZ(state[0], state[1], 0));
        }


        // Publish all four clusters for visualization
        publishClusterMarkers(left_cluster, "map", "left_raw", 0, 1.0, 0.0, 0.0);  // Red - Raw Left
        publishClusterMarkers(right_cluster, "map", "right_raw", 1, 0.0, 1.0, 0.0); // Green - Raw Right
        publishClusterMarkers(left_corrected_cluster, "map", "left_kalman", 2, 1.0, 1.0, 0.0); // Yellow - Kalman Left
        publishClusterMarkers(right_corrected_cluster, "map", "right_kalman", 3, 0.0, 1.0, 1.0); // Cyan - Kalman Right


        // Publish both detected and Kalman-corrected lanes for visualization
        //publishCurveMarkers(right_cluster, false);  // Raw observed right
        //publishCurveMarkers(left_cluster, false);   // Raw observed left
        //publishCurveMarkers(right_corrected_cluster, true);  // Kalman-filtered right
        //publishCurveMarkers(left_corrected_cluster, true);   // Kalman-filtered left

        
        //publishSingleClusterMarker(right_cluster);


        // Use Kalman-filtered clusters for curve fitting
        if (fit_side_ == true)
            publishCurveMarkers(right_cluster); // Fit curve on right

        if (fit_side_ == false)
            publishCurveMarkers(left_cluster);  // Fit curve on left

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



    

    
    void publishCurveMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
    {
        std::vector<geometry_msgs::msg::Point> shifted_curve_points;
        
        if (cluster->size() < 2)
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

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        marker_publisher_->publish(markers);
    }


    void publishClusterMarkers(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
        const std::string& frame_id,
        const std::string& ns,
        int id,
        float r, float g, float b)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        for (const auto& point : cluster->points)
        {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            marker.points.push_back(p);
        }

        cluster_marker_pub_->publish(marker);
    }


    void publishSingleClusterMarker(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker cluster_marker;
    
        cluster_marker.header.frame_id = "map";
        cluster_marker.header.stamp = this->get_clock()->now();
        cluster_marker.ns = "single_cluster";
        cluster_marker.id = 0;  // Single cluster
        cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        cluster_marker.scale.x = 0.05;
        cluster_marker.scale.y = 0.05;
        cluster_marker.scale.z = 0.05;
        cluster_marker.color.r = 1.0;  // Red color
        cluster_marker.color.g = 0.0;
        cluster_marker.color.b = 0.0;
        cluster_marker.color.a = 1.0;
    
        // Add points to marker
        for (const auto& point : cluster->points)
        {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            cluster_marker.points.push_back(p);
        }
    
        // Add marker to array
        marker_array.markers.push_back(cluster_marker);
    
        // Publish the MarkerArray
        marker_publisher2_->publish(marker_array);
    }
    



    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cluster_marker_pub_;


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