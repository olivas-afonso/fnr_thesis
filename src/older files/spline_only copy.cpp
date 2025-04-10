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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // For tf2::Quaternion and tf2::Matrix3x3




#include <vector>
#include <cmath>
#include <numeric>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>





class KalmanCircleLocalization : public rclcpp::Node
{
public:
    KalmanCircleLocalization() : Node("kalman_spline_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&KalmanCircleLocalization::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10,
            std::bind(&KalmanCircleLocalization::poseCallback, this, std::placeholders::_1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);
    
        curve_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_original", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        test_cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("test_clusters", 10);


        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);

        // Initialize state [x_c, y_c, r, theta]
        state = Eigen::VectorXf::Zero(8);
        state << 0, 0, 0, 0.1, 0, 0.1, 0, 1.4; 

        // Initialize covariance matrix
        P = Eigen::MatrixXf::Identity(8,8) * 0.1;

        // Process and measurement noise
        Q = Eigen::MatrixXf::Identity(8,8) * 0.01;
        R = Eigen::MatrixXf::Identity(8,8) * 0.5;
        I = Eigen::MatrixXf::Identity(8,8);

        // State transition model (assume slow-moving lane changes)
        F = Eigen::MatrixXf::Identity(8,8);



    }

private:
    Eigen::VectorXf state; // [ref_x, ref_y, ref_theta, left_k0, left_k1, right_k0, right_k1, width]
    Eigen::MatrixXf P, Q, R, I, F;
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;


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
        /*
        if (selected_clusters.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return;
        }
        */

        publishClusterMarkers(white_cloud, selected_clusters);

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

        // Fit circles to left and right lane markings
        Eigen::Vector2f center;
        float left_radius, right_radius;
        this->get_parameter("fit_side", fit_side_);
        Eigen::VectorXf Z(8); // Ensure correct size
        left_detected=true;
        right_detected=true;
        /*
        if (left_detected && right_detected) {
            // Both clusters detected: fit circles as usual
            fitCircles(white_cloud, leftmost_cluster, rightmost_cluster, current_pose_.pose, center, left_radius, right_radius);
            Z << center[0], center[1], left_radius, right_radius;
        } else if (left_detected) {
            // Only left cluster detected: fit a single circle to the left cluster
            fitSingleCircle(white_cloud, leftmost_cluster, current_pose_.pose, center, left_radius);
        
            // Derive the right cluster based on the known lane width
            float lane_width = 1.4f;  // Known lane width in meters
            Eigen::Vector2f right_center = center; 
            float right_radius = left_radius + lane_width;  // Assume the same radius for simplicity
        
            Z << center[0], center[1], left_radius, right_radius;
        } else if (right_detected) {
            // Only right cluster detected: fit a single circle to the right cluster
            fitSingleCircle(white_cloud, rightmost_cluster, current_pose_.pose, center, right_radius);
        
            // Derive the left cluster based on the known lane width
            float lane_width = 1.4f;  // Known lane width in meters
            Eigen::Vector2f left_center = center;  // Shift to the left
            float left_radius = right_radius - lane_width;  // Assume the same radius for simplicity
        
            Z << center[0], center[1], left_radius, right_radius;
        }
        else
        {
            return;
        }
            */
           if (fit_side_) {
            SplineFitResult result;
            if (fitSplines(white_cloud, leftmost_cluster, rightmost_cluster, current_pose_.pose, result)) {
                RCLCPP_INFO(this->get_logger(), "GOT FIT");
                // Extract spline parameters for Kalman measurement
                float left_k0 = computeSplineCurvature(result.left_spline, result.left_accel, result.left_x, 0.0f);
                float left_k1 = computeSplineCurvature(result.left_spline, result.left_accel, result.left_x, 1.0f);
                float right_k0 = computeSplineCurvature(result.right_spline, result.right_accel, result.right_x, 0.0f);
                float right_k1 = computeSplineCurvature(result.right_spline, result.right_accel, result.right_x, 1.0f);
                
                // Get reference angle from spline direction at start
                double start_deriv = gsl_spline_eval_deriv(result.left_spline, result.left_x.front(), result.left_accel);
                float ref_theta = atan2(start_deriv, 1.0);
                
                // Create measurement vector
                Z.resize(8);
                Z << result.common_reference_point.x(),  // ref_x
                     result.common_reference_point.y(),  // ref_y
                     ref_theta,                         // ref_theta
                     left_k0, left_k1,                  // left spline curvatures
                     right_k0, right_k1,                // right spline curvatures
                     1.4f;                             // lane width
                
                visualizeSplines(result, left_detected, right_detected);
            }

        } else if (!fit_side_) {
            // Only left cluster detected: fit a single circle to the left cluster
            //fitSingleCircle(white_cloud, leftmost_cluster, current_pose_.pose, center, left_radius);
        
            // Derive the right cluster based on the known lane width
            //float lane_width = 1.35f;  // Known lane width in meters
            //Eigen::Vector2f right_center = center; 
            //float right_radius = left_radius + lane_width;  // Assume the same radius for simplicity
            
            //Z << center[0], center[1], left_radius, right_radius;
        }
            
        /*
        } else if (!fit_side_) {
            // Only right cluster detected: fit a single circle to the right cluster
            fitSingleCircle(white_cloud, rightmost_cluster, current_pose_.pose, center, right_radius);
        
            // Derive the left cluster based on the known lane width
            float lane_width = 1.35f;  // Known lane width in meters
            Eigen::Vector2f left_center = center;  // Shift to the left
            float left_radius = right_radius - lane_width;  // Assume the same radius for simplicity
        
            Z << center[0], center[1], left_radius, right_radius;
        }
        */
        else
        {
            return;
        }

        // Kalman Prediction Step
        state = F * state;
        P = F * P * F.transpose() + Q;

        // Kalman Update Step
        Eigen::MatrixXf S = P + R;
        Eigen::MatrixXf K = P * S.inverse();

        state = state + K * (Z - state);
        P = (I - K) * P;

        Eigen::Vector2f common_center = Z.head<2>(); 
        //(float left_start_angle, left_end_angle, right_start_angle, right_end_angle;
        /*
        if(fit_side_) 
        {
            computeArcAngles(white_cloud, rightmost_cluster, common_center, right_start_angle, right_end_angle);
            computeArcAngles(white_cloud, leftmost_cluster, common_center, left_start_angle, left_end_angle);
        }
            
        else if(!fit_side_) computeArcAngles(white_cloud, leftmost_cluster, common_center, left_start_angle, left_end_angle);

        */
        //if(!fit_side_) computeArcAngles(white_cloud, rightmost_cluster, common_center, right_start_angle, right_end_angle);

        //computeArcAngles(white_cloud, leftmost_cluster, left_circle, left_start_angle, left_end_angle);
        //computeArcAngles(white_cloud, rightmost_cluster, right_circle, right_start_angle, right_end_angle);
        
        //std::cout << "left start angle: " << left_start_angle << "\n";
        //std::cout << "right start angle: " << right_start_angle << "\n";
        if(fit_side_){
            left_detected=true;
            right_detected=true;
        }
        else if (!fit_side_)
        {
            left_detected=true;
            right_detected=false;
        }
        //visualizeCircles(common_center, left_radius, right_radius, state, left_start_angle, left_end_angle, right_start_angle, right_end_angle, left_detected, right_detected);
        //visualizeCircles(common_center, left_radius, right_radius, state, left_start_angle, left_end_angle, left_start_angle, left_end_angle, left_detected, left_detected);
        //visualizeCircles(common_center, left_radius, right_radius, state, right_start_angle, right_end_angle, right_start_angle, right_end_angle, right_detected, right_detected);
        //visualizeSplines(result, left_detected, right_detected);
    }

    
    
    // Replace your SplineFitResult struct with this:
    struct SplineFitResult {
        gsl_interp_accel* left_accel = nullptr;
        gsl_spline* left_spline = nullptr;
        gsl_interp_accel* right_accel = nullptr;
        gsl_spline* right_spline = nullptr;
        Eigen::Vector2f common_reference_point;
        std::vector<double> left_x, left_y;
        std::vector<double> right_x, right_y;

        ~SplineFitResult() {
            if (left_spline) gsl_spline_free(left_spline);
            if (left_accel) gsl_interp_accel_free(left_accel);
            if (right_spline) gsl_spline_free(right_spline);
            if (right_accel) gsl_interp_accel_free(right_accel);
        }
    };

    // Replace your fitSplines function with this:
    bool fitSplines(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
        const pcl::PointIndices &left_indices,
        const pcl::PointIndices &right_indices,
        const geometry_msgs::msg::Pose &current_pose,
        SplineFitResult &result)
    {
        // Prepare left points with proper parameterization
        auto [left_t, left_x, left_y] = preparePoints(left_indices, cloud);
        if (left_t.size() < 4) return false;
        
        // Prepare right points with proper parameterization
        auto [right_t, right_x, right_y] = preparePoints(right_indices, cloud);
        if (right_t.size() < 4) return false;

        // Fit splines using the parameter t [0,1]
        result.left_accel = gsl_interp_accel_alloc();
        result.left_spline = gsl_spline_alloc(gsl_interp_cspline, left_t.size());
        gsl_spline_init(result.left_spline, left_t.data(), left_x.data(), left_t.size());
        
        result.right_accel = gsl_interp_accel_alloc();
        result.right_spline = gsl_spline_alloc(gsl_interp_cspline, right_t.size());
        gsl_spline_init(result.right_spline, right_t.data(), right_x.data(), right_t.size());

        return true;
    }

    // Replace your computeSplineCurvature function with this:
    float computeSplineCurvature(gsl_spline* spline, gsl_interp_accel* accel, 
                            const std::vector<double>& x, float t_param) 
    {
        if (!spline || !accel || x.empty()) return 0.0f;
        
        // Convert parameter t [0,1] to actual x value
        double x_min = x.front();
        double x_max = x.back();
        double t = x_min + t_param * (x_max - x_min);
        
        // Clamp to valid range
        t = std::clamp(t, x_min, x_max);
        
        double y, d1, d2;
        y = gsl_spline_eval(spline, t, accel);
        d1 = gsl_spline_eval_deriv(spline, t, accel);
        d2 = gsl_spline_eval_deriv2(spline, t, accel);
        
        // Curvature formula: (d1*d2) / (1 + d1^2)^(3/2)
        double denominator = std::pow(1.0 + d1*d1, 1.5);
        if (denominator < 1e-6) return 0.0f;
        
        return static_cast<float>(d2 / denominator);
    }


    // Replace your visualizeSplines function with this:
    void visualizeSplines(const SplineFitResult &fit_result, bool left_detected, bool right_detected)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        auto createSplineMarker = [&](gsl_spline* spline, gsl_interp_accel* accel, 
                                    const std::vector<double>& x, int id, float r, float g, float b) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->now();
            marker.ns = "spline_fit";
            marker.id = id;
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.05;
            marker.color.r = r;
            marker.color.g = g;
            marker.color.b = b;
            marker.color.a = 1.0;

            double x_min = x.front();
            double x_max = x.back();
            for (double t = x_min; t <= x_max; t += 0.1) {
                geometry_msgs::msg::Point p;
                p.x = t;
                p.y = gsl_spline_eval(spline, t, accel);
                p.z = 0.0;
                marker.points.push_back(p);
            }
            return marker;
        };

        if (left_detected && fit_result.left_spline) {
            marker_array.markers.push_back(
                createSplineMarker(fit_result.left_spline, fit_result.left_accel, 
                                fit_result.left_x, 0, 0.0, 0.0, 1.0));
        }

        if (right_detected && fit_result.right_spline) {
            marker_array.markers.push_back(
                createSplineMarker(fit_result.right_spline, fit_result.right_accel, 
                                fit_result.right_x, 1, 1.0, 0.0, 0.0));
        }

        curve_publisher_->publish(marker_array);
    }


    



    std::vector<pcl::PointIndices> clusterWhitePoints( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,const geometry_msgs::msg::Pose &current_pose_) // Camera pose
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
    
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(5000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);
    
        struct ClusterData
        {
            pcl::PointIndices indices;
            float min_distance;
            float depth; // Width in the camera direction (Z-axis)
        };
    
        std::vector<ClusterData> cluster_data;
    
        // Extract camera position
        Eigen::Vector3f camera_position(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);
    
        // Extract camera orientation
        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);
    
        // Convert to Eigen rotation matrix
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R(i, j) = rotation_matrix[i][j];
    
        // Extract forward axis (Z direction in camera frame)
        Eigen::Vector3f forward_dir = R.col(2); // Z-axis
    
        for (const auto &cluster : clusters)
        {
            float min_proj = std::numeric_limits<float>::max();
            float max_proj = std::numeric_limits<float>::lowest();
            float sum_x = 0.0, sum_z = 0.0;
            int num_points = cluster.indices.size();
    
            for (int idx : cluster.indices)
            {
                Eigen::Vector3f point(
                    cloud->points[idx].x,
                    cloud->points[idx].y,
                    cloud->points[idx].z);
    
                // Transform to camera-aligned frame
                Eigen::Vector3f relative_point = R.transpose() * (point - camera_position);
    
                // Project onto the camera direction (Z-axis in camera frame)
                float projection = relative_point.dot(Eigen::Vector3f(0, 0, 1));
    
                min_proj = std::min(min_proj, projection);
                max_proj = std::max(max_proj, projection);
    
                sum_x += relative_point.x();
                sum_z += relative_point.z();
            }
    
            float centroid_x = sum_x / num_points;
            float centroid_z = sum_z / num_points;
            float distance = std::sqrt(centroid_x * centroid_x + centroid_z * centroid_z);
    
            float cluster_depth = max_proj - min_proj; // Width along camera direction
    
            cluster_data.push_back({cluster, distance, cluster_depth});
        }
    
        // Filter clusters based on depth (remove clusters with small Z-width)
        std::vector<pcl::PointIndices> filtered_clusters;
        for (const auto &cd : cluster_data)
        {
            if (cd.depth >= 0.2) // Adjust this threshold as needed
            {
                filtered_clusters.push_back(cd.indices);
            }
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
        cluster_marker_pub_->publish(cluster_markers);
    }

    void testClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const pcl::PointIndices& left_indices, const pcl::PointIndices& right_indices)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;

        // Define colors for left (red) and right (blue) clusters
        std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0},  // Left cluster (Red)
                                                {0.0, 0.0, 1.0}}; // Right cluster (Blue)

        // Helper lambda to create markers
        auto createMarker = [&](const pcl::PointIndices& indices, int id, const std::tuple<float, float, float>& color) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "clusters";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = 0.05;

        auto [r, g, b] = color;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        for (int idx : indices.indices)
        {
        geometry_msgs::msg::Point p;
        p.x = cloud->points[idx].x;
        p.y = cloud->points[idx].y;
        p.z = cloud->points[idx].z;
        marker.points.push_back(p);
        }

        return marker;
        };

        // Add left and right cluster markers
        cluster_markers.markers.push_back(createMarker(left_indices, 0, colors[0]));  // Left
        cluster_markers.markers.push_back(createMarker(right_indices, 1, colors[1])); // Right

        // Publish the markers
        test_cluster_marker_pub_->publish(cluster_markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr test_cluster_marker_pub_;
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

std::pair<std::vector<double>, std::vector<double>> 
KalmanCircleLocalization::preparePoints(
    const pcl::PointIndices& indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // 1. Compute centroid
    Eigen::Vector2d centroid(0,0);
    for (int idx : indices.indices) {
        centroid.x() += cloud->points[idx].x;
        centroid.y() += cloud->points[idx].y;
    }
    centroid /= indices.indices.size();

    // 2. Create vector of points with angles
    std::vector<std::pair<Eigen::Vector2d, double>> points_with_angles;
    for (int idx : indices.indices) {
        Eigen::Vector2d point(
            cloud->points[idx].x,
            cloud->points[idx].y);
        double angle = atan2(point.y() - centroid.y(), point.x() - centroid.x());
        points_with_angles.emplace_back(point, angle);
    }

    // 3. Sort by angle
    std::sort(points_with_angles.begin(), points_with_angles.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    // 4. Parameterize by cumulative chord length
    std::vector<double> x, y, t;
    double cumulative_length = 0;
    t.push_back(0);
    
    // First point
    x.push_back(points_with_angles[0].first.x());
    y.push_back(points_with_angles[0].first.y());
    
    // Remaining points
    for (size_t i = 1; i < points_with_angles.size(); ++i) {
        const auto& prev = points_with_angles[i-1].first;
        const auto& curr = points_with_angles[i].first;
        cumulative_length += (curr - prev).norm();
        t.push_back(cumulative_length);
        x.push_back(curr.x());
        y.push_back(curr.y());
    }

    // Normalize t to [0,1]
    if (cumulative_length > 0) {
        for (auto& val : t) val /= cumulative_length;
    }

    return {t, x, y}; // Now t is our parameter, x and y are coordinates
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanCircleLocalization>());
    rclcpp::shutdown();
    return 0;
}
