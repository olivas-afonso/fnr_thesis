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
    
        curve_publisher_right_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("right_spline_fit_original", 10);
        curve_publisher_left_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("left_spline_fit_original", 10);
        kalman_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("kalman_state", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        test_cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("test_clusters", 10);


        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);

        // Initialize state [x_c, y_c, r, theta]
        state = Eigen::VectorXf::Zero(12);
        state << 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,   // Left lane (slight initial curvature)
         0.01, 0.0, 1.4, 0.0, 0.0, 0.0;  // Right lane (1.4m offset)


        // Initialize covariance matrix
        P = Eigen::MatrixXf::Identity(12,12) * 0.1;


        // Process noise
        Q = Eigen::MatrixXf::Identity(12,12) * 0.01;
        // Higher uncertainty for derivative terms
        Q.block(3,3,3,3) = Eigen::MatrixXf::Identity(3,3) * 0.1;  // Left derivatives
        Q.block(9,9,3,3) = Eigen::MatrixXf::Identity(3,3) * 0.1;  // Right derivatives
        
        //Measurement noise
        R = Eigen::MatrixXf::Identity(6,6) * 0.05;  // For full observation case
        I = Eigen::MatrixXf::Identity(12,12);

        // State transition model with coupling between lanes
        F = Eigen::MatrixXf::Identity(12,12);
        float dt = 0.1; // Time step
        // Left lane dynamics
        F(0,3) = dt; F(1,4) = dt; F(2,5) = dt;
        // Right lane dynamics
        F(6,9) = dt; F(7,10) = dt; F(8,11) = dt;
        // Coupling terms (left-right influence)
        F.block(0,6,3,3) = 0.05 * Eigen::Matrix3f::Identity();
        F.block(6,0,3,3) = 0.05 * Eigen::Matrix3f::Identity();


        // Measurement matrix - can observe up to 6 values (both lanes)
        H_full = Eigen::MatrixXf::Zero(6,12);
        H_full.block(0,0,3,3) = Eigen::Matrix3f::Identity();  // Left coeffs
        H_full.block(3,6,3,3) = Eigen::Matrix3f::Identity();  // Right coeffs
        
        H_left_only = Eigen::MatrixXf::Zero(3,12);
        H_left_only.block(0,0,3,3) = Eigen::Matrix3f::Identity();

        H_right_only = Eigen::MatrixXf::Zero(3,12);
        H_right_only.block(0,6,3,3) = Eigen::Matrix3f::Identity();



    }

private:
    /////////////////////////////////////
    /*
    State Vecotr Initialize 12D state vector [a_L, b_L, c_L, ȧ_L, ḃ_L, ċ_L, a_R, b_R, c_R, ȧ_R, ḃ_R, ċ_R]

    */
    
    Eigen::VectorXf state; // [x_c, y_c, r, theta]
    Eigen::MatrixXf P, Q, R, I, F, H, H_full, H_left_only, H_right_only;
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;
    Eigen::Vector3f lane_transform_;


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
        this->get_parameter("fit_side", fit_side_);
        //fitCircle(white_cloud, leftmost_cluster, left_circle);
        //RCLCPP_INFO(this->get_logger(), "Circle 1: x=%f, y=%f, r=%f", left_circle[0], left_circle[1], left_circle[2]);

        //fitCircle(white_cloud, rightmost_cluster, right_circle);
        //RCLCPP_INFO(this->get_logger(), "Circle 1: x=%f, y=%f, r=%f", right_circle[0], right_circle[1], right_circle[2]);
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

        // Measurement Update Logic
        Eigen::VectorXf Z;
        Eigen::MatrixXf H;
        Eigen::MatrixXf R_current;
        Eigen::Vector3f coeffs_right, coeffs_left;

         // Stores [Δa, Δb, Δc]
        
        if(!fit_side_)
        {
            left_detected=true;
            right_detected=false;
        }
        if (left_detected && right_detected) {
            
            

            pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *right_cluster);
            coeffs_right=fitQuadraticCurve(right_cluster, curve_publisher_right_);

            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            coeffs_left=fitQuadraticCurve(left_cluster, curve_publisher_left_);

            lane_transform_ = calculateAndStoreTransform(coeffs_left, coeffs_right);

            Z.resize(6);

            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;


            // Both clusters detected: fit circles as usual
            //fitCircles(white_cloud, leftmost_cluster, rightmost_cluster, current_pose_.pose, center, left_radius, right_radius);
            //Z << center[0], center[1], left_radius, right_radius;
        
        } else if (left_detected) {
            // Only left cluster detected: fit a single circle to the left cluster
            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            Eigen::Vector3f coeffs_left = fitQuadraticCurve(left_cluster, curve_publisher_left_);
            Eigen::Vector3f coeffs_right = estimateRightFromLeft(coeffs_left, lane_transform_);

            Z.resize(6);
            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;
            R_current.block(3,3,3,3) *= 2.0; // Higher uncertainty for estimated right lane
        }

        else if (right_detected) {
            // Only right lane visible - estimate left lane
            pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *right_cluster);
            Eigen::Vector3f coeffs_right = fitQuadraticCurve(right_cluster,curve_publisher_right_);
            Eigen::Vector3f coeffs_left = estimateLeftFromRight(coeffs_right, lane_transform_);
        
            Z.resize(6);
            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                 coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;
            R_current.block(0,0,3,3) *= 2.0; // Higher uncertainty for estimated left lane
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
        Eigen::MatrixXf S = H * P * H.transpose() + R_current;
        Eigen::MatrixXf K = P * H.transpose() * S.inverse();

        
        state = state + K * (Z - H * state);
        P = (I - K * H) * P;

        visualizeKalmanState(
            state,               // Your 12D Kalman state vector
            kalman_publisher_,    // Your marker publisher
            left_detected,       // Boolean from detection
            right_detected,      // Boolean from detection
            -5.0f,               // Min x-range
            10.0f,               // Max x-range
            0.25f                // Point spacing
        );
        //(float left_start_angle, left_end_angle, right_start_angle, right_end_angle;
        
        //if(!fit_side_) computeArcAngles(white_cloud, rightmost_cluster, common_center, right_start_angle, right_end_angle);

        //computeArcAngles(white_cloud, leftmost_cluster, left_circle, left_start_angle, left_end_angle);
        //computeArcAngles(white_cloud, rightmost_cluster, right_circle, right_start_angle, right_end_angle);
        
        //std::cout << "left start angle: " << left_start_angle << "\n";
        //std::cout << "right start angle: " << right_start_angle << "\n";
    
        //visualizeCircles(common_center, left_radius, right_radius, state, left_start_angle, left_end_angle, left_start_angle, left_end_angle, left_detected, left_detected);
        //visualizeCircles(common_center, left_radius, right_radius, state, right_start_angle, right_end_angle, right_start_angle, right_end_angle, right_detected, right_detected);

    }

    Eigen::Vector3f fitQuadraticCurve(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub)
    {
        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points)
        {
            x_vals.push_back(point.x);
            y_vals.push_back(point.y);
        }
    
        Eigen::MatrixXd A(x_vals.size(), 3);
        Eigen::VectorXd Z(y_vals.size());
    
        for (size_t i = 0; i < x_vals.size(); i++) {
            A(i, 0) = x_vals[i] * x_vals[i]; // x² term
            A(i, 1) = x_vals[i];             // x term
            A(i, 2) = 1.0;                   // constant term
            Z(i) = y_vals[i];
        }
    
        // Solve for coefficients
        Eigen::Vector3d coeffs = A.colPivHouseholderQr().solve(Z);
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

        visualization_msgs::msg::Marker curve_marker;
        curve_marker.header.frame_id = "map";
        curve_marker.header.stamp = this->get_clock()->now();
        curve_marker.ns = "curve_fit";
        curve_marker.id = 1;
        curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        curve_marker.scale.x = 0.05; 
        curve_marker.color.r = 0.0;
        curve_marker.color.g = 0.0;
        curve_marker.color.b = 1.0;
        curve_marker.color.a = 1.0;

        // Generate curve points
        for (float x = *std::min_element(x_vals.begin(), x_vals.end());
            x <= *std::max_element(x_vals.begin(), x_vals.end()); x += 0.05)
        {
            float y = a * x * x + b * x + c;
            Eigen::Vector3f point_in_map(x, y, 0.0);
            Eigen::Vector3f point_in_camera = rotation_matrix.inverse() * (point_in_map - translation_vector);

            // Transform back to map frame
            Eigen::Vector3f translated_point = (rotation_matrix * point_in_camera) + translation_vector;

            geometry_msgs::msg::Point p;
            p.x = translated_point[0];
            p.y = translated_point[1];
            p.z = 0.0;

            curve_marker.points.push_back(p);

        }


        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        pub->publish(markers);

        return coeffs.cast<float>();
    }


    float calculateOffset(float a, float b, float x_ref, float lane_width) {
        float slope = 2*a*x_ref + b;  // dy/dx at reference point
        return lane_width * sqrt(1 + slope*slope);
    }
    
    Eigen::Vector3f estimateRightFromLeft(const Eigen::Vector3f& left_coeffs, Eigen::Vector3f  &lane_transform_ ) {
    
        Eigen::Vector3f aux;
        aux[0]=left_coeffs[0] + lane_transform_[0];
        aux [1] = left_coeffs[1] + lane_transform_[1];
        aux [2] = left_coeffs[2] + lane_transform_[2];
        return aux;
    }
    
    Eigen::Vector3f estimateLeftFromRight(const Eigen::Vector3f& right_coeffs, Eigen::Vector3f  &lane_transform_ ) {
        
        Eigen::Vector3f aux;
        aux[0]=right_coeffs[0] - lane_transform_[0];
        aux [1] = right_coeffs[1] - lane_transform_[1];
        aux [2] = right_coeffs[2] - lane_transform_[2];

        return aux;
    }


    void visualizeKalmanState(
        const Eigen::VectorXf& state,
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
        bool left_detected,
        bool right_detected,
        float x_min = -5.0f,
        float x_max = 20.0f,
        float step = 0.25f) 
    {
        visualization_msgs::msg::MarkerArray marker_array;
    
        // Extract coefficients from state vector
        Eigen::Vector3f left_coeffs(state[0], state[1], state[2]);
        Eigen::Vector3f right_coeffs(state[6], state[7], state[8]);
    
        // Left lane marker (blue for detected, light blue for estimated)
        visualization_msgs::msg::Marker left_marker;
        left_marker.header.frame_id = "map";
        left_marker.header.stamp = this->now();
        left_marker.ns = "kalman_lanes";
        left_marker.id = 0;
        left_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        left_marker.action = visualization_msgs::msg::Marker::ADD;
        left_marker.scale.x = 0.05;
        left_marker.color.a = 1.0;
        left_marker.color.b = 1.0; // Blue base
        left_marker.color.g = left_detected ? 0.0 : 0.5; // Light blue if estimated
        left_marker.pose.orientation.w = 1.0;
    
        // Right lane marker (red for detected, pink for estimated)
        visualization_msgs::msg::Marker right_marker = left_marker;
        right_marker.id = 1;
        right_marker.color.r = 1.0; // Red base
        right_marker.color.b = right_detected ? 0.0 : 0.5; // Pink if estimated
        right_marker.color.g = 0.0;
    
        // Generate curve points
        for (float x = x_min; x <= x_max; x += step) {
            // Left lane point
            geometry_msgs::msg::Point p_left;
            p_left.x = x;
            p_left.y = left_coeffs[0] * x * x + left_coeffs[1] * x + left_coeffs[2];
            p_left.z = 0;
            left_marker.points.push_back(p_left);
    
            // Right lane point
            geometry_msgs::msg::Point p_right;
            p_right.x = x;
            p_right.y = right_coeffs[0] * x * x + right_coeffs[1] * x + right_coeffs[2];
            p_right.z = 0;
            right_marker.points.push_back(p_right);
        }
    
        // Add lane width indicator every 5 meters
        for (float x = x_min; x <= x_max; x += 5.0f) {
            visualization_msgs::msg::Marker width_marker;
            width_marker.header = left_marker.header;
            width_marker.ns = "lane_width";
            width_marker.id = x * 100; // Unique ID based on x-position
            width_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            width_marker.scale.x = 0.05;
            width_marker.color.r = 1.0;
            width_marker.color.g = 1.0;
            width_marker.color.a = 0.5;
    
            geometry_msgs::msg::Point p_left, p_right;
            p_left.x = x;
            p_left.y = left_coeffs[0] * x * x + left_coeffs[1] * x + left_coeffs[2];
            p_right.x = x;
            p_right.y = right_coeffs[0] * x * x + right_coeffs[1] * x + right_coeffs[2];
    
            width_marker.points.push_back(p_left);
            width_marker.points.push_back(p_right);
            marker_array.markers.push_back(width_marker);
        }
    
        // Add markers to array
        marker_array.markers.push_back(left_marker);
        marker_array.markers.push_back(right_marker);
    
        // Add text markers for coefficients
        visualization_msgs::msg::Marker text_marker;
        text_marker.header = left_marker.header;
        text_marker.ns = "coefficients";
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.scale.z = 0.3;
        text_marker.color.a = 1.0;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
    
        // Left coefficients text
        text_marker.id = 100;
        text_marker.pose.position.x = x_min + 1.0;
        text_marker.pose.position.y = left_coeffs[0] * x_min * x_min + left_coeffs[1] * x_min + left_coeffs[2] + 1.0;
        text_marker.text = "Left: " + 
                          std::to_string(left_coeffs[0]) + "x² + " + 
                          std::to_string(left_coeffs[1]) + "x + " + 
                          std::to_string(left_coeffs[2]) + 
                          (left_detected ? "" : " (estimated)");
        marker_array.markers.push_back(text_marker);
    
        // Right coefficients text
        text_marker.id = 101;
        text_marker.pose.position.x = x_min + 1.0;
        text_marker.pose.position.y = right_coeffs[0] * x_min * x_min + right_coeffs[1] * x_min + right_coeffs[2] - 1.0;
        text_marker.text = "Right: " + 
                           std::to_string(right_coeffs[0]) + "x² + " + 
                           std::to_string(right_coeffs[1]) + "x + " + 
                           std::to_string(right_coeffs[2]) + 
                           (right_detected ? "" : " (estimated)");
        marker_array.markers.push_back(text_marker);
    
        publisher->publish(marker_array);
    }


    Eigen::Vector3f calculateAndStoreTransform(const Eigen::Vector3f& left_coeffs, 
        const Eigen::Vector3f& right_coeffs) 
    {
        Eigen::Vector3f lane_transform_;
        lane_transform_ = right_coeffs - left_coeffs;

        return lane_transform_;
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

    void testClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
        const pcl::PointIndices& left_indices, 
        const pcl::PointIndices& right_indices)
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
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_left_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_right_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr kalman_publisher_;
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

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanCircleLocalization>());
    rclcpp::shutdown();
    return 0;
}
