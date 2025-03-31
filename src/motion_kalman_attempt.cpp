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

#include <deque>
#include <utility> // for std::pair
#include "std_msgs/msg/float32.hpp"

using std::placeholders::_1;




class KalmanCircleLocalization : public rclcpp::Node
{
public:
    KalmanCircleLocalization() : Node("kalman_circ_node"), last_speed_time_(this->now()), last_servo_time_(this->now())
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&KalmanCircleLocalization::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10,
            std::bind(&KalmanCircleLocalization::poseCallback, this, std::placeholders::_1));

        speed_sub_ = create_subscription<std_msgs::msg::Float32>(
            "/commands/motor/speed", 10,
            std::bind(&KalmanCircleLocalization::speedCallback, this, _1));
        servo_sub_ = create_subscription<std_msgs::msg::Float32>(
            "/commands/servo/position", 10,
            std::bind(&KalmanCircleLocalization::servoCallback, this, _1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);
    
        curve_publisher_right_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("right_spline_fit_original", 10);
        curve_publisher_left_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("left_spline_fit_original", 10);
        kalman_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("kalman_state", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        test_cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("test_clusters", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);



        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);

        this->declare_parameter("time_jump_threshold", 1.0);
        this->get_parameter("time_jump_threshold", time_jump_threshold_);

        // Initialize state [a_L, b_L, c_L, ȧ_L, ḃ_L, ċ_L, a_R, b_R, c_R, ȧ_R, ḃ_R, ċ_R]
        state = Eigen::VectorXf::Zero(12);
        state << 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,   // Left lane (slight initial curvature)
         0.01, 0.0, 1.4, 0.0, 0.0, 0.0;  // Right lane (1.4m offset)


        // Initialize covariance matrix
        P = Eigen::MatrixXf::Identity(12,12) * 0.1;
        P.block(0,0,3,3) *= 0.1f;  // Higher confidence in a,b,c
        P.block(6,6,3,3) *= 0.1f;
        P.block(3,3,3,3) *= 1.0f;  // Higher uncertainty for derivatives
        P.block(9,9,3,3) *= 1.0f;

        // Process noise
        Q = Eigen::MatrixXf::Identity(12,12) * 0.05;  // Increased base from 0.01
        Q.block(0,0,3,3) *= 0.01f;  // a,b,c process noise (slightly higher)
        Q.block(6,6,3,3) *= 0.01f;
        Q.block(3,3,3,3) *= 0.1f;   // Derivative process noise (10x higher)
        Q.block(9,9,3,3) *= 0.1f;
                
        //Measurement noise
        R = Eigen::MatrixXf::Identity(6,6) * 0.01;  // For full observation case
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
        
            // In constructor:
        last_speed_time_ = this->now();
        last_servo_time_ = this->now();

        current_control_[0] = 0.0f; // Default to stopped
        last_speed_value_ = 0.0f;
    }

private:
    /////////////////////////////////////
    /*
    State Vecotr Initialize 12D state vector [a_L, b_L, c_L, ȧ_L, ḃ_L, ċ_L, a_R, b_R, c_R, ȧ_R, ḃ_R, ċ_R]

    */
    
    Eigen::VectorXf state;
    Eigen::MatrixXf P, Q, R, I, F, H, H_full;
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;
    Eigen::Vector3f lane_transform_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Time last_callback_time_; 
    Eigen::Vector2f current_control_ = Eigen::Vector2f::Zero();
    rclcpp::Time last_update_time_;
    bool first_update_ = true;
    std::deque<std::pair<rclcpp::Time, float>> speed_history_;
    std::deque<std::pair<rclcpp::Time, float>> servo_history_;

    rclcpp::Time last_speed_time_;
    rclcpp::Time last_servo_time_;
    float speed_timeout_ = 0.5;  // 500ms timeout
    float servo_timeout_ = 0.5;

    float last_speed_value_ = 0.0f;

    double time_jump_threshold_ = 1.0; // Configurable threshold in seconds





    const float wheel_radius_ = 0.1f;      // 10cm
    const float wheelbase_L_ = 0.22f;      // 22cm
    const float rpm_to_ms_ = (2.0 * M_PI * wheel_radius_) / 60.0f;  // Exact conversion factor
    const float servo_to_rad_ = 0.6f;      // Assuming servo uses [-1,1] range for ±0.5rad (28.6°)

    void initializeFilter() {
        // Reset to initial state
        state = Eigen::VectorXf::Zero(12);
        state << 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,   // Left lane
                 0.01, 0.0, 1.4, 0.0, 0.0, 0.0;    // Right lane
        
        // Reset covariance
        P = Eigen::MatrixXf::Identity(12,12) * 0.1;
        P.block(0,0,3,3) *= 0.1f;
        P.block(6,6,3,3) *= 0.1f;
        P.block(3,3,3,3) *= 1.0f;
        P.block(9,9,3,3) *= 1.0f;
        
        first_update_ = true;
    }


    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
    }

    void speedCallback(const std_msgs::msg::Float32::SharedPtr msg) {
        last_speed_time_ = this->now();
        current_control_[0] = msg->data * rpm_to_ms_;
    }
    
    void servoCallback(const std_msgs::msg::Float32::SharedPtr msg) {
        last_servo_time_ = this->now();
        current_control_[1] = msg->data * servo_to_rad_;
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {

        rclcpp::Time current_time = msg->header.stamp;
        float dt;

        if (first_update_) {
            dt = 1.0/30.0;  // Assume 30Hz for first update
            first_update_ = false;
        } else {
            dt = (current_time - last_update_time_).seconds();
            
            // More robust bag loop detection
            const double time_jump_threshold = 1.0; // 1 second
            if (current_time < last_update_time_) {
                // Only reset if significant backward jump (not just small timestamp fluctuation)
                if ((last_update_time_ - current_time).seconds() > time_jump_threshold) {
                    RCLCPP_INFO(this->get_logger(), "Significant time jump detected (%.3fs), resetting filter", 
                            (last_update_time_ - current_time).seconds());
                    initializeFilter();
                    dt = 1.0/30.0;
                } else {
                    // Small backward fluctuation - just warn and continue
                    RCLCPP_DEBUG(this->get_logger(), 
                                "Minor timestamp fluctuation (%.3fs), ignoring",
                                (last_update_time_ - current_time).seconds());
                    dt = 0.001; // Small positive dt to keep moving forward
                }
            }
            
            // Sanity checks
            if (dt <= 0 || dt > 0.5) {
                dt = 0.1;
                RCLCPP_WARN(this->get_logger(), "Invalid dt=%.3f, using default", dt);
            }
        }
        last_update_time_ = current_time;

        // Add this check for control commands (NEW)
        if ((current_time - last_speed_time_).seconds() > speed_timeout_) {
            current_control_[0] = 0.0f;  // Zero speed if no recent commands
            RCLCPP_DEBUG(this->get_logger(), "Speed command timeout, assuming zero");
        }

        if ((current_time - last_servo_time_).seconds() > servo_timeout_) {
            current_control_[1] = 0.0f;  // Zero steering if no recent commands
            RCLCPP_DEBUG(this->get_logger(), "Servo command timeout, assuming straight");
        }
        
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
        
        if (!selected_clusters.empty()) {
            // Camera position and orientation in map frame
            Eigen::Vector3f camera_position(
                current_pose_.pose.position.x,
                current_pose_.pose.position.y,
                current_pose_.pose.position.z);
        
            // Get camera orientation
            tf2::Quaternion q(
                current_pose_.pose.orientation.x,
                current_pose_.pose.orientation.y,
                current_pose_.pose.orientation.z,
                current_pose_.pose.orientation.w);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            const float cos_yaw = cos(yaw);
            const float sin_yaw = sin(yaw);
        
            // Variables to track left/right clusters
            float max_left_score = -std::numeric_limits<float>::infinity();
            float min_right_score = std::numeric_limits<float>::infinity();
            pcl::PointIndices left_candidate, right_candidate;
        
            for (const auto& cluster : selected_clusters) {
                // Voting system for left/right determination
                int left_votes = 0;
                int right_votes = 0;
                float total_y_cam = 0.0f;
        
                // Analyze all points in the cluster
                for (int idx : cluster.indices) {
                    const auto& point = white_cloud->points[idx];
                    Eigen::Vector3f relative_pos(
                        point.x - camera_position[0],
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
        
                // Calculate cluster score (positive for left, negative for right)
                float mean_y_cam = total_y_cam / cluster.indices.size();
                float cluster_score = (left_votes - right_votes) * std::abs(mean_y_cam);
        
                // Determine if this is the strongest left or right candidate
                if (cluster_score > 0 && cluster_score > max_left_score) {
                    max_left_score = cluster_score;
                    left_candidate = cluster;
                } else if (cluster_score < 0 && cluster_score < min_right_score) {
                    min_right_score = cluster_score;
                    right_candidate = cluster;
                }
            }
        
            // Assign detected clusters
            if (max_left_score > -std::numeric_limits<float>::infinity()) {
                leftmost_cluster = left_candidate;
                left_detected = true;
            }
            if (min_right_score < std::numeric_limits<float>::infinity()) {
                rightmost_cluster = right_candidate;
                right_detected = true;
            }
        
            // Special case: if only one cluster found, still classify it properly
            if (selected_clusters.size() == 1) {
                if (left_detected && !right_detected) {
                    RCLCPP_INFO(this->get_logger(), "Single cluster detected on left side");
                } else if (!left_detected && right_detected) {
                    RCLCPP_INFO(this->get_logger(), "Single cluster detected on right side");
                }
            }
        }

        // Fit circles to left and right lane markings
        this->get_parameter("fit_side", fit_side_);
        //fitCircle(white_cloud, leftmost_cluster, left_circle);
        //RCLCPP_INFO(this->get_logger(), "Circle 1: x=%f, y=%f, r=%f", left_circle[0], left_circle[1], left_circle[2]);

        // Measurement Update Logic
        Eigen::VectorXf Z;
        Eigen::MatrixXf H;
        Eigen::MatrixXf R_current;
        Eigen::Vector3f coeffs_right, coeffs_left;
        
        if(!fit_side_)
        {
            left_detected=true;
            right_detected=false;
        }
        if (left_detected && right_detected) {
            
            

            pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *right_cluster);
            coeffs_right=fitQuadraticCurve(right_cluster, curve_publisher_right_, current_pose_.pose);

            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            coeffs_left=fitQuadraticCurve(left_cluster, curve_publisher_left_, current_pose_.pose);

            lane_transform_ = calculateAndStoreTransform(coeffs_left, coeffs_right);

            Z.resize(6);

            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;


        
        } else if (left_detected) {
            // Only left cluster detected: fit a single circle to the left cluster
            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            Eigen::Vector3f coeffs_left = fitQuadraticCurve(left_cluster, curve_publisher_left_, current_pose_.pose);
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
            Eigen::Vector3f coeffs_right = fitQuadraticCurve(right_cluster,curve_publisher_right_, current_pose_.pose);
            Eigen::Vector3f coeffs_left = estimateLeftFromRight(coeffs_right, lane_transform_);
        
            Z.resize(6);
            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                 coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;
            R_current.block(0,0,3,3) *= 2.0; // Higher uncertainty for estimated left lane
        }
            
        else
        {
            return;
        }

        // Kalman Prediction Step
        predictionStep(current_time, dt);


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
            leftmost_cluster,
            rightmost_cluster,
            white_cloud
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

    std::pair<float, float> findNearestControl(const std::deque<std::pair<rclcpp::Time, float>>& history, 
        rclcpp::Time query_time) 
    {
        for (auto it = history.rbegin(); it != history.rend(); ++it) 
        {
            if (it->first <= query_time) 
            {
                return {it->second, (query_time - it->first).seconds()};
            }
        }
        return {0.0f, 0.0f};  // Default if no recent control
    }

    // Kalman Prediction Step with Motion Model
    void predictionStep(rclcpp::Time current_time, float dt) {
        // 1. Handle command timeouts
        const bool speed_timed_out = (current_time - last_speed_time_).seconds() > speed_timeout_;
        const bool servo_timed_out = (current_time - last_servo_time_).seconds() > servo_timeout_;
        
        // For servo: timeout → assume straight (0.0)
        if (servo_timed_out) {
            current_control_[1] = 0.0f;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "Servo command timeout, assuming straight");
        }
    
        // 2. Try to update controls if not timed out
        if (!speed_timed_out || !servo_timed_out) {
            auto [speed_value, speed_lag] = findNearestControl(speed_history_, current_time);
            auto [servo_value, servo_lag] = findNearestControl(servo_history_, current_time);
            
            // For speed: only update if we got a fresh value
            if (speed_lag < 0.1 && !speed_timed_out) {
                current_control_[0] = speed_value * rpm_to_ms_;
                last_speed_value_ = current_control_[0]; // Store last good value
            }
            
            // For servo: update if fresh and not timed out
            if (servo_lag < 0.1 && !servo_timed_out) {
                current_control_[1] = servo_value * servo_to_rad_;
            }
        }
    
        // 3. Apply bicycle model if we have motion
        if (std::abs(current_control_[0]) > 0.01f) {  // Small threshold to avoid drift
            const float v = current_control_[0];
            const float gamma = current_control_[1];
            tf2::Quaternion q(
                current_pose_.pose.orientation.x,
                current_pose_.pose.orientation.y,
                current_pose_.pose.orientation.z,
                current_pose_.pose.orientation.w);
            const float theta = q.getAngle();
            
            // Calculate motion deltas
            const float delta_theta = (v * tan(gamma) / wheelbase_L_) * dt;
            const float delta_y = v * sin(theta) * dt;
            
            // Update state
            state[1] -= delta_theta;  // b_L
            state[2] -= delta_y;      // c_L
            state[7] -= delta_theta;  // b_R 
            state[8] -= delta_y;      // c_R
            
            // Adaptive process noise
            const float turn_factor = 1.0f + (2.0f * std::abs(gamma));
            Q.block(1,1,2,2) = Eigen::Matrix2f::Identity() * 0.01f * turn_factor;
            Q.block(7,7,2,2) = Eigen::Matrix2f::Identity() * 0.01f * turn_factor;
        }
    
        // 4. State transition matrix
        F = Eigen::MatrixXf::Identity(12,12);
        F(0,3) = dt; F(1,4) = dt; F(2,5) = dt;   // Left lane
        F(6,9) = dt; F(7,10) = dt; F(8,11) = dt;  // Right lane
    
        // 5. Covariance prediction
        P = F * P * F.transpose() + Q;
    }


    Eigen::Vector3f fitQuadraticCurve(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub,
        const geometry_msgs::msg::Pose& camera_pose)
    {
        // Get camera transform
        tf2::Quaternion q( camera_pose.orientation.x, camera_pose.orientation.y, camera_pose.orientation.z, camera_pose.orientation.w);
        tf2::Matrix3x3 rotation(q);

        Eigen::Matrix3f rot_matrix;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
        rot_matrix(i,j) = rotation[i][j];

        Eigen::Vector3f camera_pos(
        camera_pose.position.x,
        camera_pose.position.y,
        camera_pose.position.z);

        // Transform points to camera frame
        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points) {
            Eigen::Vector3f pt_map(point.x, point.y, point.z);
            Eigen::Vector3f pt_cam = rot_matrix.transpose() * (pt_map - camera_pos);
            x_vals.push_back(pt_cam.x());
            y_vals.push_back(pt_cam.y());
        }

        // Fit quadratic in camera frame (y = ax² + bx + c)
        Eigen::MatrixXd A(x_vals.size(), 3);
        Eigen::VectorXd Z(y_vals.size());

        for (size_t i = 0; i < x_vals.size(); i++) {
            A(i, 0) = x_vals[i] * x_vals[i]; // x² term
            A(i, 1) = x_vals[i];             // x term
            A(i, 2) = 1.0;                   // constant term
            Z(i) = y_vals[i];
        }

        Eigen::Vector3d coeffs_cam = A.colPivHouseholderQr().solve(Z);

        // Visualize curve in map frame
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

        // Generate and transform curve points back to map frame
        float x_min = *std::min_element(x_vals.begin(), x_vals.end());
        float x_max = *std::max_element(x_vals.begin(), x_vals.end());

        for (float x = x_min; x <= x_max; x += 0.05) {
            float y = coeffs_cam(0)*x*x + coeffs_cam(1)*x + coeffs_cam(2);
            Eigen::Vector3f pt_cam(x, y, 0);
            Eigen::Vector3f pt_map = rot_matrix * pt_cam + camera_pos;

            geometry_msgs::msg::Point p;
            p.x = pt_map.x();
            p.y = pt_map.y();
            p.z = 0.0;
            curve_marker.points.push_back(p);
        }

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        pub->publish(markers);

        return coeffs_cam.cast<float>();
    }

    
    Eigen::Vector3f estimateRightFromLeft(const Eigen::Vector3f& left_coeffs_cam, 
        const Eigen::Vector3f& lane_transform_cam) 
    {
        return left_coeffs_cam + lane_transform_cam;
    }

    Eigen::Vector3f estimateLeftFromRight(const Eigen::Vector3f& right_coeffs_cam, 
            const Eigen::Vector3f& lane_transform_cam) 
    {
        return right_coeffs_cam - lane_transform_cam;
    }


    void visualizeKalmanState(
        const Eigen::VectorXf& state,
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher,
        bool left_detected,
        bool right_detected,
        const pcl::PointIndices& left_cluster_indices,
        const pcl::PointIndices& right_cluster_indices,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud)
        {
            visualization_msgs::msg::MarkerArray marker_array;
            
            // 1. Get current camera pose and orientation
            tf2::Quaternion q(
                current_pose_.pose.orientation.x,
                current_pose_.pose.orientation.y,
                current_pose_.pose.orientation.z,
                current_pose_.pose.orientation.w);
            tf2::Matrix3x3 rot(q);
            Eigen::Vector3f cam_pos(
                current_pose_.pose.position.x,
                current_pose_.pose.position.y,
                current_pose_.pose.position.z);
        
            // Convert tf2::Matrix3x3 to Eigen::Matrix3f
            Eigen::Matrix3f rot_eigen;
            for(int i=0; i<3; i++) {
                for(int j=0; j<3; j++) {
                    rot_eigen(i,j) = rot[i][j];
                }
            }
        
            // 2. Extract camera-relative coefficients from state
            Eigen::Vector3f left_coeffs(state[0], state[1], state[2]);  // a_L, b_L, c_L
            Eigen::Vector3f right_coeffs(state[6], state[7], state[8]); // a_R, b_R, c_R
        
            // 3. Determine visualization range in camera X coordinates
            auto getCameraXRange = [&](const pcl::PointIndices& indices) {
                if (indices.indices.empty()) {
                    return std::make_pair(-5.0f, 5.0f); // Default 10m forward
                }
        
                float min_x = std::numeric_limits<float>::max();
                float max_x = std::numeric_limits<float>::lowest();
                
                for (int idx : indices.indices) {
                    const auto& point = white_cloud->points[idx];
                    Eigen::Vector3f pt_map(point.x, point.y, point.z);
                    Eigen::Vector3f pt_cam = rot_eigen.transpose() * (pt_map - cam_pos);
                    
                    min_x = std::min(min_x, pt_cam.x());
                    max_x = std::max(max_x, pt_cam.x());
                }
                return std::make_pair(min_x - 0.0f, max_x + 0.0f); // Add small padding
            };
        
            auto [left_min_x, left_max_x] = getCameraXRange(left_cluster_indices);
            auto [right_min_x, right_max_x] = getCameraXRange(right_cluster_indices);
        
            // 4. Create left lane marker (green for detected, blue for estimated)
            visualization_msgs::msg::Marker left_marker;
            left_marker.header.frame_id = "map";
            left_marker.header.stamp = this->now();
            left_marker.ns = "kalman_lanes";
            left_marker.id = 0;
            left_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            left_marker.action = visualization_msgs::msg::Marker::ADD;
            left_marker.scale.x = 0.05; // Line width
            left_marker.color.a = 1.0;
            left_marker.color.g = left_detected ? 1.0 : 0.5;  // Green if detected
            left_marker.color.b = left_detected ? 0.0 : 1.0;  // Blue if estimated
            left_marker.pose.orientation.w = 1.0;
        
            // 5. Create right lane marker (red for detected, pink for estimated)
            visualization_msgs::msg::Marker right_marker = left_marker;
            right_marker.id = 1;
            right_marker.color.r = 1.0;
            right_marker.color.g = right_detected ? 0.0 : 0.5;
            right_marker.color.b = right_detected ? 0.0 : 0.5;
        
            // 6. Generate points for left lane
            const float step_size = 0.2f; // meters between points
            for (float x_cam = left_min_x; x_cam <= left_max_x; x_cam += step_size) {
                // Calculate y in camera frame
                float y_cam = left_coeffs[0]*x_cam*x_cam + left_coeffs[1]*x_cam + left_coeffs[2];
                
                // Transform to map frame
                Eigen::Vector3f p_cam(x_cam, y_cam, 0);
                Eigen::Vector3f p_map = rot_eigen * p_cam + cam_pos;
                
                geometry_msgs::msg::Point p;
                p.x = p_map.x();
                p.y = p_map.y();
                p.z = 0;
                left_marker.points.push_back(p);
            }
        
            // 7. Generate points for right lane
            for (float x_cam = right_min_x; x_cam <= right_max_x; x_cam += step_size) {
                // Calculate y in camera frame
                float y_cam = right_coeffs[0]*x_cam*x_cam + right_coeffs[1]*x_cam + right_coeffs[2];
                
                // Transform to map frame
                Eigen::Vector3f p_cam(x_cam, y_cam, 0);
                Eigen::Vector3f p_map = rot_eigen * p_cam + cam_pos;
                
                geometry_msgs::msg::Point p;
                p.x = p_map.x();
                p.y = p_map.y();
                p.z = 0;
                right_marker.points.push_back(p);
            }
        
            // 8. Add markers to array and publish
            marker_array.markers.push_back(left_marker);
            marker_array.markers.push_back(right_marker);
            publisher->publish(marker_array);
        }


    Eigen::Vector3f calculateAndStoreTransform(const Eigen::Vector3f& left_coeffs_cam, 
                                            const Eigen::Vector3f& right_coeffs_cam) 
    {
        // Transform is calculated in camera-relative coordinates
        return right_coeffs_cam - left_coeffs_cam;
    }

    std::vector<pcl::PointIndices> clusterWhitePoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const geometry_msgs::msg::Pose &current_pose_)
    {
        // 1. Verify frame_id is map
        if (cloud->header.frame_id != "map") {
            //RCLCPP_WARN(this->get_logger(), 
                //"Expected point cloud in map frame, got %s. Proceeding anyway.",
            //    cloud->header.frame_id.c_str());
        }
    
        // 2. Perform Euclidean clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
        
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.06);  // 8cm
        ec.setMinClusterSize(250);     // Minimum points per cluster
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

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr servo_sub_;




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
