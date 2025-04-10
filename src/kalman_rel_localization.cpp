#include "kalman_rel_localization/kalman_rel_localization.hpp"
#include "kalman_rel_localization/utilities.hpp"
#include "kalman_rel_localization/visualization.hpp"


using namespace lane_detection_utils;
using namespace lane_detection_visualization;



KalmanRelLocalization::KalmanRelLocalization() 
: Node("kalman_rel_node"), 
  last_speed_time_(this->now()), 
  last_servo_time_(this->now())
{
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ground_plane", rclcpp::SensorDataQoS(),
        std::bind(&KalmanRelLocalization::pointCloudCallback, this, std::placeholders::_1));

    pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/zed/zed_node/pose", 10,
        std::bind(&KalmanRelLocalization::poseCallback, this, std::placeholders::_1));

    speed_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/commands/motor/speed", 10,
        std::bind(&KalmanRelLocalization::speedCallback, this, std::placeholders::_1));
    servo_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/commands/servo/position", 10,
        std::bind(&KalmanRelLocalization::servoCallback, this, std::placeholders::_1));

    white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);

    curve_publisher_right_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("right_spline_fit_original", 10);
    curve_publisher_left_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("left_spline_fit_original", 10);
    kalman_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("kalman_state", 10);
    middle_lane_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("middle_lane", 10);
    cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
    test_cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("test_clusters", 10);

    distance_orientation_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("distance_orientation", 10);
    distance_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("distance_marker", 10);
    orientation_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("orientation_marker", 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);



    this->declare_parameter("fit_side", true);
    this->get_parameter("fit_side", fit_side_);

    // Add this at the beginning of the constructor

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

void KalmanRelLocalization::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    current_pose_ = *msg;
}

void KalmanRelLocalization::speedCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    last_speed_time_ = this->now();
    current_control_[0] = msg->data * rpm_to_ms_;
}

void KalmanRelLocalization::servoCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    last_servo_time_ = this->now();
    current_control_[1] = msg->data * servo_to_rad_;
}

void KalmanRelLocalization::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rclcpp::Time current_time = msg->header.stamp;
    float dt;

    if (first_update_) {
        dt = 1.0/30.0;
        first_update_ = false;
    } else {
        // Add check for valid timestamps
        if (last_update_time_.nanoseconds() == 0 || current_time.nanoseconds() == 0) {
            dt = 1.0/30.0;
        } else {
            dt = (current_time - last_update_time_).seconds();
            
            // Handle time jumps
            if (current_time < last_update_time_) {
                if ((last_update_time_ - current_time).seconds() > time_jump_threshold_) {
                    RCLCPP_INFO(this->get_logger(), "Significant time jump detected (%.3fs), resetting filter", 
                            (last_update_time_ - current_time).seconds());
                    initializeFilter();
                    dt = 1.0/30.0;
                } else {
                    RCLCPP_DEBUG(this->get_logger(), 
                                "Minor timestamp fluctuation (%.3fs), ignoring",
                                (last_update_time_ - current_time).seconds());
                    dt = 0.001;
                }
            }
            
            if (dt <= 0 || dt > 0.5) {
                dt = 0.1;
                RCLCPP_WARN(this->get_logger(), "Invalid dt=%.3f, using default", dt);
            }
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

    for (const auto &point : cloud->points) {
        float h, s, v;
        RGBtoHSV(point.r, point.g, point.b, h, s, v);
        if (v > 0.65f && s < 0.2f) {
            white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
        }
    }

    publishCameraAxes(current_pose_, marker_pub_, current_time);

    if (white_cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "No white points detected!");
        return;
    }

    std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud, current_pose_.pose);
    publishClusterMarkers(white_cloud, selected_clusters, current_pose_.pose, cluster_marker_pub_, current_time);

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


        this->get_parameter("fit_side", fit_side_);


        // Measurement Update Logic
        Eigen::VectorXf Z;
        Eigen::MatrixXf H;
        Eigen::MatrixXf R_current;
        Eigen::Vector3f coeffs_right, coeffs_left;
        
        if (left_detected && right_detected) {
            
            

            pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *right_cluster);
            coeffs_right=fitQuadraticCurve(right_cluster, curve_publisher_right_, current_pose_.pose, current_time);

            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            coeffs_left=fitQuadraticCurve(left_cluster, curve_publisher_left_, current_pose_.pose, current_time);

            lane_transform_ = calculateAndStoreTransform(coeffs_left, coeffs_right);

            Z.resize(6);

            Z << coeffs_left(0), coeffs_left(1), coeffs_left(2),
                coeffs_right(0), coeffs_right(1), coeffs_right(2);
            H = H_full;
            R_current = R;


        
        } else if (left_detected) {
            // Only left cluster detected: fit a single  to the left cluster
            pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *left_cluster);
            Eigen::Vector3f coeffs_left = fitQuadraticCurve(left_cluster, curve_publisher_left_, current_pose_.pose, current_time);
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
            Eigen::Vector3f coeffs_right = fitQuadraticCurve(right_cluster,curve_publisher_right_, current_pose_.pose, current_time);
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



        Eigen::Vector3f middle_coeffs_cam;
        if (fit_side_) {
            // Use left curve shifted right
            middle_coeffs_cam = shiftToMiddle(Eigen::Vector3f(state[0], state[1], state[2]), 
                                lane_transform_, true);
        } else {
            // Use right curve shifted left
            middle_coeffs_cam = shiftToMiddle(Eigen::Vector3f(state[6], state[7], state[8]), 
                                lane_transform_, false);
        }

        // Get camera position and orientation for visualization
        tf2::Quaternion q(
            current_pose_.pose.orientation.x,
            current_pose_.pose.orientation.y,
            current_pose_.pose.orientation.z,
            current_pose_.pose.orientation.w);
        tf2::Matrix3x3 rot(q);
        Eigen::Matrix3f rot_eigen;
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                rot_eigen(i,j) = rot[i][j];
            }
        }
        Eigen::Vector3f cam_pos(
            current_pose_.pose.position.x,
            current_pose_.pose.position.y,
            current_pose_.pose.position.z);

        // Calculate reasonable bounds for middle lane visualization
        float x_min_cam = 0.5f;  // Start 0.5m in front of camera
        float x_max_cam = 3.0f;  // Extend 3.0m forward
        
        // Create middle lane marker
        visualization_msgs::msg::MarkerArray middle_markers;
        visualization_msgs::msg::Marker middle_marker;
        middle_marker.header.frame_id = "map";
        middle_marker.header.stamp = this->now();
        middle_marker.ns = "middle_lane";
        middle_marker.id = 0;
        middle_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        middle_marker.action = visualization_msgs::msg::Marker::ADD;
        middle_marker.scale.x = 0.05;  // Thicker line for better visibility
        middle_marker.color.r = 1.0;
        middle_marker.color.g = 1.0;
        middle_marker.color.b = 0.0;  // Yellow color
        middle_marker.color.a = 1.0;
        middle_marker.pose.orientation.w = 1.0;

        // Generate points for middle curve within the calculated bounds
        for (float x_cam = x_min_cam; x_cam <= x_max_cam; x_cam += 0.1f) {
            float y_cam = middle_coeffs_cam[0]*x_cam*x_cam + middle_coeffs_cam[1]*x_cam + middle_coeffs_cam[2];
            Eigen::Vector3f p_cam(x_cam, y_cam, 0);
            Eigen::Vector3f p_map = rot_eigen * p_cam + cam_pos;
            
            geometry_msgs::msg::Point p;
            p.x = p_map.x();
            p.y = p_map.y();
            p.z = 0;
            middle_marker.points.push_back(p);
        }
        middle_markers.markers.push_back(middle_marker);
        middle_lane_->publish(middle_markers);

        // Calculate distance and orientation
        float distance = 0.0f;
        float orientation_diff = 0.0f;
        float min_distance = 0.5f;
        float max_distance = 3.0f;

        calculateDistanceAndOrientation(
            middle_coeffs_cam,
            cam_pos,
            rot,
            min_distance,
            max_distance,
            distance,
            orientation_diff);

        // Get yaw angle for orientation marker
        double roll, pitch, yaw;
        rot.getRPY(roll, pitch, yaw);

        // Publish results
        std_msgs::msg::Float32MultiArray distance_orientation_msg;
        distance_orientation_msg.data.push_back(distance);
        distance_orientation_msg.data.push_back(orientation_diff);
        distance_orientation_pub_->publish(distance_orientation_msg);

        // Publish distance visualization
        visualization_msgs::msg::Marker distance_marker;
        distance_marker.header.frame_id = "map";
        distance_marker.header.stamp = this->now();
        distance_marker.ns = "distance";
        distance_marker.id = 0;
        distance_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        distance_marker.text = "D: " + std::to_string(distance).substr(0,4) + "m";
        distance_marker.pose.position.x = current_pose_.pose.position.x;
        distance_marker.pose.position.y = current_pose_.pose.position.y;
        distance_marker.pose.position.z = current_pose_.pose.position.z + 0.5;
        distance_marker.scale.z = 0.2; // Text height
        distance_marker.color.a = 1.0;
        distance_marker.color.r = 1.0;
        distance_marker.color.g = 1.0;
        distance_marker.color.b = 1.0;
        distance_marker_pub_->publish(distance_marker);

        // Publish orientation as text only
        visualization_msgs::msg::Marker orientation_marker;
        orientation_marker.header.frame_id = "map";
        orientation_marker.header.stamp = this->now();
        orientation_marker.ns = "orientation";
        orientation_marker.id = 0;
        orientation_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        orientation_marker.text = "θ: " + std::to_string(orientation_diff * 180.0/M_PI).substr(0,4) + "°";
        orientation_marker.pose.position.x = current_pose_.pose.position.x;
        orientation_marker.pose.position.y = current_pose_.pose.position.y;
        orientation_marker.pose.position.z = current_pose_.pose.position.z + 1.5; // Below distance text
        orientation_marker.scale.z = 0.2; // Text height
        orientation_marker.color.a = 1.0;
        orientation_marker.color.r = 1.0;
        orientation_marker.color.g = 1.0;
        orientation_marker.color.b = 1.0;
        orientation_marker_pub_->publish(orientation_marker);

        visualizeKalmanState(
            state,               // Your 12D Kalman state vector
            kalman_publisher_,    // Your marker publisher
            left_detected,       // Boolean from detection
            right_detected,      // Boolean from detection
            leftmost_cluster,
            rightmost_cluster,
            white_cloud,
            current_pose_.pose,
            current_time
        );
    
}

void KalmanRelLocalization::initializeFilter() 
{
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

void KalmanRelLocalization::predictionStep(rclcpp::Time current_time, float dt) 
{
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

std::pair<float, float> KalmanRelLocalization::findNearestControl(const std::deque<std::pair<rclcpp::Time, float>>& history, rclcpp::Time query_time) 
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

rclcpp::Parameter fit_side_param_;
bool fit_side_;


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanRelLocalization>());
    rclcpp::shutdown();
    return 0;
}