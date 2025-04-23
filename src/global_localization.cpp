#include "global_localization/global_localization.hpp"
#include "kalman_rel_localization/utilities.hpp"
#include "kalman_rel_localization/visualization.hpp"


using namespace lane_detection_utils;
using namespace lane_detection_visualization;



GlobalLocalization::GlobalLocalization() 
: Node("global_local_node"), 
  last_speed_time_(this->now()), 
  last_servo_time_(this->now())
{
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ground_plane", rclcpp::SensorDataQoS(),
        std::bind(&GlobalLocalization::pointCloudCallback, this, std::placeholders::_1));

    pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/zed/zed_node/pose", 10,
        std::bind(&GlobalLocalization::poseCallback, this, std::placeholders::_1));

    speed_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/commands/motor/speed", 10,
        std::bind(&GlobalLocalization::speedCallback, this, std::placeholders::_1));
    servo_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/commands/servo/position", 10,
        std::bind(&GlobalLocalization::servoCallback, this, std::placeholders::_1));

    rclcpp::QoS map_qos(10);
    map_qos.transient_local();
    map_qos.reliable();
    
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 
        map_qos,
        std::bind(&GlobalLocalization::mapCallback, this, std::placeholders::_1));

    particle_sub_ = this->create_subscription<nav2_msgs::msg::ParticleCloud>(
        "/particle_cloud", 10,
        std::bind(&GlobalLocalization::particleCallback, this, std::placeholders::_1));

    particle_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("particle_visualization", 10);


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

    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/cluster_scan", 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

    

    this->declare_parameter("fit_side", true);
    this->get_parameter("fit_side", fit_side_);

    // Add this at the beginning of the constructor

    this->declare_parameter("time_jump_threshold", 5.0); // Increased from 1.0 to 5.0
    this->get_parameter("time_jump_threshold", time_jump_threshold_);

    // Add parameters for laser scan conversion
    this->declare_parameter("min_scan_range", 0.1);
    this->declare_parameter("max_scan_range", 10.0);
    this->declare_parameter("scan_height_threshold", 0.1);

    
        // In constructor:
    last_speed_time_ = this->now();
    last_servo_time_ = this->now();

    current_control_[0] = 0.0f; // Default to stopped
    last_speed_value_ = 0.0f;
}

void GlobalLocalization::particleCallback(const nav2_msgs::msg::ParticleCloud::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker marker;
    
    marker.header = msg->header;
    marker.ns = "particles";
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 0.5;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    
    for (size_t i = 0; i < msg->particles.size(); ++i) {
        marker.id = i;
        marker.pose = msg->particles[i].pose;
        markers.markers.push_back(marker);
    }
    
    // Publish to RViz or similar
    particle_viz_pub_->publish(markers);
    RCLCPP_DEBUG(this->get_logger(), "Visualized %zu particles", msg->particles.size());
}

    // Add these new callback methods
void GlobalLocalization::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) 
{
    current_map_ = msg;
    map_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Map received with resolution %f", msg->info.resolution);
}

// Add this new method to convert clusters to laser scan
sensor_msgs::msg::LaserScan GlobalLocalization::clustersToLaserScan(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<pcl::PointIndices>& clusters,
    const geometry_msgs::msg::Pose& current_pose) 
{
    sensor_msgs::msg::LaserScan scan;
    
    // Get parameters
    double min_range, max_range, height_threshold;
    this->get_parameter("min_scan_range", min_range);
    this->get_parameter("max_scan_range", max_range);
    this->get_parameter("scan_height_threshold", height_threshold);

    // Configure scan message
    scan.header.stamp = this->now();
    scan.header.frame_id = "base_link";  // Or your sensor frame
    scan.angle_min = -M_PI;
    scan.angle_max = M_PI;
    scan.angle_increment = M_PI / 180.0;  // 1 degree resolution
    scan.range_min = min_range;
    scan.range_max = max_range;
    scan.time_increment = 0.0;
    scan.scan_time = 0.1;

    // Initialize ranges with max range (no detection)
    int num_readings = (scan.angle_max - scan.angle_min) / scan.angle_increment;
    scan.ranges.resize(num_readings, scan.range_max);

    // Convert clusters to scan
    for (const auto& cluster : clusters) {
        for (const auto& idx : cluster.indices) {
            const auto& point = cloud->points[idx];
            
            // Filter by height if needed
            if (fabs(point.z) > height_threshold) {
                continue;
            }
            
            // Convert to polar coordinates
            float range = sqrt(point.x * point.x + point.y * point.y);
            float angle = atan2(point.y, point.x);
            
            // Only consider points in valid range
            if (range < min_range || range > max_range) {
                continue;
            }
            
            // Find the appropriate bin
            int bin = (angle - scan.angle_min) / scan.angle_increment;
            if (bin >= 0 && bin < num_readings) {
                // Keep the closest point in each bin
                if (range < scan.ranges[bin]) {
                    scan.ranges[bin] = range;
                }
            }
        }
    }
    
    return scan;
}

void GlobalLocalization::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    current_pose_ = *msg;
}

void GlobalLocalization::speedCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    last_speed_time_ = this->now();
    current_control_[0] = msg->data * rpm_to_ms_;
}

void GlobalLocalization::servoCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    last_servo_time_ = this->now();
    current_control_[1] = msg->data * servo_to_rad_;
}

void GlobalLocalization::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    rclcpp::Time current_time = msg->header.stamp;
    float dt;

    if (first_update_) {
        dt = 1.0/30.0; // Default for first update
        first_update_ = false;
    } else {
        // Handle invalid timestamps
        if (last_update_time_.nanoseconds() == 0 || current_time.nanoseconds() == 0) {
            dt = 1.0/30.0;
            RCLCPP_WARN(this->get_logger(), "Invalid timestamp detected, using dt=%.3f", dt);
        } 
        else {
            dt = (current_time - last_update_time_).seconds();
            
            // Handle time jumps
            if (current_time < last_update_time_) {
                if ((last_update_time_ - current_time).seconds() > time_jump_threshold_) {
                    RCLCPP_WARN(this->get_logger(), 
                        "Significant time jump detected (%.3fs), resetting filter", 
                        (last_update_time_ - current_time).seconds());
                 
                    dt = 1.0/30.0;
                } else {
                    // Small time fluctuation - use last valid dt
                    dt = last_valid_dt_;
                    RCLCPP_DEBUG(this->get_logger(), 
                        "Minor timestamp fluctuation, using last valid dt=%.3f", dt);
                }
            }
            
            // Validate dt range
            const float MIN_DT = 0.001f; // 1ms
            const float MAX_DT = 1.0f;   // 1 second (increased from 0.5)
            
            if (dt <= MIN_DT || dt > MAX_DT) {
                dt = std::clamp(dt, MIN_DT, MAX_DT);
                RCLCPP_WARN(this->get_logger(), 
                    "Clamping dt=%.3f to [%.3f, %.3f]", 
                    dt, MIN_DT, MAX_DT);
            }
        }
    }
    
    last_valid_dt_ = dt; // Store for future use
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
    for (size_t i = 0; i < selected_clusters.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Cluster %zu has %zu points", i, selected_clusters[i].indices.size());
    }
    publishClusterMarkers(white_cloud, selected_clusters, current_pose_.pose, cluster_marker_pub_, current_time);

    sensor_msgs::msg::PointCloud2 white_msg;
    pcl::toROSMsg(*white_cloud, white_msg);
    white_msg.header = msg->header;
    white_publisher_->publish(white_msg);


    //MAKE AMCL HERE, TAKE CLUSTER TO LASER SCAN AND DO AMCL NAV2 THING FOR LOCALIZATION

    // Convert clusters to laser scan
    auto scan = clustersToLaserScan(white_cloud, selected_clusters, current_pose_.pose);
    scan_publisher_->publish(scan);
    
    
}





std::pair<float, float> GlobalLocalization::findNearestControl(const std::deque<std::pair<rclcpp::Time, float>>& history, rclcpp::Time query_time) 
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
    rclcpp::spin(std::make_shared<GlobalLocalization>());
    rclcpp::shutdown();
    return 0;
}