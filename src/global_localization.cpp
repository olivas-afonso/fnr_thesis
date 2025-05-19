#include "global_localization/global_localization.hpp"
#include "kalman_rel_localization/utilities.hpp"
#include "kalman_rel_localization/visualization.hpp"


using namespace lane_detection_utils;
using namespace lane_detection_visualization;



GlobalLocalization::GlobalLocalization(const rclcpp::NodeOptions & options) 
: Node("global_local_node", options), 
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
    

    
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 
        map_qos,
        std::bind(&GlobalLocalization::mapCallback, this, std::placeholders::_1));

    /*
    particle_sub_ = this->create_subscription<nav2_msgs::msg::ParticleCloud>(
        "/particle_cloud", 
        rclcpp::QoS(10).reliable(),
        std::bind(&GlobalLocalization::particleCallback, this, std::placeholders::_1));

    particle_viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "particle_visualization", rclcpp::QoS(10).reliable());
    */

    accumulated_white_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    persistent_white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/persistent_white_cloud", rclcpp::SensorDataQoS().reliable().transient_local());

    
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

    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
        "/cluster_scan", 
        rclcpp::QoS(10).best_effort()  // Match AMCL's default
    );

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

    debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("scan_debug", 10);

    

    this->declare_parameter("fit_side", true);
    this->get_parameter("fit_side", fit_side_);

    // Add this at the beginning of the constructor

    this->declare_parameter("time_jump_threshold", 5.0); // Increased from 1.0 to 5.0
    this->get_parameter("time_jump_threshold", time_jump_threshold_);

    // Add parameters for laser scan conversion
    this->declare_parameter("min_scan_range", 0.1);
    this->declare_parameter("max_scan_range", 10.0);

    
        // In constructor:
    last_speed_time_ = this->now();
    last_servo_time_ = this->now();

    current_control_[0] = 0.0f; // Default to stopped
    last_speed_value_ = 0.0f;
}

/*
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
*/
    // Add these new callback methods
void GlobalLocalization::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) 
{
    // Just store the map, no special handling needed
    current_map_ = msg;
    map_received_ = true;
}

sensor_msgs::msg::LaserScan GlobalLocalization::pointCloudToLaserScan(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::string& cloud_frame_id,
    const rclcpp::Time& cloud_time)
{
    sensor_msgs::msg::LaserScan scan;
    scan.header.stamp = cloud_time;
    scan.header.frame_id = "base_link";  // Use ZED as base frame

    // 1. Load scan parameters
    double min_range = 0.1, max_range = 10.0;
    this->get_parameter("min_scan_range", min_range);
    this->get_parameter("max_scan_range", max_range);

    // 2. LaserScan configuration
    scan.angle_min = -M_PI;
    scan.angle_max = M_PI;
    scan.angle_increment = M_PI / 180.0;  // 0.5 degrees
    scan.scan_time = 1.0 / 30.0; // match your data rate (adjust if needed)
    scan.range_min = static_cast<float>(min_range);
    scan.range_max = static_cast<float>(max_range);

    size_t num_ranges = static_cast<size_t>((scan.angle_max - scan.angle_min) / scan.angle_increment);
    scan.ranges.assign(num_ranges, scan.range_max + 1.0f);  // invalid by default (too far)

    // 3. Lookup transform from camera cloud frame to zed_camera_link
    geometry_msgs::msg::TransformStamped tf_camera_to_zed;
    try {
        tf_camera_to_zed = tf_buffer_->lookupTransform(
            "base_link", cloud_frame_id, cloud_time,
            tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(),
            "TF lookup failed from '%s' to '%s': %s",
            cloud_frame_id.c_str(), "base_link", ex.what());
        return scan;  // return scan filled with default range values
    }

    tf2::Transform tf;
    tf2::fromMsg(tf_camera_to_zed.transform, tf);

    // 4. Transform and populate scan
    for (const auto &point : cloud->points) {
        tf2::Vector3 pt_camera(point.x, point.y, point.z);
        tf2::Vector3 pt_zed = tf * pt_camera;

        float x = pt_zed.x();
        float y = pt_zed.y();
        float range = std::hypot(x, y);
        if (range < scan.range_min || range > scan.range_max) continue;

        float angle = std::atan2(y, x);  
        if (angle < scan.angle_min || angle > scan.angle_max) continue;

        size_t bin = static_cast<size_t>((angle - scan.angle_min) / scan.angle_increment);
        if (bin >= num_ranges) continue;

        if (range < scan.ranges[bin]) {
            scan.ranges[bin] = range;
        }
    }

    RCLCPP_INFO(this->get_logger(), "Publishing scan with %zu ranges, frame_id: %s", 
    scan.ranges.size(), scan.header.frame_id.c_str());
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



void GlobalLocalization::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Ensure we have the latest transforms
    try {
        auto transform = tf_buffer_->lookupTransform(
            "map", "zed_camera_link",
            msg->header.stamp,
            tf2::durationFromSec(0.1));
        
        // Convert to PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto &point : cloud->points) {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
            if (v > 0.65f && s < 0.2f) {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
                
                // Transform point to map frame before accumulating
                geometry_msgs::msg::PointStamped pt_camera, pt_map;
                pt_camera.header.frame_id = "map";
                pt_camera.header.stamp = msg->header.stamp;
                pt_camera.point.x = point.x;
                pt_camera.point.y = point.y;
                pt_camera.point.z = point.z;
                
                try {
                    tf_buffer_->transform(pt_camera, pt_map, "map");
                    accumulated_white_cloud_->push_back(pcl::PointXYZ(
                        pt_map.point.x, 
                        pt_map.point.y, 
                        pt_map.point.z
                    ));
                } catch (const tf2::TransformException &ex) {
                    RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
                }
            }
        }

        if (white_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }

        // Convert to laser scan
        auto scan = pointCloudToLaserScan(
            white_cloud, 
            msg->header.frame_id,
            msg->header.stamp);
        scan_publisher_->publish(scan);

        // Your existing cluster processing
        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud, current_pose_.pose);
        publishClusterMarkers(white_cloud, selected_clusters, current_pose_.pose, cluster_marker_pub_, msg->header.stamp);

        // Publish white cloud
        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

        // Publish the accumulated white cloud with persistent QoS
        sensor_msgs::msg::PointCloud2 persistent_msg;
        pcl::toROSMsg(*accumulated_white_cloud_, persistent_msg);
        persistent_msg.header.frame_id = "map";  // Or "map" if you transform it
        persistent_msg.header.stamp = this->now();
        persistent_white_publisher_->publish(persistent_msg);

    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
    }
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
    auto node = std::make_shared<GlobalLocalization>(rclcpp::NodeOptions().use_intra_process_comms(false).append_parameter_override("use_sim_time", true));
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}