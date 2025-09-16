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

    //accumulated_white_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    //persistent_white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/persistent_white_cloud", rclcpp::SensorDataQoS().reliable().transient_local());

    
    white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);


    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
        "/cluster_scan", 
        rclcpp::QoS(10).reliable()  // Match AMCL's default
    );

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

    

    this->declare_parameter("fit_side", true);
    this->get_parameter("fit_side", fit_side_);

    this->declare_parameter("time_jump_threshold", 5.0); // Increased from 1.0 to 5.0
    this->get_parameter("time_jump_threshold", time_jump_threshold_);

    this->declare_parameter("min_scan_range", 0.1);
    this->declare_parameter("max_scan_range", 10.0);

    this->declare_parameter("save_path", false);
    this->declare_parameter("path_filename", "car_path.csv");
    this->declare_parameter("save_interval", 0.1);  // seconds between saves
    
    this->get_parameter("save_path", save_path_);
    this->get_parameter("path_filename", path_filename_);
    this->get_parameter("save_interval", save_interval_);

    last_save_time_ = this->now();
        // In constructor:
    last_speed_time_ = this->now();
    last_servo_time_ = this->now();

    current_control_[0] = 0.0f; // Default to stopped
    last_speed_value_ = 0.0f;
}

void GlobalLocalization::saveCarPath(const geometry_msgs::msg::PoseStamped& pose)
{
    if (!save_path_) {
        return;
    }
    
    // Check if pose has changed significantly
    bool pose_changed = false;
    if (last_saved_pose_.header.stamp.sec == 0) { // First pose
        pose_changed = true;
    } else {
        double dx = pose.pose.position.x - last_saved_pose_.pose.position.x;
        double dy = pose.pose.position.y - last_saved_pose_.pose.position.y;
        double dz = pose.pose.position.z - last_saved_pose_.pose.position.z;
        double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > 0.01) { // Save if moved more than 1cm
            pose_changed = true;
        }
    }
    
    if (pose_changed) {
        std::ofstream file(path_filename_, std::ios_base::app);
        
        if (file.is_open()) {
            auto current_time = this->now();
            auto nanoseconds = current_time.nanoseconds();
            auto seconds = nanoseconds / 1000000000;
            auto fractional = nanoseconds % 1000000000;
            
            file << seconds << "." << std::setw(9) << std::setfill('0') << fractional << ","
                 << pose.pose.position.x << ","
                 << pose.pose.position.y << ","
                 << pose.pose.position.z << ","
                 << pose.pose.orientation.x << ","
                 << pose.pose.orientation.y << ","
                 << pose.pose.orientation.z << ","
                 << pose.pose.orientation.w << "\n";
            
            file.close();
            last_save_time_ = current_time;
            last_saved_pose_ = pose;
            
            RCLCPP_DEBUG(this->get_logger(), "Saved pose to %s", path_filename_.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to open file %s for writing", path_filename_.c_str());
        }
    }
}

// Add this method to initialize the path file
void GlobalLocalization::initializePathFile()
{
    if (save_path_) {
        std::ofstream file(path_filename_);
        if (file.is_open()) {
            file << "timestamp,x,y,z,qx,qy,qz,qw\n";
            file.close();
            RCLCPP_INFO(this->get_logger(), "Initialized path file: %s", path_filename_.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to initialize path file: %s", path_filename_.c_str());
        }
    }
}


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
    scan.header.frame_id = "base_link";  // Standard frame for laser scans

    // Get scan parameters
    double min_range, max_range;
    this->get_parameter("min_scan_range", min_range);
    this->get_parameter("max_scan_range", max_range);

    // LaserScan configuration
    scan.angle_min = -M_PI * 40.0 / 180.0;  // -40º
    scan.angle_max =  M_PI * 40.0 / 180.0;  // +40º
    scan.angle_increment = M_PI / 180.0;  // 1 degree resolution
    scan.scan_time = 1.0 / 30.0;          // Assuming 30Hz data rate
    scan.range_min = static_cast<float>(min_range);
    scan.range_max = static_cast<float>(max_range);
    scan.time_increment = scan.scan_time / (2 * M_PI / scan.angle_increment);

    size_t num_ranges = static_cast<size_t>((scan.angle_max - scan.angle_min) / scan.angle_increment);
    scan.ranges.assign(num_ranges, std::numeric_limits<float>::infinity());  // Initialize with "no detection"

    // Get transform from cloud frame to base_link
    geometry_msgs::msg::TransformStamped tf_cloud_to_base;
    try {
        tf_cloud_to_base = tf_buffer_->lookupTransform(
            "base_link", 
            cloud_frame_id, 
            cloud_time,
            tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(),
            "TF lookup failed from '%s' to 'base_link': %s",
            cloud_frame_id.c_str(), ex.what());
        return scan;
    }

    // Convert transform to tf2::Transform for efficient computation
    tf2::Transform tf;
    tf2::fromMsg(tf_cloud_to_base.transform, tf);

    // Transform points and populate scan
    for (const auto &point : cloud->points) {
        // Skip invalid points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            continue;
        }

        // Transform point to base_link frame
        tf2::Vector3 pt_cloud(point.x, point.y, point.z);
        tf2::Vector3 pt_base = tf * pt_cloud;

        // Calculate range and angle in base_link frame
        float x = pt_base.x();
        float y = pt_base.y();
        float range = std::hypot(x, y);
        float angle = std::atan2(y, x);

        // Check if within valid range
        if (range < scan.range_min || range > scan.range_max) {
            continue;
        }

        // Find appropriate bin
        int bin = static_cast<int>((angle - scan.angle_min) / scan.angle_increment);
        if (bin < 0 || bin >= static_cast<int>(num_ranges)) {
            continue;
        }

        // Keep shortest range in each bin
        if (range < scan.ranges[bin]) {
            scan.ranges[bin] = range;
        }
    }

    RCLCPP_DEBUG(this->get_logger(), "Published scan with %zu ranges in frame %s", 
                scan.ranges.size(), scan.header.frame_id.c_str());
    return scan;
}

void GlobalLocalization::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{


    current_pose_ = *msg;
    saveCarPath(*msg);
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
        // Convert to PCL point cloud
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


        // Publish white cloud
        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

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
    // Initialize path file
    node->initializePathFile();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}