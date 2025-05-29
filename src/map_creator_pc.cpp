#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <rcpputils/filesystem_helper.hpp>
#include <ctime>
#include <iomanip>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.h>

class MapCreator : public rclcpp::Node
{
public:
    MapCreator() : Node("map_creator"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Parameters
        this->declare_parameter("map_resolution", 0.05);
        this->declare_parameter("map_width", 40.0);
        this->declare_parameter("map_height", 40.0);
        this->declare_parameter("map_frame", "map");
        this->declare_parameter("output_dir", "/home/olivas/maps");
        
        resolution_ = this->get_parameter("map_resolution").as_double();
        width_ = this->get_parameter("map_width").as_double();
        height_ = this->get_parameter("map_height").as_double();
        map_frame_ = this->get_parameter("map_frame").as_string();
        output_dir_ = this->get_parameter("output_dir").as_string();
        
        // Subscriber to white point cloud
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/white_only", 10, std::bind(&MapCreator::cloudCallback, this, std::placeholders::_1));
            
        // Publisher for visualization
        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/custom_map", 10);
        
        // Timer to save map periodically
        save_timer_ = this->create_wall_timer(
            std::chrono::seconds(10),  // Save every 10 seconds
            std::bind(&MapCreator::saveMap, this));
            
        RCLCPP_INFO(this->get_logger(), "Map creator node initialized");
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);
        
        // Downsample the cloud
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(resolution_, resolution_, resolution_);
        voxel_grid.filter(*cloud);
        
        // Transform to map frame if needed
        if (msg->header.frame_id != map_frame_) {
            try {
                auto transform = tf_buffer_.lookupTransform(
                    map_frame_, msg->header.frame_id, 
                    msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
                
                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                for (const auto& point : cloud->points) {
                    geometry_msgs::msg::PointStamped pt_in, pt_out;
                    pt_in.header = msg->header;
                    pt_in.point.x = point.x;
                    pt_in.point.y = point.y;
                    pt_in.point.z = point.z;
                    
                    tf_buffer_.transform(pt_in, pt_out, map_frame_);
                    
                    pcl::PointXYZ transformed_point;
                    transformed_point.x = pt_out.point.x;
                    transformed_point.y = pt_out.point.y;
                    transformed_point.z = pt_out.point.z;
                    transformed_cloud->push_back(transformed_point);
                }
                cloud = transformed_cloud;
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
                return;
            }
        }
        
        // Update the map with new points
        updateMap(cloud);
    }
    
    void updateMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        // Initialize map if not done yet
        if (map_.info.width == 0) {
            initializeMap();
        }
        
        // Mark occupied cells
        for (const auto& point : cloud->points) {
            int mx = static_cast<int>((point.x - map_.info.origin.position.x) / resolution_);
            int my = static_cast<int>((point.y - map_.info.origin.position.y) / resolution_);
            
            if (mx >= 0 && mx < static_cast<int>(map_.info.width) && 
                my >= 0 && my < static_cast<int>(map_.info.height)) {
                int index = my * map_.info.width + mx;
                map_.data[index] = 100;  // 100 means occupied
            }
        }
        
        // Publish for visualization
        map_.header.stamp = this->now();
        map_pub_->publish(map_);
    }
    
    void initializeMap()
    {
        map_.header.frame_id = map_frame_;
        map_.info.resolution = resolution_;
        map_.info.width = static_cast<unsigned int>(width_ / resolution_);
        map_.info.height = static_cast<unsigned int>(height_ / resolution_);
        
        // Set origin to center the map
        map_.info.origin.position.x = -width_ / 2.0;
        map_.info.origin.position.y = -height_ / 2.0;
        map_.info.origin.position.z = 0.0;
        map_.info.origin.orientation.w = 1.0;
        
        // Initialize all cells as unknown (-1)
        map_.data.resize(map_.info.width * map_.info.height, -1);
    }
    
    void saveMap()
    {
        if (map_.info.width == 0) {
            RCLCPP_WARN(this->get_logger(), "Map not initialized yet, not saving");
            return;
        }
        
        // Create output directory if it doesn't exist
        if (!rcpputils::fs::exists(output_dir_)) {
            rcpputils::fs::create_directories(output_dir_);
        }
        
        // Generate filename with timestamp
        auto now = this->now();
        std::time_t time_t = static_cast<std::time_t>(now.seconds());
        std::tm tm = *std::localtime(&time_t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string filename_base = "map_" + oss.str();
        std::string pgm_path = output_dir_ + "/" + filename_base + ".pgm";
        std::string yaml_path = output_dir_ + "/" + filename_base + ".yaml";
        
        // Save PGM file
        std::ofstream pgm_file(pgm_path, std::ios::binary);
        if (!pgm_file) {
            RCLCPP_ERROR(this->get_logger(), "Could not open file %s for writing", pgm_path.c_str());
            return;
        }
        
        pgm_file << "P5\n";
        pgm_file << map_.info.width << " " << map_.info.height << "\n";
        pgm_file << "255\n";
        
        for (int y = static_cast<int>(map_.info.height) - 1; y >= 0; --y) {
            for (unsigned int x = 0; x < map_.info.width; ++x) {
                int index = y * map_.info.width + x;
                unsigned char value;
                
                if (map_.data[index] == 100) {  // Occupied
                    value = 0;
                } else if (map_.data[index] == 0) {  // Free
                    value = 254;
                } else {  // Unknown
                    value = 205;
                }
                
                pgm_file.write(reinterpret_cast<char*>(&value), 1);
            }
        }
        
        pgm_file.close();
        
        std::ofstream yaml_file(yaml_path);
        if (!yaml_file) {
            RCLCPP_ERROR(this->get_logger(), "Could not open file %s for writing", yaml_path.c_str());
            return;
        }

        yaml_file << "image: " << filename_base + ".pgm" << "\n";
        yaml_file << "resolution: " << resolution_ << "\n";
        yaml_file << "origin: [" 
                << map_.info.origin.position.x << ", " 
                << map_.info.origin.position.y << ", " 
                << "0.0]\n";
        yaml_file << "negate: 0\n";
        yaml_file << "occupied_thresh: 0.65\n";
        yaml_file << "free_thresh: 0.196\n";

        yaml_file.close();
        
        RCLCPP_INFO(this->get_logger(), "Map saved to %s and %s", pgm_path.c_str(), yaml_path.c_str());
    }
    
    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::TimerBase::SharedPtr save_timer_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    nav_msgs::msg::OccupancyGrid map_;
    double resolution_;
    double width_;
    double height_;
    std::string map_frame_;
    std::string output_dir_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapCreator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}