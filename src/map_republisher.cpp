#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

class MapRepublisher : public rclcpp::Node {
    public:
        MapRepublisher() : Node("map_republisher") {
            // Hardcoded transformation parameters
            this->declare_parameter<double>("scale", 0.5);
            this->declare_parameter<double>("rotation_deg", -90.0);
            this->declare_parameter<double>("translation_x", -1.8);
            this->declare_parameter<double>("translation_y", 5.12);
    
            map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
                "/persistent_map", 10,
                [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                    if (!map_received_) {
                        original_map_ = *msg;
                        map_received_ = true;
                        timer_ = create_wall_timer(
                            std::chrono::seconds(1),
                            [this]() { republishMap(); });
                    }
                });
    
            map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
                "/map", rclcpp::QoS(1).transient_local());
        }
    
    private:
        void republishMap() {
            auto message = original_map_;
            
            // Apply hardcoded transform
            double scale = this->get_parameter("scale").as_double();
            double rotation = this->get_parameter("rotation_deg").as_double() * M_PI / 180.0;
            double tx = this->get_parameter("translation_x").as_double();
            double ty = this->get_parameter("translation_y").as_double();
    
            // Apply transformations
            message.info.resolution *= scale;
            message.info.origin.position.x += tx;
            message.info.origin.position.y += ty;
            
            tf2::Quaternion q;
            q.setRPY(0, 0, rotation);
            message.info.origin.orientation = tf2::toMsg(q);
    
            message.header.stamp = now();
            message.header.frame_id = "map";
            
            map_pub_->publish(message);
        }
    

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    nav_msgs::msg::OccupancyGrid original_map_;
    bool map_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MapRepublisher>());
    rclcpp::shutdown();
    return 0;
}