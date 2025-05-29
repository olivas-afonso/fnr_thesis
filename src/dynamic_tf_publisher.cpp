#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"

class DynamicTfPublisher : public rclcpp::Node {
public:
    DynamicTfPublisher() : Node("dynamic_tf_publisher") {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        use_sim_time_ = this->get_parameter("use_sim_time").as_bool();
        
        // Parameters for initial map->odom transform
        this->declare_parameter<double>("initial_map_x", 0.0);
        this->declare_parameter<double>("initial_map_y", 0.0);
        this->declare_parameter<double>("initial_map_z", 0.0);
        this->declare_parameter<double>("initial_map_yaw", 0.0);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // 100Hz
            std::bind(&DynamicTfPublisher::publish_transform, this));
    }

private:
    void publish_transform() {
        auto stamp = this->now();
        
        geometry_msgs::msg::TransformStamped map_to_odom;
        map_to_odom.header.stamp = stamp;
        map_to_odom.header.frame_id = "map";
        map_to_odom.child_frame_id = "odom";
        
        // Get initial transform parameters
        double x = this->get_parameter("initial_map_x").as_double();
        double y = this->get_parameter("initial_map_y").as_double();
        double z = this->get_parameter("initial_map_z").as_double();
        double yaw = this->get_parameter("initial_map_yaw").as_double();
        
        // Set translation
        map_to_odom.transform.translation.x = x;
        map_to_odom.transform.translation.y = y;
        map_to_odom.transform.translation.z = z;
        
        // Set rotation (yaw only for 2D)
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        map_to_odom.transform.rotation.x = q.x();
        map_to_odom.transform.rotation.y = q.y();
        map_to_odom.transform.rotation.z = q.z();
        map_to_odom.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(map_to_odom);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    bool use_sim_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicTfPublisher>());
    rclcpp::shutdown();
    return 0;
}