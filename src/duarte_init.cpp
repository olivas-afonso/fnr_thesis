#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"

class DistanceOrientationSubscriber : public rclcpp::Node
{
public:
    DistanceOrientationSubscriber()
        : Node("distance_orientation_subscriber")
    {
        subscription_ = this->create_subscription<visualization_msgs::msg::Marker>(
            "distance_orientation_marker", 10,
            std::bind(&DistanceOrientationSubscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const visualization_msgs::msg::Marker::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received Text: %s", msg->text.c_str());

        // Extract distance and orientation from text (assuming fixed format)
        std::istringstream iss(msg->text);
        std::string temp;
        float distance, orientation;
        
        iss >> temp >> distance >> temp >> orientation; // Parse "Dist: <value>m Orient: <value> rad"
        
        RCLCPP_INFO(this->get_logger(), "Distance: %.2f m, Orientation: %.2f rad", distance, orientation);
    }

    rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DistanceOrientationSubscriber>());
    rclcpp::shutdown();
    return 0;
}
