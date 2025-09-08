#include <rclcpp/rclcpp.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std::chrono_literals;

class ZedObjectVisualizer : public rclcpp::Node
{
public:
    ZedObjectVisualizer() : Node("zed_object_visualizer")
    {
        // Initialize label map and color map
        initialize_label_map();
        initialize_color_map();
        
        // Subscribe to object detection topic
        subscription_ = this->create_subscription<zed_msgs::msg::ObjectsStamped>(
            "/zed/zed_node/obj_det/objects",
            10,
            std::bind(&ZedObjectVisualizer::objects_callback, this, std::placeholders::_1));
        
        // Subscribe to camera image topic
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/left/image_rect_color",
            10,
            std::bind(&ZedObjectVisualizer::image_callback, this, std::placeholders::_1));
        
        // Publisher for visualization markers
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/zed/visualization/objects",
            10);
        
        // Create a timer to check for keyboard input
        timer_ = this->create_wall_timer(
            100ms, std::bind(&ZedObjectVisualizer::check_keyboard_input, this));
        
        // Setup non-blocking keyboard input
        setup_non_blocking_io();
        
        RCLCPP_INFO(this->get_logger(), "ZED Object Visualizer started");
        RCLCPP_INFO(this->get_logger(), "Press SPACE to save images of detected objects");
    }

    ~ZedObjectVisualizer() {
        // Restore terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &original_terminal_settings_);
    }

private:
    void setup_non_blocking_io() {
        // Save original terminal settings
        tcgetattr(STDIN_FILENO, &original_terminal_settings_);
        
        // Set terminal to non-blocking mode
        struct termios new_settings = original_terminal_settings_;
        new_settings.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_settings);
        
        // Set stdin to non-blocking
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }
    
    void initialize_label_map()
    {
        // Map sublabel numbers to human-readable class names
        label_map_ = {
            {"0", "animal"},
            {"1", "bump"},
            {"2", "bus"},
            {"3", "crosswalk"},
            {"4", "hospital"},
            {"5", "lanes"},
            {"6", "left"},
            {"7", "lights"},
            {"8", "parking"},
            {"9", "roundabout"},
            {"10", "speed"},
            {"11", "warning"}
        };
    }
    
    void initialize_color_map()
    {
        // Define colors for each object type
        color_map_ = {
            {"animal",     {0.0, 0.5, 0.0, 0.7}},    // Dark Green
            {"bump",       {1.0, 0.5, 0.0, 0.7}},    // Orange
            {"bus",        {0.0, 0.0, 1.0, 0.7}},    // Blue
            {"crosswalk",  {0.5, 0.0, 0.5, 0.7}},    // Purple
            {"hospital",   {1.0, 0.0, 0.0, 0.7}},    // Red
            {"lanes",      {0.0, 1.0, 1.0, 0.7}},    // Cyan
            {"left",       {1.0, 1.0, 0.0, 0.7}},    // Yellow
            {"lights",     {0.5, 0.5, 0.0, 0.7}},    // Olive
            {"parking",    {0.0, 0.5, 0.5, 0.7}},    // Teal
            {"roundabout", {0.5, 0.0, 0.0, 0.7}},    // Maroon
            {"speed",      {0.0, 0.0, 0.5, 0.7}},    // Navy
            {"warning",    {1.0, 0.0, 1.0, 0.7}}     // Magenta
        };
    }
    
    std::string get_label_name(const zed_msgs::msg::Object& obj)
    {
        // If label is not empty, use it directly
        if (!obj.label.empty() && obj.label != "") {
            return obj.label;
        }
        
        // Otherwise, use sublabel to look up the class name
        auto it = label_map_.find(obj.sublabel);
        if (it != label_map_.end()) {
            return it->second;
        }
        
        // Fallback: use sublabel as is
        return "class_" + obj.sublabel;
    }
    
    std_msgs::msg::ColorRGBA get_color_for_label(const std::string& label)
    {
        std_msgs::msg::ColorRGBA color;
        
        // Look up color in the map, use gray if not found
        auto it = color_map_.find(label);
        if (it != color_map_.end()) {
            color.r = it->second[0];
            color.g = it->second[1];
            color.b = it->second[2];
            color.a = it->second[3];
        } else {
            // Default gray for unknown labels
            color.r = 0.5;
            color.g = 0.5;
            color.b = 0.5;
            color.a = 0.7;
        }
        
        return color;
    }

    void objects_callback(const zed_msgs::msg::ObjectsStamped::SharedPtr msg)
    {
        // Store the latest objects for potential image capture
        latest_objects_ = msg;
        
        auto marker_array = std::make_shared<visualization_msgs::msg::MarkerArray>();
        
        // Clear previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header = msg->header;
        clear_marker.ns = "zed_objects";
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array->markers.push_back(clear_marker);
        
        int marker_id = 0;
        int object_count = 0;
        
        // Debug output for detected objects
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                           "Received %zu objects", msg->objects.size());
        
        for (const auto& obj : msg->objects)
        {
            // Skip objects with low confidence or invalid positions
            if (obj.confidence < 30.0 || 
                std::isnan(obj.position[0]) || 
                std::isnan(obj.position[1]) || 
                std::isnan(obj.position[2]))
            {
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                    "Skipping object with confidence: %.2f", obj.confidence);
                continue;
            }
            
            // Get the proper label name
            std::string label_name = get_label_name(obj);
            
            // Debug output for each valid object
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "Object %d: %s (sublabel: %s) - Confidence: %.2f%% at [%.2f, %.2f, %.2f]",
                               object_count, label_name.c_str(), obj.sublabel.c_str(),
                               obj.confidence, obj.position[0], obj.position[1], obj.position[2]);
            
            // Create bounding box marker with unique ID per object type
            create_bounding_box_marker(marker_array, obj, msg->header, marker_id, label_name);
            marker_id++;
            
            // Create text label marker with unique ID
            create_label_marker(marker_array, obj, msg->header, marker_id, label_name);
            marker_id++;
            
            // Create confidence marker with unique ID
            create_confidence_marker(marker_array, obj, msg->header, marker_id);
            marker_id++;
            
            object_count++;
        }
        
        // Publish markers
        marker_pub_->publish(*marker_array);
        
        // Summary debug output
        if (object_count > 0) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "Published %zu markers for %d objects", 
                               marker_array->markers.size(), object_count);
        }
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Store the latest image for potential capture
        latest_image_ = msg;
    }
    
    void check_keyboard_input()
    {
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1) {
            if (c == ' ') {
                RCLCPP_INFO(this->get_logger(), "Space key pressed - saving object images");
                
                // Save images if we have both objects and an image
                if (latest_objects_ && latest_image_) {
                    save_object_images();
                } else {
                    if (!latest_objects_) {
                        RCLCPP_WARN(this->get_logger(), "No objects available to save");
                    }
                    if (!latest_image_) {
                        RCLCPP_WARN(this->get_logger(), "No image available to save");
                    }
                }
            }
        }
    }
    
    void save_object_images()
    {
        try {

            
            // Check if we have a valid image
            if (!latest_image_) {
                RCLCPP_WARN(this->get_logger(), "No image available to process");
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "Processing image with encoding: %s", latest_image_->encoding.c_str());
            
            // Convert ROS image to OpenCV format
            cv_bridge::CvImagePtr cv_ptr;
            try {
                // Check the image encoding and convert appropriately
                if (latest_image_->encoding == "bgra8") {
                    cv_ptr = cv_bridge::toCvCopy(latest_image_, sensor_msgs::image_encodings::BGRA8);
                } else if (latest_image_->encoding == "rgb8") {
                    cv_ptr = cv_bridge::toCvCopy(latest_image_, sensor_msgs::image_encodings::RGB8);
                } else if (latest_image_->encoding == "bgr8") {
                    cv_ptr = cv_bridge::toCvCopy(latest_image_, sensor_msgs::image_encodings::BGR8);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Unsupported image encoding: %s", latest_image_->encoding.c_str());
                    return;
                }
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            
            cv::Mat image;
            if (cv_ptr->encoding == "bgra8") {
                // Convert BGRA to BGR by removing alpha channel
                cv::cvtColor(cv_ptr->image, image, cv::COLOR_BGRA2BGR);
            } else {
                image = cv_ptr->image;
            }
            RCLCPP_INFO(this->get_logger(), "Image converted successfully. Dimensions: %d x %d", image.cols, image.rows);
            
            // Check if we have objects
            if (!latest_objects_) {
                RCLCPP_WARN(this->get_logger(), "No objects available to process");
                return;
            }
            
            int saved_count = 0;
            
            for (const auto& obj : latest_objects_->objects)
            {
                // Skip objects with low confidence or invalid positions
                if (obj.confidence < 30.0 || 
                    std::isnan(obj.position[0]) || 
                    std::isnan(obj.position[1]) || 
                    std::isnan(obj.position[2]))
                {
                    continue;
                }
                
                // Get the proper label name
                std::string label_name = get_label_name(obj);
                
                // Check if 2D bounding box is available
                if (obj.bounding_box_2d.corners.size() >= 4) {
                    // Get the 2D bounding box coordinates
                    float min_x = std::numeric_limits<float>::max();
                    float min_y = std::numeric_limits<float>::max();
                    float max_x = std::numeric_limits<float>::lowest();
                    float max_y = std::numeric_limits<float>::lowest();

                    // Scale factor: coordinates are for 1280x720, image is 640x360
                    const float scale_x = 2.0f;
                    const float scale_y = 2.0f;

                    for (const auto& corner : obj.bounding_box_2d.corners) {
                        // Scale down the coordinates to match the image resolution
                        float x = static_cast<float>(corner.kp[0]) / scale_x;
                        float y = static_cast<float>(corner.kp[1]) / scale_y;
                        
                        min_x = std::min(min_x, x);
                        min_y = std::min(min_y, y);
                        max_x = std::max(max_x, x);
                        max_y = std::max(max_y, y);
                    }

                    RCLCPP_INFO(this->get_logger(), "Scaled bounds: min_x=%.1f, min_y=%.1f, max_x=%.1f, max_y=%.1f",
                                min_x, min_y, max_x, max_y);
                    // Convert to integer coordinates
                    int x1 = static_cast<int>(min_x);
                    int y1 = static_cast<int>(min_y);
                    int x2 = static_cast<int>(max_x);
                    int y2 = static_cast<int>(max_y);
                    
                    RCLCPP_INFO(this->get_logger(), "Integer bounds: x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2);
                    RCLCPP_INFO(this->get_logger(), "Image dimensions: width=%d, height=%d", image.cols, image.rows);
                    
                    // Ensure coordinates are within image bounds
                    x1 = std::max(0, x1);
                    y1 = std::max(0, y1);
                    x2 = std::min(image.cols - 1, x2);
                    y2 = std::min(image.rows - 1, y2);
                    
                    RCLCPP_INFO(this->get_logger(), "After clamping: x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2);
                    
                    // Check if the bounding box is valid
                    if (x2 > x1 && y2 > y1) {
                        int width = x2 - x1;
                        int height = y2 - y1;
                        
                        RCLCPP_INFO(this->get_logger(), "ROI dimensions: width=%d, height=%d", width, height);
                        
                        // Extract the region of interest
                        cv::Mat roi = image(cv::Rect(x1, y1, width, height));
                        
                        // Generate filename with timestamp and object info
                        int count = label_counters_[label_name]++;
                        std::stringstream ss;
                        ss << label_name << "_" << count << ".jpg";
                        
                        // Save the image
                        cv::imwrite(ss.str(), roi);
                        RCLCPP_INFO(this->get_logger(), "Saved image: %s", ss.str().c_str());
                        
                        saved_count++;
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Invalid bounding box for object: %s (x1=%d, y1=%d, x2=%d, y2=%d)", 
                                label_name.c_str(), x1, y1, x2, y2);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "No 2D bounding box available for object: %s", label_name.c_str());
                }
            }
            
            if (saved_count > 0) {
                RCLCPP_INFO(this->get_logger(), "Saved %d object images", saved_count);
            } else {
                RCLCPP_WARN(this->get_logger(), "No valid bounding boxes found to save");
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error saving images: %s", e.what());
        }
    }
    
    void create_bounding_box_marker(
        visualization_msgs::msg::MarkerArray::SharedPtr marker_array,
        const zed_msgs::msg::Object& obj,
        const std_msgs::msg::Header& header,
        int marker_id,
        const std::string& label_name)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "zed_objects_bbox";
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Set position (center of the object)
        marker.pose.position.x = obj.position[0];
        marker.pose.position.y = obj.position[1];
        marker.pose.position.z = obj.position[2];
        marker.pose.orientation.w = 1.0;
        
        // Set scale (dimensions of the object)
        if (!obj.dimensions_3d.empty() && obj.dimensions_3d.size() >= 3)
        {
            marker.scale.x = obj.dimensions_3d[0];
            marker.scale.y = obj.dimensions_3d[1];
            marker.scale.z = obj.dimensions_3d[2];
        }
        else
        {
            // Default size if dimensions not available
            marker.scale.x = 0.3;
            marker.scale.y = 1.0;
            marker.scale.z = 0.3;
        }
        
        // Set color based on object type using the color map
        marker.color = get_color_for_label(label_name);
        
        marker.lifetime = rclcpp::Duration(200ms); // Slightly longer lifetime
        
        marker_array->markers.push_back(marker);
    }
    
    void create_label_marker(
        visualization_msgs::msg::MarkerArray::SharedPtr marker_array,
        const zed_msgs::msg::Object& obj,
        const std_msgs::msg::Header& header,
        int marker_id,
        const std::string& label_name)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "zed_labels_text";
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Position text slightly above the object
        marker.pose.position.x = obj.position[0];
        marker.pose.position.y = obj.position[1];
        marker.pose.position.z = obj.position[2] + 0.5;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.z = 0.2;  // Text height
        marker.text = label_name + " (" + std::to_string(static_cast<int>(obj.confidence)) + "%)";
        
        // White text
        std_msgs::msg::ColorRGBA color;
        color.r = 1.0;
        color.g = 1.0;
        color.b = 1.0;
        color.a = 1.0;
        marker.color = color;
        
        marker.lifetime = rclcpp::Duration(200ms);
        
        marker_array->markers.push_back(marker);
    }
    
    void create_confidence_marker(
        visualization_msgs::msg::MarkerArray::SharedPtr marker_array,
        const zed_msgs::msg::Object& obj,
        const std_msgs::msg::Header& header,
        int marker_id)
    {
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "zed_confidence_bars";
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Position confidence bar below the object
        marker.pose.position.x = obj.position[0];
        marker.pose.position.y = obj.position[1];
        marker.pose.position.z = obj.position[2] - 0.3;
        marker.pose.orientation.w = 1.0;
        
        // Scale based on confidence (0-100% mapped to 0-0.3m height)
        marker.scale.x = 0.05;
        marker.scale.y = 0.05;
        marker.scale.z = obj.confidence / 100.0 * 0.3;
        
        // Color gradient from red (low) to green (high)
        std_msgs::msg::ColorRGBA color;
        color.a = 0.8;
        if (obj.confidence < 50.0) {
            color.r = 1.0;
            color.g = obj.confidence / 50.0;
            color.b = 0.0;
        } else {
            color.r = (100.0 - obj.confidence) / 50.0;
            color.g = 1.0;
            color.b = 0.0;
        }
        marker.color = color;
        
        marker.lifetime = rclcpp::Duration(200ms);
        
        marker_array->markers.push_back(marker);
    }
    
    rclcpp::Subscription<zed_msgs::msg::ObjectsStamped>::SharedPtr subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unordered_map<std::string, int> label_counters_;
    
    std::unordered_map<std::string, std::string> label_map_;
    std::unordered_map<std::string, std::vector<double>> color_map_;
    
    zed_msgs::msg::ObjectsStamped::SharedPtr latest_objects_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    
    struct termios original_terminal_settings_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ZedObjectVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}