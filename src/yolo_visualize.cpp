#include <rclcpp/rclcpp.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

using namespace std::chrono_literals;

class MobileNetVisualizationNode : public rclcpp::Node
{
public:
    MobileNetVisualizationNode() : Node("mobilenet_visualization_node")
    {
        // Initialize MobileNet model
        initialize_mobilenet_model();
        
        // Subscribe to object detection topic
        objects_subscription_ = this->create_subscription<zed_msgs::msg::ObjectsStamped>(
            "/zed/zed_node/obj_det/objects",
            10,
            std::bind(&MobileNetVisualizationNode::objects_callback, this, std::placeholders::_1));
        
        // Subscribe to camera image topic
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/left/image_rect_color",
            10,
            std::bind(&MobileNetVisualizationNode::image_callback, this, std::placeholders::_1));
        
        // Create image publisher for visualization
        image_publisher_ = image_transport::create_publisher(this, "/mobilenet/visualization");
        
        RCLCPP_INFO(this->get_logger(), "MobileNet Visualization Node started");
    }

private:
    void initialize_mobilenet_model()
    {
        try {
            // Load the trained MobileNet model
            std::string model_path = "/home/jetson/test_ws/src/fnr_thesis/weights/traffic_sign_classifier_great.onnx";
            std::string classes_path = "/home/jetson/test_ws/src/fnr_thesis/weights/classes.txt";
            
            // Load the model
            net_ = cv::dnn::readNet(model_path);
            
            if (net_.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load MobileNet model from: %s", model_path.c_str());
                return;
            }
            
            // Load class names
            load_class_names(classes_path);
            
            // Set backend preferences
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            RCLCPP_INFO(this->get_logger(), "MobileNet model loaded successfully");
            RCLCPP_INFO(this->get_logger(), "Number of classes: %zu", class_names_.size());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading MobileNet model: %s", e.what());
        }
    }
    
    void load_class_names(const std::string& classes_path)
    {
        std::ifstream file(classes_path);
        if (!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open classes file: %s", classes_path.c_str());
            return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                class_names_.push_back(line);
            }
        }
        file.close();
    }
    
    std::pair<std::string, float> run_inference(const cv::Mat& image)
    {
        if (net_.empty() || class_names_.empty()) {
            return {"Model not loaded", 0.0f};
        }
        
        try {
            if (image.empty() || image.cols <= 0 || image.rows <= 0) {
                RCLCPP_ERROR(this->get_logger(), "Invalid input image for inference: %dx%d", image.cols, image.rows);
                return {"Invalid image", 0.0f};
            }
            
            // Resize to match training size (64x64)
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(64, 64));
            
            // Preprocess image
            cv::Mat blob;
            cv::dnn::blobFromImage(resized_image, blob, 1.0/255.0,
                                cv::Size(64, 64),
                                cv::Scalar(0, 0, 0), true, false);
            
            blob = 2.0 * blob - 1.0;
            
            // Set input and run inference
            net_.setInput(blob);
            cv::Mat output = net_.forward();
            
            // Apply softmax
            cv::Mat exp_output;
            cv::exp(output, exp_output);
            cv::Mat probabilities = exp_output / cv::sum(exp_output)[0];
            
            // Get the predicted class and confidence
            cv::Point class_id_point;
            double confidence;
            cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &class_id_point);
            
            int class_id = class_id_point.x;
            
            if (class_id < static_cast<int>(class_names_.size())) {
                return {class_names_[class_id], static_cast<float>(confidence)};
            } else {
                return {"Unknown class", static_cast<float>(confidence)};
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
            return {"Inference failed", 0.0f};
        }
    }
    
    void objects_callback(const zed_msgs::msg::ObjectsStamped::SharedPtr msg)
    {
        latest_objects_ = msg;
        process_and_visualize();
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        latest_image_ = msg;
    }
    
    void process_and_visualize()
    {
        if (!latest_objects_ || !latest_image_) {
            return;
        }
        
        try {
            // Convert ROS image to OpenCV format
            cv_bridge::CvImagePtr cv_ptr;
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
            
            cv::Mat visualization_image;
            if (cv_ptr->encoding == "bgra8") {
                cv::cvtColor(cv_ptr->image, visualization_image, cv::COLOR_BGRA2BGR);
            } else {
                visualization_image = cv_ptr->image.clone();
            }
            
            int processed_count = 0;
            
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
                
                // Get YOLO label for reference
                std::string yolo_label = get_label_name(obj);
                
                // Extract the cropped image from YOLO bounding box
                if (obj.bounding_box_2d.corners.size() >= 4) {
                    // Calculate bounding box coordinates
                    float min_x = std::numeric_limits<float>::max();
                    float min_y = std::numeric_limits<float>::max();
                    float max_x = std::numeric_limits<float>::lowest();
                    float max_y = std::numeric_limits<float>::lowest();

                    // Scale factor: coordinates are for 1280x720, image is 640x360
                    const float scale_x = 2.0f;
                    const float scale_y = 2.0f;

                    for (const auto& corner : obj.bounding_box_2d.corners) {
                        float x = static_cast<float>(corner.kp[0]) / scale_x;
                        float y = static_cast<float>(corner.kp[1]) / scale_y;
                        
                        min_x = std::min(min_x, x);
                        min_y = std::min(min_y, y);
                        max_x = std::max(max_x, x);
                        max_y = std::max(max_y, y);
                    }

                    // Convert to integer coordinates and clamp to image bounds
                    int x1 = std::max(0, static_cast<int>(min_x));
                    int y1 = std::max(0, static_cast<int>(min_y));
                    int x2 = std::min(visualization_image.cols - 1, static_cast<int>(max_x));
                    int y2 = std::min(visualization_image.rows - 1, static_cast<int>(max_y));
                    
                    // Check if the bounding box is valid
                    if (x2 > x1 && y2 > y1) {
                        // Extract the region of interest for inference
                        cv::Mat roi = visualization_image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                        
                        if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) {
                            continue;
                        }
                        
                        // Run MobileNet inference on the cropped image
                        auto [mobilenet_class, confidence] = run_inference(roi);
                        
                        // Draw YOLO bounding box
                        cv::Scalar box_color = cv::Scalar(0, 255, 0); // Green for YOLO boxes
                        cv::rectangle(visualization_image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);
                        
                        // Prepare text for display
                        //std::string yolo_text = "YOLO: " + yolo_label + " (" + std::to_string(static_cast<int>(obj.confidence)) + "%)";
                        std::string mobilenet_text = mobilenet_class;
                        
                        // Choose text position (above or below the box depending on position)
                        int text_y = y1 - 10;
                        if (text_y < 20) {
                            text_y = y2 + 20;
                        }
                        
                        // Draw YOLO label
                        //cv::putText(visualization_image, yolo_text, cv::Point(x1, text_y),
                         //          cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
                        
                        // Draw MobileNet classification result
                        cv::putText(visualization_image, mobilenet_text, cv::Point(x1 + 25, text_y + 20),
                                   cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 0, 0), 1); // Blue for MobileNet
                        
                        // Log results
                        //RCLCPP_INFO(this->get_logger(), 
                         //       "Object %d - YOLO: %s (%.1f%%) | MobileNet: %s (%.1f%%)",
                         //       processed_count,
                          //      yolo_label.c_str(),
                         //       obj.confidence,
                          //      mobilenet_class.c_str(),
                          //      confidence * 100.0f);
                        
                        processed_count++;
                    }
                }
            }
            
            // Add timestamp to visualization
            auto now = this->now();
            //std::string timestamp_text = "Time: " + std::to_string(now.seconds());
            //cv::putText(visualization_image, timestamp_text, cv::Point(10, 30),
             //          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            // Add object count
            //std::string count_text = "Objects: " + std::to_string(processed_count);
            //cv::putText(visualization_image, count_text, cv::Point(10, 60),
            //           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            // Convert back to ROS image and publish
            auto visualization_msg = cv_bridge::CvImage(
                std_msgs::msg::Header(), "bgr8", visualization_image
            ).toImageMsg();
            
            image_publisher_.publish(visualization_msg);
            
            if (processed_count > 0) {
                RCLCPP_INFO(this->get_logger(), "Published visualization with %d objects", processed_count);
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing visualization: %s", e.what());
        }
    }
    
    std::string get_label_name(const zed_msgs::msg::Object& obj)
    {
        static const std::unordered_map<std::string, std::string> label_map = {
            {"0", "bus"}, {"1", "left"}, {"2", "bump"}, {"3", "animal"},
            {"4", "lights"}, {"5", "lanes"}, {"6", "hospital"}, {"7", "roundabout"},
            {"8", "parking"}, {"9", "crosswalk"}, {"10", "speed"}, {"11", "warning"}
        };
        
        if (!obj.label.empty()) {
            return obj.label;
        }
        
        auto it = label_map.find(obj.sublabel);
        if (it != label_map.end()) {
            return it->second;
        }
        
        return "class_" + obj.sublabel;
    }
    
    // ROS subscribers and publishers
    rclcpp::Subscription<zed_msgs::msg::ObjectsStamped>::SharedPtr objects_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    image_transport::Publisher image_publisher_;
    
    // OpenCV DNN network
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    
    // Latest messages
    zed_msgs::msg::ObjectsStamped::SharedPtr latest_objects_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MobileNetVisualizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}