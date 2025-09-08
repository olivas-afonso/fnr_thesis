#include <rclcpp/rclcpp.hpp>
#include <zed_msgs/msg/objects_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
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

class MobileNetInferenceNode : public rclcpp::Node
{
public:
    MobileNetInferenceNode() : Node("mobilenet_inference_node")
    {
        // Initialize MobileNet model
        initialize_mobilenet_model();
        
        // Subscribe to object detection topic
        objects_subscription_ = this->create_subscription<zed_msgs::msg::ObjectsStamped>(
            "/zed/zed_node/obj_det/objects",
            10,
            std::bind(&MobileNetInferenceNode::objects_callback, this, std::placeholders::_1));
        
        // Subscribe to camera image topic
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/left/image_rect_color",
            10,
            std::bind(&MobileNetInferenceNode::image_callback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "MobileNet Inference Node started");
    }

private:
    void initialize_mobilenet_model()
    {
        try {
            // Load the trained MobileNet model
            // UPDATE THESE PATHS TO YOUR ACTUAL MODEL FILES
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
            // Check if input image is valid
            if (image.empty() || image.cols <= 0 || image.rows <= 0) {
                RCLCPP_ERROR(this->get_logger(), "Invalid input image for inference: %dx%d", image.cols, image.rows);
                return {"Invalid image", 0.0f};
            }
            
            // Resize to match training size (64x64)
            cv::Mat resized_image;
            cv::resize(image, resized_image, cv::Size(64, 64));
            
            // Preprocess image with the SAME normalization as training
            cv::Mat blob;
            cv::dnn::blobFromImage(resized_image, blob, 1.0/255.0,  // Scale to 0-1
                                cv::Size(64, 64),  // MATCH TRAINING SIZE: 64x64
                                cv::Scalar(0, 0, 0), true, false);
            
            // Apply the same normalization as training: (x - 0.5) / 0.5 = 2*x - 1
            // This converts [0,1] range to [-1,1] range
            blob = 2.0 * blob - 1.0;
            
            // Set input
            net_.setInput(blob);
            
            // Run inference
            cv::Mat output = net_.forward();
            
            // Apply softmax to convert logits to probabilities
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
        process_objects_with_mobilenet();
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        latest_image_ = msg;
    }
    
    void process_objects_with_mobilenet()
    {
        if (!latest_objects_ || !latest_image_) {
            return;
        }
        
        try {
            // Convert ROS image to OpenCV format - USE THE SAME LOGIC AS save_object_images!
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
            
            cv::Mat image;
            if (cv_ptr->encoding == "bgra8") {
                cv::cvtColor(cv_ptr->image, image, cv::COLOR_BGRA2BGR);
            } else {
                image = cv_ptr->image;
            }
            
           // RCLCPP_INFO(this->get_logger(), "Image dimensions: %d x %d", image.cols, image.rows);
            
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

                    // USE THE SAME SCALING AS save_object_images!
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

                    // Convert to integer coordinates and clamp to image bounds
                    int x1 = std::max(0, static_cast<int>(min_x));
                    int y1 = std::max(0, static_cast<int>(min_y));
                    int x2 = std::min(image.cols - 1, static_cast<int>(max_x));
                    int y2 = std::min(image.rows - 1, static_cast<int>(max_y));
                    
                    // Check if the bounding box is valid
                    if (x2 > x1 && y2 > y1) {
                        // Extract the region of interest
                        cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                        
                        // Check if ROI is valid before inference
                        if (roi.empty() || roi.cols <= 0 || roi.rows <= 0) {
                            RCLCPP_WARN(this->get_logger(), "Invalid ROI for %s: %dx%d", 
                                    yolo_label.c_str(), roi.cols, roi.rows);
                            continue;
                        }
                        
                        // Run MobileNet inference on the cropped image
                        auto [mobilenet_class, confidence] = run_inference(roi);
                        
                        // Display results
                        RCLCPP_INFO(this->get_logger(), 
                                "Object %d - YOLO: %s (%.1f%%) | MobileNet: %s (%.1f%%) | BBox: [%d,%d]-[%d,%d]",
                                processed_count,
                                yolo_label.c_str(),
                                obj.confidence,
                                mobilenet_class.c_str(),
                                confidence * 100.0f,
                                x1, y1, x2, y2);
                        
                        processed_count++;
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Invalid bounding box for %s: [%d,%d]-[%d,%d] (image: %dx%d)",
                                yolo_label.c_str(), x1, y1, x2, y2, image.cols, image.rows);
                    }
                }
            }
            
            if (processed_count > 0) {
                //RCLCPP_INFO(this->get_logger(), "Processed %d objects with MobileNet", processed_count);
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing objects: %s", e.what());
        }
    }
    
    std::string get_label_name(const zed_msgs::msg::Object& obj)
    {
        // Simple label mapping (same as your original code)
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
    
    // ROS subscribers and members
    rclcpp::Subscription<zed_msgs::msg::ObjectsStamped>::SharedPtr objects_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    
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
    auto node = std::make_shared<MobileNetInferenceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}