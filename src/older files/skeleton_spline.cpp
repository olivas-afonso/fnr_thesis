#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <visualization_msgs/msg/marker.hpp>



#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


#include <pcl/common/pca.h>


#include <vector>
#include <cmath>
#include <numeric>
#include <ceres/ceres.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> 


class KalmanCircleLocalization : public rclcpp::Node
{
public:
    KalmanCircleLocalization() : Node("kalman_circ_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&KalmanCircleLocalization::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10,
            std::bind(&KalmanCircleLocalization::poseCallback, this, std::placeholders::_1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        skeleton_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/skeleton", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);
    
        curve_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_original", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);
        test_cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("test_clusters", 10);


        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);

        // Initialize state [x_c, y_c, r, theta]
        state = Eigen::VectorXf::Zero(4);
        state << 0, 0, 3.0, 3.0; // Assume initial curve with 5m radius

        // Initialize covariance matrix
        P = Eigen::MatrixXf::Identity(4,4) * 0.1;

        // Process and measurement noise
        Q = Eigen::MatrixXf::Identity(4,4) * 0.01;
        R = Eigen::MatrixXf::Identity(4,4) * 0.5;
        I = Eigen::MatrixXf::Identity(4,4);

        // State transition model (assume slow-moving lane changes)
        F = Eigen::MatrixXf::Identity(4,4);



    }

private:
    Eigen::VectorXf state; // [x_c, y_c, r, theta]
    Eigen::MatrixXf P, Q, R, I, F;
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;


    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &point : cloud->points)
        {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
            if (v > 0.65f && s < 0.2f)
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }
        publishCameraAxes(current_pose_);
        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }

        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud, current_pose_.pose);
        pcl::PointCloud<pcl::PointXYZ>::Ptr thinned_cloud(new pcl::PointCloud<pcl::PointXYZ>());

            // Iterate over each cluster and skeletonize
        for (const auto& cluster : selected_clusters) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            
            // Extract points for this cluster
            for (int idx : cluster.indices) {
                cluster_cloud->push_back(white_cloud->points[idx]);
            }

            // Skeletonize this cluster
            pcl::PointCloud<pcl::PointXYZ>::Ptr skeleton = skeletonizeCluster(cluster_cloud);
            
            // Append to the final thinned cloud
            *thinned_cloud += *skeleton;
        }

        // Publish the skeletonized cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*thinned_cloud, output_msg);
        output_msg.header = msg->header;
        skeleton_publisher_->publish(output_msg);
        /*
        if (selected_clusters.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return;
        }
        */

        publishClusterMarkersindex(white_cloud, selected_clusters);
        
        
        
        //publishClusterMarkers(thinned_clusters);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        Eigen::Vector2f left_observed, right_observed;
        bool left_detected = false, right_detected = false;
        float max_angle = -std::numeric_limits<float>::infinity();
        float min_angle = std::numeric_limits<float>::infinity();
            
        Eigen::Vector3f camera_position(0.0, 0.0, 0.0);  
        camera_position[0]=current_pose_.pose.position.x;
        camera_position[1]=current_pose_.pose.position.y;
        camera_position[2]=current_pose_.pose.position.z;
        
        if (selected_clusters.size() == 1)
        {
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            pcl::PointIndices cluster = selected_clusters[0]; 
            for (int idx : cluster.indices)
            {
                centroid[0] += white_cloud->points[idx].x;
                centroid[1] += white_cloud->points[idx].y;
                centroid[2] += white_cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();  // Compute centroid
    
            float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);
            
            if (angle >= 0)  // Cluster is on the left side
            {
                leftmost_cluster = cluster;
                left_observed = centroid.head<2>();
                left_detected = true;
                RCLCPP_INFO(this->get_logger(), "OI");
            }
            else  // Cluster is on the right side
            {
                rightmost_cluster = cluster;
                right_observed = centroid.head<2>();
                right_detected = true;
            }
        }
        else  // Standard case: multiple clusters
        {
            for (const auto& cluster : selected_clusters)
            {
                Eigen::Vector3f centroid(0.0, 0.0, 0.0);
                for (int idx : cluster.indices)
                {
                    centroid[0] += white_cloud->points[idx].x;
                    centroid[1] += white_cloud->points[idx].y;
                    centroid[2] += white_cloud->points[idx].z;
                }
                centroid /= cluster.indices.size();  // Compute centroid
            
                // Compute relative angle to the camera
                float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);
            
                if (angle > max_angle) 
                {
                    max_angle = angle;
                    leftmost_cluster = cluster;
                    left_observed = centroid.head<2>();
                    left_detected = true;
                }
            
                if (angle < min_angle) 
                {
                    min_angle = angle;
                    rightmost_cluster = cluster;
                    right_observed = centroid.head<2>();
                    right_detected = true;
                }
            }
        }


        // Kalman Prediction Step
        /*
        state = F * state;
        P = F * P * F.transpose() + Q;

        // Kalman Update Step
        Eigen::MatrixXf S = P + R;
        Eigen::MatrixXf K = P * S.inverse();

        state = state + K * (Z - state);
        P = (I - K) * P;
        */

    //visualizeCircles(common_center, left_radius, right_radius, state, left_start_angle, left_end_angle, left_start_angle, left_end_angle, left_detected, left_detected);
        //visualizeCircles(common_center, left_radius, right_radius, state, right_start_angle, right_end_angle, right_start_angle, right_end_angle, right_detected, right_detected);

    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr skeletonizeCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud) {
        // Step 1: Find min/max bounds for scaling
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);
    
        float scale = 10.0f;  // Scaling factor to convert to image size
        int img_width = static_cast<int>((max_pt.x() - min_pt.x()) * scale);
        int img_height = static_cast<int>((max_pt.z() - min_pt.z()) * scale);
    
        // Avoid zero-size image
        if (img_width == 0 || img_height == 0) {
            return pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        }
    
        // Create a binary image
        cv::Mat binary_image = cv::Mat::zeros(img_height, img_width, CV_8UC1);
    
        // Step 2: Populate the binary image with XZ projection
        for (const auto& p : cluster_cloud->points) {
            int x = static_cast<int>((p.x - min_pt.x()) * scale);
            int z = static_cast<int>((p.z - min_pt.z()) * scale);
            if (x >= 0 && x < img_width && z >= 0 && z < img_height) {
                binary_image.at<uint8_t>(z, x) = 255;  // Set pixel white
            }
        }
        cv::imwrite("/tmp/binary_image.png", binary_image);
        RCLCPP_INFO(this->get_logger(), "Binary image saved to /tmp/binary_image.png");

    
        // Step 3: Apply skeletonization
        cv::Mat skeleton;
        cv::ximgproc::thinning(binary_image, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);

        cv::imwrite("/tmp/skeleton_image.png", skeleton);
        RCLCPP_INFO(this->get_logger(), "Skeleton image saved to /tmp/skeleton_image.png");

    
        // Step 4: Convert the skeleton back to a point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr skeleton_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (int z = 0; z < skeleton.rows; ++z) {
            for (int x = 0; x < skeleton.cols; ++x) {
                if (skeleton.at<uint8_t>(z, x) > 0) {
                    float world_x = min_pt.x() + (x / scale);
                    float world_z = min_pt.z() + (z / scale);
                    skeleton_cloud->push_back(pcl::PointXYZ(world_x, 0.0f, world_z)); // Y will be set later
                }
            }
        }
    
        // Step 5: Assign Y values using nearest neighbor search in the original cluster
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cluster_cloud);
        for (auto& point : skeleton_cloud->points) {
            std::vector<int> nearest_index(1);
            std::vector<float> nearest_dist(1);
            if (kdtree.nearestKSearch(point, 1, nearest_index, nearest_dist) > 0) {
                point.y = cluster_cloud->points[nearest_index[0]].y;  // Assign closest Y value
            }
        }
    
        return skeleton_cloud;
    }
    


    pcl::PointCloud<pcl::PointXYZ>::Ptr extractCenterline(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // Step 1: Compute PCA
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    
    Eigen::Vector3f main_axis = pca.getEigenVectors().col(0); // Main direction

    // Step 2: Project points onto the main axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : cloud->points) {
        Eigen::Vector3f p(point.x, point.y, point.z);
        float projection_length = p.dot(main_axis);
        Eigen::Vector3f projected_point = projection_length * main_axis;
        projected_cloud->push_back(pcl::PointXYZ(projected_point.x(), projected_point.y(), projected_point.z()));
    }

    // Step 3: Reduce density by keeping one representative point per small segment
    pcl::PointCloud<pcl::PointXYZ>::Ptr thinned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    float segment_size = 0.05;  // Adjust based on point cloud resolution
    std::map<int, pcl::PointXYZ> segment_map;

    for (const auto& point : projected_cloud->points) {
        int segment_id = static_cast<int>(point.x / segment_size); // Group by X-axis
        if (segment_map.find(segment_id) == segment_map.end()) {
            segment_map[segment_id] = point;
        }
    }

    // Convert map back to point cloud
    for (const auto& [_, point] : segment_map) {
        thinned_cloud->push_back(point);
    }

    return thinned_cloud;
}

    std::vector<pcl::PointIndices> clusterWhitePoints( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,const geometry_msgs::msg::Pose &current_pose_) // Camera pose
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
    
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(5000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);
    
        struct ClusterData
        {
            pcl::PointIndices indices;
            float min_distance;
            float depth; // Width in the camera direction (Z-axis)
        };
    
        std::vector<ClusterData> cluster_data;
    
        // Extract camera position
        Eigen::Vector3f camera_position(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);
    
        // Extract camera orientation
        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);
    
        // Convert to Eigen rotation matrix
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R(i, j) = rotation_matrix[i][j];
    
        // Extract forward axis (Z direction in camera frame)
        Eigen::Vector3f forward_dir = R.col(2); // Z-axis
    
        for (const auto &cluster : clusters)
        {
            float min_proj = std::numeric_limits<float>::max();
            float max_proj = std::numeric_limits<float>::lowest();
            float sum_x = 0.0, sum_z = 0.0;
            int num_points = cluster.indices.size();
    
            for (int idx : cluster.indices)
            {
                Eigen::Vector3f point(
                    cloud->points[idx].x,
                    cloud->points[idx].y,
                    cloud->points[idx].z);
    
                // Transform to camera-aligned frame
                Eigen::Vector3f relative_point = R.transpose() * (point - camera_position);
    
                // Project onto the camera direction (Z-axis in camera frame)
                float projection = relative_point.dot(Eigen::Vector3f(0, 0, 1));
    
                min_proj = std::min(min_proj, projection);
                max_proj = std::max(max_proj, projection);
    
                sum_x += relative_point.x();
                sum_z += relative_point.z();
            }
    
            float centroid_x = sum_x / num_points;
            float centroid_z = sum_z / num_points;
            float distance = std::sqrt(centroid_x * centroid_x + centroid_z * centroid_z);
    
            float cluster_depth = max_proj - min_proj; // Width along camera direction
    
            cluster_data.push_back({cluster, distance, cluster_depth});
        }
    
        // Filter clusters based on depth (remove clusters with small Z-width)
        std::vector<pcl::PointIndices> filtered_clusters;
        for (const auto &cd : cluster_data)
        {
            if (cd.depth >= 0.2) // Adjust this threshold as needed
            {
                filtered_clusters.push_back(cd.indices);
            }
        }
    
        return filtered_clusters;
    }
    
    void RGBtoHSV(int r, int g, int b, float &h, float &s, float &v)
    {
        float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
        float maxC = std::max({rf, gf, bf});
        float minC = std::min({rf, gf, bf});
        v = maxC;

        float delta = maxC - minC;
        if (delta < 1e-5)
        {
            h = 0.0f;
            s = 0.0f;
            return;
        }

        s = (maxC > 0.0f) ? (delta / maxC) : 0.0f;

        if (maxC == rf)
            h = 60.0f * (fmod(((gf - bf) / delta), 6));
        else if (maxC == gf)
            h = 60.0f * (((bf - rf) / delta) + 2);
        else
            h = 60.0f * (((rf - gf) / delta) + 4);

        if (h < 0)
            h += 360.0f;
    }

    void publishClusterMarkers(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& thinned_clusters)
{
    visualization_msgs::msg::MarkerArray cluster_markers;
    std::vector<std::tuple<float, float, float>> colors = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    for (size_t i = 0; i < thinned_clusters.size(); ++i)
    {
        visualization_msgs::msg::Marker cluster_marker;
        cluster_marker.header.frame_id = "map";
        cluster_marker.header.stamp = this->get_clock()->now();
        cluster_marker.ns = "clusters";
        cluster_marker.id = i;
        cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        cluster_marker.scale.x = 0.05;
        cluster_marker.scale.y = 0.05;
        cluster_marker.scale.z = 0.05;

        auto [r, g, b] = colors[i % colors.size()];
        cluster_marker.color.r = r;
        cluster_marker.color.g = g;
        cluster_marker.color.b = b;
        cluster_marker.color.a = 1.0;

        // Loop through the thinned cluster points instead of using indices
        for (const auto& point : thinned_clusters[i]->points)
        {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            cluster_marker.points.push_back(p);
        }
        cluster_markers.markers.push_back(cluster_marker);
    }
    cluster_marker_pub_->publish(cluster_markers);
}

    
    void publishClusterMarkersindex(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::vector<pcl::PointIndices>& clusters)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;
        std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            visualization_msgs::msg::Marker cluster_marker;
            cluster_marker.header.frame_id = "map";
            cluster_marker.header.stamp = this->get_clock()->now();
            cluster_marker.ns = "clusters";
            cluster_marker.id = i;
            cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            cluster_marker.scale.x = 0.03;
            cluster_marker.scale.y = 0.03;
            cluster_marker.scale.z = 0.03;

            auto [r, g, b] = colors[i % colors.size()];
            cluster_marker.color.r = r;
            cluster_marker.color.g = g;
            cluster_marker.color.b = b;
            cluster_marker.color.a = 1.0;

            for (int idx : clusters[i].indices)
            {
                geometry_msgs::msg::Point p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                cluster_marker.points.push_back(p);
            }
            cluster_markers.markers.push_back(cluster_marker);
        }
        cluster_marker_pub_->publish(cluster_markers);
    }

    void testClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const pcl::PointIndices& left_indices, const pcl::PointIndices& right_indices)
    {
    visualization_msgs::msg::MarkerArray cluster_markers;

    // Define colors for left (red) and right (blue) clusters
    std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0},  // Left cluster (Red)
                                            {0.0, 0.0, 1.0}}; // Right cluster (Blue)

    // Helper lambda to create markers
    auto createMarker = [&](const pcl::PointIndices& indices, int id, const std::tuple<float, float, float>& color) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "clusters";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    auto [r, g, b] = color;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;

    for (int idx : indices.indices)
    {
    geometry_msgs::msg::Point p;
    p.x = cloud->points[idx].x;
    p.y = cloud->points[idx].y;
    p.z = cloud->points[idx].z;
    marker.points.push_back(p);
    }

    return marker;
    };

    // Add left and right cluster markers
    cluster_markers.markers.push_back(createMarker(left_indices, 0, colors[0]));  // Left
    cluster_markers.markers.push_back(createMarker(right_indices, 1, colors[1])); // Right

    // Publish the markers
    test_cluster_marker_pub_->publish(cluster_markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr skeleton_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr test_cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;


    rclcpp::Parameter fit_side_param_;
    bool fit_side_;

    void publishCameraAxes(const geometry_msgs::msg::PoseStamped &msg)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        // Extract camera position
        Eigen::Vector3f camera_position(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z);

        // Extract rotation matrix from quaternion
        tf2::Quaternion q(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        // Extract direction vectors
        Eigen::Vector3f x_axis(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]); // Right (Red)
        Eigen::Vector3f y_axis(rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]); // Up (Green)
        Eigen::Vector3f z_axis(rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]); // Forward (Blue)

        auto createArrowMarker = [&](int id, const Eigen::Vector3f &dir, const std_msgs::msg::ColorRGBA &color)
        {
            visualization_msgs::msg::Marker arrow;
            arrow.header.frame_id = "map"; // Change if needed
            arrow.header.stamp = this->now();
            arrow.ns = "camera_axes";
            arrow.id = id;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.scale.x = 0.05;  // Shaft diameter
            arrow.scale.y = 0.1;   // Head diameter
            arrow.scale.z = 0.1;   // Head length

            geometry_msgs::msg::Point start, end;
            start.x = camera_position.x();
            start.y = camera_position.y();
            start.z = camera_position.z();

            end.x = start.x + dir.x() * 0.5; // Scale for visibility
            end.y = start.y + dir.y() * 0.5;
            end.z = start.z + dir.z() * 0.5;

            arrow.points.push_back(start);
            arrow.points.push_back(end);

            arrow.color = color;
            arrow.lifetime = rclcpp::Duration::from_seconds(0.5); // Short lifespan to update
            arrow.frame_locked = true;

            return arrow;
        };

        // Define colors
        std_msgs::msg::ColorRGBA red, green, blue;
        red.r = 1.0; red.a = 1.0;
        green.g = 1.0; green.a = 1.0;
        blue.b = 1.0; blue.a = 1.0;

        marker_array.markers.push_back(createArrowMarker(0, x_axis, red));   // X-axis (Right)
        marker_array.markers.push_back(createArrowMarker(1, y_axis, green)); // Y-axis (Up)
        marker_array.markers.push_back(createArrowMarker(2, z_axis, blue));  // Z-axis (Forward)

        marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanCircleLocalization>());
    rclcpp::shutdown();
    return 0;
}
