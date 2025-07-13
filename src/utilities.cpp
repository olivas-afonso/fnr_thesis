#include "kalman_rel_localization/utilities.hpp"
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Dense>

namespace lane_detection_utils {

    Eigen::Vector3f fitQuadraticCurve(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub,
    const geometry_msgs::msg::Pose& camera_pose,
    const rclcpp::Time& current_time)
    {
            // Points are already in base_link frame - use directly
        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points) {
            x_vals.push_back(point.x);  // Use raw coordinates
            y_vals.push_back(point.y);
        }

        // Fit quadratic in camera frame (y = ax² + bx + c)
        Eigen::MatrixXd A(x_vals.size(), 3);
        Eigen::VectorXd Z(y_vals.size());

        for (size_t i = 0; i < x_vals.size(); i++) {
            A(i, 0) = x_vals[i] * x_vals[i]; // x² term
            A(i, 1) = x_vals[i];             // x term
            A(i, 2) = 1.0;                   // constant term
            Z(i) = y_vals[i];
        }

        Eigen::Vector3d coeffs_cam = A.colPivHouseholderQr().solve(Z);

        // Visualize curve in base_link frame
        visualization_msgs::msg::Marker curve_marker;
        curve_marker.header.frame_id = "base_link";  // Changed from "map"
        curve_marker.header.stamp = current_time;
        curve_marker.ns = "curve_fit";
        curve_marker.id = 1;
        curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        curve_marker.scale.x = 0.05; 
        curve_marker.color.r = 0.0;
        curve_marker.color.g = 0.0;
        curve_marker.color.b = 1.0;
        curve_marker.color.a = 1.0;

        // Generate curve points directly in camera frame
        float x_min = *std::min_element(x_vals.begin(), x_vals.end());
        float x_max = *std::max_element(x_vals.begin(), x_vals.end());

        for (float x = x_min; x <= x_max; x += 0.05) {
            float y = coeffs_cam(0)*x*x + coeffs_cam(1)*x + coeffs_cam(2);
            
            geometry_msgs::msg::Point p;
            p.x = x;  // Directly use camera-relative coordinates
            p.y = y;
            p.z = 0.0;
            curve_marker.points.push_back(p);
        }

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        pub->publish(markers);

        return coeffs_cam.cast<float>();
    }

    Eigen::Vector3f estimateRightFromLeft(const Eigen::Vector3f& left_coeffs_cam, const Eigen::Vector3f& lane_transform_cam) 
    {
        return left_coeffs_cam + lane_transform_cam;
    }

    Eigen::Vector3f estimateLeftFromRight(const Eigen::Vector3f& right_coeffs_cam, const Eigen::Vector3f& lane_transform_cam) 
    {
        return right_coeffs_cam - lane_transform_cam;
    }


    Eigen::Vector3f calculateAndStoreTransform(const Eigen::Vector3f& left_coeffs_cam, 
        const Eigen::Vector3f& right_coeffs_cam) 
    {
        // Transform is calculated in camera-relative coordinates
        return right_coeffs_cam - left_coeffs_cam;
    }    

    Eigen::Vector3f shiftToMiddle(const Eigen::Vector3f& coeffs, const Eigen::Vector3f& lane_transform, bool is_left) 
    {
        if (is_left) {
            // Shift right by half the lane width
            return Eigen::Vector3f(
            coeffs[0],  // a stays same (curvature)
            coeffs[1],  // b stays same (linear term)
            coeffs[2] - 0.7  // Offset c by half lane width
            );
        } else {
            // Shift left by half the lane width
            return Eigen::Vector3f(
            coeffs[0],  // a stays same
            coeffs[1],  // b stays same
            coeffs[2] + 0.7  // Offset c by half lane width
            );
        }
    }

    void calculateDistanceAndOrientation(
    const Eigen::Vector3f& middle_coeffs_cam,
    const Eigen::Vector3f& camera_pos,
    const tf2::Matrix3x3& rotation_matrix,
    float min_distance,
    float max_distance,
    float& distance,
    float& orientation_diff) 
    {
        // Simplified since we're working in camera frame
        const float a = middle_coeffs_cam[0];
        const float b = middle_coeffs_cam[1];
        const float c = middle_coeffs_cam[2];
        
        // Search in camera frame
        float x_min = min_distance;
        float x_max = max_distance;
        
        // Find closest point on curve
        float x = x_min;
        float min_dist_sq = std::numeric_limits<float>::max();
        
        for (float test_x = x_min; test_x <= x_max; test_x += 0.1f) {
            float y = a*test_x*test_x + b*test_x + c;
            float dist_sq = test_x*test_x + y*y;
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                x = test_x;
            }
        }
        
        // Calculate final distance (in camera frame)
        float y = a*x*x + b*x + c;
        distance = sqrt(x*x + y*y);
        
        // Calculate orientation difference (relative to camera's forward direction)
        float tangent_slope = 2*a*x + b;
        orientation_diff = atan2(tangent_slope, 1.0f);
        
        // Normalize to [-π, π]
        while (orientation_diff > M_PI) orientation_diff -= 2*M_PI;
        while (orientation_diff < -M_PI) orientation_diff += 2*M_PI;
    }

    std::vector<pcl::PointIndices> clusterWhitePoints(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const geometry_msgs::msg::Pose &current_pose_)
    {
        // Input cloud should already be in base_link frame
        RCLCPP_INFO(rclcpp::get_logger("clusterWhitePoints"), 
            "Input cloud in frame: %s", cloud->header.frame_id.c_str());
        
        // Perform clustering directly in base_link frame
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
        
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.06);
        ec.setMinClusterSize(150);
        ec.setMaxClusterSize(750);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);

        // Debug: Check found clusters
        RCLCPP_INFO(rclcpp::get_logger("clusterWhitePoints"), 
            "Found %zu clusters before filtering", clusters.size());

        // Filter clusters based on distance (exact same logic as old version, but simplified for base_link frame)
        std::vector<pcl::PointIndices> filtered_clusters;
        const float max_distance = 2.0f; // Maximum distance from robot (in meters)
        const int max_clusters = 2;

        std::vector<std::pair<pcl::PointIndices, float>> cluster_distance_pairs;

        for (const auto &cluster : clusters) {
            // Calculate centroid in base_link frame
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            for (int idx : cluster.indices) {
                centroid[0] += cloud->points[idx].x;
                centroid[1] += cloud->points[idx].y;
                centroid[2] += cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();

            // Distance is simply the norm from origin (since we're in base_link frame)
            float distance = centroid.norm();  // sqrt(x² + y² + z²)

            if (distance <= max_distance) {
                cluster_distance_pairs.emplace_back(cluster, distance);
            }
        }

        // Sort clusters by distance (closest first) - exact same as old version
        std::sort(cluster_distance_pairs.begin(), cluster_distance_pairs.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

        // Select up to max_clusters closest clusters - exact same as old version
        for (size_t i = 0; i < std::min(cluster_distance_pairs.size(), static_cast<size_t>(max_clusters)); ++i) {
            filtered_clusters.push_back(cluster_distance_pairs[i].first);
        }

        RCLCPP_INFO(rclcpp::get_logger("clusterWhitePoints"), 
            "Returning %zu filtered clusters", filtered_clusters.size());

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

} // namespace lane_detection_utils