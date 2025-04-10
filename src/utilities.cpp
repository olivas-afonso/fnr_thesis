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
        // Get camera transform
        tf2::Quaternion q( camera_pose.orientation.x, camera_pose.orientation.y, camera_pose.orientation.z, camera_pose.orientation.w);
        tf2::Matrix3x3 rotation(q);

        Eigen::Matrix3f rot_matrix;
        for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
        rot_matrix(i,j) = rotation[i][j];

        Eigen::Vector3f camera_pos(
        camera_pose.position.x,
        camera_pose.position.y,
        camera_pose.position.z);

        // Transform points to camera frame
        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points) {
            Eigen::Vector3f pt_map(point.x, point.y, point.z);
            Eigen::Vector3f pt_cam = rot_matrix.transpose() * (pt_map - camera_pos);
            x_vals.push_back(pt_cam.x());
            y_vals.push_back(pt_cam.y());
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

        // Visualize curve in map frame
        visualization_msgs::msg::Marker curve_marker;
        curve_marker.header.frame_id = "map";
        curve_marker.header.stamp = current_time;
        curve_marker.ns = "curve_fit";
        curve_marker.id = 1;
        curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        curve_marker.scale.x = 0.05; 
        curve_marker.color.r = 0.0;
        curve_marker.color.g = 0.0;
        curve_marker.color.b = 1.0;
        curve_marker.color.a = 1.0;

        // Generate and transform curve points back to map frame
        float x_min = *std::min_element(x_vals.begin(), x_vals.end());
        float x_max = *std::max_element(x_vals.begin(), x_vals.end());

        for (float x = x_min; x <= x_max; x += 0.05) {
            float y = coeffs_cam(0)*x*x + coeffs_cam(1)*x + coeffs_cam(2);
            Eigen::Vector3f pt_cam(x, y, 0);
            Eigen::Vector3f pt_map = rot_matrix * pt_cam + camera_pos;

            geometry_msgs::msg::Point p;
            p.x = pt_map.x();
            p.y = pt_map.y();
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
            coeffs[2] + lane_transform[2]/2.0f  // Offset c by half lane width
            );
        } else {
            // Shift left by half the lane width
            return Eigen::Vector3f(
            coeffs[0],  // a stays same
            coeffs[1],  // b stays same
            coeffs[2] - lane_transform[2]/2.0f  // Offset c by half lane width
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
        // Convert camera orientation to Eigen matrix
        Eigen::Matrix3f rot_eigen;
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                rot_eigen(i,j) = rotation_matrix[i][j];
            }
        }

        const float a = middle_coeffs_cam[0];
        const float b = middle_coeffs_cam[1];
        const float c = middle_coeffs_cam[2];
        
        // Search only within our distance bounds (camera frame)
        float x_min = min_distance;
        float x_max = max_distance;
        
        // Sample points along curve to find closest in bounded region
        const int num_samples = 20;
        float min_dist_sq = std::numeric_limits<float>::max();
        float best_x = 0.0f;
        
        for (int i = 0; i <= num_samples; ++i) {
            float x = x_min + (x_max - x_min) * (i / float(num_samples));
            float y = a*x*x + b*x + c;
            float dist_sq = x*x + y*y;  // Distance squared from camera (0,0)
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_x = x;
            }
        }
        
        // Refine with Newton-Raphson around the best point
        float x = best_x;
        const float tolerance = 1e-5f;
        const int max_iterations = 10;
        
        for (int i = 0; i < max_iterations; ++i) {
            float y = a*x*x + b*x + c;
            float dy_dx = 2*a*x + b;
            
            float f = 2*x + 2*y*dy_dx;
            float df_dx = 2 + 2*(dy_dx*dy_dx + y*2*a);
            
            float delta = f / df_dx;
            x -= delta;
            
            // Clamp to our search region
            x = std::max(x_min, std::min(x_max, x));
            
            if (std::abs(delta) < tolerance) {
                break;
            }
        }
        
        // Calculate final closest point
        float y = a*x*x + b*x + c;
        Eigen::Vector3f closest_pt_cam(x, y, 0);
        
        // Transform back to map frame
        Eigen::Vector3f closest_pt_map = rot_eigen * closest_pt_cam + camera_pos;
        
        // Calculate distance in map frame
        distance = (closest_pt_map - camera_pos).norm();
        
        // Calculate tangent at closest point (in camera frame)
        float tangent_slope = 2*a*x + b;
        float curve_angle = atan2(tangent_slope, 1.0f); // Use atan2 for proper quadrant
        
        // Get camera yaw angle (in camera frame, looking along +X)
        double roll, pitch, yaw;
        rotation_matrix.getRPY(roll, pitch, yaw);
        
        // Calculate orientation difference (curve_angle is relative to camera's +X axis)
        // Positive when curve turns right (relative to camera), negative when left
        orientation_diff = curve_angle;
        
        // Normalize to [-π, π]
        while (orientation_diff > M_PI) orientation_diff -= 2*M_PI;
        while (orientation_diff < -M_PI) orientation_diff += 2*M_PI;
    }

    std::vector<pcl::PointIndices> clusterWhitePoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const geometry_msgs::msg::Pose &current_pose_)
    {
        // 1. Verify frame_id is map
        if (cloud->header.frame_id != "map") {
            //RCLCPP_WARN(this->get_logger(), 
                //"Expected point cloud in map frame, got %s. Proceeding anyway.",
            //    cloud->header.frame_id.c_str());
        }

        // 2. Perform Euclidean clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
        
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.07);  // 8cm
        ec.setMinClusterSize(220);     // Minimum points per cluster
        ec.setMaxClusterSize(5000);    // Maximum points per cluster
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);

        // 3. Process clusters directly in map frame
        std::vector<pcl::PointIndices> filtered_clusters;
        const Eigen::Vector3f camera_pos(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);

        // Parameters for cluster selection
        const float max_distance_from_camera = 1.5f; // Maximum distance in meters
        const int max_clusters_to_select = 2;        // Maximum number of clusters to select

        // Vector to store clusters with their distances
        std::vector<std::pair<pcl::PointIndices, float>> cluster_distance_pairs;

        for (const auto &cluster : clusters) {
            // Calculate centroid
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            for (int idx : cluster.indices) {
                centroid[0] += cloud->points[idx].x;
                centroid[1] += cloud->points[idx].y;
                centroid[2] += cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();

            // Calculate distance from camera
            float distance = (centroid - camera_pos).norm();

            // Filter based on distance
            if (distance <= max_distance_from_camera) {
                cluster_distance_pairs.emplace_back(cluster, distance);
                //RCLCPP_DEBUG(this->get_logger(),
                //    "Cluster candidate: distance=%.2fm, points=%lu",
                //    distance, cluster.indices.size());
            }
        }

        // Sort clusters by distance (closest first)
        std::sort(cluster_distance_pairs.begin(), cluster_distance_pairs.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

        // Select up to max_clusters_to_select closest clusters
        for (size_t i = 0; i < std::min(cluster_distance_pairs.size(), static_cast<size_t>(max_clusters_to_select)); ++i) {
            filtered_clusters.push_back(cluster_distance_pairs[i].first);
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

} // namespace lane_detection_utils