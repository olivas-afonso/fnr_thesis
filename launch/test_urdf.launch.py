from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
import os

def generate_launch_description():
    # Get URDF via xacro
    pkg_path = get_package_share_directory('point_cloud_processor')
    urdf_path = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')
    
    # Convert xacro to URDF string
    robot_description = Command(['xacro ', urdf_path])

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': False  # Disable for testing
            }]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': False}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(pkg_path, 'rviz', 'urdf_config.rviz')],
            output='screen',
            parameters=[{'use_sim_time': False}]
        )
    ])