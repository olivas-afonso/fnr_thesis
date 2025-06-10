import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


slam_config_path = PathJoinSubstitution([
    FindPackageShare('point_cloud_processor'),
    'config',
    'slam_offline.yaml'  # offline config!
])

def generate_launch_description():

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )

    # Sync (offline) slam_toolbox node
    slam_toolbox_sync_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox_offline',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            slam_config_path
        ],
        remappings=[
            ('/scan', '/cluster_scan'),  # match your scan topic
            ('/odom', '/odometry/filtered')
            # No odometry remapping needed; it ignores it
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        slam_toolbox_sync_node
    ])