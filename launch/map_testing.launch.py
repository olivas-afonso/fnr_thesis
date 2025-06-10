import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction, ExecuteProcess
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory



ekf_config_path = PathJoinSubstitution([
    FindPackageShare('point_cloud_processor'),
    'config',
    'ekf_config.yaml'
])

def generate_launch_description():
    # Package and directory setup
    pkg_name = 'point_cloud_processor'
    pkg_dir = get_package_share_directory(pkg_name)
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    

    point_cloud_processor_node = Node(
        package=pkg_name,
        executable='point_cloud_processor',
        name='point_cloud_processor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    

    
    global_localization = Node(
        package=pkg_name,
        executable='global_localization',
        name='global_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'fit_side': True,
            'time_jump_threshold': 5.0,
            'min_scan_range': 0.1,
            'max_scan_range': 10.0
        }]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            ekf_config_path,
            {'use_sim_time': use_sim_time}
        ]
    )

    slam_config_path = PathJoinSubstitution([
        FindPackageShare('point_cloud_processor'),
        'config',
        'slam_toolbox_config.yaml'  # New config file
    ])

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            slam_config_path  # Load custom config
        ],
        remappings=[
            ('/scan', '/cluster_scan'),
            ('/odom', '/odometry/filtered'),
        ]
    )



    # Delay remaining nodes AND bag playback
    delayed_nodes = TimerAction(
        period=3.0,
        actions=[

            # map_creator_node,
            #slam_toolbox_node,
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        point_cloud_processor_node,
        ekf_node,
        global_localization,
        delayed_nodes
    ])
