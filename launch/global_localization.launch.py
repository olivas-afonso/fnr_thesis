import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction, ExecuteProcess

def generate_launch_description():
    # Package and directory setup
    pkg_name = 'point_cloud_processor'
    pkg_dir = get_package_share_directory(pkg_name)
    
    bag_play_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/home/olivas/camera_ws/bags/cross_left_2',
             '--loop', '--clock'],
        output='screen'
    )
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    
    # TF Tree Publishers
    #tf_publishers = [
        # map -> odom
    #    Node(
    #        package='tf2_ros',
    #        executable='static_transform_publisher',
    #        arguments=['0', '0', '0', '0', '0', '0', 'map', 'zed/zed_node/odom'],
    #        parameters=[{'use_sim_time': use_sim_time}],
    #        output='screen'
    #    ),
    #    Node(
    #        package='tf2_ros',
    #        executable='static_transform_publisher',
    #        arguments=['0', '0', '0', '0', '0', '0', 'zed/zed_node/odom', 'zed_camera_link'],
    #        parameters=[{'use_sim_time': use_sim_time}],
    #        output='screen'
    #    )
    #   ]

    map_transformer_node = Node(
        package=pkg_name,
        executable='map_transformer',
        name='map_transformer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # Map server node
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'yaml_filename': PathJoinSubstitution([
                FindPackageShare(pkg_name),
                'maps',
                'map.yaml'
            ]),
            'use_sim_time': use_sim_time,
            'topic_name': 'persistent_map',
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # Add a static map republisher
    map_republisher = Node(
        package='point_cloud_processor',
        executable='map_republisher',
        name='map_republisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # AMCL node with updated parameters
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            PathJoinSubstitution([FindPackageShare(pkg_name), 'config', 'amcl_config.yaml']),
            {
                'use_sim_time': use_sim_time,
                'odom_frame_id': 'odom',
                'base_frame_id': 'zed_camera_link',
                'global_frame_id': 'map',
                'transform_tolerance': 1.0
            }
        ],
        remappings=[
            ('/scan', '/cluster_scan'),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # Lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # Global Localization node
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

    tf_publisher_node = Node(
        package='point_cloud_processor',
        executable='dynamic_tf_publisher',
        name='dynamic_tf_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Delay launching nodes to let /clock start ticking
    delayed_nodes = TimerAction(
        period=1.0,
        actions=[
            tf_publisher_node,
            map_transformer_node,
            map_server,
            map_republisher,
            amcl,
            lifecycle_manager,
            global_localization
        ]
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        bag_play_process,
        delayed_nodes,
    ])