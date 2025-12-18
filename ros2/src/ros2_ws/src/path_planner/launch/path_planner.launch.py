from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='path_planner',
            executable='path_planner_node',
            name='path_planner',
            output='screen',
            parameters=[{
                'robot_radius': 0.5,
                'safety_margin': 0.2,
                'grid_resolution': 0.1,
                'field_size_x': 10.0,
                'field_size_y': 10.0,
            }]
        ),
    ])
