from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vel_tunner',
            executable='vel_tunner_node',
            name='vel_tunner_node',
            output='screen'
        ),
        Node(
            package='weapon_controller',
            executable='weapon_controller',
            name='weapon_controller',
            output='screen'
        ),
        Node(
            package='velocity_help_node',
            executable='velocity_help_node',
            name='velocity_help_node',
            output='screen'
        ),
    ])
