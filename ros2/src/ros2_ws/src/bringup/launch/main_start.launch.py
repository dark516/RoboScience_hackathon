from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Контроллер вооружения
        Node(
            package='weapon_controller',
            executable='weapon_controller',
            output='screen',
            name='weapon_controller'
        ),
        
        # PID контроллер для следования по пути
        Node(
            package='behavior_server',
            executable='path_follower_pid_node',
            output='screen',
            name='path_follower_pid_node',
            parameters=[{
                'kp_angular': -0.18,
                'ki_angular': -0.0,
                'kd_angular': -0.015,
                'lookahead_distance': 0.4,
                'base_linear_speed': 0.3,
            }]
        ),
        
        # Сервер поведения
        Node(
            package='behavior_server',
            executable='behavior_server_node',
            output='screen',
            name='behavior_server_node'
        ),
        
        # Конвертер скорости в команды для моторов
        Node(
            package='vel_to_sticks',
            executable='vel_to_sticks',
            output='screen',
            name='vel_to_sticks',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'failsafe_timeout': 1.0,
            }]
        )
    ])
