from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    # Declare the enable_ai parameter
    enable_ai = LaunchConfiguration('enable_ai', default='true')

    return LaunchDescription([
        # Define the argument for launching
        DeclareLaunchArgument(
            'enable_ai',
            default_value='true',
            description='Enable or disable AI inference and covariance corrections'
        ),
        Node(
            package='ai_tools',
            executable='ai_covariance_node',
            name='ai_covariance_node',
            output='screen',
            parameters=[{'enable_ai': enable_ai}],
        ),
        Node(
            package='ai_tools',
            executable='ai_covariance_updater',
            name='ai_covariance_updater',
            output='screen',
            parameters=[{'enable_ai': enable_ai}],
        ),
    ])