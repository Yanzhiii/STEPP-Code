"""
ROS 2 launch file for STEPP traversability estimation.

Usage:
    ros2 launch stepp_ros2_humble STEPP_launch.py \
        model_path:=/path/to/checkpoint.pth \
        camera_type:=zed2

All launch arguments have defaults that can be overridden at runtime.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # ----------------------------------------------------------------
            # Launch arguments (with defaults from the original ROS 1 file)
            # ----------------------------------------------------------------
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Path to trained MLP checkpoint (.pth file). REQUIRED.",
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="true",
                description="Publish traversability overlay image (slows inference).",
            ),
            DeclareLaunchArgument(
                "ump",
                default_value="false",
                description="Use mixed precision for model inference.",
            ),
            DeclareLaunchArgument(
                "cutoff",
                default_value="0.45",
                description="Max normalized reconstruction error (reference: 0.45).",
            ),
            DeclareLaunchArgument(
                "camera_type",
                default_value="zed2",
                description="Camera model for intrinsics: zed2 | D455 | cmu_sim.",
            ),
            DeclareLaunchArgument(
                "decay_time",
                default_value="8.0",
                description="Depth pointcloud decay time in seconds (feature WIP).",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera/color/image_raw/compressed",
                description="Input RGB image topic (compressed).",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/aligned_depth_to_color/image_raw",
                description="Input aligned depth image topic (raw).",
            ),
            DeclareLaunchArgument(
                "odom_topic",
                default_value="/state_estimation",
                description="Input odometry topic.",
            ),
            # ----------------------------------------------------------------
            # Inference Node
            # ----------------------------------------------------------------
            Node(
                package="stepp_ros2_humble",
                executable="inference_node.py",
                name="inference_node",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "visualize": LaunchConfiguration("visualize"),
                        "ump": LaunchConfiguration("ump"),
                        "cutoff": LaunchConfiguration("cutoff"),
                    }
                ],
                remappings=[
                    (
                        "/camera/color/image_raw/compressed",
                        LaunchConfiguration("rgb_topic"),
                    ),
                ],
            ),
            # ----------------------------------------------------------------
            # Depth Projection Node
            # ----------------------------------------------------------------
            Node(
                package="stepp_ros2_humble",
                executable="depth_projection_node.py",
                name="depth_projection",
                output="screen",
                parameters=[
                    {
                        "camera_type": LaunchConfiguration("camera_type"),
                        "decay_time": LaunchConfiguration("decay_time"),
                    }
                ],
                remappings=[
                    (
                        "/camera/aligned_depth_to_color/image_raw",
                        LaunchConfiguration("depth_topic"),
                    ),
                    (
                        "/state_estimation",
                        LaunchConfiguration("odom_topic"),
                    ),
                ],
            ),
        ]
    )
