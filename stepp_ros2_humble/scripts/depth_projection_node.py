#!/usr/bin/env python3
"""
Depth Projection Node — ROS 2 Humble (Python rewrite of C++ version).

Synchronises depth image, odometry, and traversability cost,
projects depth to 3D point cloud, maintains a persistent terrain
cloud with temporal decay, and publishes PointCloud2 with cost.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters

from stepp_ros2_humble.msg import Float32Stamped

# ---------------------------------------------------------------------------
# Camera intrinsics (same hardcoded values as the C++ version)
# ---------------------------------------------------------------------------
CAMERA_INTRINSICS = {
    "D455": {
        "fx": 634.3491821289062,
        "fy": 632.8595581054688,
        "cx": 631.8179931640625,
        "cy": 375.0325622558594,
        "height": 720,
        "width": 1280,
    },
    "zed2": {
        "fx": 534.3699951171875,
        "fy": 534.47998046875,
        "cx": 477.2049865722656,
        "cy": 262.4590148925781,
        "height": 540,
        "width": 958,  # 960 - 2*azimuth_buff (azimuth_buff = 1 in C++ → 958)
    },
    "cmu_sim": {
        "fx": 205.46963709898583,
        "fy": 205.46963709898583,
        "cx": 320.5,
        "cy": 180.5,
        "height": 360,
        "width": 638,  # 640 - 2*azimuth_buff (azimuth_buff = 1 in C++ → 638)
    },
}

# Hardcoded camera-to-map transform (from C++ code)
CAMERA_TO_MAP = np.array(
    [
        [0.01165962, -0.02415892, 0.99964014, 0.482],
        [-0.99953617, 0.02784553, 0.01233136, 0.04],
        [-0.02813342, -0.99932026, -0.02382304, 0.249],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class DepthProjectionNode(Node):
    def __init__(self):
        super().__init__("depth_projection")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("camera_type", "zed2")
        self.declare_parameter("decay_time", 8.0)

        camera_type = (
            self.get_parameter("camera_type").get_parameter_value().string_value
        )
        self.decay_time = (
            self.get_parameter("decay_time").get_parameter_value().double_value
        )

        if camera_type not in CAMERA_INTRINSICS:
            self.get_logger().fatal(
                f"Unknown camera_type '{camera_type}'. "
                f"Choose from: {list(CAMERA_INTRINSICS.keys())}"
            )
            raise ValueError(f"Invalid camera_type: {camera_type}")

        intr = CAMERA_INTRINSICS[camera_type]
        self.fx = intr["fx"]
        self.fy = intr["fy"]
        self.cx = intr["cx"]
        self.cy = intr["cy"]
        self.img_height = intr["height"]
        self.img_width = intr["width"]

        # Derived FOV (matching C++ computation)
        self.fovy = 2 * math.atan(self.img_height / (2 * self.fy))
        self.fovx = 2 * math.atan(self.img_width / (2 * self.fx))

        # Hardcoded parameters (from C++ globals)
        self.voxel_size = 0.1
        self.no_decay_dis = 5.0  # noDecayDis
        self.min_dis = 1.5  # minDis
        self.clearing_dis = 3.0  # clearingDis
        self.vehicle_height = 0.5  # vehicleHeight
        self.azimuth_buff = 0  # Not used in C++ for D455? Actually 1 for zed2/cmu_sim

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        self.odom_transform = np.eye(4, dtype=np.float64)  # 4x4 odometry transform
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0
        self.sin_roll = 0.0
        self.cos_roll = 1.0
        self.sin_pitch = 0.0
        self.cos_pitch = 1.0
        self.sin_yaw = 0.0
        self.cos_yaw = 1.0

        self.system_init_time = None
        self.depth_cloud_time = 0.0
        self.new_depth_cloud = False

        # Persistent terrain cloud: Nx5 (x, y, z, cost, capture_time) in odom frame
        self.persistent_cloud = np.empty((0, 5), dtype=np.float64)
        # Newly captured cloud this frame: Mx5 (x, y, z, cost, capture_time) in camera frame
        self.pending_points = np.empty((0, 5), dtype=np.float64)

        # Layout dimensions for cost data indexing (from Float32Stamped)
        self._cost_rows = 1
        self._cost_cols = 1
        self._cost_row_stride = 1
        self._cost_col_stride = 1
        self._first_cost_msg = True

        self.cv_bridge = CvBridge()

        # ------------------------------------------------------------
        # Synchronized subscribers via message_filters
        # ------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        depth_sub = message_filters.Subscriber(
            self, Image, "/camera/aligned_depth_to_color/image_raw",
            qos_profile=sensor_qos
        )
        odom_sub = message_filters.Subscriber(
            self, Odometry, "/state_estimation",
            qos_profile=reliable_qos
        )
        cost_sub = message_filters.Subscriber(
            self, Float32Stamped, "/inference/results_stamped_post",
            qos_profile=reliable_qos
        )

        # ApproximateTime synchronizer (matching C++ ApproximateTime policy)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, odom_sub, cost_sub],
            queue_size=10,
            slop=1.5,  # seconds (same as C++ inter-message lower bound)
        )
        self._sync.registerCallback(self.sync_callback)

        # ------------------------------------------------------------
        # Publisher
        # ------------------------------------------------------------
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/depth_projection", reliable_qos
        )

        # ------------------------------------------------------------
        # Main processing timer (200 Hz ≈ 5 ms, matching C++ ros::Rate(200))
        # ------------------------------------------------------------
        self.create_timer(0.005, self.main_loop)

        self.get_logger().info(
            f"Depth projection node initialized "
            f"(camera={camera_type}, "
            f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"fovx={math.degrees(self.fovx):.1f}°, "
            f"fovy={math.degrees(self.fovy):.1f}°)"
        )

    # ------------------------------------------------------------------
    # Synchronized callback
    # ------------------------------------------------------------------
    def sync_callback(
        self, depth_msg: Image, odom_msg: Odometry, cost_msg: Float32Stamped
    ):
        """Called when depth, odometry, and cost messages are approximately aligned."""
        # ---- Odometry ----
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation

        self.vehicle_x = pos.x
        self.vehicle_y = pos.y
        self.vehicle_z = pos.z

        # Convert quaternion to RPY (same tf::Matrix3x3→getRPY as C++)
        roll, pitch, yaw = self._quat_to_rpy(ori.x, ori.y, ori.z, ori.w)
        self.sin_roll, self.cos_roll = math.sin(roll), math.cos(roll)
        self.sin_pitch, self.cos_pitch = math.sin(pitch), math.cos(pitch)
        self.sin_yaw, self.cos_yaw = math.sin(yaw), math.cos(yaw)

        # Build 4x4 odometry transform (world → vehicle, inverted later for camera→world)
        self.odom_transform = self._build_transform(
            pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w
        )

        # ---- Layout dimensions (first cost message only) ----
        if self._first_cost_msg and len(cost_msg.data.layout.dim) >= 2:
            self._cost_rows = cost_msg.data.layout.dim[0].size
            self._cost_cols = cost_msg.data.layout.dim[1].size
            self._cost_row_stride = cost_msg.data.layout.dim[0].stride
            self._cost_col_stride = cost_msg.data.layout.dim[1].stride
            self._first_cost_msg = False
            self.get_logger().info(
                f"Cost layout: {self._cost_rows}x{self._cost_cols} "
                f"(stride {self._cost_row_stride}x{self._cost_col_stride})"
            )

        # ---- Depth image → 3D point cloud ----
        self.depth_cloud_time = self._stamp_to_sec(depth_msg.header.stamp)

        if self.system_init_time is None:
            self.system_init_time = self.depth_cloud_time

        # Convert depth image to numpy
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Build cost array from Float32Stamped
        cost_data = np.array(cost_msg.data.data, dtype=np.float32)
        cost_2d = cost_data.reshape(self._cost_rows, self._cost_cols)

        # Project depth → 3D points (vectorized)
        self.pending_points = self._project_depth_to_points(depth_img, cost_2d)
        self.new_depth_cloud = True

    # ------------------------------------------------------------------
    # Main processing loop (200 Hz)
    # ------------------------------------------------------------------
    def main_loop(self):
        if not self.new_depth_cloud:
            return

        self.new_depth_cloud = False

        if self.system_init_time is None:
            return

        elapsed = self.depth_cloud_time - self.system_init_time

        # ---- Step 1: Filter persistent cloud (decay, FOV, distance) ----
        kept_from_persistent = np.empty((0, 5), dtype=np.float64)

        if self.persistent_cloud.shape[0] > 0:
            pts = self.persistent_cloud  # Nx5: x, y, z, cost, capture_time
            px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
            p_capture_time = pts[:, 4]

            # Translate to vehicle frame
            tx = px - self.vehicle_x
            ty = py - self.vehicle_y
            tz = pz - self.vehicle_z

            # Rotate to vehicle-aligned frame
            rx = self.cos_yaw * tx + self.sin_yaw * ty
            ry = -self.sin_yaw * tx + self.cos_yaw * ty
            rz = self.cos_pitch * tz - self.sin_pitch * rx

            # Planar distance
            dis = np.sqrt(rx * rx + ry * ry)

            # Azimuth & elevation angles
            angle1 = np.arctan2(ry, rx)  # azimuth
            angle2 = np.arctan2(rz, dis)  # elevation

            # Age check
            age = self.depth_cloud_time - p_capture_time

            # Filter conditions (matching C++ logic exactly):
            # Keep if:
            #   (age < decay_time OR dis < clearing_dis)
            #   AND z < vehicle_height
            #   AND ((outside FOV) OR dis < min_dis)
            age_ok = (age < self.decay_time) | (dis < self.clearing_dis)
            z_ok = pz < self.vehicle_height
            outside_fov = (
                np.abs(angle1) > (self.fovx / 2.0) - 8.0 * (math.pi / 180.0)
            ) | (np.abs(angle2) > (self.fovy / 2.0))
            close_enough = dis < self.min_dis
            fov_ok = outside_fov | close_enough

            keep = age_ok & z_ok & fov_ok

            if np.any(keep):
                kept_from_persistent = pts[keep]

        # ---- Step 2: Transform new points camera→map→odom ----
        new_in_odom = np.empty((0, 5), dtype=np.float64)

        if self.pending_points.shape[0] > 0:
            # pending_points: Nx5 in camera frame
            cam_xyz = self.pending_points[:, :3]
            cam_costs = self.pending_points[:, 3]
            cam_time = self.pending_points[:, 4]

            # Camera → Map (hardcoded transform)
            ones = np.ones((cam_xyz.shape[0], 1), dtype=np.float64)
            cam_h = np.hstack([cam_xyz, ones])  # Nx4 homogeneous
            map_h = (CAMERA_TO_MAP @ cam_h.T).T  # Nx4
            map_xyz = map_h[:, :3]

            # Map → Odometry frame
            odom_h = (self.odom_transform @ map_h.T).T  # Nx4
            odom_xyz = odom_h[:, :3]

            # Distance from vehicle
            dx = odom_xyz[:, 0] - self.vehicle_x
            dy = odom_xyz[:, 1] - self.vehicle_y
            dis = np.sqrt(dx * dx + dy * dy)

            # Filter: z < vehicle_z + vehicle_height, beyond min_dis, within no_decay_dis
            z_ok = odom_xyz[:, 2] < (self.vehicle_z + self.vehicle_height)
            dis_ok = (dis > self.min_dis) & (dis < self.no_decay_dis)
            keep = z_ok & dis_ok

            if np.any(keep):
                new_in_odom = np.column_stack(
                    [odom_xyz[keep], cam_costs[keep], cam_time[keep]]
                )

        # ---- Step 3: Combine and voxel filter ----
        combined = np.vstack([kept_from_persistent, new_in_odom])

        if combined.shape[0] == 0:
            self.persistent_cloud = combined
            return

        # Voxel grid filter (replaces PCL VoxelGrid)
        combined = self._voxel_filter(combined, self.voxel_size)

        # Update persistent cloud
        self.persistent_cloud = combined

        # ---- Step 4: Publish PointCloud2 ----
        self._publish_cloud(combined)

    # ------------------------------------------------------------------
    # Depth → 3D projection (vectorized, replaces C++ pixel loops)
    # ------------------------------------------------------------------
    def _project_depth_to_points(self, depth_img: np.ndarray, cost_2d: np.ndarray):
        """
        Project depth image to 3D camera-frame points with cost.
        Handles both 32FC1 (meters) and 16UC1 (millimeters) encodings.
        Returns Nx5 array: x, y, z, cost, capture_time.
        """
        h, w = depth_img.shape

        # Convert depth to meters
        if depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) * 0.001
        else:
            depth_m = depth_img.astype(np.float32)

        # Crop to effective width (accounting for azimuth_buff)
        # C++ loop uses u from azimuth_buff to width-azimuth_buff
        # For D455, azimuth_buff=0 so full width. For zed2/cmu_sim it's 0 in the C++ code too.
        # Actually the C++ uses int azimuth_buff = 0.0 (line 47), so no cropping.
        # The width adjustment happens via intrinsics width value.

        # Use only the valid region (matching C++ loop bounds)
        depth_valid = depth_m[:, :]
        cost_valid = cost_2d[:, :]

        # Compute pixel centers
        vv, uu = np.mgrid[0:h, 0:w]

        # Valid depth mask (>0 meters)
        valid = depth_valid > 0.0

        if not np.any(valid):
            return np.empty((0, 5), dtype=np.float64)

        # Project to 3D
        z = depth_valid[valid]
        x = (uu[valid] - self.cx) / self.fx * z
        y = (vv[valid] - self.cy) / self.fy * z
        costs = cost_valid[valid]
        capture_time = np.full_like(z, self.depth_cloud_time)

        return np.column_stack([x, y, z, costs, capture_time])

    # ------------------------------------------------------------------
    # Voxel grid filter (centroid averaging, replaces PCL VoxelGrid)
    # ------------------------------------------------------------------
    def _voxel_filter(self, points: np.ndarray, voxel_size: float):
        """
        Downsample points by averaging within each voxel cell.
        points: Nx5 (x, y, z, cost, capture_time).
        Returns centroids per occupied voxel (Mx5).
        """
        if points.shape[0] < 2:
            return points

        # Quantize positions to voxel indices
        voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)

        # Find unique voxels
        _, unique_indices, inverse = np.unique(
            voxel_idx, axis=0, return_index=True, return_inverse=True
        )

        n_voxels = len(unique_indices)

        if n_voxels == points.shape[0]:
            return points  # No duplicates

        # Sum per voxel
        result = np.zeros((n_voxels, 5), dtype=np.float64)
        np.add.at(result, inverse, points)

        # Divide by counts for centroid
        counts = np.bincount(inverse, minlength=n_voxels)
        result /= counts[:, np.newaxis]

        return result

    # ------------------------------------------------------------------
    # Publish PointCloud2
    # ------------------------------------------------------------------
    def _publish_cloud(self, cloud: np.ndarray):
        """
        Build and publish a PointCloud2 message.
        cloud: Nx5 (x, y, z, cost, capture_time).
        Output: x, y, z, intensity (=cost, matching C++ behavior).
        Frame: odom.
        """
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        # Fields: x, y, z (float32), intensity (float32 = cost/curvature)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
        ]
        msg.point_step = 16  # 4 fields × 4 bytes
        msg.is_bigendian = False
        msg.is_dense = True
        msg.height = 1
        msg.width = cloud.shape[0]

        # Pack data: [x, y, z, cost] as float32
        data = np.zeros((cloud.shape[0], 4), dtype=np.float32)
        data[:, 0] = cloud[:, 0].astype(np.float32)  # x
        data[:, 1] = cloud[:, 1].astype(np.float32)  # y
        data[:, 2] = cloud[:, 2].astype(np.float32)  # z
        data[:, 3] = cloud[:, 3].astype(np.float32)  # intensity = cost (matching C++)

        msg.row_step = msg.point_step * msg.width
        msg.data = data.tobytes()

        self.cloud_pub.publish(msg)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        """Convert a ROS 2 Time message to seconds (float)."""
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _quat_to_rpy(x, y, z, w):
        """Convert quaternion to roll, pitch, yaw (Euler intrinsic ZYX)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _build_transform(px, py, pz, qx, qy, qz, qw):
        """Build a 4x4 homogeneous transform from position + quaternion."""
        # Rotation matrix from quaternion
        R = np.zeros((3, 3), dtype=np.float64)

        R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        R[0, 1] = 2.0 * (qx * qy - qz * qw)
        R[0, 2] = 2.0 * (qx * qz + qy * qw)

        R[1, 0] = 2.0 * (qx * qy + qz * qw)
        R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        R[1, 2] = 2.0 * (qy * qz - qx * qw)

        R[2, 0] = 2.0 * (qx * qz - qy * qw)
        R[2, 1] = 2.0 * (qy * qz + qx * qw)
        R[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [px, py, pz]

        return T


def main():
    rclpy.init()
    node = DepthProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
