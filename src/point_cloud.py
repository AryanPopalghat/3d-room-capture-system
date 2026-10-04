"""
Point cloud construction from depth maps + camera poses.

Takes per-frame depth maps, applies camera intrinsics to unproject them
into 3D space, then uses the odometry poses to place each frame's points
in a shared world coordinate system. The result is a dense 3D point cloud
of the scanned room.

Processing pipeline:
    1. For each selected frame:
       a. Load depth map (16-bit PNG → meters)
       b. Load confidence map, filter low-confidence pixels
       c. Unproject 2D depth pixels → 3D points in camera space
       d. Transform camera-space points → world space using pose
    2. Merge all frames into one point cloud
    3. Downsample with voxel grid to reduce density
    4. Remove statistical outliers (noise from reflections, glass, etc.)
"""

import numpy as np
import open3d as o3d

from src.data_loader import ScanData, load_depth_frame, load_confidence_frame
from src.utils import quaternion_to_rotation_matrix, unproject_depth_to_3d, transform_points_to_world


def build_point_cloud_from_scan(scan_data, frame_step=5, confidence_threshold=1,
                                 max_depth=6.0, voxel_size=0.02):
    """
    Build a complete 3D point cloud from a scan's depth data.

    This is the main entry point for point cloud construction.

    Args:
        scan_data: ScanData object with loaded metadata
        frame_step: process every Nth frame (5 = every 5th frame)
                    higher = faster but less dense coverage
        confidence_threshold: minimum ARKit confidence level to keep
                              0=all, 1=medium+high, 2=high only
        max_depth: ignore depth readings beyond this distance (meters)
                   LiDAR gets noisy at range, 6m is a safe cutoff for rooms
        voxel_size: grid cell size for downsampling (meters)
                    0.02 = 2cm resolution, good balance of detail vs performance

    Returns:
        open3d.geometry.PointCloud in world coordinates
    """
    all_points = []
    frames_processed = 0
    frames_to_process = list(range(0, scan_data.num_frames, frame_step))
    total = len(frames_to_process)

    print(f"\nBuilding point cloud ({total} frames, step={frame_step})...")

    for i, frame_idx in enumerate(frames_to_process):
        pose = scan_data.frame_poses[frame_idx]

        # Load this frame's depth map
        depth_path = scan_data.get_depth_path(pose.frame_id)
        depth_map = load_depth_frame(depth_path)
        if depth_map is None:
            continue

        # Load and apply confidence filtering
        conf_path = scan_data.get_confidence_path(pose.frame_id)
        conf_map = load_confidence_frame(conf_path)

        if conf_map is not None:
            # The confidence map might be a different resolution than depth
            # Resize confidence to match depth if needed
            if conf_map.shape != depth_map.shape:
                import cv2
                conf_map = cv2.resize(conf_map, (depth_map.shape[1], depth_map.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # Zero out low-confidence depth pixels
            depth_map[conf_map < confidence_threshold] = 0

        # Clamp unreasonably far depth readings
        depth_map[depth_map > max_depth] = 0

        # Use per-frame intrinsics from odometry (more accurate than reference)
        fx = pose.intrinsics.fx
        fy = pose.intrinsics.fy
        cx = pose.intrinsics.cx
        cy = pose.intrinsics.cy

        # Scale intrinsics to match depth map resolution
        # RGB intrinsics are for ~1920x1440, depth maps are ~256x192
        depth_h, depth_w = depth_map.shape
        # The reference intrinsics correspond to the full RGB resolution
        # We need to scale them down to the depth map resolution
        rgb_w = cx * 2  # approximate: cx is roughly half of image width
        scale = depth_w / rgb_w if rgb_w > 0 else 1.0

        fx_scaled = fx * scale
        fy_scaled = fy * scale
        cx_scaled = cx * scale
        cy_scaled = cy * scale

        # Unproject depth pixels to 3D points in camera coordinate frame
        points_camera = unproject_depth_to_3d(depth_map, fx_scaled, fy_scaled,
                                               cx_scaled, cy_scaled)

        if len(points_camera) == 0:
            continue

        # Build the camera-to-world transform from the odometry pose
        rotation = quaternion_to_rotation_matrix(pose.qx, pose.qy, pose.qz, pose.qw)
        translation = pose.position

        # Move points from camera space into world space
        points_world = transform_points_to_world(points_camera, rotation, translation)
        all_points.append(points_world)
        frames_processed += 1

        # Progress reporting
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pts_so_far = sum(len(p) for p in all_points)
            print(f"  Frame {i+1}/{total} — {pts_so_far:,} points accumulated")

    if not all_points:
        print("  Error: no valid depth frames found!")
        return o3d.geometry.PointCloud()

    # Merge all frame point clouds into one
    merged_points = np.vstack(all_points)
    print(f"  Raw point cloud: {len(merged_points):,} points from {frames_processed} frames")

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)

    # Downsample to a uniform density using a voxel grid
    # This replaces all points within each 2cm cube with their centroid
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  After voxel downsampling ({voxel_size*100:.0f}cm): {len(pcd.points):,} points")

    # Remove statistical outliers — points that are far from their neighbors
    # are likely noise from reflective surfaces, glass, or sensor errors
    pcd = remove_outliers(pcd)

    return pcd


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove points that are statistical outliers based on distance to neighbors.

    Points whose average distance to their k nearest neighbors is more than
    std_ratio standard deviations above the mean are removed.

    This handles noise from:
    - Reflective surfaces (mirrors, glossy floors)
    - Glass (depth readings through windows)
    - Edge artifacts at object boundaries
    """
    if len(pcd.points) < nb_neighbors:
        return pcd

    cleaned, mask = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    removed = len(pcd.points) - len(cleaned.points)
    print(f"  Outlier removal: removed {removed:,} noisy points ({removed/len(pcd.points)*100:.1f}%)")

    return cleaned


def save_point_cloud(pcd, output_path):
    """Save point cloud to PLY file for visualization in MeshLab etc."""
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"  Saved point cloud: {output_path}")


def estimate_point_cloud_bounds(pcd):
    """
    Get the axis-aligned bounding box of the point cloud.

    Returns dict with min/max coordinates and room extent in each axis.
    Useful for sanity checking — a typical room is 3-8m in XZ and 2.4-3m in Y.
    """
    points = np.asarray(pcd.points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins

    bounds = {
        'min': {'x': mins[0], 'y': mins[1], 'z': mins[2]},
        'max': {'x': maxs[0], 'y': maxs[1], 'z': maxs[2]},
        'extent': {'x': extents[0], 'y': extents[1], 'z': extents[2]},
    }

    print(f"\n  Point cloud bounds:")
    print(f"    X: {mins[0]:.2f} to {maxs[0]:.2f} ({extents[0]:.2f}m)")
    print(f"    Y: {mins[1]:.2f} to {maxs[1]:.2f} ({extents[1]:.2f}m)")
    print(f"    Z: {mins[2]:.2f} to {maxs[2]:.2f} ({extents[2]:.2f}m)")

    return bounds
