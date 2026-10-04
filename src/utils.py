"""
Utility functions for 3D math operations.
Handles coordinate transforms, quaternion math, and depth unprojection.
"""

import numpy as np


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.

    The quaternion format follows the (qx, qy, qz, qw) convention
    that ARKit uses for camera pose output.

    Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    """
    # Normalize to be safe - ARKit quats should already be unit length
    # but floating point drift happens
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # Pre-compute repeated products
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rotation = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
    ])

    return rotation


def unproject_depth_to_3d(depth_map, fx, fy, cx, cy):
    """
    Convert a 2D depth map into 3D points in the camera's coordinate frame.

    Each pixel (u, v) with depth d gets mapped to:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    This is the standard pinhole camera model inversion.

    Args:
        depth_map: HxW numpy array with depth values in meters
        fx, fy: focal lengths in pixels
        cx, cy: principal point (image center) in pixels

    Returns:
        Nx3 array of 3D points in camera space, only for valid (non-zero) depths
    """
    h, w = depth_map.shape

    # Build pixel coordinate grids
    u_coords = np.arange(w)
    v_coords = np.arange(h)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # Only keep pixels that have valid depth readings
    valid_mask = depth_map > 0.01  # ignore near-zero depth (invalid)

    u_valid = u_grid[valid_mask].astype(np.float64)
    v_valid = v_grid[valid_mask].astype(np.float64)
    d_valid = depth_map[valid_mask].astype(np.float64)

    # Unproject using pinhole model
    x = (u_valid - cx) * d_valid / fx
    y = (v_valid - cy) * d_valid / fy
    z = d_valid

    points = np.stack([x, y, z], axis=-1)
    return points


def transform_points_to_world(points, rotation, translation):
    """
    Apply a rigid body transform to move points from camera space to world space.

    world_point = R @ camera_point + t

    Args:
        points: Nx3 array of 3D points
        rotation: 3x3 rotation matrix
        translation: length-3 translation vector

    Returns:
        Nx3 array of transformed points
    """
    return (rotation @ points.T).T + translation


def compute_plane_from_points(points):
    """
    Fit a plane ax + by + cz + d = 0 to a set of 3D points using SVD.

    Returns the plane normal (a, b, c) and offset d.
    The normal is oriented so that d is positive (convention).
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find the direction of least variance = plane normal
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # last row of V^T = smallest singular value direction

    # Plane offset: n . centroid = -d
    d = -np.dot(normal, centroid)

    # Convention: make d positive
    if d < 0:
        normal = -normal
        d = -d

    return normal, d, centroid


def angle_between_normals(n1, n2):
    """Angle in degrees between two unit normal vectors."""
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(cos_angle)))


def project_points_to_2d(points_3d, up_axis=1):
    """
    Project 3D points onto a 2D horizontal plane by dropping the vertical axis.

    For room scanning, the vertical axis is typically Y (ARKit convention)
    so we project onto the XZ plane to get a top-down floor plan view.

    Args:
        points_3d: Nx3 array
        up_axis: which axis is vertical (0=X, 1=Y, 2=Z)

    Returns:
        Nx2 array of 2D coordinates
    """
    axes = [i for i in range(3) if i != up_axis]
    return points_3d[:, axes]
