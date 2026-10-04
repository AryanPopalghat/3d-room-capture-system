"""
Plane detection using RANSAC on 3D point clouds.

RANSAC (Random Sample Consensus) works by:
    1. Pick 3 random points → define a plane
    2. Count how many other points are close to this plane (inliers)
    3. Repeat many times, keep the plane with the most inliers
    4. Remove those inlier points, repeat for next plane

We classify detected planes by their normal direction:
    - Horizontal planes (normal pointing up/down) → floor or ceiling
    - Vertical planes (normal pointing sideways) → walls

This is robust to furniture, clutter, and noise because RANSAC
naturally handles outliers - it only needs enough of the surface
to be visible to detect the underlying plane.
"""

import numpy as np
import open3d as o3d

from src.utils import angle_between_normals


class DetectedPlane:
    """A plane detected in the point cloud."""

    def __init__(self, normal, d, inlier_points, inlier_indices):
        self.normal = normal          # unit normal vector [nx, ny, nz]
        self.d = d                    # plane offset (ax + by + cz + d = 0)
        self.inlier_points = inlier_points  # Nx3 array of points on this plane
        self.inlier_indices = inlier_indices
        self.label = None             # 'floor', 'ceiling', 'wall'
        self.centroid = np.mean(inlier_points, axis=0)

    @property
    def num_points(self):
        return len(self.inlier_points)

    def __repr__(self):
        return (f"DetectedPlane(label={self.label}, points={self.num_points}, "
                f"normal=[{self.normal[0]:.2f}, {self.normal[1]:.2f}, {self.normal[2]:.2f}])")


def detect_planes_ransac(pcd, max_planes=10, distance_threshold=0.02,
                          min_points_ratio=0.02, ransac_n=3, num_iterations=2000):
    """
    Detect multiple planes in a point cloud using iterative RANSAC.

    After finding each plane, its inlier points are removed from the cloud
    and RANSAC runs again on the remainder. This continues until we've found
    max_planes or the remaining cloud is too small.

    Args:
        pcd: Open3D point cloud
        max_planes: maximum number of planes to detect
        distance_threshold: max distance (meters) for a point to be an inlier
                           0.02 = 2cm, good for indoor walls/floors
        min_points_ratio: stop when remaining points < this fraction of original
        ransac_n: number of points to sample for each RANSAC iteration (3 for plane)
        num_iterations: RANSAC iterations per plane (more = slower but more reliable)

    Returns:
        list of DetectedPlane objects, sorted by number of inlier points (largest first)
    """
    total_points = len(pcd.points)
    min_points = int(total_points * min_points_ratio)
    remaining_pcd = pcd
    planes = []

    print(f"\nDetecting planes (RANSAC, {total_points:,} points)...")

    for i in range(max_planes):
        if len(remaining_pcd.points) < min_points:
            print(f"  Stopping: only {len(remaining_pcd.points)} points remaining")
            break

        # Run RANSAC to find the best plane in the remaining points
        plane_model, inlier_indices = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        if len(inlier_indices) < min_points:
            print(f"  Stopping: plane {i+1} had only {len(inlier_indices)} inliers")
            break

        # Extract the plane parameters: ax + by + cz + d = 0
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal_len = np.linalg.norm(normal)
        if normal_len > 0:
            normal = normal / normal_len

        # Get the inlier points
        inlier_points = np.asarray(remaining_pcd.points)[inlier_indices]

        plane = DetectedPlane(normal, d, inlier_points, inlier_indices)
        planes.append(plane)

        coverage = len(inlier_indices) / total_points * 100
        print(f"  Plane {i+1}: {len(inlier_indices):,} points ({coverage:.1f}%), "
              f"normal=[{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]")

        # Remove the found plane's points from the cloud for the next iteration
        remaining_pcd = remaining_pcd.select_by_index(inlier_indices, invert=True)

    print(f"  Total planes found: {len(planes)}")
    return planes


def classify_planes(planes, vertical_threshold=20.0):
    """
    Label each detected plane as 'floor', 'ceiling', or 'wall' based on its
    normal vector orientation.

    Classification logic:
    - If normal is close to vertical (within threshold degrees of Y-axis):
        → horizontal plane → floor or ceiling
        → floor = lowest horizontal plane, ceiling = highest
    - If normal is close to horizontal:
        → wall

    ARKit coordinate system: Y is up, X is right, Z is towards the camera.
    So floor/ceiling normals should be close to [0, ±1, 0].

    Args:
        planes: list of DetectedPlane objects
        vertical_threshold: max angle (degrees) from vertical to classify as floor/ceiling

    Returns:
        dict with 'floor', 'ceiling', 'walls' keys
    """
    up_vector = np.array([0, 1, 0])  # ARKit Y-up convention

    horizontal_planes = []
    wall_planes = []

    for plane in planes:
        angle_to_vertical = angle_between_normals(plane.normal, up_vector)

        if angle_to_vertical < vertical_threshold:
            # Normal points up or down → horizontal surface
            horizontal_planes.append(plane)
            plane.label = 'horizontal'
        elif angle_to_vertical > (90 - vertical_threshold):
            # Normal is perpendicular to up → vertical surface (wall)
            wall_planes.append(plane)
            plane.label = 'wall'
        else:
            # Angled surface - could be a ramp or sloped ceiling
            # Treat as wall for now
            wall_planes.append(plane)
            plane.label = 'wall'

    # Among horizontal planes, classify floor vs ceiling by height (Y position)
    # Floor is the lowest, ceiling is the highest
    classified = {
        'floor': None,
        'ceiling': None,
        'walls': wall_planes
    }

    if horizontal_planes:
        # Sort by average Y position of inlier points
        horizontal_planes.sort(key=lambda p: p.centroid[1])

        classified['floor'] = horizontal_planes[0]
        classified['floor'].label = 'floor'

        if len(horizontal_planes) >= 2:
            candidate_ceiling = horizontal_planes[-1]
            # Ensure it's actually a ceiling and not just a table/bed
            # A typical room is at least 2.0m tall
            floor_y = classified['floor'].centroid[1]
            candidate_y = candidate_ceiling.centroid[1]
            
            if (candidate_y - floor_y) >= 2.0:
                classified['ceiling'] = candidate_ceiling
                classified['ceiling'].label = 'ceiling'
            else:
                print(f"  Note: Highest horizontal plane is only {candidate_y - floor_y:.2f}m above floor. "
                      f"Likely furniture, not a ceiling.")

    # Print classification summary
    print(f"\nPlane classification:")
    if classified['floor']:
        y = classified['floor'].centroid[1]
        print(f"  Floor: {classified['floor'].num_points:,} points at Y={y:.2f}m")
    else:
        print(f"  Floor: not detected")

    if classified['ceiling']:
        y = classified['ceiling'].centroid[1]
        print(f"  Ceiling: {classified['ceiling'].num_points:,} points at Y={y:.2f}m")
    else:
        print(f"  Ceiling: not detected")

    print(f"  Walls: {len(classified['walls'])} planes detected")
    for i, wall in enumerate(classified['walls']):
        n = wall.normal
        print(f"    Wall {i+1}: {wall.num_points:,} points, "
              f"normal=[{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]")

    return classified
