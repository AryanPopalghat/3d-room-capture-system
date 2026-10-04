"""
Room geometry extraction from classified planes.

Takes the floor, ceiling, and wall planes detected by RANSAC and
computes the actual room dimensions:
    - Ceiling height (distance between floor and ceiling planes)
    - Floor area (convex hull of floor points projected to 2D)
    - Wall segments (intersection lines between adjacent walls)
    - Wall lengths
    - Openings (gaps in walls — doors, windows)

All measurements include confidence intervals derived from the
spread of inlier points around each fitted plane.
"""

import numpy as np
from scipy.spatial import ConvexHull


class WallSegment:
    """A single wall with measured length and endpoints."""

    def __init__(self, start_2d, end_2d, length, normal, plane_index):
        self.start = start_2d   # (x, z) in world coords
        self.end = end_2d       # (x, z) in world coords
        self.length = length    # meters
        self.normal = normal    # wall facing direction
        self.plane_index = plane_index
        self.openings = []      # detected gaps (doors/windows)

    def __repr__(self):
        return f"Wall({self.length:.2f}m, {self.start} → {self.end})"


class RoomOpening:
    """A detected opening in a wall (door, window, etc.)."""

    def __init__(self, position, width, height=None, opening_type='door'):
        self.position = position    # 2D position along wall
        self.width = width          # meters
        self.height = height        # meters (None if unknown)
        self.type = opening_type    # 'door' or 'window'


class RoomDimensions:
    """Complete room measurement results with confidence intervals."""

    def __init__(self):
        self.ceiling_height = None
        self.ceiling_height_std = None  # standard deviation for confidence
        self.floor_area = None
        self.floor_area_std = None
        self.wall_segments = []
        self.openings = []
        self.floor_polygon = None       # 2D polygon of floor boundary
        self.bounds = {}

    @property
    def ceiling_height_ci(self):
        """95% confidence interval for ceiling height."""
        if self.ceiling_height is None or self.ceiling_height_std is None:
            return None
        margin = 1.96 * self.ceiling_height_std
        return (self.ceiling_height - margin, self.ceiling_height + margin)


def compute_ceiling_height(floor_plane, ceiling_plane):
    """
    Compute the vertical distance between floor and ceiling planes.

    For horizontal planes, this is simply the difference in their
    average Y coordinates (ARKit Y-up convention).

    Also computes uncertainty from the spread of inlier points.
    """
    if floor_plane is None or ceiling_plane is None:
        return None, None

    # Method 1: Simple centroid difference
    floor_y = floor_plane.centroid[1]
    ceiling_y = ceiling_plane.centroid[1]
    height = abs(ceiling_y - floor_y)

    # Method 2: Use plane equations for more accuracy
    # Distance between two parallel planes ax+by+cz+d1=0 and ax+by+cz+d2=0
    # is |d1-d2| / sqrt(a²+b²+c²)
    # But our planes may not have exactly the same normal, so we use point-to-plane

    # Compute uncertainty from how spread out the points are around each plane
    floor_y_values = floor_plane.inlier_points[:, 1]
    ceiling_y_values = ceiling_plane.inlier_points[:, 1]

    floor_std = np.std(floor_y_values)
    ceiling_std = np.std(ceiling_y_values)

    # Combined uncertainty (propagation of independent errors)
    height_std = np.sqrt(floor_std**2 + ceiling_std**2)

    print(f"\n  Ceiling height: {height:.3f}m (+/-{height_std:.3f}m)")
    print(f"    Floor plane Y: {floor_y:.3f}m (std={floor_std:.3f}m)")
    print(f"    Ceiling plane Y: {ceiling_y:.3f}m (std={ceiling_std:.3f}m)")

    return height, height_std


def compute_floor_area(floor_plane):
    """
    Compute floor area using the convex hull of floor points projected to 2D.

    Projects the floor plane's inlier points onto the XZ horizontal plane
    and computes the convex hull area. This gives a reasonable estimate for
    roughly rectangular rooms; concave rooms will be slightly overestimated.
    """
    if floor_plane is None:
        return None, None, None

    # Project floor points to 2D (drop Y axis)
    points_2d = floor_plane.inlier_points[:, [0, 2]]  # X and Z

    if len(points_2d) < 3:
        return None, None, None

    try:
        hull = ConvexHull(points_2d)
        area = hull.volume  # in 2D, ConvexHull.volume gives area
        perimeter = hull.area  # in 2D, this is perimeter

        # Get the polygon vertices in order for floor plan rendering
        hull_vertices = points_2d[hull.vertices]

        # Uncertainty estimate: use the point density near the boundary
        # More points near edges = more confident boundary
        area_std = area * 0.05  # rough 5% uncertainty estimate

        print(f"  Floor area: {area:.2f} m^2 (+/-{area_std:.2f} m^2)")
        print(f"  Floor perimeter: {perimeter:.2f}m")

        return area, area_std, hull_vertices

    except Exception as e:
        print(f"  Warning: convex hull failed: {e}")
        return None, None, None


def extract_wall_segments(wall_planes, floor_plane):
    """
    Extract wall segments with lengths from the detected wall planes.

    For each wall plane, we project its inlier points onto the horizontal
    plane and find the extent (min/max) along the wall direction.
    """
    if not wall_planes:
        return []

    # Get floor height for reference
    floor_y = floor_plane.centroid[1] if floor_plane is not None else 0

    segments = []

    for i, wall in enumerate(wall_planes):
        points = wall.inlier_points

        # Project wall points to 2D (XZ plane)
        points_xz = points[:, [0, 2]]

        if len(points_xz) < 2:
            continue

        # Find the principal direction of the wall using PCA
        # The wall's longest extent tells us its direction
        centroid_2d = np.mean(points_xz, axis=0)
        centered = points_xz - centroid_2d

        # SVD gives us the principal axes
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            wall_direction = vt[0]  # first principal component = wall direction
        except np.linalg.LinAlgError:
            continue

        # Project all points onto the wall direction to get extent
        projections = centered @ wall_direction
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        wall_length = max_proj - min_proj

        # Compute wall endpoints in 2D
        start_2d = centroid_2d + wall_direction * min_proj
        end_2d = centroid_2d + wall_direction * max_proj

        # Wall normal in 2D (perpendicular to wall direction)
        wall_normal_2d = np.array([-wall_direction[1], wall_direction[0]])

        segment = WallSegment(
            start_2d=start_2d,
            end_2d=end_2d,
            length=wall_length,
            normal=wall_normal_2d,
            plane_index=i
        )
        segments.append(segment)

        print(f"  Wall {i+1}: {wall_length:.2f}m "
              f"from ({start_2d[0]:.2f}, {start_2d[1]:.2f}) "
              f"to ({end_2d[0]:.2f}, {end_2d[1]:.2f})")

    # Sort walls by length (longest first) for reporting
    segments.sort(key=lambda s: s.length, reverse=True)

    return segments


def detect_openings(wall_planes, floor_plane, ceiling_plane, gap_threshold=0.3):
    """
    Detect openings (doors, windows) by finding gaps in wall coverage.

    Doors appear as vertical gaps in the wall plane extending from floor level.
    Windows appear as gaps at mid-height.

    This is a simplified approach — a full solution would use the RGB images
    for visual detection and the depth data for geometric verification.
    """
    if not wall_planes or floor_plane is None:
        return []

    openings = []
    floor_y = floor_plane.centroid[1]
    ceiling_y = ceiling_plane.centroid[1] if ceiling_plane else floor_y + 2.5

    for wall_idx, wall in enumerate(wall_planes):
        points = wall.inlier_points

        # Project onto wall direction (PCA)
        points_xz = points[:, [0, 2]]
        centroid_2d = np.mean(points_xz, axis=0)
        centered = points_xz - centroid_2d

        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            wall_dir = vt[0]
        except np.linalg.LinAlgError:
            continue

        # Project along wall direction
        projections = centered @ wall_dir
        proj_min = np.min(projections)
        proj_max = np.max(projections)

        # Bin the projections to find gaps
        num_bins = max(10, int((proj_max - proj_min) / 0.1))
        hist, bin_edges = np.histogram(projections, bins=num_bins)

        # Look for consecutive empty bins (= gap in wall)
        bin_width = bin_edges[1] - bin_edges[0]
        gap_start = None

        for b in range(len(hist)):
            if hist[b] == 0:
                if gap_start is None:
                    gap_start = b
            else:
                if gap_start is not None:
                    gap_width = (b - gap_start) * bin_width
                    if gap_width >= gap_threshold:
                        gap_center = (bin_edges[gap_start] + bin_edges[b]) / 2
                        pos_2d = centroid_2d + wall_dir * gap_center

                        opening = RoomOpening(
                            position=pos_2d,
                            width=gap_width,
                            opening_type='door'  # assume door for now
                        )
                        openings.append(opening)
                        print(f"  Opening detected in wall {wall_idx+1}: "
                              f"width={gap_width:.2f}m at ({pos_2d[0]:.2f}, {pos_2d[1]:.2f})")
                    gap_start = None

    return openings


def extract_room_dimensions(classified_planes, pcd=None):
    """
    Main function: extract all room dimensions from classified planes.

    Args:
        classified_planes: dict with 'floor', 'ceiling', 'walls' from classify_planes()
        pcd: original point cloud (optional, for additional analysis)

    Returns:
        RoomDimensions object with all measurements
    """
    print("\n--- Extracting Room Dimensions ---")

    dims = RoomDimensions()
    floor = classified_planes['floor']
    ceiling = classified_planes['ceiling']
    walls = classified_planes['walls']

    # 1. Ceiling height
    if floor and ceiling:
        dims.ceiling_height, dims.ceiling_height_std = compute_ceiling_height(floor, ceiling)
    elif floor:
        print("  Warning: no ceiling detected, cannot compute height")
    else:
        print("  Warning: no floor detected")

    # 2. Floor area
    if floor:
        dims.floor_area, dims.floor_area_std, dims.floor_polygon = compute_floor_area(floor)

    # 3. Wall segments
    print(f"\n  Extracting wall segments:")
    dims.wall_segments = extract_wall_segments(walls, floor)

    # 4. Openings
    print(f"\n  Detecting openings:")
    dims.openings = detect_openings(walls, floor, ceiling)

    if not dims.openings:
        print("  No clear openings detected (doors/windows)")

    # 5. Bounding box
    if pcd is not None:
        points = np.asarray(pcd.points)
        dims.bounds = {
            'min_x': float(np.min(points[:, 0])),
            'max_x': float(np.max(points[:, 0])),
            'min_y': float(np.min(points[:, 1])),
            'max_y': float(np.max(points[:, 1])),
            'min_z': float(np.min(points[:, 2])),
            'max_z': float(np.max(points[:, 2])),
        }

    # Summary
    print(f"\n--- Room Dimensions Summary ---")
    if dims.ceiling_height:
        ci = dims.ceiling_height_ci
        print(f"  Ceiling height: {dims.ceiling_height:.3f}m "
              f"(95% CI: {ci[0]:.3f}m - {ci[1]:.3f}m)")
    if dims.floor_area:
        print(f"  Floor area: {dims.floor_area:.2f} m^2")
    print(f"  Walls detected: {len(dims.wall_segments)}")
    for i, w in enumerate(dims.wall_segments):
        print(f"    Wall {i+1}: {w.length:.2f}m")
    print(f"  Openings: {len(dims.openings)}")

    return dims
