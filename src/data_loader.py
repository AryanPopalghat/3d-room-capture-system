"""
Data loading module for iPhone LiDAR scan data.

Each scan directory contains:
    - camera_matrix.csv   : 3x3 intrinsic matrix (focal length + principal point)
    - odometry.csv        : per-frame 6DoF camera poses + per-frame intrinsics
    - imu.csv             : high-rate accelerometer + gyroscope readings
    - depth/              : per-frame 16-bit PNG depth maps (millimeters)
    - confidence/         : per-frame 8-bit PNG confidence maps (0=low, 1=med, 2=high)
    - rgb.mp4             : RGB video of the capture

The depth maps come from the iPhone's LiDAR scanner at ~256x192 resolution.
They appear black in normal viewers because values use a small range of 16-bit space.
"""

import csv
import os
import numpy as np
import cv2
from pathlib import Path


class CameraIntrinsics:
    """Pinhole camera parameters."""

    def __init__(self, fx, fy, cx, cy):
        self.fx = fx  # horizontal focal length (pixels)
        self.fy = fy  # vertical focal length (pixels)
        self.cx = cx  # principal point x (pixels)
        self.cy = cy  # principal point y (pixels)

    def __repr__(self):
        return f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f})"


class FramePose:
    """
    Camera pose for a single frame.

    Combines the 6DoF position + orientation with per-frame camera intrinsics.
    ARKit updates intrinsics slightly per frame due to autofocus adjustments.
    """

    def __init__(self, timestamp, frame_id, x, y, z, qx, qy, qz, qw,
                 fx, fy, cx, cy):
        self.timestamp = timestamp
        self.frame_id = frame_id

        # Camera position in world coordinates (meters)
        self.position = np.array([x, y, z])

        # Camera orientation as quaternion (Hamilton convention)
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

        # Per-frame intrinsics (may vary slightly due to autofocus)
        self.intrinsics = CameraIntrinsics(fx, fy, cx, cy)


class ScanData:
    """Container holding all loaded data from a single scan directory."""

    def __init__(self, scan_dir):
        self.scan_dir = Path(scan_dir)
        self.scan_id = self.scan_dir.name

        # These get populated by load_scan()
        self.reference_intrinsics = None  # from camera_matrix.csv
        self.frame_poses = []             # list of FramePose
        self.depth_dir = None             # path to depth/ folder
        self.confidence_dir = None        # path to confidence/ folder
        self.rgb_video_path = None        # path to rgb.mp4
        self.num_frames = 0

    def get_depth_path(self, frame_id):
        """Get the file path for a specific frame's depth map."""
        return self.depth_dir / f"{frame_id:06d}.png"

    def get_confidence_path(self, frame_id):
        """Get the file path for a specific frame's confidence map."""
        return self.confidence_dir / f"{frame_id:06d}.png"


def load_camera_matrix(csv_path):
    """
    Load the 3x3 camera intrinsic matrix from camera_matrix.csv.

    The file is a plain 3x3 matrix:
        fx,  0, cx
         0, fy, cy
         0,  0,  1

    Returns CameraIntrinsics with fx, fy, cx, cy extracted.
    """
    matrix = []
    with open(csv_path, 'r') as f:
        for line in f:
            row = [float(v.strip()) for v in line.strip().split(',')]
            matrix.append(row)

    matrix = np.array(matrix)
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    cx = matrix[0, 2]
    cy = matrix[1, 2]

    return CameraIntrinsics(fx, fy, cx, cy)


def load_odometry(csv_path):
    """
    Load per-frame camera poses from odometry.csv.

    Columns: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,
             distortion_center_x, distortion_center_y

    The position (x, y, z) is in meters in a world coordinate system.
    The quaternion (qx, qy, qz, qw) represents camera orientation.
    """
    poses = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row

        for row in reader:
            if len(row) < 13:
                continue

            try:
                timestamp = float(row[0].strip())
                frame_id = int(row[1].strip())
                x = float(row[2].strip())
                y = float(row[3].strip())
                z = float(row[4].strip())
                qx = float(row[5].strip())
                qy = float(row[6].strip())
                qz = float(row[7].strip())
                qw = float(row[8].strip())
                fx = float(row[9].strip())
                fy = float(row[10].strip())
                cx = float(row[11].strip())
                cy = float(row[12].strip())

                pose = FramePose(timestamp, frame_id, x, y, z,
                                 qx, qy, qz, qw, fx, fy, cx, cy)
                poses.append(pose)
            except (ValueError, IndexError) as e:
                print(f"  Warning: skipping malformed odometry row: {e}")
                continue

    return poses


def load_depth_frame(depth_path):
    """
    Load a single depth map from a 16-bit PNG file.

    ARKit LiDAR depth maps are stored as 16-bit unsigned integers.
    The values represent depth in millimeters, which we convert to meters.

    The images look completely black in normal viewers because the depth
    values (typically 500-5000) only occupy a tiny fraction of the
    16-bit range (0-65535). This is normal and expected.

    Returns:
        HxW numpy array with depth values in meters.
        Returns None if the file doesn't exist or can't be loaded.
    """
    if not os.path.exists(depth_path):
        return None

    # cv2.IMREAD_UNCHANGED preserves the 16-bit depth values
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        return None

    # Convert from millimeters to meters
    depth_meters = depth_raw.astype(np.float32) / 1000.0

    return depth_meters


def load_confidence_frame(confidence_path):
    """
    Load a confidence map from an 8-bit PNG file.

    ARKit confidence levels:
        0 = low confidence (sensor uncertain about this pixel)
        1 = medium confidence
        2 = high confidence

    We typically filter out low-confidence pixels (keep >= 1).
    """
    if not os.path.exists(confidence_path):
        return None

    conf = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
    return conf


def load_scan(scan_dir):
    """
    Load all data from a scan directory into a ScanData object.

    This doesn't load the actual depth/confidence images into memory —
    those are loaded on-demand during point cloud construction to avoid
    running out of RAM on large scans.

    Args:
        scan_dir: path to directory containing the scan data
                  (e.g., 'data/single_room/c00a170fe1/')

    Returns:
        ScanData object with all metadata loaded
    """
    scan_dir = Path(scan_dir)
    print(f"\nLoading scan from: {scan_dir.name}")

    scan = ScanData(scan_dir)

    # Load camera intrinsics
    cam_matrix_path = scan_dir / "camera_matrix.csv"
    if cam_matrix_path.exists():
        scan.reference_intrinsics = load_camera_matrix(cam_matrix_path)
        print(f"  Camera intrinsics: {scan.reference_intrinsics}")
    else:
        print("  Warning: camera_matrix.csv not found")

    # Load odometry (camera poses for each frame)
    odom_path = scan_dir / "odometry.csv"
    if odom_path.exists():
        scan.frame_poses = load_odometry(odom_path)
        scan.num_frames = len(scan.frame_poses)
        print(f"  Loaded {scan.num_frames} frame poses")
    else:
        print("  Warning: odometry.csv not found")

    # Set up paths for depth and confidence directories
    scan.depth_dir = scan_dir / "depth"
    scan.confidence_dir = scan_dir / "confidence"
    scan.rgb_video_path = scan_dir / "rgb.mp4"

    # Verify directories exist
    depth_count = len(list(scan.depth_dir.glob("*.png"))) if scan.depth_dir.exists() else 0
    conf_count = len(list(scan.confidence_dir.glob("*.png"))) if scan.confidence_dir.exists() else 0
    print(f"  Depth maps: {depth_count}")
    print(f"  Confidence maps: {conf_count}")
    print(f"  RGB video: {'yes' if scan.rgb_video_path.exists() else 'no'}")

    return scan


def extract_video_frames(video_path, output_dir=None, max_frames=None, frame_step=30):
    """
    Extract individual frames from the RGB video.

    Useful for damage detection and visual analysis.
    By default extracts every 30th frame (roughly 1 per second at 30fps).

    Args:
        video_path: path to rgb.mp4
        output_dir: where to save extracted frames (None = return in memory)
        max_frames: maximum number of frames to extract
        frame_step: extract every Nth frame

    Returns:
        list of (frame_index, numpy_array) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames at {fps:.1f} fps ({total_frames/fps:.1f}s)")

    frames = []
    frame_idx = 0
    extracted = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            if output_dir:
                out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)

            frames.append((frame_idx, frame))
            extracted += 1

            if max_frames and extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"  Extracted {extracted} frames from video")
    return frames
