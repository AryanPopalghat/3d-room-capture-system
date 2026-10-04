#!/usr/bin/env python3
"""
3D Room Capture Pipeline
========================

Reconstructs room geometry from iPhone LiDAR scan data.
Produces dimensioned floor plans, room measurements, and damage reports.

Usage:
    python run_pipeline.py --input data/single_room/c00a170fe1
    python run_pipeline.py --input data/single_room/c00a170fe1 --output output/single_room
    python run_pipeline.py --input data/single_room/c00a170fe1 --frame-step 10 --fast

The input directory should contain:
    camera_matrix.csv, odometry.csv, imu.csv, depth/, confidence/, rgb.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path so we can import src modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_scan
from src.point_cloud import (
    build_point_cloud_from_scan,
    save_point_cloud,
    estimate_point_cloud_bounds,
)
from src.plane_detection import detect_planes_ransac, classify_planes
from src.geometry import extract_room_dimensions
from src.floor_plan import render_floor_plan, render_point_cloud_topdown
from src.output_schema import build_output_json, save_output_json, print_output_summary
from src.damage_detector import detect_damage_in_video


def run_pipeline(input_dir, output_dir, frame_step=5, confidence_threshold=1,
                 voxel_size=0.02, max_depth=6.0, skip_damage=False):
    """
    Run the complete room capture pipeline on a scan directory.

    Pipeline steps:
        1. Load scan data (camera intrinsics, odometry poses, paths)
        2. Build 3D point cloud from depth maps + poses
        3. Detect planes using RANSAC (floor, ceiling, walls)
        4. Extract room dimensions (heights, areas, wall lengths)
        5. Detect damage from RGB video
        6. Render floor plan
        7. Export JSON results

    Returns:
        dict with the output JSON structure
    """
    start_time = time.time()

    print("="*60)
    print("  3D ROOM CAPTURE PIPELINE")
    print("="*60)
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Settings: frame_step={frame_step}, voxel={voxel_size}m, "
          f"confidence>={confidence_threshold}")
    print("="*60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load scan data ──
    print("\n[1/7] Loading scan data...")
    scan_data = load_scan(input_dir)

    if scan_data.num_frames == 0:
        print("ERROR: No frames found in scan data!")
        return None

    # ── Step 2: Build 3D point cloud ──
    print("\n[2/7] Building 3D point cloud...")
    pcd = build_point_cloud_from_scan(
        scan_data,
        frame_step=frame_step,
        confidence_threshold=confidence_threshold,
        max_depth=max_depth,
        voxel_size=voxel_size,
    )

    if len(pcd.points) == 0:
        print("ERROR: Point cloud is empty - check depth map format!")
        return None

    bounds = estimate_point_cloud_bounds(pcd)

    # Save the point cloud for external visualization
    ply_path = output_dir / "point_cloud.ply"
    save_point_cloud(pcd, ply_path)

    # Render a top-down debug view
    topdown_path = output_dir / "point_cloud_topdown.png"
    render_point_cloud_topdown(pcd, topdown_path,
                                title=f"Point Cloud - {scan_data.scan_id}")

    # ── Step 3: Detect planes ──
    print("\n[3/7] Detecting planes (RANSAC)...")
    planes = detect_planes_ransac(pcd, max_planes=10, distance_threshold=0.03)

    # ── Step 4: Classify planes ──
    print("\n[4/7] Classifying planes (floor/ceiling/walls)...")
    classified = classify_planes(planes)

    # ── Step 5: Extract dimensions ──
    print("\n[5/7] Extracting room dimensions...")
    room_dims = extract_room_dimensions(classified, pcd)

    # ── Step 6: Damage detection ──
    damage_regions = []
    if not skip_damage and scan_data.rgb_video_path and scan_data.rgb_video_path.exists():
        print("\n[6/7] Running damage detection...")
        damage_regions = detect_damage_in_video(
            scan_data.rgb_video_path,
            max_frames=15,
            frame_step=60
        )
    else:
        print("\n[6/7] Skipping damage detection")

    # ── Step 7: Generate outputs ──
    print("\n[7/7] Generating outputs...")

    elapsed = time.time() - start_time

    # Build JSON output
    output_json = build_output_json(room_dims, scan_data, processing_time=elapsed)

    # Add damage regions to output
    if damage_regions:
        output_json["room"]["damage_regions"] = [d.to_dict() for d in damage_regions]

    # Save JSON
    json_path = output_dir / "room_output.json"
    save_output_json(output_json, json_path)

    # Render floor plan
    plan_path = output_dir / "floor_plan.png"
    render_floor_plan(room_dims, plan_path,
                      title=f"Floor Plan - {scan_data.scan_id}")

    # Print summary
    print_output_summary(output_json)

    print(f"\nPipeline completed in {elapsed:.1f}s")
    print(f"Outputs saved to: {output_dir}")

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description="3D Room Capture Pipeline - reconstruct room geometry from iPhone LiDAR scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --input data/single_room/c00a170fe1
  python run_pipeline.py --input data/single_room/c00a170fe1 --output results/room1
  python run_pipeline.py --input data/single_room/c00a170fe1 --fast
  python run_pipeline.py --input data/single_room/c00a170fe1 --frame-step 3 --voxel 0.01
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to scan directory (containing depth/, odometry.csv, etc.)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: output/<scan_id>/)')
    parser.add_argument('--frame-step', type=int, default=5,
                        help='Process every Nth frame (default: 5). Higher = faster, less dense.')
    parser.add_argument('--confidence', type=int, default=1, choices=[0, 1, 2],
                        help='Minimum depth confidence level: 0=all, 1=medium+high, 2=high only')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                        help='Voxel grid size in meters for downsampling (default: 0.02)')
    parser.add_argument('--max-depth', type=float, default=6.0,
                        help='Maximum depth reading to keep in meters (default: 6.0)')
    parser.add_argument('--skip-damage', action='store_true',
                        help='Skip the damage detection step')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: frame_step=10, voxel=0.03, skip damage')

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    if not (input_dir / "odometry.csv").exists():
        print(f"Error: {input_dir} does not look like a scan directory "
              "(no odometry.csv found)")
        sys.exit(1)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "output" / input_dir.name

    # Apply fast mode settings
    frame_step = args.frame_step
    voxel_size = args.voxel_size
    skip_damage = args.skip_damage

    if args.fast:
        frame_step = 10
        voxel_size = 0.03
        skip_damage = True
        print("Fast mode enabled: frame_step=10, voxel=3cm, damage detection off")

    # Run the pipeline
    result = run_pipeline(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        frame_step=frame_step,
        confidence_threshold=args.confidence,
        voxel_size=voxel_size,
        max_depth=args.max_depth,
        skip_damage=skip_damage,
    )

    if result is None:
        print("\nPipeline failed!")
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
