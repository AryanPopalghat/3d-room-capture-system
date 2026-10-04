# 3D Room Capture System

Reconstructs room geometry from iPhone LiDAR scan data. Produces dimensioned floor plans, room measurements with confidence intervals, and damage reports.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a scan
python run_pipeline.py --input data/single_room/c00a170fe1

# Fast mode (lower resolution, quicker)
python run_pipeline.py --input data/single_room/c00a170fe1 --fast
```

## What It Does

Given a directory of iPhone LiDAR capture data (depth maps, camera poses, RGB video), the pipeline:

1. **Builds a 3D point cloud** from depth maps + camera poses
2. **Detects structural planes** (floor, ceiling, walls) using RANSAC
3. **Extracts room dimensions** — ceiling height, floor area, wall lengths
4. **Detects openings** (doors, windows) from wall gaps
5. **Identifies damage** (stains, cracks, discoloration) from RGB video
6. **Renders a floor plan** with measurements
7. **Exports JSON** with all measurements and confidence intervals

## Input Data Format

Each scan directory should contain:

```
scan_directory/
├── camera_matrix.csv     # 3x3 camera intrinsic matrix
├── odometry.csv          # Per-frame camera poses (position + orientation)
├── imu.csv               # Accelerometer + gyroscope data
├── depth/                # Per-frame depth maps (16-bit PNG, millimeters)
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── confidence/           # Per-frame depth confidence (8-bit PNG, 0-2)
│   ├── 000000.png
│   └── ...
└── rgb.mp4               # RGB video of the capture
```

**Getting sample data:** Download from the provided Google Drive link and extract into a `data/` directory at the project root.

## Output

The pipeline produces:

```
output/<scan_id>/
├── room_output.json        # All measurements in machine-readable JSON
├── floor_plan.png          # Dimensioned 2D floor plan
├── point_cloud.ply         # 3D point cloud (viewable in MeshLab)
└── point_cloud_topdown.png # Debug: top-down point cloud view
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | required | Path to scan directory |
| `--output` / `-o` | `output/<scan_id>` | Output directory |
| `--frame-step` | 5 | Process every Nth frame (higher = faster) |
| `--confidence` | 1 | Min depth confidence: 0=all, 1=medium+, 2=high |
| `--voxel-size` | 0.02 | Downsampling grid size in meters |
| `--max-depth` | 6.0 | Max depth reading to keep (meters) |
| `--skip-damage` | false | Skip damage detection step |
| `--fast` | false | Fast mode (step=10, voxel=3cm, no damage) |

## Architecture

```
Data Ingestion → Point Cloud → Plane Detection → Geometry → Floor Plan
                  Builder        (RANSAC)       Extraction   Rendering
                                                              ↓
                                               Damage     → JSON
                                               Detection    Output
```

### Key Modules

- **`src/data_loader.py`** — Parses camera intrinsics, odometry poses, depth/confidence maps
- **`src/point_cloud.py`** — Fuses per-frame depth into a world-space 3D point cloud
- **`src/plane_detection.py`** — RANSAC plane fitting with floor/ceiling/wall classification
- **`src/geometry.py`** — Extracts ceiling height, floor area, wall lengths, openings
- **`src/floor_plan.py`** — Renders dimensioned 2D floor plans
- **`src/output_schema.py`** — JSON output with confidence intervals
- **`src/damage_detector.py`** — RGB-based damage detection (stains, cracks)

## Requirements

- Python 3.9+
- ~4GB RAM for large scans
- Dependencies in `requirements.txt`
