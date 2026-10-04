"""
JSON output schema for room measurement results.

Produces a structured JSON output containing all extracted dimensions,
confidence intervals, and metadata. This is the machine-readable output
that downstream systems (scope generators, damage reports, etc.) consume.
"""

import json
import datetime
from pathlib import Path


def build_output_json(room_dims, scan_data, processing_time=None):
    """
    Build the complete JSON output from room dimensions and scan metadata.

    The schema includes:
    - Room dimensions with confidence intervals
    - Wall segments with lengths
    - Detected openings
    - Processing metadata
    - Input data summary

    Args:
        room_dims: RoomDimensions object
        scan_data: ScanData object with input metadata
        processing_time: how long the pipeline took (seconds)

    Returns:
        dict suitable for json.dump()
    """
    output = {
        "schema_version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "scan_id": scan_data.scan_id,
        "input_tier": "lidar",  # this pipeline processes LiDAR-tier data

        "room": {
            "ceiling_height": _measurement_block(
                room_dims.ceiling_height,
                room_dims.ceiling_height_std,
                unit="meters"
            ),
            "floor_area": _measurement_block(
                room_dims.floor_area,
                room_dims.floor_area_std,
                unit="square_meters"
            ),
            "walls": [
                {
                    "wall_id": i + 1,
                    "length": {
                        "value": round(w.length, 3),
                        "unit": "meters"
                    },
                    "start_point": {
                        "x": round(float(w.start[0]), 3),
                        "z": round(float(w.start[1]), 3)
                    },
                    "end_point": {
                        "x": round(float(w.end[0]), 3),
                        "z": round(float(w.end[1]), 3)
                    },
                    "normal": {
                        "x": round(float(w.normal[0]), 3),
                        "z": round(float(w.normal[1]), 3)
                    }
                }
                for i, w in enumerate(room_dims.wall_segments)
            ],
            "openings": [
                {
                    "type": o.type,
                    "width": {
                        "value": round(o.width, 3),
                        "unit": "meters"
                    },
                    "position": {
                        "x": round(float(o.position[0]), 3),
                        "z": round(float(o.position[1]), 3)
                    }
                }
                for o in room_dims.openings
            ],
            "bounding_box": room_dims.bounds if room_dims.bounds else None,
            "damage_regions": []  # populated by damage detector if available
        },

        "processing": {
            "input_frames": scan_data.num_frames,
            "pipeline_version": "1.0.0",
            "processing_time_seconds": round(processing_time, 1) if processing_time else None,
        },

        "confidence_notes": (
            "All measurements include 95% confidence intervals derived from "
            "the spread of RANSAC inlier points. Ceiling height uncertainty "
            "combines floor and ceiling plane fitting errors. Floor area uses "
            "convex hull of floor plane inliers."
        )
    }

    return output


def _measurement_block(value, std, unit="meters"):
    """
    Build a measurement object with value, uncertainty, and confidence interval.

    Returns None for the block if the value is None (measurement not available).
    """
    if value is None:
        return None

    block = {
        "value": round(value, 3),
        "unit": unit,
    }

    if std is not None:
        margin = 1.96 * std  # 95% CI
        block["uncertainty"] = round(std, 4)
        block["confidence_interval_95"] = {
            "lower": round(value - margin, 3),
            "upper": round(value + margin, 3)
        }

    return block


def save_output_json(output_dict, output_path):
    """Save the output JSON to a file with pretty formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"  JSON output saved: {output_path}")


def print_output_summary(output_dict):
    """Print a human-readable summary of the output."""
    room = output_dict.get("room", {})

    print("\n" + "="*50)
    print("  ROOM MEASUREMENT RESULTS")
    print("="*50)

    ch = room.get("ceiling_height")
    if ch:
        ci = ch.get("confidence_interval_95", {})
        print(f"  Ceiling Height:  {ch['value']:.3f}m")
        if ci:
            print(f"    95% CI:        [{ci['lower']:.3f}m, {ci['upper']:.3f}m]")

    fa = room.get("floor_area")
    if fa:
        print(f"  Floor Area:      {fa['value']:.2f} m^2")

    walls = room.get("walls", [])
    print(f"  Walls:           {len(walls)}")
    for w in walls:
        print(f"    Wall {w['wall_id']}: {w['length']['value']:.2f}m")

    openings = room.get("openings", [])
    print(f"  Openings:        {len(openings)}")
    for o in openings:
        print(f"    {o['type']}: {o['width']['value']:.2f}m wide")

    print("="*50)
