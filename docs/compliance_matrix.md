# Compliance Matrix

Maps each assignment requirement to its implementation status.

## Output Contract (Part 2)

| Requirement | File/Artifact | Status | Notes |
|---|---|---|---|
| Dimensioned per-room plan with walls | `output/*/floor_plan.png` | ✅ Built | Wall segments with length annotations |
| Ceiling height | `output/*/room_output.json` → `room.ceiling_height` | ✅ Built | From floor-ceiling plane distance |
| Floor area | `output/*/room_output.json` → `room.floor_area` | ✅ Built | Convex hull of floor plane |
| Openings (doors, windows) | `output/*/room_output.json` → `room.openings` | ✅ Built | Gap detection in wall planes |
| Stitched multi-room plan | - | ❌ Not built | Only single-room data provided |
| Per-surface damage regions | `output/*/room_output.json` → `room.damage_regions` | ✅ Built | CV-based detection (stains, cracks) |
| Concealed-damage flags | - | ❌ Not built | Requires rule engine + domain knowledge |
| Scope line items | - | ❌ Not built | Requires damage-to-scope mapping |
| Confidence interval on every measurement | `output/*/room_output.json` | ✅ Built | 95% CI from RANSAC inlier spread |
| One command per capture | `run_pipeline.py --input <path>` | ✅ Built | Single CLI command |
| JSON to published schema | `output/*/room_output.json` | ✅ Built | Structured JSON with measurements |
| Rendered plan | `output/*/floor_plan.png` | ✅ Built | Matplotlib floor plan |

## Input Tiers (Part 1)

| Tier | Status | Notes |
|---|---|---|
| LiDAR (depth + poses + intrinsics) | ✅ Built | Primary pipeline, full implementation |
| Video (handheld walkthrough) | ⚠️ Partial | Video frames extracted for damage detection |
| Photos (2-8 stills per room) | ❌ Not built | Requires SfM pipeline (COLMAP) |

## Accuracy Gates (Part 2)

| Metric | Gate | Status | Notes |
|---|---|---|---|
| Opening widths | ≤ 2cm on ≥ 85% | ⚠️ To verify | Gap detection implemented |
| Ceiling height | ≤ 1.5cm per room | ⚠️ To verify | No ground truth to compare against |
| Repeatability | Same room ≤ 1cm | ❌ No repeated captures | Only one capture per room |
| Drift accountability | Report drift handling | ✅ Documented | Uses ARKit odometry as-is, acknowledged |
| Photo-tier stitch | Per-room folders → one plan | ❌ Not built | Photo tier not implemented |

## Deliverables (Part 5)

| # | Deliverable | Status | Location |
|---|---|---|---|
| 1 | Compliance matrix | ✅ | `docs/compliance_matrix.md` |
| 2 | Capture route + device matrix | ✅ | `docs/capture_protocol.md`, `docs/device_matrix.md` |
| 3 | Repo + README | ✅ | `README.md` |
| 4 | Reproduction bundle | ✅ | Pipeline regenerates all outputs from raw data |
| 5 | Benchmark report | ⚠️ Partial | No ground truth available for numerical accuracy |
| 6 | Fix loop bundle | ✅ | `docs/fix_loop.md` |
| 7 | Technical report | ✅ | `docs/technical_report.md` |
| 8 | Raw benchmark data | ✅ | Sample data provided, outputs in `output/` |

## Scoring Components

| Weight | Component | Our Coverage |
|---|---|---|
| 30% | Walk-in test | Pipeline runs cold on any LiDAR scan directory |
| 25% | Fix loop | Documented with before/after in `docs/fix_loop.md` |
| 15% | Benchmark accuracy | Outputs generated for all 3 sample datasets |
| 10% | Compliance matrix | This document |
| 10% | Head-to-head | Architecture described but no comparison data |
| 5% | Capture route | `docs/capture_protocol.md` |
| 5% | Process evidence | Multiple commits with meaningful progression |
