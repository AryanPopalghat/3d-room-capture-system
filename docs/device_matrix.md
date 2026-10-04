# Device Matrix

Which input tier runs on which hardware, and what accuracy each tier delivers.

## Tier Summary

| Tier            | Hardware                                                        | Sensors Used                         | Expected Wall Accuracy | Expected Ceiling Height Accuracy |
| --------------- | --------------------------------------------------------------- | ------------------------------------ | ---------------------- | -------------------------------- |
| **LiDAR** | iPhone 12 Pro, 13 Pro, 14 Pro, 15 Pro, 16 Pro, iPad Pro (2020+) | LiDAR depth + RGB camera + IMU       | ±1-2 cm              | ±1.5 cm                         |
| **Video** | Any iPhone 15 or newer                                          | RGB camera + IMU (no depth)          | ±3-5 cm              | ±3 cm                           |
| **Photo** | Any iPhone 15 or newer                                          | RGB camera only (no depth, no poses) | ±5-10 cm             | ±8 cm (if ceiling visible)      |

## LiDAR Tier (Implemented)

**Hardware:** Any iPhone or iPad with LiDAR scanner (Pro models, 2020+)

**Sensors:**

- LiDAR depth scanner (~256×192 @ 60Hz, range 0-5m)
- RGB camera (1920×1440 @ 30fps)
- 6-axis IMU (accelerometer + gyroscope @ 100Hz)
- ARKit visual-inertial odometry (6DoF pose per frame)

**Accuracy expectations:**

- Wall lengths: ±1-2 cm (limited by LiDAR noise + ARKit tracking drift)
- Ceiling height: ±1-1.5 cm
- Floor area: ±3-5% of true area
- Opening widths: ±2 cm

**Limitations:**

- LiDAR struggles with glass, mirrors, very dark surfaces
- Maximum reliable range ~5m (noisy beyond that)
- ARKit odometry drifts over time without loop closure

## Video Tier (Architecture Described)

**Hardware:** Any iPhone 15 or newer (no LiDAR required)

**Approach:**

- Extract frames from video at ~2fps
- Use monocular depth estimation (MiDAS/DPT) for per-frame depth
- Use visual odometry (ORB-SLAM or DSO) for camera tracking
- Same downstream pipeline (point cloud → planes → dimensions)

**Accuracy expectations:**

- Wall lengths: ±3-5 cm (monocular depth is less precise than LiDAR)
- Ceiling height: ±3 cm
- Scale recovery requires at least one known dimension or IMU integration

## Photo Tier (Architecture Described)

**Hardware:** Any iPhone 15 or newer

**Approach:**

- Structure from Motion (SfM) using COLMAP or similar
- Reconstruct sparse → dense point cloud from 2-8 photos
- Scale recovery from known object sizes or camera height assumption
- Same downstream pipeline

**Accuracy expectations:**

- Wall lengths: ±5-10 cm with wide confidence intervals
- Ceiling height: ±8 cm (only if ceiling visible in photos)
- Multi-room stitch: ±8% footprint area with calibrated intervals
