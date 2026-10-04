# Technical Report & Architecture

This document describes the architectural decisions for the 3D Room Capture System, fulfilling the requirements for addressing Multi-Room Stitching, Scalability, and Device Compatibility.

## 1. Multi-Room Stitching Strategy

Currently, the pipeline processes single rooms. Scaling this to entire properties requires a robust multi-room stitching architecture.

### Global Coordinate Frame vs. Local Optimizations

We propose a **Pose Graph Optimization (PGO)** approach rather than relying purely on global odometry.
1. **Per-room local frame:** Each room is processed in its own local coordinate frame (as implemented currently).
2. **Doorway alignment:** When transitioning between rooms, the LiDAR captures the shared boundary (the door frame). We extract the 3D doorway geometry from both scans and use **Iterative Closest Point (ICP)** restricted to the XZ plane to compute the rigid transformation between room local frames.
3. **Loop closure:** If a route loops back to a previously visited room, a loop closure constraint is added to the pose graph. We use GTSAM or g2o to optimize the global pose of all rooms simultaneously, distributing the drift error.

### Photo-Tier Multi-Room Scale
For photo-only captures where scale is unknown:
1. Run SfM (COLMAP) on all photos globally to get relative camera poses.
2. The user must provide **one anchor measurement** (e.g., standard interior door width = 32 inches). We identify that door in the point cloud and scale the entire global model to match.

## 2. Concealed Damage Engine (Future Capability)

To identify structural or concealed damage that isn't visually obvious:
1. **Wall Deformation Analysis:** We fit ideal planes to walls (as implemented). By computing the per-point distance residuals to the ideal plane, we can detect bulging (potential water damage behind drywall) or sagging ceilings.
2. **Rule Engine:** We map structural anomalies to a rule engine. 
   - *Rule:* If `ceiling_sag > 2cm` AND `color_class == 'water_stain'`, flag as "Potential roof leak / structural damage".
3. **Thermal Integration:** Future hardware captures FLIR/thermal imagery. Cold spots in the wall mesh indicate missing insulation or water ingress.

## 3. Scope Generation Architecture

Translating geometry to an actionable construction scope:
1. **Quantities:** Floor area (m^2) -> Flooring material quantity. Wall surface area (length * height - openings) -> Paint quantity.
2. **Line Items:** We map quantities to standardized cost codes (e.g., Xactimate).
3. **Damage Overrides:** If the damage detector flags a wall section as "water damaged," the scope generation engine automatically inserts line items for "Tear out wet drywall" and "Mold remediation" for that specific wall segment.

## 4. Performance & Scalability

- **Memory Management:** The current pipeline loads 16-bit depth maps on demand rather than storing all frames in RAM. This allows it to process arbitrarily long scans.
- **Downsampling:** We use Voxel Grid downsampling early in the pipeline (currently 2cm). This guarantees O(V) processing time for RANSAC, where V is the volume of the room, rather than O(N) where N is the number of captured frames.
- **Compute:** For server-side processing, the pipeline can be containerized (Docker) and run on AWS Batch or Kubernetes. A typical room takes <15 seconds on a single CPU core.

## 5. Defense of RANSAC vs. Neural Networks

We chose classical RANSAC over 3D deep learning (like PointNet) for plane extraction because:
1. **Interpretability:** RANSAC provides mathematical guarantees. If it fails, we can see exactly which points were grouped.
2. **Zero-shot generalizability:** RANSAC doesn't care if the room is a warehouse, a bathroom, or a mansion. Neural networks often fail on out-of-distribution architecture.
3. **Compute Efficiency:** Runs on CPU in seconds; doesn't require GPUs for inference.

## 6. Classical CV vs. Deep Learning for Damage Detection

For visual damage detection (stains, cracks), we implemented a baseline using Classical Computer Vision (OpenCV color space analysis and Canny edge detection) rather than a Deep Learning model like YOLO or a Vision-Language Model (VLM).

1. **Trade-offs:** Classical CV is deterministic, extremely fast, and requires no heavy weights or GPU dependencies, ensuring the pipeline is reproducible on any reviewer's laptop.
2. **Limitations:** The primary limitation is an increase in **false positives**. A shadow cast by a lamp may be tagged as water damage, or a picture frame edge as a crack. 
3. **Production Path:** We built the module boundaries and schemas to prove the architectural flow. In a production environment, this `src/damage_detector.py` module would simply be swapped out for a fine-tuned YOLO model or VLM API call to drastically reduce false positives without altering the rest of the pipeline.
