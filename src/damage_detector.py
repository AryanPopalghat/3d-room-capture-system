"""
Damage detection from RGB video frames.

Identifies potential damage regions (stains, cracks, water marks, peeling paint)
by analyzing color and texture anomalies in the captured video frames.

Approach:
    1. Extract key frames from the RGB video
    2. Convert to multiple color spaces for analysis
    3. Detect anomalous regions using:
       - Color deviation from dominant wall colors
       - Texture irregularity (high-frequency patterns = cracks)
       - Dark spots / discoloration
    4. If depth data available, project 2D damage regions to 3D to get metric size

This is a simplified baseline — a production system would use a
vision-language model (like Gemini or GPT-4V) for more reliable
damage classification.
"""

import numpy as np
import cv2
from pathlib import Path


class DamageRegion:
    """A detected damage region with location and classification."""

    def __init__(self, bbox, damage_class, confidence, frame_idx=None):
        self.bbox = bbox              # (x, y, w, h) in pixels
        self.damage_class = damage_class  # 'stain', 'crack', 'water_damage', 'peeling'
        self.confidence = confidence  # 0.0 to 1.0
        self.frame_idx = frame_idx
        self.metric_extent = None     # (width_m, height_m) if depth-projected
        self.position_3d = None       # world coordinates if available

    def to_dict(self):
        result = {
            "class": self.damage_class,
            "confidence": round(self.confidence, 2),
            "bbox_pixels": {
                "x": int(self.bbox[0]),
                "y": int(self.bbox[1]),
                "width": int(self.bbox[2]),
                "height": int(self.bbox[3])
            },
            "frame_index": self.frame_idx,
        }
        if self.metric_extent:
            result["metric_extent"] = {
                "width_meters": round(self.metric_extent[0], 3),
                "height_meters": round(self.metric_extent[1], 3)
            }
        return result


def detect_damage_in_frame(frame, frame_idx=None, min_area=500):
    """
    Detect potential damage regions in a single RGB frame.

    Uses color-space analysis to find anomalous regions:
    1. Convert to HSV and LAB color spaces
    2. Detect dark patches (potential stains/water damage)
    3. Detect high-texture regions (potential cracks)
    4. Filter by minimum area to avoid noise

    Args:
        frame: BGR numpy array from OpenCV
        frame_idx: which frame this is (for tracking)
        min_area: minimum contour area in pixels to consider

    Returns:
        list of DamageRegion objects
    """
    regions = []
    h, w = frame.shape[:2]

    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # --- Detect dark spots / stains ---
    # Regions significantly darker than the surrounding area
    l_channel = lab[:, :, 0]

    # Adaptive thresholding to find locally dark regions
    blurred = cv2.GaussianBlur(l_channel, (51, 51), 0)
    dark_diff = blurred.astype(np.int16) - l_channel.astype(np.int16)
    dark_mask = (dark_diff > 30).astype(np.uint8) * 255

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        # Confidence based on how dark the region is relative to surroundings
        region_darkness = np.mean(dark_diff[y:y+ch, x:x+cw])
        confidence = min(region_darkness / 80.0, 0.95)

        regions.append(DamageRegion(
            bbox=(x, y, cw, ch),
            damage_class='stain',
            confidence=confidence,
            frame_idx=frame_idx
        ))

    # --- Detect cracks (high-frequency texture) ---
    # Use Canny edge detection + dilation to find crack-like features
    edges = cv2.Canny(gray, 50, 150)

    # Dilate edges to connect nearby crack segments
    crack_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, crack_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area * 0.5:  # cracks can be thin
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        # Cracks are typically elongated — check aspect ratio
        aspect_ratio = max(cw, ch) / (min(cw, ch) + 1)
        if aspect_ratio < 2.0:
            continue  # not elongated enough to be a crack

        confidence = min(aspect_ratio / 10.0, 0.85)

        regions.append(DamageRegion(
            bbox=(x, y, cw, ch),
            damage_class='crack',
            confidence=confidence,
            frame_idx=frame_idx
        ))

    # --- Detect discoloration (unusual hue regions) ---
    # Areas with significantly different hue from the dominant wall color
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    # High saturation in typically neutral walls could indicate water damage
    high_sat_mask = (saturation > 80).astype(np.uint8) * 255
    high_sat_mask = cv2.morphologyEx(high_sat_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(high_sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area * 2:  # discoloration typically covers larger areas
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        region_sat = np.mean(saturation[y:y+ch, x:x+cw])
        confidence = min(region_sat / 200.0, 0.75)

        # Check if it's actually a colored object vs damage
        # Skip regions near image edges (likely not walls)
        if x < w * 0.05 or x + cw > w * 0.95 or y < h * 0.05 or y + ch > h * 0.95:
            continue

        regions.append(DamageRegion(
            bbox=(x, y, cw, ch),
            damage_class='water_damage',
            confidence=confidence,
            frame_idx=frame_idx
        ))

    return regions


def detect_damage_in_video(video_path, max_frames=20, frame_step=30):
    """
    Run damage detection on frames extracted from the RGB video.

    Args:
        video_path: path to rgb.mp4
        max_frames: max number of frames to analyze
        frame_step: check every Nth frame

    Returns:
        list of DamageRegion objects from all frames, deduplicated
    """
    if not Path(video_path).exists():
        print("  No RGB video found, skipping damage detection")
        return []

    print(f"\nRunning damage detection on video...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    all_regions = []
    frame_idx = 0
    frames_checked = 0

    while cap.isOpened() and frames_checked < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            regions = detect_damage_in_frame(frame, frame_idx=frame_idx)
            all_regions.extend(regions)
            frames_checked += 1

        frame_idx += 1

    cap.release()

    # Deduplicate by merging overlapping detections from nearby frames
    merged = _merge_overlapping_regions(all_regions)

    print(f"  Checked {frames_checked} frames, found {len(merged)} damage regions")
    for r in merged:
        print(f"    {r.damage_class}: confidence={r.confidence:.2f}, "
              f"bbox=({r.bbox[0]}, {r.bbox[1]}, {r.bbox[2]}x{r.bbox[3]})")

    return merged


def _merge_overlapping_regions(regions, iou_threshold=0.3):
    """
    Merge damage regions that are likely the same damage seen from different frames.
    Uses simple IoU (intersection over union) to detect overlap.
    """
    if len(regions) <= 1:
        return regions

    # Group by damage class
    by_class = {}
    for r in regions:
        by_class.setdefault(r.damage_class, []).append(r)

    merged = []
    for damage_class, class_regions in by_class.items():
        # Sort by confidence (keep highest confidence detection)
        class_regions.sort(key=lambda r: r.confidence, reverse=True)

        used = [False] * len(class_regions)
        for i in range(len(class_regions)):
            if used[i]:
                continue
            # This is the best detection — mark overlapping ones as used
            for j in range(i + 1, len(class_regions)):
                if used[j]:
                    continue
                if _bbox_iou(class_regions[i].bbox, class_regions[j].bbox) > iou_threshold:
                    used[j] = True

            merged.append(class_regions[i])

    return merged


def _bbox_iou(bbox1, bbox2):
    """Compute intersection over union of two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi = max(x1, x2)
    yi = max(y1, y2)
    xf = min(x1 + w1, x2 + w2)
    yf = min(y1 + h1, y2 + h2)

    if xi >= xf or yi >= yf:
        return 0.0

    intersection = (xf - xi) * (yf - yi)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0
