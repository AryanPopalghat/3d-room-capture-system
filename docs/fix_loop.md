# Fix Loop: Robust Ceiling Detection

**Assignment Requirement:** "A 25% component of the score is given for submitting a 'fix loop' sequence where a failing or edge-case input is diagnosed, corrected in the code, and a fix is pushed."

## The Failing Run

While testing the pipeline on the `single_room` dataset, we noticed anomalous output in the JSON file for the ceiling height:

```json
"ceiling_height": {
  "value": 0.748,
  "unit": "meters",
  "uncertainty": 0.123,
  ...
}
```

A ceiling height of 0.75m (approx. 2.5 feet) is impossible for a standard residential room.

## Diagnosis

We inspected the plane classification logs:
```
Plane classification:
  Floor: 30,765 points at Y=-1.48m
  Ceiling: 6,852 points at Y=-0.73m
```

The iPhone LiDAR coordinate system (ARKit) uses a Y-up convention where the camera starts at roughly Y=0.
- `Y = -1.48m` corresponds to the floor (meaning the camera was held ~1.5m above the ground, which is normal chest height).
- `Y = -0.73m` corresponds to a horizontal surface roughly 0.75m above the floor.

**Root cause:** The `single_room` dataset did not scan the ceiling (the user only swept the walls and floor). The RANSAC algorithm found the *next largest horizontal plane* after the floor — which was a table or bed sitting at 0.75m height. Our naive logic simply assigned the highest horizontal plane as the ceiling, falsely classifying the furniture.

## The Fix

We updated `src/plane_detection.py` to add a heuristic for ceiling validation:
If the highest horizontal plane is not at least 2.0 meters above the floor, it is rejected as furniture.

```python
if len(horizontal_planes) >= 2:
    candidate_ceiling = horizontal_planes[-1]
    # Ensure it's actually a ceiling and not just a table/bed
    floor_y = classified['floor'].centroid[1]
    candidate_y = candidate_ceiling.centroid[1]
    
    if (candidate_y - floor_y) >= 2.0:
        classified['ceiling'] = candidate_ceiling
        classified['ceiling'].label = 'ceiling'
    else:
        print(f"  Note: Highest horizontal plane is only {candidate_y - floor_y:.2f}m above floor. "
              f"Likely furniture, not a ceiling.")
```

## Result

Re-running the pipeline correctly identifies that there is no ceiling in the scan and handles it gracefully, leaving the `ceiling_height` field null and accurately mapping the floor area and walls without being distorted by the furniture plane.
