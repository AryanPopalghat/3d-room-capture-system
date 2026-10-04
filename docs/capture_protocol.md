# Capture Protocol

Instructions for a non-engineer to capture a room using an iPhone with LiDAR.

## Required Equipment

- iPhone 12 Pro or newer (any Pro model with LiDAR scanner)
- The capture app installed (Record3D or custom TestFlight build)
- Sufficient battery (>30% recommended)
- Adequate lighting in the room

## Step-by-Step Capture Instructions

### Before You Start

1. **Clear the path.** Walk the perimeter of the room once without the phone to plan your route. Make sure you won't trip on anything.
2. **Open the capture app** and verify it shows a depth preview (you should see a colorful depth overlay on the camera feed).
3. **Hold the phone at chest height**, roughly centered, with the screen facing you and the cameras facing the room.

### Capture Walk

1. **Start recording** by tapping the red record button.
2. **Stand in the center of the room** and slowly rotate 360° to capture the floor.
3. **Walk slowly around the room perimeter**, keeping walls about 1–2 meters away. Move at a normal walking pace — don't rush.
4. **Point the camera at each wall** as you pass it. Spend 2–3 seconds facing each wall directly.
5. **Scan the ceiling** by tilting the phone up ~45° and doing another slow rotation from the center.
6. **Capture doorways and windows** — pause for 2 seconds when pointing at any opening.
7. **Return to your starting position** to close the loop. This helps with drift correction.
8. **Stop recording.**

### Timing

- Small room (bathroom): ~30 seconds
- Medium room (bedroom): ~45–60 seconds
- Large room (living room): ~90 seconds
- Full property (3+ rooms): ~3–5 minutes total

### What to Avoid

| Don't | Why |
|---|---|
| Don't move too fast | Creates motion blur in depth readings |
| Don't point at mirrors directly | LiDAR reads through mirrors, creating phantom geometry |
| Don't capture in complete darkness | LiDAR works but RGB frames are useless for damage detection |
| Don't cover the LiDAR sensor | It's the small dot next to the camera lenses |
| Don't shake the phone | Smooth movements give better odometry tracking |

### After Capture

1. The app will save the scan data automatically.
2. Export / upload the scan from the app's library.
3. The exported folder should contain: `depth/`, `confidence/`, `rgb.mp4`, `camera_matrix.csv`, `odometry.csv`, `imu.csv`

## Tips for Best Results

- **Lighting:** Natural daylight is best. Avoid extremely bright windows that create high contrast.
- **Coverage:** Make sure every surface you want measured is visible from at least 2 different angles.
- **Speed:** Slower is always better. The LiDAR has a limited capture rate (~60Hz) and fast movement reduces frame overlap.
- **Loop closure:** Starting and ending at the same spot significantly improves accuracy by allowing drift correction.
