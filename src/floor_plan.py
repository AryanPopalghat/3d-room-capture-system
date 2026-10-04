"""
2D floor plan rendering from extracted room geometry.

Takes the wall segments, floor polygon, openings, and dimensions
and renders a clean top-down floor plan with measurements annotated.
The output is a PNG image that a homeowner would recognize as their room.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no window needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


def render_floor_plan(room_dims, output_path, title=None, figsize=(12, 10)):
    """
    Render a dimensioned 2D floor plan.

    The plan shows:
    - Floor boundary polygon
    - Wall lines with length annotations
    - Opening locations (doors/windows)
    - Key measurements (ceiling height, floor area)

    Args:
        room_dims: RoomDimensions object from geometry extraction
        output_path: where to save the PNG
        title: optional title for the plot
        figsize: figure size in inches
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')

    # Style
    wall_color = '#2c3e50'
    floor_color = '#ecf0f1'
    dimension_color = '#e74c3c'
    opening_color = '#3498db'
    text_color = '#2c3e50'

    # Draw floor polygon if available
    if room_dims.floor_polygon is not None and len(room_dims.floor_polygon) > 2:
        polygon = plt.Polygon(room_dims.floor_polygon, closed=True,
                              facecolor=floor_color, edgecolor=wall_color,
                              linewidth=2.5, zorder=1)
        ax.add_patch(polygon)

    # Draw wall segments
    for i, wall in enumerate(room_dims.wall_segments):
        # Wall line
        ax.plot([wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                color=wall_color, linewidth=3, solid_capstyle='round', zorder=2)

        # Dimension annotation
        mid_x = (wall.start[0] + wall.end[0]) / 2
        mid_z = (wall.start[1] + wall.end[1]) / 2

        # Offset the label slightly from the wall
        offset = wall.normal * 0.15  # 15cm offset
        label_x = mid_x + offset[0]
        label_z = mid_z + offset[1]

        ax.annotate(f'{wall.length:.2f}m',
                    xy=(mid_x, mid_z),
                    xytext=(label_x, label_z),
                    fontsize=9, fontweight='bold', color=dimension_color,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=dimension_color, alpha=0.9),
                    zorder=5)

    # Draw openings
    for opening in room_dims.openings:
        ax.plot(opening.position[0], opening.position[1],
                's', color=opening_color, markersize=10, zorder=4)
        ax.annotate(f'{opening.type}\n{opening.width:.2f}m',
                    xy=(opening.position[0], opening.position[1]),
                    xytext=(opening.position[0] + 0.3, opening.position[1] + 0.3),
                    fontsize=8, color=opening_color,
                    arrowprops=dict(arrowstyle='->', color=opening_color),
                    zorder=5)

    # Add room info text box
    info_lines = []
    if room_dims.ceiling_height is not None:
        ci = room_dims.ceiling_height_ci
        info_lines.append(f'Ceiling Height: {room_dims.ceiling_height:.2f}m')
        if ci:
            info_lines.append(f'  (95% CI: {ci[0]:.2f}m - {ci[1]:.2f}m)')
    if room_dims.floor_area is not None:
        info_lines.append(f'Floor Area: {room_dims.floor_area:.1f} m²')
    info_lines.append(f'Walls: {len(room_dims.wall_segments)}')
    info_lines.append(f'Openings: {len(room_dims.openings)}')

    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace', zorder=10)

    # Formatting
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Z (meters)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Floor Plan (Top-Down View)', fontsize=14, fontweight='bold', pad=15)

    # Add a scale bar
    _add_scale_bar(ax)

    # Add compass indicator
    _add_compass(ax)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"  Floor plan saved: {output_path}")


def _add_scale_bar(ax):
    """Add a 1-meter scale bar to the bottom-right of the plot."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    bar_length = 1.0  # 1 meter
    bar_x = xlim[1] - bar_length - 0.3
    bar_y = ylim[0] + 0.2

    ax.plot([bar_x, bar_x + bar_length], [bar_y, bar_y],
            color='black', linewidth=3, solid_capstyle='butt')
    ax.plot([bar_x, bar_x], [bar_y - 0.05, bar_y + 0.05],
            color='black', linewidth=2)
    ax.plot([bar_x + bar_length, bar_x + bar_length], [bar_y - 0.05, bar_y + 0.05],
            color='black', linewidth=2)
    ax.text(bar_x + bar_length/2, bar_y + 0.1, '1m',
            ha='center', va='bottom', fontsize=9, fontweight='bold')


def _add_compass(ax):
    """Add a simple north arrow to indicate orientation."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    compass_x = xlim[1] - 0.4
    compass_y = ylim[1] - 0.4

    ax.annotate('N', xy=(compass_x, compass_y),
                xytext=(compass_x, compass_y - 0.4),
                fontsize=12, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))


def render_point_cloud_topdown(pcd, output_path, title="Point Cloud Top-Down View"):
    """
    Render a top-down scatter plot of the raw point cloud.

    Useful for debugging - shows the raw spatial distribution of points
    before any plane fitting or geometry extraction.
    """
    points = np.asarray(pcd.points)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect('equal')

    # Plot XZ plane (top-down), color by Y (height)
    scatter = ax.scatter(points[:, 0], points[:, 2],
                         c=points[:, 1], cmap='viridis',
                         s=0.1, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Height Y (m)', shrink=0.7)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Top-down view saved: {output_path}")
