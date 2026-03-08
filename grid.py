import cv2
import numpy as np

GRID_SIZE = 50  # Grid spacing for roads

def create_road_map(image_shape, markers):
    """Creates a visible road network and accurately connects markers."""
    height, width, _ = image_shape
    road_map = np.zeros((height, width, 3), dtype=np.uint8)  # Black base image

    # Draw structured roads (grid-based)
    for y in range(0, height, GRID_SIZE):
        cv2.line(road_map, (0, y), (width, y), (0, 0, 255), 2)  # Red horizontal roads

    for x in range(0, width, GRID_SIZE):
        cv2.line(road_map, (x, 0), (x, height), (0, 0, 255), 2)  # Red vertical roads

    # Ensure markers are properly connected to the grid
    if len(markers) == 2:
        start, end = markers

        # Find the closest grid point to each marker
        start_x, start_y = round(start[0] / GRID_SIZE) * GRID_SIZE, round(start[1] / GRID_SIZE) * GRID_SIZE
        end_x, end_y = round(end[0] / GRID_SIZE) * GRID_SIZE, round(end[1] / GRID_SIZE) * GRID_SIZE

        # Draw a temporary connection from markers to their closest grid points
        cv2.line(road_map, start, (start_x, start_y), (0, 255, 0), 2)  # Green connection
        cv2.line(road_map, end, (end_x, end_y), (0, 255, 0), 2)  # Green connection

        # Draw the main path along the grid
        cv2.line(road_map, (start_x, start_y), (end_x, start_y), (0, 255, 0), 3)  # Move horizontally first
        cv2.line(road_map, (end_x, start_y), (end_x, end_y), (0, 255, 0), 3)  # Then move vertically

    return road_map
