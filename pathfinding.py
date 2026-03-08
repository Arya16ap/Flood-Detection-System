import cv2
import numpy as np
import heapq

def find_path(terrain_map, start, end, flood_mask, flood_penalty=10):
    """
    Finds the optimal path from start to end, avoiding flooded areas when possible.
    If no completely dry path exists, finds the path with least flooding.
    
    Args:
        terrain_map: The elevation map of the terrain
        start: Starting point as (x, y) coordinates
        end: Ending point as (x, y) coordinates
        flood_mask: Boolean mask where True indicates flooded areas
        flood_penalty: Cost multiplier for traversing flooded areas
        
    Returns:
        Numpy array of path coordinates or None if no path is found
    """
    height, width = terrain_map.shape if len(terrain_map.shape) == 2 else terrain_map.shape[:2]
    
    # Convert to tuple for dictionary keys
    start = tuple(start)
    end = tuple(end)
    
    def heuristic(a, b):
        """Euclidean distance heuristic for A* algorithm"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(node):
        """Get neighboring points in 8 directions"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor = (node[0] + dx, node[1] + dy)
                if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                    neighbors.append(neighbor)
        return neighbors
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            # Reconstruct path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return np.array(path)
        
        for neighbor in get_neighbors(current):
            # Calculate movement cost (diagonal movement costs more)
            dx, dy = neighbor[0] - current[0], neighbor[1] - current[1]
            base_cost = np.sqrt(dx*dx + dy*dy)  # Euclidean distance
            
            # Apply flood penalty if the neighbor is in a flooded area
            is_flooded = flood_mask[neighbor[1], neighbor[0]] if neighbor[1] < height and neighbor[0] < width else False
            movement_cost = base_cost * (flood_penalty if is_flooded else 1)
            
            new_cost = cost_so_far[current] + movement_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(end, neighbor)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    
    # If no path found, return a straight line between start and end
    return create_straight_line_path(start, end)

def optimize_path(path, step_size=5):
    """
    Optimizes a path by removing unnecessary points and smoothing.
    
    Args:
        path: Numpy array of path coordinates
        step_size: Interval for selecting points
        
    Returns:
        Optimized path with fewer points
    """
    if path is None or len(path) <= 2:
        return path
        
    # Take every nth point, always keeping start and end
    indices = [0]
    indices.extend(range(step_size, len(path) - 1, step_size))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
        
    return path[indices]

def create_straight_line_path(start, end):
    """
    Creates a straight line path between start and end points.
    
    Args:
        start: Starting point as (x, y) coordinates
        end: Ending point as (x, y) coordinates
        
    Returns:
        Numpy array of path coordinates
    """
    # Calculate number of points needed based on distance
    distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    num_points = max(int(distance / 5), 2)  # At least 2 points, one every 5 pixels
    
    # Generate points along the straight line
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    
    # Combine into path array
    path = np.column_stack((x, y)).astype(int)
    return path

def visualize_path(image, path, marker_color=(44, 107, 156), thickness=3, show_flood=True, flood_mask=None):
    """
    Visualize the path on the given image.
    
    Args:
        image: The image to draw on
        path: Numpy array of path coordinates
        marker_color: BGR color tuple for the path (default #2C6B9C in BGR format)
        thickness: Line thickness
        show_flood: Whether to highlight flood-affected path segments
        flood_mask: Boolean mask where True indicates flooded areas
        
    Returns:
        Image with the path drawn on it
    """
    result = image.copy()
    
    if path is None or len(path) < 2:
        return result
    
    # Draw the path
    for i in range(len(path) - 1):
        pt1 = (int(path[i][0]), int(path[i][1]))
        pt2 = (int(path[i+1][0]), int(path[i+1][1]))
        
        if show_flood and flood_mask is not None:
            # Check if segment crosses flood
            is_pt1_flooded = flood_mask[pt1[1], pt1[0]] if pt1[1] < flood_mask.shape[0] and pt1[0] < flood_mask.shape[1] else False
            is_pt2_flooded = flood_mask[pt2[1], pt2[0]] if pt2[1] < flood_mask.shape[0] and pt2[0] < flood_mask.shape[1] else False
            
            if is_pt1_flooded or is_pt2_flooded:
                # Red for flooded segments
                segment_color = (0, 0, 255)
            else:
                segment_color = marker_color
        else:
            segment_color = marker_color
            
        cv2.line(result, pt1, pt2, segment_color, thickness)
    
    # Add markers for start and end points
    start = (int(path[0][0]), int(path[0][1]))
    end = (int(path[-1][0]), int(path[-1][1]))
    cv2.circle(result, start, 8, marker_color, -1)  # Use marker color for start
    cv2.circle(result, end, 8, marker_color, -1)    # Use marker color for end
    
    return result

def get_flood_percentage(path, flood_mask):
    """
    Calculate the percentage of the path that passes through flooded areas.
    
    Args:
        path: Numpy array of path coordinates
        flood_mask: Boolean mask where True indicates flooded areas
        
    Returns:
        Percentage of the path that is flooded (0-100)
    """
    if path is None or len(path) < 2:
        return 0
        
    flooded_segments = 0
    total_segments = len(path) - 1
    
    for i in range(total_segments):
        pt1 = (int(path[i][0]), int(path[i][1]))
        pt2 = (int(path[i+1][0]), int(path[i+1][1]))
        
        # Ensure coordinates are within bounds
        if (0 <= pt1[1] < flood_mask.shape[0] and 0 <= pt1[0] < flood_mask.shape[1] and
            0 <= pt2[1] < flood_mask.shape[0] and 0 <= pt2[0] < flood_mask.shape[1]):
            
            is_pt1_flooded = flood_mask[pt1[1], pt1[0]]
            is_pt2_flooded = flood_mask[pt2[1], pt2[0]]
            
            if is_pt1_flooded or is_pt2_flooded:
                flooded_segments += 1
            
    return (flooded_segments / total_segments) * 100 if total_segments > 0 else 0