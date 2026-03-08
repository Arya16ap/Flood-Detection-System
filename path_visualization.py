import cv2
import numpy as np

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

def display_path(image, path, flood_mask, flood_percent):
    """
    Display the path on a new window.
    
    Args:
        image: The original image with flood overlay
        path: Numpy array of path coordinates
        flood_mask: Boolean mask indicating flooded areas
        flood_percent: Percentage of the path that is flooded
    """
    result = visualize_path(image, path, show_flood=True, flood_mask=flood_mask)
    
    # Add text information about the path and flood percentage
    cv2.putText(result, f"Flood-affected path: {flood_percent:.1f}%", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    
    # Display the result
    cv2.imshow("Optimal Flood-Aware Navigation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()