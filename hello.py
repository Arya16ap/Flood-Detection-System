import cv2
import numpy as np
from flood_simulation import apply_flood_overlay, get_weather_parameters
from pathfinding import find_path, optimize_path, get_flood_percentage, create_straight_line_path
from path_visualization import display_path

# Global variables
markers = []
current_image = None
flood_mask = None

def draw_markers():
    """Draw visual indicators for the home and destination markers."""
    global markers, current_image
    
    # Create a copy of the current image to draw on
    display_image = current_image.copy()
    
    # Draw each marker with labels
    for i, (x, y) in enumerate(markers):
        # Draw circle for marker
        color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for home, Red for destination
        cv2.circle(display_image, (x, y), 8, color, -1)
        
        # Add label
        label = "Home" if i == 0 else "Destination"
        cv2.putText(display_image, label, (x + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display the image with markers
    cv2.imshow("Place markers (home and destination)", display_image)

def click_event(event, x, y, flags, params):
    global markers, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(markers) < 2:
            markers.append((x, y))
            # Draw markers and labels
            draw_markers()
            if len(markers) == 2:
                print(f"Home set at: {markers[0]}")
                print(f"Destination set at: {markers[1]}")
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow("Place markers (home and destination)")
                find_and_display_path()

def modified_find_path(gray_image, start, end, flood_mask, flood_penalty=1000):
    """
    A custom pathfinding wrapper that uses binary flood avoidance first,
    and only falls back to penalty-based pathfinding if necessary.
    """
    # Step 1: Try to find a path that completely avoids flooded areas
    # Create a binary mask where flooded areas are completely blocked
    binary_mask = np.ones_like(gray_image, dtype=np.uint8) * 255
    binary_mask[flood_mask] = 0
    
    # Try with binary approach first (flood areas completely blocked)
    print("Attempting to find path with complete flood avoidance...")
    
    # Method 1: Use the modified gray image as terrain
    modified_gray = gray_image.copy()
    modified_gray[flood_mask] = 0  # Make flooded areas appear as obstacles
    
    path = find_path(modified_gray, start, end, flood_mask, flood_penalty=float('inf'))
    
    if path is not None and get_flood_percentage(path, flood_mask) == 0:
        print("Found path with 0% flooding!")
        return path, 0
        
    # Method 2: If still no path, try with extremely high penalty
    print("Trying with extreme flood penalty...")
    max_penalty = 10000
    path = find_path(gray_image, start, end, flood_mask, flood_penalty=max_penalty)
    
    if path is not None:
        flood_percent = get_flood_percentage(path, flood_mask)
        print(f"Found path with {flood_percent:.1f}% flooding using penalty {max_penalty}")
        return path, flood_percent
        
    # Method 3: Fall back to progressive penalties if still no path
    print("Falling back to progressive penalties...")
    flood_penalties = [1000, 500, 200, 100, 50]  # Try high to low
    best_path = None
    best_flood_percent = 100
    
    for penalty in flood_penalties:
        path = find_path(gray_image, start, end, flood_mask, flood_penalty=penalty)
        
        if path is not None:
            flood_percent = get_flood_percentage(path, flood_mask)
            print(f"Path with penalty {penalty}: flood percentage = {flood_percent:.1f}%")
            
            if flood_percent < best_flood_percent:
                best_path = path
                best_flood_percent = flood_percent
    
    if best_path is not None:
        return best_path, best_flood_percent
    
    # Method 4: Last resort - direct path
    print("Using direct path as last resort")
    direct_path = create_straight_line_path(start, end)
    direct_flood_percent = get_flood_percentage(direct_path, flood_mask)
    return direct_path, direct_flood_percent

def find_and_display_path():
    global markers, current_image, flood_mask
    start, end = markers
    
    # Check if start or end is in a flooded area
    start_y, start_x = int(start[1]), int(start[0])
    end_y, end_x = int(end[1]), int(end[0])
    
    # Ensure coordinates are within bounds
    h, w = flood_mask.shape[:2]
    start_y, start_x = min(max(start_y, 0), h-1), min(max(start_x, 0), w-1)
    end_y, end_x = min(max(end_y, 0), h-1), min(max(end_x, 0), w-1)
    
    if flood_mask[start_y, start_x]:
        print("Warning: Starting point is in a flooded area")
    
    if flood_mask[end_y, end_x]:
        print("Warning: Destination is in a flooded area")
    
    # Create a grayscale image for pathfinding
    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    
    # Find path using our modified approach
    path, flood_percent = modified_find_path(gray_image, start, end, flood_mask)
    
    # Optimize path
    if path is not None:
        path = optimize_path(path, step_size=5)  # Smaller step size for more precision
        # Recalculate flood percentage after optimization
        flood_percent = get_flood_percentage(path, flood_mask)
        print(f"Final path flood percentage: {flood_percent:.1f}%")
        display_path(current_image, path, flood_mask, flood_percent)
    else:
        print("No path found.")

def main():
    global markers, current_image, flood_mask
    
    try:
        # Get weather parameters for flood simulation
        weather_params = get_weather_parameters()
        
        # Load and process map
        image_path = "image.png"
        
        # Apply flood overlay based on weather conditions
        flooded_image = apply_flood_overlay(image_path, weather_params)
        
        # Extract flood mask - more aggressive threshold to identify more flooded areas
        b_channel = flooded_image[:,:,0]
        # Lower threshold to catch more blue areas
        _, flood_mask_img = cv2.threshold(b_channel, 180, 255, cv2.THRESH_BINARY)  
        flood_mask = flood_mask_img > 0
        
        # Dilate the flood mask to create a buffer zone around flooded areas
        kernel = np.ones((5,5), np.uint8)
        flood_mask = cv2.dilate(flood_mask.astype(np.uint8), kernel, iterations=1) > 0
        
        # Set current image for the click event function
        current_image = flooded_image.copy()
        
        # Let user choose home and destination locations
        cv2.namedWindow("Place markers (home and destination)")
        cv2.setMouseCallback("Place markers (home and destination)", click_event)
        
        # Display instructions
        instruction_image = current_image.copy()
        cv2.putText(instruction_image, "Click to place Home, then Destination", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Place markers (home and destination)", instruction_image)
        
        # Wait until user places both markers
        cv2.waitKey(0)
        
        # If user closed window without setting both markers, use defaults
        if len(markers) < 2:
            print("Using default marker positions.")
            markers = [(135, 275), (540, 725)]  # Example marker positions
            find_and_display_path()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()