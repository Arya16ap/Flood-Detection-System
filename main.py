import cv2
import numpy as np
from flood_simulation import apply_flood_overlay
from weather import get_weather_parameters
from pathfinding import find_path, visualize_path, optimize_path, get_flood_percentage, create_straight_line_path

# Global variables to store marker positions
markers = []
current_image = None

def click_event(event, x, y, flags, params):
    """
    Mouse callback function to handle marker placement.
    Allows user to place up to 2 markers (home and destination).
    """
    global markers, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only allow 2 markers (home and destination)
        if len(markers) < 2:
            markers.append((x, y))
            
            # Create a copy of the image to draw markers
            temp_image = current_image.copy()
            
            # Draw markers with different colors for home and destination
            for i, marker in enumerate(markers):
                # First marker (home) is green, second marker (destination) is blue
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.circle(temp_image, marker, 8, color, -1)
                
                # Add label
                label = "Home" if i == 0 else "Destination"
                cv2.putText(temp_image, label, (marker[0]+10, marker[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show image with markers
            cv2.imshow("Place markers (home and destination)", temp_image)
            
            # When both markers are placed, close the window
            if len(markers) == 2:
                print(f"Home set at: {markers[0]}")
                print(f"Destination set at: {markers[1]}")
                cv2.destroyWindow("Place markers (home and destination)")

def main():
    global markers, current_image
    
    # Get weather parameters for flood simulation
    weather_params = get_weather_parameters()
    
    # Load and process map
    image_path = "image.png"
    
    # Apply flood overlay based on weather conditions
    flooded_image = apply_flood_overlay(image_path, weather_params)
    
    # Extract flood mask (where blue intensity is high)
    # This assumes the flood overlay uses blue color (BGR where B is high)
    b_channel = flooded_image[:,:,0]
    _, flood_mask = cv2.threshold(b_channel, 200, 255, cv2.THRESH_BINARY)
    flood_mask = flood_mask > 0
    
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
    
    start, end = markers
    
    # Custom marker color (BGR format for OpenCV)
    marker_color = (156, 107, 44)  # #2C6B9C in BGR format
    
    # Find optimal path (potentially going through floods if necessary)
    # Using flood penalty to discourage but not prevent flood traversal
    gray_image = cv2.cvtColor(flooded_image, cv2.COLOR_BGR2GRAY)
    
    # Try with a high penalty first to avoid floods if possible
    path = find_path(gray_image, start, end, flood_mask, flood_penalty=20)
    
    # If A* algorithm couldn't find a path (which shouldn't happen with our fallback),
    # create a straight line as a backup
    if path is None:
        print("Using straight line connection between markers.")
        path = create_straight_line_path(start, end)
    else:
        # Optimize path to remove unnecessary zigzagging
        path = optimize_path(path, step_size=10)
    
    # Calculate flood percentage
    flood_percent = get_flood_percentage(path, flood_mask)
    print(f"Path found with {flood_percent:.1f}% going through flooded areas")
    
    # Visualize the path on the flooded image with custom color
    result = visualize_path(flooded_image, path, marker_color=marker_color, show_flood=True, flood_mask=flood_mask)
    
    # Add text information about the path and flood percentage
    cv2.putText(result, f"Flood-affected path: {flood_percent:.1f}%", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the result
    cv2.imshow("Optimal Flood-Aware Navigation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()