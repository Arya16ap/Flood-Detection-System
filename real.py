import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label
from PIL import Image, ImageTk
from flood_simulation import apply_flood_overlay, generate_perlin_noise
from grid import create_road_map, GRID_SIZE
from pathfinding import find_path, visualize_path

class FloodNavigationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flood Navigation System")
        
        # Initialize variables
        self.image_path = None
        self.original_image = None
        self.flooded_image = None
        self.flood_mask = None
        self.markers = []
        self.path = None
        
        # Weather parameters (can be adjusted through sliders)
        self.weather_params = {
            "Humidity": 80,
            "Pressure": 1000
        }
        
        # Create UI elements
        self.create_ui()
    
    def create_ui(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Load Image Button
        load_btn = Button(control_frame, text="Load Map", command=self.load_image)
        load_btn.pack(pady=5)
        
        # Weather controls
        Label(control_frame, text="Humidity:").pack(anchor=tk.W)
        self.humidity_slider = Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     command=self.update_weather_param)
        self.humidity_slider.set(self.weather_params["Humidity"])
        self.humidity_slider.pack(fill=tk.X)
        
        Label(control_frame, text="Pressure:").pack(anchor=tk.W)
        self.pressure_slider = Scale(control_frame, from_=970, to=1030, orient=tk.HORIZONTAL,
                                     command=self.update_weather_param)
        self.pressure_slider.set(self.weather_params["Pressure"])
        self.pressure_slider.pack(fill=tk.X)
        
        # Path controls
        Label(control_frame, text="Path Settings:").pack(anchor=tk.W, pady=(10,0))
        Button(control_frame, text="Clear Markers", command=self.clear_markers).pack(fill=tk.X, pady=5)
        Button(control_frame, text="Find Safe Path", command=self.find_safe_path).pack(fill=tk.X)
        
        # Status label
        self.status_label = Label(control_frame, text="Status: Ready")
        self.status_label.pack(pady=10)
        
        # Canvas for displaying the image
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind click event for marker placement
        self.canvas.bind("<Button-1>", self.place_marker)
    
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not self.image_path:
            return
            
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            self.status_label.config(text="Status: Failed to load image")
            return
            
        # Reset markers and path
        self.markers = []
        self.path = None
        
        # Apply initial flood simulation
        self.update_flood_simulation()
    
    def update_weather_param(self, _):
        # Update weather parameters from sliders
        self.weather_params["Humidity"] = self.humidity_slider.get()
        self.weather_params["Pressure"] = self.pressure_slider.get()
        
        # Update flood simulation if an image is loaded
        if self.original_image is not None:
            self.update_flood_simulation()
    
    def update_flood_simulation(self):
        # Apply flood overlay based on current weather parameters
        self.flooded_image = apply_flood_overlay(self.image_path, self.weather_params)
        
        # Extract flood mask (blue channel thresholding)
        b_channel = self.flooded_image[:,:,0]
        _, flood_mask = cv2.threshold(b_channel, 200, 255, cv2.THRESH_BINARY)
        self.flood_mask = flood_mask > 0
        
        # Generate road network if markers exist
        if len(self.markers) > 0:
            road_map = create_road_map(self.flooded_image.shape, self.markers)
            # Overlay roads onto flood map
            self.display_image = cv2.addWeighted(self.flooded_image, 0.8, road_map, 1, 0)
        else:
            self.display_image = self.flooded_image.copy()
        
        # Draw the grid
        self.draw_grid()
        
        # Draw markers on the canvas
        self.draw_markers()
        
        # Draw path if it exists
        if self.path is not None:
            self.display_image = visualize_path(self.display_image, self.path)
        
        # Update canvas
        self.show_image()
    
    def draw_grid(self):
        # Draw a grid on the canvas
        grid_color = "lightgray"
        grid_size = GRID_SIZE
        
        for i in range(0, self.canvas.winfo_width(), grid_size):
            self.canvas.create_line(i, 0, i, self.canvas.winfo_height(), fill=grid_color)
        for j in range(0, self.canvas.winfo_height(), grid_size):
            self.canvas.create_line(0, j, self.canvas.winfo_width(), j, fill=grid_color)
    
    def draw_markers(self):
        # Clear previous markers
        self.canvas.delete("marker")
        
        # Draw each marker as a circle
        for (x, y) in self.markers:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags="marker")
    
    def place_marker(self, event):
        if self.original_image is None:
            self.status_label.config(text="Status: Load an image first")
            return
            
        # Calculate scaling factors
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = self.original_image.shape[1]
        img_height = self.original_image.shape[0]
        
        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height
        
        # Convert canvas coordinates to image coordinates
        x = int(event.x * scale_x)
        y = int(event.y * scale_y)
        
        # Snap to nearest grid intersection
        x = round(x / GRID_SIZE) * GRID_SIZE
        y = round(y / GRID_SIZE) * GRID_SIZE
        
        # Add marker (maximum 2 markers)
        if len(self.markers) < 2:
            self.markers.append((x, y))
            self.status_label.config(text=f"Status: Marker {len(self.markers)} placed at ({x}, {y})")
        else:
            self.status_label.config(text="Status: Maximum of 2 markers reached")
        
        # Reset path
        self.path = None
        
        # Update display
        self.update_flood_simulation()
    
    def clear_markers(self):
        self.markers = []
        self.path = None
        self.status_label.config(text="Status: Markers cleared")
        self.update_flood_simulation()
    
    def find_safe_path(self):
        if len(self.markers) != 2:
            self.status_label.config(text="Status: Need exactly 2 markers")
            return
            
        # Generate road network with current markers
        road_map = create_road_map(self.flooded_image.shape, self.markers)
        
        # Find path from start to end avoiding flooded areas
        gray_image = cv2.cvtColor(self.flooded_image, cv2.COLOR_BGR2GRAY)
        self.path = find_path(gray_image, self.markers[0], self.markers[1], self.flood_mask)
        
        if self.path is not None:
            self.status_label.config(text="Status: Safe path found")
        else:
            self.status_label.config(text="Status: No safe path available")
        
        # Update display
        self.update_flood_simulation()
    
    def show_image(self):
        # Convert OpenCV image to PIL format
        rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
            # Calculate aspect ratio preserving resize
            img_ratio = pil_image.width / pil_image.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                # Image is wider than canvas
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                # Image is taller than canvas
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            
            pil_image = pil_image.resize((new_width, new_height), Image .LANCZOS)
        
        # Convert to PhotoImage and display
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FloodNavigationApp(root)
    root.geometry("1200x800")
    root.mainloop()