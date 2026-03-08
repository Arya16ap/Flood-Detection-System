import cv2
import numpy as np
import noise
from weather import get_weather_parameters

FLOOD_SCALE = 100  

def generate_perlin_noise(height, width, scale):
    """Generates Perlin noise for flood simulation."""
    perlin_noise = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            perlin_noise[i, j] = noise.pnoise2(i / scale, j / scale, octaves=3)
    
    perlin_noise = (perlin_noise - perlin_noise.min()) / (perlin_noise.max() - perlin_noise.min())
    return perlin_noise

def calculate_flood_intensity(weather_params):
    """Determines flood intensity based on storm severity."""
    storm_intensity = (weather_params["Humidity"] / 100 + (1010 - weather_params["Pressure"]) / 200) / 2
    return min(max(storm_intensity, 0.3), 0.9)  

def apply_flood_overlay(image_path, weather_params):
    """Applies Perlin noise-based flood overlay on the map."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image '{image_path}'.")

    height, width, _ = image.shape
    perlin_noise = generate_perlin_noise(height, width, FLOOD_SCALE)
    flood_intensity = calculate_flood_intensity(weather_params)

    # Apply flood mask
    flood_mask = perlin_noise > (1 - flood_intensity)

    blue_overlay = np.zeros_like(image, dtype=np.uint8)
    blue_overlay[:, :, 0] = 255  

    flooded_image = image.copy()
    flooded_image[flood_mask] = cv2.addWeighted(flooded_image[flood_mask], 1 - flood_intensity, 
                                                blue_overlay[flood_mask], flood_intensity, 0)

    return flooded_image
