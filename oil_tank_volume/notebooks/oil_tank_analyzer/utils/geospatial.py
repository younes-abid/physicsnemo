"""
Geospatial utilities for oil tank processing.
"""
import math
import numpy as np
import rasterio
from rasterio.mask import mask
import cv2
from shapely.geometry import Polygon
from typing import List, Tuple, Dict, Any, Optional

from ..core.types import TankMeasurement, ProcessedTank


class GeospatialProcessor:
    """Processor for geospatial operations on oil tanks."""
    
    # Earth radius in meters
    EARTH_RADIUS_M = 6371000
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def calculate_pixel_resolution(self, raster_path: str) -> float:
        """Calculate average pixel resolution in meters."""
        with rasterio.open(raster_path) as src:
            transform = src.transform
            crs = src.crs
            
            resolution_x = transform[0]
            resolution_y = -transform[4]
            
            if crs and crs.is_geographic:
                # Convert degrees to meters
                bounds = src.bounds
                center_lat = (bounds.top + bounds.bottom) / 2
                meters_per_degree_lat = 111319.0
                meters_per_degree_lon = 111319.0 * abs(math.cos(math.radians(center_lat)))
                
                resolution_x_m = abs(resolution_x) * meters_per_degree_lon
                resolution_y_m = abs(resolution_y) * meters_per_degree_lat
            else:
                resolution_x_m = abs(resolution_x)
                resolution_y_m = abs(resolution_y)
            
            resolution = (resolution_x_m + resolution_y_m) / 2
            
            if self.verbose:
                print(f"Pixel resolution: {resolution:.2f} meters")
            
            return resolution
    
    def calculate_obb_rotation_angle(self, geometry: Polygon) -> Tuple[float, str]:
        """Calculate rotation angle to align long axis vertically."""
        coords = list(geometry.exterior.coords)
        
        # Calculate edge vectors
        edge1 = np.array(coords[1]) - np.array(coords[0])
        edge2 = np.array(coords[2]) - np.array(coords[1])
        
        len1 = np.linalg.norm(edge1)
        len2 = np.linalg.norm(edge2)
        
        long_edge = edge1 if len1 >= len2 else edge2
        
        # Calculate current angle
        current_angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
        
        # Find best rotation to make vertical
        rotation_1 = current_angle - 90
        rotation_2 = current_angle - (-90)
        
        rotation_1 = (rotation_1 + 180) % 360 - 180
        rotation_2 = (rotation_2 + 180) % 360 - 180
        
        if abs(rotation_1) <= abs(rotation_2):
            return rotation_1, "up"
        else:
            return rotation_2, "down"
    
    def calculate_obb_measurements(self, geometry: Polygon) -> TankMeasurement:
        """Calculate OBB measurements using Haversine distance."""
        coords = list(geometry.exterior.coords)
        
        def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
            """Calculate great-circle distance between two points in meters."""
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)
            
            a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon/2) * math.sin(delta_lon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            return self.EARTH_RADIUS_M * c
        
        # Calculate all edge distances
        edge_distances = []
        for i in range(4):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[(i + 1) % 4]
            distance = haversine_distance(lon1, lat1, lon2, lat2)
            edge_distances.append(distance)
        
        # Sort and identify height/diameter
        edge_distances.sort(reverse=True)
        height_m = edge_distances[0]  # Longest edge
        diameter_m = edge_distances[2]  # Third longest (should be the shorter dimension)
        
        aspect_ratio = height_m / diameter_m if diameter_m > 0 else 0.0
        
        if self.verbose:
            print(f"  Edge distances: {[f'{e:.2f}' for e in edge_distances]} meters")
        
        return TankMeasurement(
            height_m=height_m,
            diameter_m=diameter_m,
            aspect_ratio=aspect_ratio,
            measurement_type="OBB"
        )
    
    def crop_tank_from_raster(self, raster_path: str, geometry: Polygon) -> np.ndarray:
        """Crop tank region from raster."""
        with rasterio.open(raster_path) as src:
            cropped_data, transform = mask(
                src, [geometry], crop=True, all_touched=True, pad=False
            )
            
            # Remove initial padding
            return self._remove_padding(cropped_data)
    
    def _remove_padding(self, image: np.ndarray) -> np.ndarray:
        """Remove zero-padding from image."""
        if image.shape[0] == 1:
            data_mask = image[0] > 0
        else:
            data_mask = np.any(image > 0, axis=0)
        
        rows = np.any(data_mask, axis=1)
        cols = np.any(data_mask, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            return image[:, y_min:y_max+1, x_min:x_max+1]
        
        return image
    
    def rotate_image_to_vertical(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate image to make tank vertical and remove padding."""
        if abs(angle_degrees) < 0.1:
            return image
        
        # Convert to counter-clockwise for OpenCV
        ccw_angle = -angle_degrees
        
        rotated_bands = []
        for band_idx in range(image.shape[0]):
            band_data = image[band_idx].astype(np.float32)
            height, width = band_data.shape
            center = (width // 2, height // 2)
            
            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, ccw_angle, 1.0)
            
            # Calculate new dimensions
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # Adjust rotation matrix for new dimensions
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            rotated_band = cv2.warpAffine(
                band_data, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            rotated_bands.append(rotated_band)
        
        rotated_image = np.stack(rotated_bands, axis=0)
        
        # Remove padding from rotated image
        return self._remove_padding(rotated_image)


class TankProcessor:
    """Main processor for oil tank data."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.geospatial_processor = GeospatialProcessor(verbose)
    
    def process_single_tank(self, raster_path: str, geometry: Polygon, 
                          annotation_id: int) -> ProcessedTank:
        """Process a single oil tank."""
        if self.verbose:
            print(f"Processing tank {annotation_id}...")
        
        # Calculate pixel resolution
        pixel_resolution = self.geospatial_processor.calculate_pixel_resolution(raster_path)
        
        # Calculate rotation angle
        rotation_angle, vertical_direction = self.geospatial_processor.calculate_obb_rotation_angle(geometry)
        
        # Calculate OBB measurements
        measurements = self.geospatial_processor.calculate_obb_measurements(geometry)
        
        # Crop tank from raster
        non_rotated_crop = self.geospatial_processor.crop_tank_from_raster(raster_path, geometry)
        
        # Rotate to vertical
        rotated_crop = self.geospatial_processor.rotate_image_to_vertical(
            non_rotated_crop, rotation_angle
        )
        
        if self.verbose:
            print(f"  {measurements}")
            print(f"  Rotation: {rotation_angle:.2f}° ({vertical_direction})")
            print(f"  Crop shape: {non_rotated_crop.shape} → {rotated_crop.shape}")
        
        return ProcessedTank(
            annotation_id=annotation_id,
            non_rotated_crop=non_rotated_crop,
            rotated_crop=rotated_crop,
            rotation_angle=rotation_angle,
            vertical_direction=vertical_direction,
            measurements=measurements,
            geometry=geometry,
            pixel_resolution=pixel_resolution
        )