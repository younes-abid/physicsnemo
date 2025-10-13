"""
Type definitions for oil tank analysis.
"""
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon

@dataclass
class TankMeasurement:
    """Measurements for a single oil tank."""
    height_m: float
    diameter_m: float
    aspect_ratio: float
    measurement_type: str = "OBB"  # OBB = Oriented Bounding Box
    
    def __str__(self):
        return (f"Height: {self.height_m:.2f}m, "
                f"Diameter: {self.diameter_m:.2f}m, "
                f"Aspect: {self.aspect_ratio:.2f} ({self.measurement_type})")

@dataclass
class ProcessedTank:
    """Container for processed tank data."""
    annotation_id: int
    non_rotated_crop: np.ndarray
    rotated_crop: np.ndarray
    rotation_angle: float
    vertical_direction: str  # 'up' or 'down'
    measurements: TankMeasurement
    geometry: Polygon
    pixel_resolution: float
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the rotated crop."""
        if len(self.rotated_crop.shape) == 3:
            return self.rotated_crop.shape[1], self.rotated_crop.shape[2]
        return self.rotated_crop.shape
    
    def __str__(self):
        return (f"Tank {self.annotation_id}: {self.measurements} | "
                f"Rotation: {self.rotation_angle:.1f}° | "
                f"Shape: {self.shape}")