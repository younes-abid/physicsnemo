from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class DenoisingConfig:
    method: str = "median"
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class ContourConfig:
    method: str = "adaptive_threshold"
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class CircleDetectionConfig:
    method: str = "contour_based"
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class PipelineConfig:
    denoising: DenoisingConfig = None
    contour_detection: ContourConfig = None
    circle_detection: CircleDetectionConfig = None
    pixel_resolution: float = 0.51
    
    def __post_init__(self):
        if self.denoising is None:
            self.denoising = DenoisingConfig()
        if self.contour_detection is None:
            self.contour_detection = ContourConfig()
        if self.circle_detection is None:
            self.circle_detection = CircleDetectionConfig()
    
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=lambda x: x.__dict__)