from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json

@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    method: str = "identity_preprocessor"
    params: Dict[str, Any] = field(default_factory=dict)
    
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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    contour_detection: ContourConfig = field(default_factory=ContourConfig)
    circle_detection: CircleDetectionConfig = field(default_factory=CircleDetectionConfig)
    pixel_resolution: float = 0.51  # meters per pixel
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'preprocessing': {
                'method': self.preprocessing.method,
                'params': self.preprocessing.params
            },
            'denoising': {
                'method': self.denoising.method,
                'params': self.denoising.params
            },
            'contour_detection': {
                'method': self.contour_detection.method,
                'params': self.contour_detection.params
            },
            'circle_detection': {
                'method': self.circle_detection.method,
                'params': self.circle_detection.params
            },
            'pixel_resolution': self.pixel_resolution
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        return cls(
            preprocessing=PreprocessingConfig(
                method=config_dict.get('preprocessing', {}).get('method', 'identity_preprocessor'),
                params=config_dict.get('preprocessing', {}).get('params', {})
            ),
            denoising=DenoisingConfig(
                method=config_dict.get('denoising', {}).get('method', 'enhanced_lee'),
                params=config_dict.get('denoising', {}).get('params', {})
            ),
            contour_detection=ContourConfig(
                method=config_dict.get('contour_detection', {}).get('method', 'canny_tank'),
                params=config_dict.get('contour_detection', {}).get('params', {})
            ),
            circle_detection=CircleDetectionConfig(
                method=config_dict.get('circle_detection', {}).get('method', 'multi_scale_tank'),
                params=config_dict.get('circle_detection', {}).get('params', {})
            ),
            pixel_resolution=config_dict.get('pixel_resolution', 0.51)
        )