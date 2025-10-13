from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type
import numpy as np
from .config import PipelineConfig

class BaseProcessor(ABC):
    """Base class for all processing modules."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = {}
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> Any:
        pass
    
    def get_results(self) -> Dict:
        return self.results
    
    def reset(self):
        self.results = {}

class BaseRegistry:
    """Base registry class with common functionality."""
    
    _registry: Dict[str, Type[BaseProcessor]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a processor class."""
        def decorator(processor_class):
            cls._registry[name] = processor_class
            return processor_class
        return decorator
    
    @classmethod
    def create_processor(cls, name: str, config: PipelineConfig, **kwargs) -> BaseProcessor:
        """Create a processor instance by name."""
        if name not in cls._registry:
            raise ValueError(f"Processor '{name}' not found in {cls.__name__}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](config, **kwargs)
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List all registered processor names."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a processor is registered."""
        return name in cls._registry

class PreprocessorRegistry(BaseRegistry):
    """Registry for image preprocessing algorithms."""
    _registry: Dict[str, Type[BaseProcessor]] = {}

class DenoiserRegistry(BaseRegistry):
    """Registry for image denoising algorithms."""
    _registry: Dict[str, Type[BaseProcessor]] = {}

class EdgeDetectorRegistry(BaseRegistry):
    """Registry for edge detection algorithms."""
    _registry: Dict[str, Type[BaseProcessor]] = {}

class CircleDetectorRegistry(BaseRegistry):
    """Registry for circle detection algorithms."""
    _registry: Dict[str, Type[BaseProcessor]] = {}