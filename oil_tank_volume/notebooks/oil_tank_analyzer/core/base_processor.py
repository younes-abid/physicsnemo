from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
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

class ProcessorRegistry:
    """Registry pattern for managing different algorithms."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(processor_class):
            cls._registry[name] = processor_class
            return processor_class
        return decorator
    
    @classmethod
    def create_processor(cls, name: str, config: PipelineConfig, **kwargs) -> BaseProcessor:
        if name not in cls._registry:
            raise ValueError(f"Processor '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name](config, **kwargs)
    
    @classmethod
    def list_processors(cls) -> List[str]:
        return list(cls._registry.keys())