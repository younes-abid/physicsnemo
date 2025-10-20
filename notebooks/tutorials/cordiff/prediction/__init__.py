"""
CorrDiff Prediction Pipeline

A modular prediction pipeline for the CorrDiff model consisting of:
1. Data loading and preprocessing
2. Regression model inference 
3. Diffusion model inference
4. Visualization and metrics

Usage:
    from prediction import DataManager, RegressionPipeline, DiffusionPipeline, Visualizer
"""

from .data_manager import DataManager
from .regression_pipeline import RegressionPipeline
from .diffusion_pipeline import DiffusionPipeline
from .visualizer import Visualizer
from .metrics import MetricsCalculator

__all__ = [
    "DataManager",
    "RegressionPipeline", 
    "DiffusionPipeline",
    "Visualizer",
    "MetricsCalculator"
]

__version__ = "1.0.0"