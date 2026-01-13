"""
SAR2Height Prediction Pipeline

A modular pipeline for SAR-to-height prediction using regression and diffusion models.
Supports ensemble prediction with uncertainty quantification.

Modules:
- data_manager: Handles SAR2Height data loading, normalization, and preprocessing
- regression_pipeline: Regression model loading and inference
- diffusion_pipeline: Diffusion model loading and ensemble inference
- ensemble_analyzer: Statistical analysis of ensemble predictions
- metrics_calculator: Comprehensive evaluation metrics
- visualizer: Advanced visualization and plotting
- experiment_manager: Experiment configuration and coordination
"""

__version__ = "1.0.0"
__author__ = "SAR2Height Team"

from .data_manager import SAR2HeightDataManager
from .regression_pipeline import SAR2HeightRegressionPipeline
from .diffusion_pipeline import SAR2HeightDiffusionPipeline
from .ensemble_analyzer import EnsembleAnalyzer
from .metrics_calculator import SAR2HeightMetrics
from .visualizer import SAR2HeightVisualizer
from .experiment_manager import ExperimentManager

__all__ = [
    "SAR2HeightDataManager",
    "SAR2HeightRegressionPipeline", 
    "SAR2HeightDiffusionPipeline",
    "EnsembleAnalyzer",
    "SAR2HeightMetrics",
    "SAR2HeightVisualizer",
    "ExperimentManager"
]