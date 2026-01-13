# SAR2Height Prediction Pipeline

A modular pipeline for SAR-to-height prediction using regression and diffusion models with ensemble prediction and uncertainty quantification.

## Overview

This pipeline implements a complete workflow for predicting digital surface model (DSM) heights from SAR (Synthetic Aperture Radar) data using:

1. **Regression Model**: Initial height prediction from SAR features
2. **Diffusion Model**: Refinement and ensemble generation using the regression output as conditioning
3. **Ensemble Analysis**: Statistical analysis of multiple diffusion predictions with uncertainty quantification
4. **Comprehensive Evaluation**: Metrics calculation, visualization, and comparison

## Pipeline Components

### Core Modules

- **`data_manager.py`**: Handles SAR2Height data loading, normalization, and preprocessing
- **`regression_pipeline.py`**: Regression model loading and inference
- **`diffusion_pipeline.py`**: Diffusion model loading and ensemble inference  
- **`ensemble_analyzer.py`**: Statistical analysis of ensemble predictions
- **`metrics_calculator.py`**: Comprehensive evaluation metrics
- **`visualizer.py`**: Advanced visualization and plotting
- **`experiment_manager.py`**: Coordinates all components and manages workflow

### Key Features

- **Modular Design**: Each component can be used independently or as part of the complete pipeline
- **Ensemble Prediction**: Generate multiple predictions with different random seeds for uncertainty quantification
- **Comprehensive Metrics**: Standard regression metrics, height-specific accuracy measures, spatial correlation analysis
- **Advanced Visualization**: Input channels, prediction comparisons, ensemble analysis, statistical plots
- **Experiment Management**: Automated workflow execution with result organization and saving

## Quick Start

### Complete Experiment Workflow

```python
from prediction_pipeline import ExperimentManager

# Model configurations
regression_config = {
    "model_channels": 192,
    "channel_mult": [1, 1, 2, 2, 4, 4],
    "num_res_blocks": 2,
    "attention_resolutions": [16, 8],
    "num_heads": 8,
    "num_head_channels": 64,
    "dropout": 0.1
}

diffusion_config = {
    "model_channels": 192,
    "channel_mult": [1, 1, 2, 2, 4, 4],
    "num_res_blocks": 2,
    "attention_resolutions": [16, 8],
    "num_heads": 8,
    "num_head_channels": 64,
    "dropout": 0.1
}

# Create experiment manager
experiment = ExperimentManager(
    experiment_name="sar2height_prediction",
    output_dir="./results"
)

# Run complete experiment
success = experiment.run_complete_experiment(
    data_file_path="path/to/preprocessed_patches.nc",
    stats_dir="path/to/statistics",
    regression_config=regression_config,
    diffusion_config=diffusion_config,
    regression_checkpoint="path/to/regression/checkpoint.mdlus",
    diffusion_checkpoint="path/to/diffusion/checkpoint.mdlus",
    sample_idx=0,
    ensemble_size=10,
    save_plots=True
)

# Get results summary
if success:
    summary = experiment.get_results_summary()
    print(f"Experiment completed! RMSE: {summary['model_performance']}")
```

### Step-by-Step Usage

```python
from prediction_pipeline import (
    SAR2HeightDataManager,
    SAR2HeightRegressionPipeline,
    SAR2HeightDiffusionPipeline,
    EnsembleAnalyzer,
    SAR2HeightMetrics,
    SAR2HeightVisualizer
)

# 1. Data management
data_manager = SAR2HeightDataManager(data_file, stats_dir)
data_manager.load_statistics()
input_data, output_data, metadata = data_manager.load_sample_data(0)
input_tensor = data_manager.prepare_input_tensor()

# 2. Regression prediction
regression_pipeline = SAR2HeightRegressionPipeline(config, checkpoint)
regression_pipeline.load_model(n_channels)
regression_output, regression_pred = regression_pipeline.predict(input_tensor, data_manager)

# 3. Diffusion ensemble
diffusion_pipeline = SAR2HeightDiffusionPipeline(config, checkpoint)
diffusion_pipeline.load_model(n_channels)
seeds = [42, 123, 456, 789, 101112]
_, ensemble_preds = diffusion_pipeline.predict_ensemble(
    input_tensor, regression_output, data_manager, seeds
)

# 4. Ensemble analysis
analyzer = EnsembleAnalyzer()
stats = analyzer.analyze_ensemble(ensemble_preds)

# 5. Metrics and visualization
metrics_calc = SAR2HeightMetrics()
visualizer = SAR2HeightVisualizer()
# ... continue with analysis
```

## Data Format

The pipeline expects NetCDF files with the following structure:

```
- input group: SAR features (intensity_db, intensity_percentile_rescaled, etc.)
- output group: DSM height data (dsm_height)
- patch_metadata group: Patch metadata for filtering and analysis
```

## Model Requirements

- **Regression Model**: UNet architecture for SAR-to-height direct prediction
- **Diffusion Model**: UNet architecture for conditional height generation
- Both models should be trained on the same SAR2Height dataset with compatible preprocessing

## Output Structure

```
results/
└── experiment_name/
    ├── input_channels.png              # SAR input visualization
    ├── prediction_comparison.png       # Model predictions comparison
    ├── ensemble_analysis.png          # Ensemble uncertainty analysis
    ├── metrics_comparison.png         # Performance metrics
    ├── height_distributions.png       # Statistical distributions
    ├── summary_dashboard.png          # Comprehensive dashboard
    └── experiment_results.json        # Numerical results and metadata
```

## Key Metrics

- **Standard Regression**: RMSE, MAE, R², Correlation
- **Height-Specific**: Vertical accuracy at 1m, 2m, 5m thresholds
- **Ensemble**: Uncertainty calibration, confidence intervals
- **Spatial**: Local correlation, gradient preservation
- **Statistical**: Distribution comparison, significance testing

## Ensemble Analysis

- **Multiple Seeds**: Generate diverse predictions using different random seeds
- **Uncertainty Quantification**: Pixel-wise standard deviation and confidence intervals
- **Calibration Analysis**: Relationship between predicted uncertainty and actual errors
- **Statistical Summaries**: Percentiles, coverage analysis, ensemble spread

## Advanced Features

- **Adaptive Visualization**: Automatic colormap selection and range adjustment
- **Flexible Configuration**: Support for different model architectures and hyperparameters  
- **Result Organization**: Automatic experiment tracking and result storage
- **Error Handling**: Robust error handling with informative messages
- **Memory Efficiency**: Careful memory management for large ensembles

## Dependencies

- PyTorch
- NumPy
- xarray (for NetCDF data handling)
- matplotlib, seaborn (for visualization)
- scikit-learn (for metrics)
- scipy (for statistical analysis)
- physicsnemo (for model architectures)

## Integration with Existing Workflow

This pipeline is designed to integrate seamlessly with your existing SAR2Height training pipeline:

1. Use the same preprocessed NetCDF files generated by your training pipeline
2. Load the trained regression and diffusion model checkpoints
3. Apply the same normalization statistics used during training
4. Generate predictions and analysis for scientific paper preparation

## Scientific Applications

Perfect for:

- **Model Comparison**: Systematic comparison of regression vs diffusion approaches
- **Uncertainty Analysis**: Understanding model confidence and reliability
- **Performance Evaluation**: Comprehensive metrics for scientific publication
- **Visualization**: Professional-quality figures for papers and presentations
- **Ablation Studies**: Testing different ensemble sizes and configurations

## Example Configuration

See `example_usage.py` for complete examples of how to use the pipeline for different scenarios.

---

**Note**: Update the file paths, model configurations, and parameters according to your specific SAR2Height setup and trained models.