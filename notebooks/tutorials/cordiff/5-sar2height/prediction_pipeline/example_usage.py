"""
SAR2Height Prediction Pipeline - Example Usage

This script demonstrates how to use the SAR2Height prediction pipeline
for running regression and diffusion models with ensemble prediction.

Example usage for setting up and running a complete prediction experiment.
"""

# Example configuration and usage
def example_usage():
    """Example of how to use the SAR2Height prediction pipeline."""
    
    from prediction_pipeline import ExperimentManager
    
    # Example model configurations (adjust based on your actual model architectures)
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
    
    # Example paths (update these to match your actual file paths)
    data_file_path = "/path/to/your/preprocessed_patches_test.nc"
    stats_dir = "/path/to/your/statistics/directory"
    regression_checkpoint = "/path/to/regression/checkpoint/UNet.0.123456.mdlus"
    diffusion_checkpoint = "/path/to/diffusion/checkpoint/UNet.0.789012.mdlus"
    
    # Create experiment manager
    experiment = ExperimentManager(
        experiment_name="sar2height_prediction_demo",
        output_dir="./sar2height_results",
        config={
            "description": "SAR2Height prediction with regression and diffusion",
            "ensemble_size": 10,
            "model_type": "UNet"
        }
    )
    
    # Run complete experiment
    success = experiment.run_complete_experiment(
        data_file_path=data_file_path,
        stats_dir=stats_dir,
        regression_config=regression_config,
        diffusion_config=diffusion_config,
        regression_checkpoint=regression_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        sample_idx=0,  # Which sample to analyze
        ensemble_size=10,  # Number of diffusion ensemble members
        ensemble_base_seed=42,
        diffusion_steps=50,  # Number of diffusion sampling steps
        save_plots=True
    )
    
    if success:
        # Get results summary
        summary = experiment.get_results_summary()
        print("Experiment completed successfully!")
        print(f"Results summary: {summary}")
    else:
        print("Experiment failed!")

# Alternative: Step-by-step usage
def step_by_step_usage():
    """Example of step-by-step pipeline usage for more control."""
    
    from prediction_pipeline import (
        SAR2HeightDataManager, 
        SAR2HeightRegressionPipeline,
        SAR2HeightDiffusionPipeline,
        EnsembleAnalyzer,
        SAR2HeightMetrics,
        SAR2HeightVisualizer
    )
    
    # Step 1: Setup data manager
    data_manager = SAR2HeightDataManager(
        data_file_path="/path/to/preprocessed_patches_test.nc",
        stats_dir="/path/to/statistics"
    )
    
    # Load statistics and sample data
    data_manager.load_statistics()
    input_data, output_data, metadata = data_manager.load_sample_data(sample_idx=0)
    
    # Prepare input tensor
    input_tensor = data_manager.prepare_input_tensor()
    ground_truth = data_manager.get_ground_truth()
    
    # Step 2: Setup and run regression
    regression_config = {...}  # Your regression model config
    regression_pipeline = SAR2HeightRegressionPipeline(
        regression_config, 
        "/path/to/regression/checkpoint"
    )
    regression_pipeline.load_model(n_input_channels=input_tensor.shape[1])
    regression_output, regression_pred = regression_pipeline.predict(input_tensor, data_manager)
    
    # Step 3: Setup and run diffusion ensemble
    diffusion_config = {...}  # Your diffusion model config
    diffusion_pipeline = SAR2HeightDiffusionPipeline(
        diffusion_config,
        "/path/to/diffusion/checkpoint"
    )
    diffusion_pipeline.load_model(n_input_channels=input_tensor.shape[1])
    
    # Generate ensemble
    seeds = [42, 123, 456, 789, 101112]  # Example seeds
    ensemble_normalized, ensemble_denormalized = diffusion_pipeline.predict_ensemble(
        input_tensor, regression_output, data_manager, seeds
    )
    
    # Step 4: Analyze ensemble
    analyzer = EnsembleAnalyzer()
    ensemble_stats = analyzer.analyze_ensemble(ensemble_denormalized)
    ensemble_mean = ensemble_stats['basic_statistics']['mean']
    ensemble_std = ensemble_stats['basic_statistics']['std']
    
    # Step 5: Calculate metrics
    metrics_calc = SAR2HeightMetrics()
    comparison_metrics = metrics_calc.compare_models(
        regression_pred, ensemble_mean, ground_truth
    )
    
    # Step 6: Generate visualizations
    visualizer = SAR2HeightVisualizer()
    
    # Input channels
    input_dict = {var: input_data[var].values for var in input_data.data_vars}
    fig1 = visualizer.plot_input_channels(input_dict)
    
    # Prediction comparison
    fig2 = visualizer.plot_prediction_comparison(
        ground_truth, regression_pred, ensemble_mean
    )
    
    # Ensemble analysis
    fig3 = visualizer.plot_ensemble_analysis(
        ensemble_mean, ensemble_std, ground_truth, ensemble_denormalized
    )
    
    print("Step-by-step analysis completed!")


if __name__ == "__main__":
    print("SAR2Height Prediction Pipeline Example")
    print("=" * 50)
    print("1. Complete experiment workflow")
    example_usage()
    
    print("\n2. Step-by-step workflow")
    step_by_step_usage()