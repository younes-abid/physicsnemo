"""
SAR2Height Experiment Manager

Coordinates all components of the SAR2Height prediction pipeline.
Manages experiment configuration, execution flow, and result organization.
"""

import os
import json
import time
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from data_manager import SAR2HeightDataManager
from regression_pipeline import SAR2HeightRegressionPipeline
from diffusion_pipeline import SAR2HeightDiffusionPipeline
from ensemble_analyzer import EnsembleAnalyzer
from metrics_calculator import SAR2HeightMetrics
from visualizer import SAR2HeightVisualizer

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages the complete SAR2Height prediction experiment workflow.
    
    Coordinates:
    - Data loading and preprocessing
    - Model loading and configuration
    - Prediction execution (regression + diffusion + ensemble)
    - Metrics calculation and analysis
    - Visualization generation
    - Results saving and organization (NetCDF for raw data, JSON for analysis)
    """
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: str = "./results",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ExperimentManager.
        
        Args:
            experiment_name: Name for this experiment
            output_dir: Directory to save results
            config: Experiment configuration dictionary
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Create experiment directory
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized storage
        self.predictions_dir = self.experiment_dir / "predictions"
        self.analysis_dir = self.experiment_dir / "analysis" 
        self.visualizations_dir = self.experiment_dir / "visualizations"
        
        for subdir in [self.predictions_dir, self.analysis_dir, self.visualizations_dir]:
            subdir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_manager = None
        self.regression_pipeline = None
        self.diffusion_pipeline = None
        self.ensemble_analyzer = EnsembleAnalyzer()
        self.metrics_calculator = SAR2HeightMetrics()
        self.visualizer = SAR2HeightVisualizer()
        
        # Store checkpoint paths for diffusion loss function
        self.regression_checkpoint_path = None
        self.diffusion_checkpoint_path = None
        
        # Results storage
        self.results = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'config': self.config
            },
            'data': {},
            'predictions': {},
            'metrics': {},
            'ensemble_analysis': {},
            'visualizations': {},
            'file_paths': {
                'predictions_netcdf': None,
                'ensemble_netcdf': None,
                'metrics_json': None,
                'analysis_json': None
            }
        }
        
        print(f"🚀 SAR2Height Experiment Manager initialized: {experiment_name}")
        print(f"📁 Results will be saved to: {self.experiment_dir}")
        print(f"📊 Predictions: {self.predictions_dir}")
        print(f"📈 Analysis: {self.analysis_dir}")
        print(f"🎨 Visualizations: {self.visualizations_dir}")
    
    def setup_data(self, 
                   data_file_path: str,
                   stats_dir: str,
                   sample_idx: int = 0,
                   selected_variables: Optional[List[str]] = None) -> bool:
        """
        Setup data manager and load sample data.
        
        Args:
            data_file_path: Path to preprocessed NetCDF file
            stats_dir: Directory containing statistics files
            sample_idx: Index of sample to load
            selected_variables: List of input variables to use
            
        Returns:
            Success status
        """
        print("=== Setting up Data Manager ===")
        
        try:
            # Initialize data manager
            self.data_manager = SAR2HeightDataManager(data_file_path, stats_dir)
            
            # Load statistics and data
            self.data_manager.load_statistics()
            input_data, output_data, metadata = self.data_manager.load_sample_data(sample_idx)
            
            # Store data information
            self.results['data'] = {
                'file_path': data_file_path,
                'stats_dir': stats_dir,
                'sample_idx': sample_idx,
                'input_variables': self.data_manager.input_variables,
                'output_variables': self.data_manager.output_variables,
                'selected_variables': selected_variables or self.data_manager.input_variables,
                'data_shape': dict(input_data.dims),
                'metadata': self.data_manager.get_patch_metadata()
            }
            
            print(f"✓ Data setup completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Data setup failed: {e}")
            return False
    
    def setup_models(self,
                    regression_config: Dict[str, Any],
                    diffusion_config: Dict[str, Any],
                    regression_checkpoint: str,
                    diffusion_checkpoint: str) -> bool:
        """
        Setup regression and diffusion pipelines.
        
        Args:
            regression_config: Regression model configuration
            diffusion_config: Diffusion model configuration  
            regression_checkpoint: Path to regression checkpoint
            diffusion_checkpoint: Path to diffusion checkpoint
            
        Returns:
            Success status
        """
        print("=== Setting up Model Pipelines ===")
        
        if self.data_manager is None:
            print("✗ Data manager not initialized. Call setup_data() first.")
            return False
        
        try:
            # Store checkpoint paths for diffusion loss function
            self.regression_checkpoint_path = regression_checkpoint
            self.diffusion_checkpoint_path = diffusion_checkpoint
            
            # Determine number of input channels
            selected_vars = self.results['data']['selected_variables']
            n_input_channels = len(selected_vars)
            
            # Setup regression pipeline
            print("🤖 Setting up regression pipeline...")
            self.regression_pipeline = SAR2HeightRegressionPipeline(
                regression_config, regression_checkpoint
            )
            regression_loaded = self.regression_pipeline.load_model(n_input_channels)
            
            # Setup diffusion pipeline  
            print("🌀 Setting up diffusion pipeline...")
            self.diffusion_pipeline = SAR2HeightDiffusionPipeline(
                diffusion_config, diffusion_checkpoint
            )
            diffusion_loaded = self.diffusion_pipeline.load_model(n_input_channels)
            
            # NEW: Create diffusion loss function with regression checkpoint
            if diffusion_loaded:
                print("🔧 Creating diffusion loss function with regression network...")
                loss_function_created = self.diffusion_pipeline.create_loss_function(
                    regression_checkpoint
                )
                if not loss_function_created:
                    print("✗ Failed to create diffusion loss function")
                    diffusion_loaded = False
                else:
                    print("✓ Diffusion loss function created successfully")
            
            # Test models
            # if regression_loaded:
            #     self.regression_pipeline.test_model(n_input_channels)
            
            # if diffusion_loaded:
            #     self.diffusion_pipeline.test_model(n_input_channels)
            
            # Store model information
            self.results['models'] = {
                'regression': self.regression_pipeline.get_model_info(),
                'diffusion': self.diffusion_pipeline.get_model_info(),
                'n_input_channels': n_input_channels,
                'regression_checkpoint': regression_checkpoint,
                'diffusion_checkpoint': diffusion_checkpoint
            }
            
            success = regression_loaded and diffusion_loaded
            if success:
                print(f"✓ Model pipelines setup completed successfully")
            else:
                print(f"⚠ Model setup partially completed (regression: {regression_loaded}, diffusion: {diffusion_loaded})")
            
            return success
            
        except Exception as e:
            print(f"✗ Model setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_predictions(self,
                       ensemble_size: int = 10,
                       ensemble_base_seed: int = 42,
                       diffusion_steps: Optional[int] = None) -> bool:
        """
        Run complete prediction pipeline: regression → diffusion → ensemble.
        
        Args:
            ensemble_size: Number of ensemble members to generate
            ensemble_base_seed: Base seed for ensemble generation
            diffusion_steps: Number of diffusion sampling steps
            
        Returns:
            Success status
        """
        print("=== Running SAR2Height Predictions ===")
        
        if self.data_manager is None or self.regression_pipeline is None or self.diffusion_pipeline is None:
            print("✗ Prerequisites not met. Ensure setup_data() and setup_models() completed successfully.")
            return False
            
        try:
            # Prepare input tensor
            selected_vars = self.results['data']['selected_variables']
            input_tensor = self.data_manager.prepare_input_tensor(
                selected_variables=selected_vars, 
                apply_normalization=True
            )
            ground_truth = self.data_manager.get_ground_truth()
            
            print(f"📊 Input prepared: {input_tensor.shape}, Variables: {selected_vars}")
            print(f"🎯 Ground truth shape: {ground_truth.shape}")
            
            # Step 1: Regression prediction
            print("\n🤖 Running regression prediction...")
            regression_output, regression_denorm = self.regression_pipeline.predict(
                input_tensor, self.data_manager
            )
            print(f"✓ Regression completed - Output shape: {regression_denorm.shape}")
            
            # Step 2: Single diffusion prediction for testing
            print("\n🌀 Running single diffusion prediction...")
            diffusion_output, diffusion_denorm = self.diffusion_pipeline.predict(
                input_tensor, regression_output, self.data_manager
            )
            print(f"✓ Single diffusion completed - Output shape: {diffusion_denorm.shape}")
            
            # Step 3: Ensemble diffusion prediction
            print(f"\n🎲 Running ensemble diffusion prediction (size: {ensemble_size})...")
            ensemble_seeds = self.diffusion_pipeline.generate_ensemble_seeds(
                ensemble_size, ensemble_base_seed
            )
            
            # Set diffusion steps if specified
            if diffusion_steps:
                self.diffusion_pipeline.set_sampling_steps(diffusion_steps)
            
            ensemble_normalized, ensemble_denormalized = self.diffusion_pipeline.predict_ensemble(
                input_tensor, regression_output, self.data_manager, 
                ensemble_seeds, diffusion_steps
            )
            print(f"✓ Ensemble completed - {len(ensemble_denormalized)} members generated")
            
            # Store prediction results
            self.results['predictions'] = {
                'ground_truth': ground_truth,
                'regression_output': regression_denorm,
                'diffusion_output': diffusion_denorm,
                'ensemble_members': ensemble_denormalized,
                'ensemble_seeds': ensemble_seeds,
                'input_tensor_shape': list(input_tensor.shape),
                'selected_variables': selected_vars
            }
            
            print(f"✅ All predictions completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_ensemble_analysis(self, 
                            confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> bool:
        """
        Run ensemble analysis on generated predictions.
        
        Args:
            confidence_levels: Confidence levels for uncertainty quantification
            
        Returns:
            Success status
        """
        print("=== Running Ensemble Analysis ===")
        
        if 'ensemble_members' not in self.results.get('predictions', {}):
            print("✗ No ensemble predictions available. Run run_predictions() first.")
            return False
        
        try:
            ensemble_members = self.results['predictions']['ensemble_members']
            
            # Run ensemble analysis
            ensemble_stats = self.ensemble_analyzer.analyze_ensemble(
                ensemble_members, confidence_levels=confidence_levels
            )
            
            # Store ensemble analysis results
            self.results['ensemble_analysis'] = ensemble_stats
            
            # Add ensemble statistics to predictions for easy access
            self.results['predictions'].update({
                'ensemble_mean': ensemble_stats['basic_statistics']['mean'],
                'ensemble_std': ensemble_stats['basic_statistics']['std'],
                'ensemble_confidence_intervals': ensemble_stats['confidence_intervals']
            })
            
            print(f"✓ Ensemble analysis completed")
            print(f"  - Basic statistics computed")
            print(f"  - Confidence intervals: {confidence_levels}")
            print(f"  - Spatial statistics calculated")
            
            return True
            
        except Exception as e:
            print(f"✗ Ensemble analysis failed: {e}")
            return False
    
    def calculate_metrics(self) -> bool:
        """
        Calculate comprehensive prediction metrics.
        
        Returns:
            Success status
        """
        print("=== Calculating Prediction Metrics ===")
        
        predictions = self.results.get('predictions', {})
        if not all(key in predictions for key in ['ground_truth', 'regression_output', 'ensemble_mean']):
            print("✗ Missing required predictions. Run run_predictions() and run_ensemble_analysis() first.")
            return False
        
        try:
            ground_truth = predictions['ground_truth']
            regression_pred = predictions['regression_output']
            ensemble_mean = predictions['ensemble_mean']
            ensemble_std = predictions.get('ensemble_std')
            
            # Model comparison metrics
            comparison_metrics = self.metrics_calculator.compare_models(
                regression_pred, ensemble_mean, ground_truth
            )
            
            # Ensemble-specific metrics
            if ensemble_std is not None:
                ensemble_metrics = self.metrics_calculator.calculate_ensemble_metrics(
                    ensemble_mean, ensemble_std, ground_truth
                )
            else:
                ensemble_metrics = {}
            
            # Store metrics
            self.results['metrics'] = {
                'comparison_metrics': comparison_metrics,
                'ensemble_metrics': ensemble_metrics,
                'model_performance': {
                    'regression_rmse': comparison_metrics.get('regression_metrics', {}).get('rmse'),
                    'diffusion_rmse': comparison_metrics.get('diffusion_metrics', {}).get('rmse'),
                    'ensemble_rmse': comparison_metrics.get('diffusion_metrics', {}).get('rmse'),  # Same as diffusion for ensemble mean
                }
            }
            
            print(f"✓ Metrics calculation completed")
            
            # Display key metrics
            reg_rmse = comparison_metrics.get('regression_metrics', {}).get('rmse')
            diff_rmse = comparison_metrics.get('diffusion_metrics', {}).get('rmse')
            
            if reg_rmse is not None:
                print(f"  🤖 Regression RMSE: {reg_rmse:.4f} m")
            if diff_rmse is not None:
                print(f"  🌀 Ensemble RMSE: {diff_rmse:.4f} m")
            
            return True
            
        except Exception as e:
            print(f"✗ Metrics calculation failed: {e}")
            return False
    
    def save_predictions_to_netcdf(self) -> str:
        """
        Save prediction data to NetCDF files.
        
        Returns:
            Path to main predictions NetCDF file
        """
        print("=== Saving Predictions to NetCDF ===")
        
        predictions = self.results.get('predictions', {})
        if not predictions:
            raise ValueError("No predictions to save. Run predictions first.")
        
        # Main predictions file
        predictions_file = self.predictions_dir / f"{self.experiment_name}_predictions.nc"
        
        # Create xarray datasets for structured storage
        data_vars = {}
        coords = {}
        attrs = {}
        
        # Get dimensions from ground truth
        ground_truth = predictions['ground_truth']
        height, width = ground_truth.shape
        
        # Create coordinates
        coords['y'] = np.arange(height)
        coords['x'] = np.arange(width)
        
        # Add prediction arrays
        data_vars['ground_truth'] = (['y', 'x'], ground_truth.astype(np.float32))
        data_vars['regression_prediction'] = (['y', 'x'], predictions['regression_output'].astype(np.float32))
        data_vars['diffusion_prediction'] = (['y', 'x'], predictions['diffusion_output'].astype(np.float32))
        
        if 'ensemble_mean' in predictions:
            data_vars['ensemble_mean'] = (['y', 'x'], predictions['ensemble_mean'].astype(np.float32))
        if 'ensemble_std' in predictions:
            data_vars['ensemble_std'] = (['y', 'x'], predictions['ensemble_std'].astype(np.float32))
        
        # Add metadata as attributes
        attrs.update({
            'experiment_name': self.experiment_name,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sample_idx': self.results['data']['sample_idx'],
            'selected_variables': ','.join(predictions['selected_variables']),
            'description': 'SAR2Height prediction results'
        })
        
        # Create and save dataset
        ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        ds.to_netcdf(predictions_file, mode='w')
        
        print(f"💾 Predictions saved to: {predictions_file.name}")
        
        # Save ensemble members separately due to size
        if 'ensemble_members' in predictions:
            ensemble_file = self.predictions_dir / f"{self.experiment_name}_ensemble_members.nc"
            self._save_ensemble_members_netcdf(predictions['ensemble_members'], ensemble_file)
            self.results['file_paths']['ensemble_netcdf'] = str(ensemble_file)
        
        self.results['file_paths']['predictions_netcdf'] = str(predictions_file)
        return str(predictions_file)
    
    def _save_ensemble_members_netcdf(self, ensemble_members: List[np.ndarray], file_path: Path):
        """Save ensemble members to separate NetCDF file."""
        if not ensemble_members:
            return
        
        # Stack ensemble members
        ensemble_array = np.stack(ensemble_members, axis=0).astype(np.float32)
        n_members, height, width = ensemble_array.shape
        
        # Create coordinates
        coords = {
            'member': np.arange(n_members),
            'y': np.arange(height),
            'x': np.arange(width)
        }
        
        # Create dataset
        data_vars = {
            'ensemble_predictions': (['member', 'y', 'x'], ensemble_array)
        }
        
        attrs = {
            'experiment_name': self.experiment_name,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_ensemble_members': n_members,
            'description': 'SAR2Height ensemble prediction members'
        }
        
        ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        ds.to_netcdf(file_path, mode='w')
        
        print(f"💾 Ensemble members saved to: {file_path.name}")
    
    def save_analysis_to_json(self) -> str:
        """
        Save analysis results (metrics, ensemble analysis) to JSON files.
        
        Returns:
            Path to main analysis JSON file
        """
        print("=== Saving Analysis to JSON ===")
        
        # Main analysis file
        analysis_file = self.analysis_dir / f"{self.experiment_name}_analysis.json"
        
        # Prepare analysis data (exclude large arrays)
        analysis_data = {
            'experiment_info': self.results['experiment_info'],
            'data_summary': {
                'sample_idx': self.results['data']['sample_idx'],
                'selected_variables': self.results['data']['selected_variables'],
                'data_shape': self.results['data']['data_shape']
            },
            'model_info': self.results.get('models', {}),
            'metrics': self.results.get('metrics', {}),
            'file_paths': self.results['file_paths']
        }
        
        # Add ensemble analysis summary (without large arrays)
        if 'ensemble_analysis' in self.results:
            ensemble_summary = self._create_ensemble_summary()
            analysis_data['ensemble_summary'] = ensemble_summary
        
        # Save analysis file
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=self._json_serialize)
        
        print(f"📊 Analysis saved to: {analysis_file.name}")
        
        # Save detailed metrics separately
        if 'metrics' in self.results:
            metrics_file = self.analysis_dir / f"{self.experiment_name}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.results['metrics'], f, indent=2, default=self._json_serialize)
            print(f"📈 Detailed metrics saved to: {metrics_file.name}")
            self.results['file_paths']['metrics_json'] = str(metrics_file)
        
        self.results['file_paths']['analysis_json'] = str(analysis_file)
        return str(analysis_file)
    
    def _create_ensemble_summary(self) -> Dict[str, Any]:
        """Create a summary of ensemble analysis without large arrays."""
        ensemble_stats = self.results['ensemble_analysis']
        
        return {
            'ensemble_info': ensemble_stats.get('ensemble_info', {}),
            'global_statistics': ensemble_stats.get('global_statistics', {}),
            'spread_analysis': {
                'pairwise_correlations': ensemble_stats.get('spread_analysis', {}).get('pairwise_correlations', {}),
                'rank_statistics': {
                    k: v for k, v in ensemble_stats.get('spread_analysis', {}).get('rank_statistics', {}).items() 
                    if not isinstance(v, np.ndarray)
                },
                'range_statistics': {
                    k: v for k, v in ensemble_stats.get('spread_analysis', {}).get('range_statistics', {}).items() 
                    if not isinstance(v, np.ndarray)
                }
            },
            'uncertainty_summary': {
                'mean_uncertainty': float(np.mean(ensemble_stats['basic_statistics']['std'])),
                'max_uncertainty': float(np.max(ensemble_stats['basic_statistics']['std'])),
                'min_uncertainty': float(np.min(ensemble_stats['basic_statistics']['std']))
            }
        }
    
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def save_results(self) -> bool:
        """
        Save complete experiment results using appropriate file formats.
        
        Returns:
            Success status
        """
        print("=== Saving Experiment Results ===")
        
        try:
            # Save predictions to NetCDF
            if self.results.get('predictions'):
                self.save_predictions_to_netcdf()
            
            # Save analysis to JSON
            self.save_analysis_to_json()
            
            # Save experiment metadata
            metadata_path = self.experiment_dir / "experiment_metadata.json"
            with open(metadata_path, 'w') as f:
                metadata = {
                    'experiment_info': self.results['experiment_info'],
                    'data_summary': self.results.get('data', {}),
                    'model_summary': self.results.get('models', {}),
                    'file_paths': self.results['file_paths'],
                    'directory_structure': {
                        'predictions': str(self.predictions_dir),
                        'analysis': str(self.analysis_dir),
                        'visualizations': str(self.visualizations_dir)
                    }
                }
                json.dump(metadata, f, indent=2, default=self._json_serialize)
            
            print(f"💾 Experiment metadata saved: {metadata_path.name}")
            print(f"✅ Results saved successfully to: {self.experiment_dir}")
            
            # Print summary of saved files
            print("\n📁 Saved Files:")
            for key, path in self.results['file_paths'].items():
                if path:
                    print(f"  {key}: {Path(path).name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Results saving failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @classmethod
    def load_experiment(cls, experiment_dir: str) -> 'ExperimentManager':
        """
        Load a previously saved experiment.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Loaded ExperimentManager instance
        """
        experiment_path = Path(experiment_dir)
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load experiment metadata
        metadata_file = experiment_path / "experiment_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Experiment metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create new instance
        experiment_name = metadata['experiment_info']['name']
        output_dir = str(experiment_path.parent)
        config = metadata['experiment_info'].get('config', {})
        
        manager = cls(experiment_name, output_dir, config)
        
        # Load saved results
        manager.results.update({
            'experiment_info': metadata['experiment_info'],
            'data': metadata.get('data_summary', {}),
            'models': metadata.get('model_summary', {}),
            'file_paths': metadata.get('file_paths', {})
        })
        
        print(f"📂 Experiment loaded: {experiment_name}")
        print(f"📅 Created: {metadata['experiment_info'].get('timestamp', 'Unknown')}")
        
        return manager
    
    def load_predictions(self) -> Dict[str, Any]:
        """
        Load prediction data from NetCDF files.
        
        Returns:
            Dictionary containing loaded prediction data
        """
        predictions_file = self.results['file_paths'].get('predictions_netcdf')
        if not predictions_file or not Path(predictions_file).exists():
            raise FileNotFoundError("Predictions NetCDF file not found")
        
        print(f"📊 Loading predictions from: {Path(predictions_file).name}")
        
        # Load main predictions
        ds = xr.open_dataset(predictions_file)
        predictions = {}
        
        for var in ds.data_vars:
            predictions[var] = ds[var].values
        
        # Load ensemble members if available
        ensemble_file = self.results['file_paths'].get('ensemble_netcdf')
        if ensemble_file and Path(ensemble_file).exists():
            print(f"🎲 Loading ensemble members from: {Path(ensemble_file).name}")
            ensemble_ds = xr.open_dataset(ensemble_file)
            ensemble_array = ensemble_ds['ensemble_predictions'].values
            predictions['ensemble_members'] = [ensemble_array[i] for i in range(ensemble_array.shape[0])]
            ensemble_ds.close()
        
        ds.close()
        
        # Store in results
        self.results['predictions'] = predictions
        
        print(f"✓ Predictions loaded successfully")
        return predictions
    
    def load_analysis(self) -> Dict[str, Any]:
        """
        Load analysis data from JSON files.
        
        Returns:
            Dictionary containing loaded analysis data
        """
        analysis_file = self.results['file_paths'].get('analysis_json')
        if not analysis_file or not Path(analysis_file).exists():
            raise FileNotFoundError("Analysis JSON file not found")
        
        print(f"📈 Loading analysis from: {Path(analysis_file).name}")
        
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Load detailed metrics if available
        metrics_file = self.results['file_paths'].get('metrics_json')
        if metrics_file and Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                detailed_metrics = json.load(f)
                analysis['detailed_metrics'] = detailed_metrics
        
        # Store in results
        self.results.update({
            'metrics': analysis.get('metrics', {}),
            'ensemble_analysis': {'summary': analysis.get('ensemble_summary', {})}
        })
        
        print(f"✓ Analysis loaded successfully")
        return analysis
    
    def generate_visualizations(self, save_plots: bool = True) -> bool:
        """
        Generate all visualizations for the experiment.
        
        Args:
            save_plots: Whether to save plots to files
            
        Returns:
            Success status
        """
        print("=== Generating Visualizations ===")
        
        predictions = self.results.get('predictions', {})
        ensemble_analysis = self.results.get('ensemble_analysis', {})
        
        if not predictions:
            # Try to load predictions if not in memory
            try:
                predictions = self.load_predictions()
            except FileNotFoundError:
                print("✗ Missing prediction data. Run complete pipeline first.")
                return False
        
        try:
            # Prepare visualization data
            ground_truth = predictions['ground_truth']
            regression_pred = predictions['regression_output']
            ensemble_mean = predictions['ensemble_mean']
            ensemble_std = predictions['ensemble_std']
            ensemble_members = predictions['ensemble_members']
            selected_vars = predictions['selected_variables']
            
            # Prepare input data for visualization
            input_data_dict = {}
            for var in selected_vars:
                input_data_dict[var] = self.data_manager.input_data[var].values
            
            visualization_paths = {}
            
            # Generate visualizations
            print("🎨 Generating input channels visualization...")
            fig1 = self.visualizer.plot_input_channels(input_data_dict)
            if save_plots:
                path1 = self.visualizations_dir / "input_channels.png"
                fig1.savefig(path1, dpi=150, bbox_inches='tight')
                visualization_paths['input_channels'] = str(path1)
                print(f"  💾 Saved: {path1.name}")
            
            print("🎨 Generating prediction comparison...")
            fig2 = self.visualizer.plot_prediction_comparison(
                ground_truth, regression_pred, ensemble_mean
            )
            if save_plots:
                path2 = self.visualizations_dir / "prediction_comparison.png"
                fig2.savefig(path2, dpi=150, bbox_inches='tight')
                visualization_paths['prediction_comparison'] = str(path2)
                print(f"  💾 Saved: {path2.name}")
            
            print("🎨 Generating ensemble analysis...")
            fig3 = self.visualizer.plot_ensemble_analysis(
                ensemble_mean, ensemble_std, ground_truth, ensemble_members
            )
            if save_plots:
                path3 = self.visualizations_dir / "ensemble_analysis.png"
                fig3.savefig(path3, dpi=150, bbox_inches='tight')
                visualization_paths['ensemble_analysis'] = str(path3)
                print(f"  💾 Saved: {path3.name}")
            
            # Store visualization information
            self.results['visualizations'] = {
                'generated': True,
                'save_plots': save_plots,
                'paths': visualization_paths,
                'figures_generated': ['input_channels', 'prediction_comparison', 'ensemble_analysis']
            }
            
            print(f"✅ Visualization generation completed")
            if save_plots:
                print(f"📁 Saved {len(visualization_paths)} plots to: {self.visualizations_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_experiment(self,
                              data_file_path: str,
                              stats_dir: str,
                              regression_config: Dict[str, Any],
                              diffusion_config: Dict[str, Any],
                              regression_checkpoint: str,
                              diffusion_checkpoint: str,
                              sample_idx: int = 0,
                              selected_variables: Optional[List[str]] = None,
                              ensemble_size: int = 10,
                              ensemble_base_seed: int = 42,
                              diffusion_steps: Optional[int] = None,
                              save_plots: bool = True) -> bool:
        """
        Run the complete SAR2Height prediction experiment workflow.
        
        This method orchestrates the entire pipeline:
        1. Data setup and loading
        2. Model initialization and loading
        3. Prediction generation (regression + diffusion + ensemble)
        4. Ensemble analysis and uncertainty quantification
        5. Metrics calculation
        6. Visualization generation
        7. Results saving (NetCDF for predictions, JSON for analysis)
        
        Returns:
            Success status
        """
        print("🚀 STARTING COMPLETE SAR2HEIGHT PREDICTION EXPERIMENT")
        print("=" * 80)
        
        start_time = time.time()
        
        # Track which steps completed successfully
        steps_completed = []
        
        # Step 1: Data Setup
        if self.setup_data(data_file_path, stats_dir, sample_idx, selected_variables):
            steps_completed.append("data_setup")
        else:
            print("❌ Experiment failed at data setup step")
            return False
        
        # Step 2: Model Setup
        if self.setup_models(regression_config, diffusion_config, 
                            regression_checkpoint, diffusion_checkpoint):
            steps_completed.append("model_setup")
        else:
            print("❌ Experiment failed at model setup step")
            return False
        
        # Step 3: Prediction Generation
        if self.run_predictions(ensemble_size, ensemble_base_seed, diffusion_steps):
            steps_completed.append("predictions")
        else:
            print("❌ Experiment failed at prediction step")
            return False
        
        # Step 4: Ensemble Analysis
        if self.run_ensemble_analysis():
            steps_completed.append("ensemble_analysis")
        else:
            print("❌ Experiment failed at ensemble analysis step")
            return False
        
        # Step 5: Metrics Calculation
        if self.calculate_metrics():
            steps_completed.append("metrics")
        else:
            print("❌ Experiment failed at metrics calculation step")
            return False
        
        # Step 6: Visualization Generation
        if self.generate_visualizations(save_plots):
            steps_completed.append("visualizations")
        else:
            print("❌ Experiment failed at visualization step")
            return False
        
        # Step 7: Results Saving (NEW: Proper file formats)
        if self.save_results():
            steps_completed.append("results_saving")
        else:
            print("❌ Experiment failed at results saving step")
            return False
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Update experiment info with completion details
        self.results['experiment_info'].update({
            'completed_successfully': True,
            'steps_completed': steps_completed,
            'total_time_seconds': total_time,
            'completion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print("=" * 80)
        print("🎉 COMPLETE SAR2HEIGHT PREDICTION EXPERIMENT FINISHED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Results location: {self.experiment_dir}")
        print(f"✅ All {len(steps_completed)} pipeline steps completed")
        
        # Print file summary
        print("\n📂 Generated Files:")
        print(f"  📊 Predictions (NetCDF): {self.predictions_dir}")
        print(f"  📈 Analysis (JSON): {self.analysis_dir}")
        print(f"  🎨 Visualizations: {self.visualizations_dir}")
        print("=" * 80)
        
        return True
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of experiment results.
        
        Returns:
            Dictionary containing experiment summary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_directory': str(self.experiment_dir),
            'completed': self.results['experiment_info'].get('completed_successfully', False),
            'file_structure': {
                'predictions_dir': str(self.predictions_dir),
                'analysis_dir': str(self.analysis_dir),
                'visualizations_dir': str(self.visualizations_dir)
            }
        }
        
        # Data summary
        if 'data' in self.results:
            summary['data_summary'] = {
                'sample_idx': self.results['data'].get('sample_idx'),
                'selected_variables': self.results['data'].get('selected_variables'),
                'data_shape': self.results['data'].get('data_shape')
            }
        
        # Model performance
        if 'metrics' in self.results:
            summary['model_performance'] = self.results['metrics'].get('model_performance', {})
        
        # File paths
        summary['saved_files'] = self.results['file_paths']
        
        # Timing information
        if 'total_time_seconds' in self.results['experiment_info']:
            summary['total_time_seconds'] = self.results['experiment_info']['total_time_seconds']
        
        return summary