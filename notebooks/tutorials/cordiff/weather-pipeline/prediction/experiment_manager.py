"""
Experiment Manager for Weather Pipeline - Streamlit Demo

Handles saving, loading, and tracking of regression experiments.
"""

import os
import json
import yaml
import hashlib
import datetime
import pickle
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd


class ExperimentManager:
    """Manages regression experiment tracking and persistence."""
    
    def __init__(self, experiments_dir: str = "/app/outputs/weather-pipeline"):
        """
        Initialize ExperimentManager.
        
        Args:
            experiments_dir: Directory for saving experiments
        """
        
        self.experiments_dir = experiments_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    def generate_experiment_hash(self, model_path: str, data_path: str, sample_idx: int = None, dependent_model_path: str = None, seed: int = None) -> str:
        """
        Generate a unique hash for the experiment based on model and data paths.
        
        Args:
            model_path: Path to the model checkpoint
            data_path: Path to the data file
            sample_idx: Sample index (optional)
            dependent_model_path: Path to dependent model (e.g., regression model for diffusion) (optional)
            seed: Random seed for reproducibility (optional, used by diffusion)
            
        Returns:
            Hexadecimal hash string
        """
        # Create string to hash
        hash_string = f"{model_path}_{data_path}"
        if sample_idx is not None:
            hash_string += f"_{sample_idx}"
        if dependent_model_path is not None:
            hash_string += f"_{dependent_model_path}"
        if seed is not None:
            hash_string += f"_seed{seed}"
        
        # Generate MD5 hash
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]  # Use first 8 characters
    
    def check_experiment_exists(self, experiment_hash: str) -> bool:
        """
        Check if an experiment with the given hash already exists.
        
        Args:
            experiment_hash: Hash of the experiment
            
        Returns:
            True if experiment exists, False otherwise
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_hash)
        return os.path.exists(experiment_dir)
    
    def get_experiment_info(self, experiment_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an existing experiment.
        
        Args:
            experiment_hash: Hash of the experiment
            
        Returns:
            Dictionary containing experiment metadata or None if not found
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_hash)
        metadata_file = os.path.join(experiment_dir, "metadata.yaml")
        
        if not os.path.exists(metadata_file):
            return None
        
        with open(metadata_file, 'r') as f:
            return yaml.safe_load(f)
    
    def save_experiment(self, 
                       experiment_hash: str,
                       model_path: str,
                       data_path: str, 
                       sample_idx: int,
                       u10_pred: Any,
                       v10_pred: Any,
                       u10_true: Any,
                       v10_true: Any,
                       metrics: Dict[str, Any],
                       model_info: Dict[str, Any],
                       execution_time: float,
                       additional_info: Dict[str, Any] = None) -> str:
        """
        Save experiment results and metadata.
        
        Args:
            experiment_hash: Unique hash for the experiment
            model_path: Path to the model checkpoint
            data_path: Path to the data file
            sample_idx: Sample index used
            u10_pred: U10 predictions array
            v10_pred: V10 predictions array
            u10_true: U10 ground truth array
            v10_true: V10 ground truth array
            metrics: Computed metrics dictionary
            model_info: Model information dictionary
            execution_time: Time taken for prediction
            additional_info: Any additional information
            
        Returns:
            Path to the saved experiment directory
        """
        
        # Create experiment directory
        experiment_dir = os.path.join(self.experiments_dir, experiment_hash)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save predictions as pickle files
        predictions = {
            'u10_pred': u10_pred,
            'v10_pred': v10_pred,
            'u10_true': u10_true,
            'v10_true': v10_true
        }
        
        predictions_file = os.path.join(experiment_dir, "predictions.pkl")
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions, f)
        
        # Create metadata
        metadata = {
            'experiment_info': {
                'hash': experiment_hash,
                'created_at': datetime.datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
            },
            'input_data': {
                'model_path': model_path,
                'data_path': data_path,
                'sample_idx': sample_idx,
            },
            'model_info': model_info,
            'metrics': {
                'overall': {
                    'r2': float(metrics['overall_r2']),
                    'mae': float(metrics['overall_mae']),
                    'rmse': float(metrics['overall_rmse']),
                },
                'u10': {
                    'r2': float(metrics['u10']['r2']),
                    'mae': float(metrics['u10']['mae']),
                    'rmse': float(metrics['u10']['rmse']),
                    'correlation': float(metrics['u10']['correlation']),
                    'bias': float(metrics['u10']['bias']),
                },
                'v10': {
                    'r2': float(metrics['v10']['r2']),
                    'mae': float(metrics['v10']['mae']),
                    'rmse': float(metrics['v10']['rmse']), 
                    'correlation': float(metrics['v10']['correlation']),
                    'bias': float(metrics['v10']['bias']),
                }
            },
            'additional_info': additional_info or {}
        }
        
        # Save metadata as YAML
        metadata_file = os.path.join(experiment_dir, "metadata.yaml")
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)
        
        # Save metrics as JSON for easy reading
        metrics_file = os.path.join(experiment_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metadata['metrics'], f, indent=2)
        
        return experiment_dir
    
    def load_experiment(self, experiment_hash: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Load experiment results and metadata.
        
        Args:
            experiment_hash: Hash of the experiment to load
            
        Returns:
            Tuple of (predictions_dict, metadata_dict) or None if not found
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_hash)
        
        if not os.path.exists(experiment_dir):
            return None
        
        # Load predictions
        predictions_file = os.path.join(experiment_dir, "predictions.pkl")
        if not os.path.exists(predictions_file):
            return None
        
        with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)
        
        # Load metadata
        metadata_file = os.path.join(experiment_dir, "metadata.yaml")
        if not os.path.exists(metadata_file):
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        
        return predictions, metadata
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all saved experiments.
        
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        
        if not os.path.exists(self.experiments_dir):
            return experiments
        
        for experiment_hash in os.listdir(self.experiments_dir):
            experiment_path = os.path.join(self.experiments_dir, experiment_hash)
            
            if not os.path.isdir(experiment_path):
                continue
            
            metadata = self.get_experiment_info(experiment_hash)
            if metadata:
                experiments.append(metadata)
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x['experiment_info']['created_at'], reverse=True)
        
        return experiments
    
    def get_experiments_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all experiments.
        
        Returns:
            DataFrame with experiment summaries
        """
        experiments = self.list_experiments()
        
        if not experiments:
            return pd.DataFrame()
        
        data = []
        for exp in experiments:
            data.append({
                'Hash': exp['experiment_info']['hash'],
                'Created': exp['experiment_info']['created_at'][:19].replace('T', ' '),
                'Model': os.path.basename(exp['input_data']['model_path']),
                'Data': os.path.basename(exp['input_data']['data_path']),
                'Sample': exp['input_data']['sample_idx'],
                'Overall R²': f"{exp['metrics']['overall']['r2']:.4f}",
                'Overall RMSE': f"{exp['metrics']['overall']['rmse']:.4f}",
                'Execution Time': f"{exp['experiment_info']['execution_time_seconds']:.2f}s",
            })
        
        return pd.DataFrame(data)
    
    def delete_experiment(self, experiment_hash: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_hash: Hash of the experiment to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_hash)
        
        if not os.path.exists(experiment_dir):
            return False
        
        try:
            import shutil
            shutil.rmtree(experiment_dir)
            return True
        except Exception:
            return False