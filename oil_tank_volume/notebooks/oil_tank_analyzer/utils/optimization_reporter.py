"""
Reporting utilities for optimization results.
"""
import numpy as np
from typing import List, Dict, Any
from ..core.config import PipelineConfig


class OptimizationReporter:
    """Generates reports for optimization results."""
    
    @staticmethod
    def generate_summary_report(results: List[Dict], image_shape: tuple, tank_id: str):
        """Generate a comprehensive summary report."""
        print(f"\n=== GRID SEARCH SUMMARY REPORT ===")
        print(f"Target Tank: {tank_id}")
        print(f"Image Shape: {image_shape}")
        print(f"Total Configurations: {len(results)}")
        
        # Method frequency analysis
        denoise_methods, edge_methods, circle_methods = OptimizationReporter._analyze_method_frequency(results)
        
        OptimizationReporter._print_method_frequency(denoise_methods, edge_methods, circle_methods, len(results))
        OptimizationReporter._print_score_statistics(results)
        OptimizationReporter._print_success_rate(results)
    
    @staticmethod
    def _analyze_method_frequency(results: List[Dict]):
        """Analyze frequency of methods in results."""
        denoise_methods = {}
        edge_methods = {}
        circle_methods = {}
        
        for result in results:
            config = result['config']
            denoise_methods[config.denoising.method] = denoise_methods.get(config.denoising.method, 0) + 1
            edge_methods[config.contour_detection.method] = edge_methods.get(config.contour_detection.method, 0) + 1
            circle_methods[config.circle_detection.method] = circle_methods.get(config.circle_detection.method, 0) + 1
        
        return denoise_methods, edge_methods, circle_methods
    
    @staticmethod
    def _print_method_frequency(denoise_methods: Dict, edge_methods: Dict, circle_methods: Dict, total_configs: int):
        """Print method frequency analysis."""
        print(f"\nMethod Frequency in Top {total_configs} Configurations:")
        
        print("Denoising Methods:")
        for method, count in sorted(denoise_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} configurations")
        
        print("Edge Detection Methods:")
        for method, count in sorted(edge_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} configurations")
        
        print("Circle Detection Methods:")
        for method, count in sorted(circle_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count} configurations")
    
    @staticmethod
    def _print_score_statistics(results: List[Dict]):
        """Print score statistics."""
        scores = [r['score'] for r in results]
        print(f"\nScore Statistics:")
        print(f"  Best: {max(scores):.3f}")
        print(f"  Worst: {min(scores):.3f}")
        print(f"  Average: {np.mean(scores):.3f}")
        print(f"  Std Dev: {np.std(scores):.3f}")
    
    @staticmethod
    def _print_success_rate(results: List[Dict]):
        """Print success rate analysis."""
        successful_configs = len([r for r in results if r['score'] > 0.5])
        success_rate = (successful_configs / len(results)) * 100
        print(f"Success Rate (score > 0.5): {success_rate:.1f}%")
    
    @staticmethod
    def save_best_configuration(best_config: PipelineConfig, tank_id: str, score: float):
        """Save the best configuration to a file."""
        best_methods = f"{best_config.denoising.method}_{best_config.contour_detection.method}_{best_config.circle_detection.method}"
        filename = f"best_pipeline_config_tank_{tank_id}_{best_methods}_score_{score:.3f}.json"
        
        best_config.to_json(filename)
        print(f"\nBest configuration saved to {filename}")
        return filename