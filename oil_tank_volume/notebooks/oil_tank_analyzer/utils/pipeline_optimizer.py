"""
Pipeline optimization utilities for finding the best tank detection configuration.
"""
import cv2
import numpy as np
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from ..core.config import PipelineConfig, DenoisingConfig, ContourConfig, CircleDetectionConfig
from ..pipelines.tank_pipeline import TankAnalysisPipeline
from .grid_search import PipelineGridSearch
from .plot_results import GridSearchPlotter


class PipelineOptimizer:
    """Optimizes pipeline configurations for oil tank detection."""
    
    def __init__(self, verbose: bool = True, random_seed: Optional[int] = 42):
        self.verbose = verbose
        self.grid_search = PipelineGridSearch()
        self.plotter = GridSearchPlotter()
        
        # Set random seed for reproducible shuffling
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def prepare_tank_image(self, tank_data: Dict) -> np.ndarray:
        """Prepare a tank image for processing."""
        rotated_crop = tank_data.rotated_crop
        
        # Extract single band if needed
        if rotated_crop.shape[0] == 1:
            image = rotated_crop[0]
        else:
            image = rotated_crop[0]  # Use first band
        
        # Normalize to 8-bit
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image
    
    def run_optimization_pipeline(self, 
                                processed_tanks: List[Dict],
                                test_tank_index: int = 0,
                                strategies: List[str] = None,
                                shuffle: bool = True,
                                max_configs_per_strategy: Optional[List[int]] = None,
                                top_k_final: int = 15) -> Tuple[List[Dict], Any]:
        """
        Run complete optimization pipeline.
        
        Args:
            processed_tanks: List of processed tank data
            test_tank_index: Index of tank to use for optimization
            strategies: List of search strategies to use
            shuffle: Whether to shuffle configurations before testing
            max_configs_per_strategy: List of maximum configs per strategy (same order as strategies)
            top_k_final: Number of top configurations to keep
            
        Returns:
            Tuple of (final_results, best_config)
        """
        if strategies is None:
            strategies = ['recommended', 'smart']
        
        # Validate max_configs_per_strategy
        if max_configs_per_strategy is None:
            max_configs_per_strategy = [20] * len(strategies)  # Default 20 for each strategy
        elif len(max_configs_per_strategy) != len(strategies):
            raise ValueError("max_configs_per_strategy must have same length as strategies")
        
        if not processed_tanks:
            raise ValueError("No processed tanks available. Run tank processing first.")
        
        # Prepare test image
        test_tank = processed_tanks[test_tank_index]
        test_image = self.prepare_tank_image(test_tank)
        tank_id = test_tank.annotation_id
        
        if self.verbose:
            print("=== OIL TANK GRID SEARCH OPTIMIZATION ===")
            print(f"Using tank {tank_id} for optimization")
            print(f"Test image shape: {test_image.shape}")
            print(f"Strategies: {strategies}")
            print(f"Shuffle: {shuffle}")
            print(f"Max configs per strategy: {max_configs_per_strategy}")
        
        # Run grid search with multiple strategies
        all_search_results = self._run_multi_strategy_search(
            test_image, strategies, max_configs_per_strategy, shuffle
        )
        
        # Get final results
        final_results = self._get_final_results(all_search_results, test_image, top_k_final)
        
        if not final_results:
            raise RuntimeError("No valid configurations found!")
        
        # Get best configuration
        best_config = final_results[0]['config']
        
        if self.verbose:
            self._print_final_results(final_results, tank_id)
            self.plotter.plot_top_configurations(final_results[:5], test_image, top_k=5)
        
        return final_results, best_config
    
    def _run_multi_strategy_search(self, 
                                 test_image: np.ndarray,
                                 strategies: List[str],
                                 max_configs_per_strategy: List[int],
                                 shuffle: bool) -> List[Dict]:
        """Run grid search with multiple strategies."""
        all_results = []
        
        for i, strategy in enumerate(strategies):
            max_configs = max_configs_per_strategy[i]
            
            if self.verbose:
                print(f"\n--- Testing strategy: {strategy.upper()} ---")
                print(f"Max configurations: {max_configs}")
                print(f"Shuffle: {shuffle}")
            
            start_time = time.time()
            
            # Run grid search with shuffle parameter
            strategy_results = self.grid_search.run_grid_search(
                test_image, 
                max_configs=max_configs, 
                strategy=strategy,
                shuffle=shuffle
            )
            
            elapsed_time = time.time() - start_time
            
            if self.verbose:
                print(f"Strategy '{strategy}' completed in {elapsed_time:.1f}s")
                print(f"Found {len(strategy_results)} valid configurations")
                self._print_strategy_top_results(strategy, strategy_results)
            
            all_results.extend(strategy_results)
        
        return all_results
    
    def _get_final_results(self, 
                         all_results: List[Dict],
                         test_image: np.ndarray,
                         top_k: int) -> List[Dict]:
        """Combine and filter results from all strategies."""
        if not all_results:
            if self.verbose:
                print("No valid configurations found! Trying exhaustive search...")
            return self.grid_search.run_grid_search(
                test_image, 
                max_configs=30, 
                strategy='exhaustive',
                shuffle=True  # Shuffle exhaustive search too
            )
        
        # Sort by score and take top K
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _print_strategy_top_results(self, strategy: str, results: List[Dict]):
        """Print top results for a strategy."""
        if not results:
            print(f"  No valid configurations found for {strategy}")
            return
        
        print(f"Top 3 from {strategy}:")
        for i, result in enumerate(results[:3]):
            config = result['config']
            print(f"  {i+1}. Score: {result['score']:.3f} | "
                  f"Denoise: {config.denoising.method} | "
                  f"Edge: {config.contour_detection.method} | "
                  f"Circle: {config.circle_detection.method}")
    
    def _print_final_results(self, final_results: List[Dict], tank_id: str):
        """Print final optimization results."""
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total configurations tested: {len(final_results)}")
        print(f"Best score: {final_results[0]['score']:.3f}")
        
        print(f"\nTop 10 configurations:")
        for i, result in enumerate(final_results[:10]):
            config = result['config']
            print(f"{i+1}. Score: {result['score']:.3f} | "
                  f"Denoise: {config.denoising.method} | "
                  f"Edge: {config.contour_detection.method} | "
                  f"Circle: {config.circle_detection.method}")
            
            # Show parameters for top 3
            if i < 3:
                print(f"   Params - Denoise: {config.denoising.params}")
                print(f"   Params - Edge: {config.contour_detection.params}")
                print(f"   Params - Circle: {config.circle_detection.params}")
                print()


class PipelineTester:
    """Utilities for testing pipeline configurations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def test_configuration_on_tank(self,
                                config: PipelineConfig,
                                tank_data: Dict,
                                plot_results: bool = True) -> Dict[str, Any]:
        """Test a single configuration on a specific tank."""
        # Prepare image - REMOVE the automatic normalization
        rotated_crop = tank_data.rotated_crop
        if rotated_crop.shape[0] == 1:
            test_image = rotated_crop[0]
        else:
            test_image = rotated_crop[0]
        
        # REMOVE THIS LINE - preprocessing will handle normalization
        # test_image = cv2.normalize(test_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if self.verbose:
            print(f"Testing configuration: {config.preprocessing.method} + "  # UPDATED
                f"{config.denoising.method} + "
                f"{config.contour_detection.method} + {config.circle_detection.method}")
            if config.preprocessing.params:  # NEW
                print(f"  Preprocess params: {config.preprocessing.params}")
            if config.denoising.params:
                print(f"  Denoise params: {config.denoising.params}")
            if config.contour_detection.params:
                print(f"  Edge params: {config.contour_detection.params}")
            if config.circle_detection.params:
                print(f"  Circle params: {config.circle_detection.params}")
        
        try:
            pipeline = TankAnalysisPipeline(config)
            result = pipeline.process_tank(test_image)
            
            if plot_results:
                plotter = GridSearchPlotter()
                mock_result = {
                    'config': config,
                    'result': result,
                    'score': result.get('quality_metrics', {}).get('overall_quality', 0.5)
                }
                plotter.plot_top_configurations([mock_result], test_image, top_k=1)
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"Configuration failed: {e}")
            return {'error': str(e)}
    
    def test_configuration_on_all_tanks(self,
                                      config: PipelineConfig,
                                      all_tanks: List[Dict]) -> Dict[str, Any]:
        """Test configuration on all available tanks."""
        if self.verbose:
            print(f"\nTesting configuration on {len(all_tanks)} tanks...")
        
        all_scores = []
        detailed_results = []
        
        for tank_data in all_tanks:
            tank_id = tank_data.annotation_id
            result = self.test_configuration_on_tank(config, tank_data, plot_results=False)
            
            # Calculate score
            circles = result.get('circles', [])
            identified = result.get('identified_circles', {})
            score = min(len(circles) / 3, 1.0) * 0.3 + (len(identified) / 3) * 0.7
            
            all_scores.append(score)
            detailed_results.append({
                'tank_id': tank_id,
                'score': score,
                'circles_found': len(circles),
                'components_identified': len(identified),
                'success': score > 0.5
            })
            
            if self.verbose:
                status = "✓" if score > 0.5 else "✗"
                print(f"  Tank {tank_id}: {status} Score: {score:.3f} | "
                      f"Circles: {len(circles)} | Identified: {len(identified)}")
        
        # Calculate summary statistics
        successful_tanks = [r for r in detailed_results if r['success']]
        success_rate = (len(successful_tanks) / len(all_tanks)) * 100
        
        if self.verbose:
            print(f"\nConfiguration Performance Summary:")
            print(f"  Average Score: {np.mean(all_scores):.3f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Works on {len(successful_tanks)}/{len(all_tanks)} tanks")
        
        return {
            'summary': {
                'average_score': np.mean(all_scores),
                'success_rate': success_rate,
                'successful_tanks': len(successful_tanks),
                'total_tanks': len(all_tanks)
            },
            'detailed_results': detailed_results,
            'all_scores': all_scores
        }


def quick_test_configuration(processed_tanks: List[Dict],
                           edge_method: str,
                           circle_method: str,
                           preprocessing_method: str = "identity_preprocessor", 
                           denoise_method: str = "identity_denoiser",
                           preprocessing_params: Dict = None, 
                           denoise_params: Dict = None,
                           edge_params: Dict = None,
                           circle_params: Dict = None,
                           tank_index: int = 0) -> Dict[str, Any]:
    """
    Quick test of a single configuration.
    
    Args:
        processed_tanks: List of processed tanks
        preprocessing_method: Preprocessing method name (NEW)
        denoise_method: Denoising method name
        edge_method: Edge detection method name  
        circle_method: Circle detection method name
        preprocessing_params: Parameters for preprocessing (NEW)
        denoise_params: Parameters for denoising
        edge_params: Parameters for edge detection
        circle_params: Parameters for circle detection
        tank_index: Index of tank to test on
        
    Returns:
        Processing results
    """
    tester = PipelineTester(verbose=True)
    
    # Import the new PreprocessingConfig
    from ..core.config import PreprocessingConfig, PipelineConfig, DenoisingConfig, ContourConfig, CircleDetectionConfig
    
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(  # NEW
            method=preprocessing_method,
            params=preprocessing_params or {}
        ),
        denoising=DenoisingConfig(
            method=denoise_method,
            params=denoise_params or {}
        ),
        contour_detection=ContourConfig(
            method=edge_method,
            params=edge_params or {}
        ),
        circle_detection=CircleDetectionConfig(
            method=circle_method,
            params=circle_params or {}
        )
    )
    
    return tester.test_configuration_on_tank(config, processed_tanks[tank_index], plot_results=True)