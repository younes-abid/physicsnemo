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
        
        
        if self.verbose:
            print(f"Testing configuration: {config.preprocessing.method} + "  # UPDATED
                f"{config.denoising.method} + "
                f"{config.contour_detection.method} + {config.circle_detection.method}")
            
            # Show actual parameters being used (including defaults)
            try:
                pipeline = TankAnalysisPipeline(config)
                actual_params = self._get_actual_parameters(pipeline, config)
                
                print(f"  Preprocess params: {actual_params['preprocessing']}")
                print(f"  Denoise params: {actual_params['denoising']}")  
                print(f"  Edge params: {actual_params['edge_detection']}")
                print(f"  Circle params: {actual_params['circle_detection']}")
                
            except Exception as e:
                # Fallback to showing config params if pipeline creation fails
                if config.preprocessing.params:
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
    
    def _get_actual_parameters(self, pipeline: 'TankAnalysisPipeline', config: PipelineConfig) -> Dict[str, Dict]:
        """Get the actual parameters that will be used by each processor, including defaults."""
        actual_params = {}
        
        # Get preprocessing parameters
        actual_params['preprocessing'] = self._merge_with_defaults(
            config.preprocessing.params,
            self._get_preprocessing_defaults(config.preprocessing.method)
        )
        
        # Get denoising parameters  
        actual_params['denoising'] = self._merge_with_defaults(
            config.denoising.params,
            self._get_denoising_defaults(config.denoising.method)
        )
        
        # Get edge detection parameters
        actual_params['edge_detection'] = self._merge_with_defaults(
            config.contour_detection.params,
            self._get_edge_detection_defaults(config.contour_detection.method)
        )
        
        # Get circle detection parameters
        actual_params['circle_detection'] = self._merge_with_defaults(
            config.circle_detection.params,
            self._get_circle_detection_defaults(config.circle_detection.method)
        )
        
        return actual_params
    
    def _merge_with_defaults(self, provided_params: Dict, default_params: Dict) -> Dict:
        """Merge provided parameters with defaults, showing what's actually used."""
        merged = default_params.copy()
        merged.update(provided_params)
        return merged
    
    def _get_preprocessing_defaults(self, method: str) -> Dict:
        """Get default parameters for preprocessing methods."""
        defaults = {
            'identity_preprocessor': {},
            'basic_normalization': {
                'enhance_contrast': False,
                'clahe_clip_limit': 2.0,
                'clahe_grid_size': 8
            },
            'tank_enhancement': {
                'enhance_contrast': True,
                'enhance_circular': True,
                'suppress_background': False,
                'clahe_clip_limit': 2.0,
                'clahe_grid_size': 8,
                'dog_sigma1': 1.0,
                'dog_sigma2': 2.0
            },
            'multi_scale_enhancement': {
                'scales': [0.5, 1.0, 2.0],
                'weights': [0.2, 0.6, 0.2]
            }
        }
        return defaults.get(method, {})
    
    def _get_denoising_defaults(self, method: str) -> Dict:
        """Get default parameters for denoising methods."""
        defaults = {
            'identity_denoiser': {
                'preserve_dynamic_range': True
            },
            'enhanced_lee': {
                'window_size': 7,
                'damping_factor': 1.0,
                'cu': 0.523,
                'max_sigma': 2.0,
                'enhance_circular': True,
                'edge_preserving': True,
                'circular_kernel_size': 3,
                'edge_threshold': 0.1,
                'preserve_dynamic_range': True
            },
            'lee_sigma': {
                'window_size': 7,
                'sigma_range': 2.0,
                'enhance_circular': True,
                'edge_preserving': True,
                'preserve_dynamic_range': True
            },
            'frost': {
                'window_size': 7,
                'damping_factor': 2.0,
                'preserve_dynamic_range': True
            },
            'gamma_map': {
                'window_size': 7,
                'looks': 1,
                'preserve_dynamic_range': True
            },
            'kuan': {
                'window_size': 7,
                'looks': 1,
                'preserve_dynamic_range': True
            },
            'sar_bilateral': {
                'd': 9,
                'sigma_color': 0.1,
                'sigma_space': 75,
                'preserve_dynamic_range': True
            },
            'nonlocal_means_sar': {
                'h': 7,
                'template_size': 5,
                'search_size': 15,
                'preserve_dynamic_range': True
            },
            'wavelet_sar': {
                'level': 3,
                'threshold_factor': 0.1,
                'preserve_dynamic_range': True
            },
            'median_tank': {
                'kernel_size': 5,
                'preserve_dynamic_range': True
            },
            'adaptive_tank': {
                'enhance_tank_features': True,
                'edge_threshold': 0.05,
                'preserve_dynamic_range': True
            },
            'multi_scale_tank': {
                'scales': [1.0, 0.5, 2.0],
                'weights': [0.6, 0.2, 0.2],
                'preserve_dynamic_range': True
            },
            'gaussian': {
                'kernel_size': 5,
                'sigma': 1.0,
                'preserve_dynamic_range': True
            }
        }
        return defaults.get(method, {})
    
    def _get_edge_detection_defaults(self, method: str) -> Dict:
        """Get default parameters for edge detection methods."""
        defaults = {
            'xarray_contour_edges': {
                'levels': 5,
                'robust': True,
                'clean_edges': True,
                'enhance_contrast': True,
                'reduce_speckle': True,
                'remove_small_edges': True,
                'min_edge_length': 20,
                'enhance_horizontal': True
            },
            'canny_tank': {
                'aperture_size': 5,
                'l2_gradient': True,
                'enhance_contrast': True,
                'reduce_speckle': True
            },
            'adaptive_threshold_tank': {
                'block_size': 15,
                'c': 3
            },
            'multi_scale_tank': {
                'scales': [1.0, 0.75, 1.25]
            },
            'combined_tank': {
                'enhance_circular': True
            },
            'sobel': {
                'ksize': 3
            },
            'laplacian': {
                'ksize': 3
            }
        }
        return defaults.get(method, {})
    
    def _get_circle_detection_defaults(self, method: str) -> Dict:
        """Get default parameters for circle detection methods."""
        defaults = {
            'hough_tank': {
                'dp': 1.1,
                'param1': 30,
                'param2': 25,
                'center_tolerance': 0.2,
                'min_diameter_ratio': 0.6,
                'max_diameter_ratio': 1.1,
                'max_circles': 5
            },
            'ellipse_tank': {
                'min_area': 1000,
                'max_area': 10000,
                'center_tolerance': 0.2,
                'max_circles': 5
            },
            'contour_tank': {
                'min_area': 500,
                'max_area': 15000,
                'min_circularity': 0.5,
                'center_tolerance': 0.2,
                'max_circles': 5
            },
            'multi_scale_tank': {
                'center_tolerance': 0.2,
                'min_diameter_ratio': 0.6,
                'max_diameter_ratio': 1.1,
                'max_circles': 5
            },
            'template_tank': {
                'center_tolerance': 0.2,
                'max_circles': 5
            },
            'intensity_tank': {
                'intensity_threshold': 0.3,
                'center_tolerance': 0.2,
                'max_circles': 5
            },
            'blob_detector': {
                'center_tolerance': 0.2,
                'max_circles': 5
            }
        }
        return defaults.get(method, {})
    
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
                           preprocessing_method: str = "identity_preprocessor",
                           denoise_method: str = "identity_denoiser", 
                           edge_method: str = "xarray_contour_edges",
                           circle_method: str ='multi_scale_tank',
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