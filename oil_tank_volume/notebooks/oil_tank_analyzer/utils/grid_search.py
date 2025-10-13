import itertools
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from oil_tank_analyzer.core.config import PipelineConfig, DenoisingConfig, ContourConfig, CircleDetectionConfig
from oil_tank_analyzer.pipelines.tank_pipeline import TankAnalysisPipeline
import copy
import random

class PipelineGridSearch:
    """Enhanced grid search for finding optimal pipeline configuration with SAR optimizations."""
    
    def __init__(self):
        # Updated parameter grid with all our new SAR-optimized methods
        self.param_grid = {
            'denoising': {
                'enhanced_lee': {
                    'window_size': [5, 7, 9],
                    'damping_factor': [0.5, 1.0, 2.0],
                    'cu': [0.4, 0.523, 0.7],
                    'max_sigma': [1.5, 2.0, 3.0]
                },
                'lee_sigma': {
                    'window_size': [5, 7, 9],
                    'sigma_range': [1.5, 2.0, 2.5]
                },
                'frost': {
                    'window_size': [5, 7, 9],
                    'damping_factor': [1.0, 2.0, 3.0]
                },
                'median_tank': {
                    'kernel_size': [3, 5, 7]
                },
                'sar_bilateral': {
                    'd': [5, 9, 13],
                    'sigma_color': [0.05, 0.1, 0.15],
                    'sigma_space': [50, 75, 100]
                }
            },
            'edge_detection': {
                'combined_tank': {
                    'block_size': [11, 15, 19],
                    'c': [2, 3, 4],
                    'enhance_horizontal': [True],
                    'remove_small_edges': [True],
                    'min_edge_length': [20, 30, 40]
                },
                'canny_tank': {
                    'threshold1': [20, 30, 40],
                    'threshold2': [40, 60, 80],
                    'aperture_size': [3, 5]
                },
                'adaptive_threshold_tank': {
                    'block_size': [11, 15, 19],
                    'c': [2, 3, 4]
                },
                'multi_scale_tank': {
                    'scales': [[1.0, 0.75, 1.25], [1.0, 0.5, 1.5]]
                },
                'morphological_tank': {
                    'kernel_size': [3, 5, 7],
                    'morph_threshold': [5, 10, 15]
                }
            },
            'circle_detection': {
                'ellipse_tank': {
                    'center_tolerance': [0.1, 0.15, 0.2],
                    'min_diameter_ratio': [0.6, 0.7, 0.8],
                    'max_diameter_ratio': [1.0, 1.05, 1.1],
                    'min_area': [500, 1000, 2000],
                    'max_area': [5000, 10000, 15000]
                },
                'multi_scale_tank': {
                    'min_radius_ratio': [0.2, 0.25, 0.3],
                    'max_radius_ratio': [0.5, 0.6, 0.7],
                    'center_tolerance': [0.1, 0.15, 0.2]
                },
                'hough_tank': {
                    'dp': [1.0, 1.1, 1.2],
                    'min_dist_ratio': [0.6, 0.8, 1.0],
                    'param1': [20, 30, 40],
                    'param2': [20, 25, 30],
                    'min_radius_ratio': [0.2, 0.25, 0.3],
                    'max_radius_ratio': [0.5, 0.6, 0.7]
                },
                'contour_tank': {
                    'min_circularity': [0.5, 0.6, 0.7],
                    'min_area': [500, 1000, 2000],
                    'max_area': [5000, 10000, 15000]
                },
                'intensity_tank': {
                    'intensity_threshold': [0.2, 0.3, 0.4],
                    'min_area': [500, 1000, 2000]
                }
            }
        }
        
        # Method combinations that are known to work well together
        self.recommended_combinations = [
            ('enhanced_lee', 'combined_tank', 'ellipse_tank'),
            ('lee_sigma', 'canny_tank', 'multi_scale_tank'),
            ('frost', 'adaptive_threshold_tank', 'hough_tank'),
            ('median_tank', 'morphological_tank', 'contour_tank')
        ]
    
    def generate_configs(self, strategy: str = "smart") -> List[PipelineConfig]:
        """Generate pipeline configurations using different strategies."""
        if strategy == "exhaustive":
            return self._generate_exhaustive_configs()
        elif strategy == "smart":
            return self._generate_smart_configs()
        elif strategy == "recommended":
            return self._generate_recommended_configs()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _generate_exhaustive_configs(self) -> List[PipelineConfig]:
        """Generate all possible combinations (can be very large)."""
        configs = []
        
        for denoise_method, denoise_params in self.param_grid['denoising'].items():
            for edge_method, edge_params in self.param_grid['edge_detection'].items():
                for circle_method, circle_params in self.param_grid['circle_detection'].items():
                    
                    denoise_param_combos = self._generate_param_combinations(denoise_params)
                    edge_param_combos = self._generate_param_combinations(edge_params)
                    circle_param_combos = self._generate_param_combinations(circle_params)
                    
                    # Limit combinations to avoid explosion
                    max_combos = 3
                    for dp in denoise_param_combos[:max_combos]:
                        for ep in edge_param_combos[:max_combos]:
                            for cp in circle_param_combos[:max_combos]:
                                # FIX: Create DEEP copies of all parameters to avoid reference sharing
                                config = PipelineConfig(
                                    denoising=DenoisingConfig(
                                        method=denoise_method, 
                                        params=copy.deepcopy(dp)  # Deep copy params
                                    ),
                                    contour_detection=ContourConfig(
                                        method=edge_method, 
                                        params=copy.deepcopy(ep)  # Deep copy params
                                    ),
                                    circle_detection=CircleDetectionConfig(
                                        method=circle_method, 
                                        params=copy.deepcopy(cp)  # Deep copy params
                                    )
                                )
                                configs.append(config)
        
        print(f"Generated {len(configs)} exhaustive configurations")
        return configs
    
    def _generate_smart_configs(self) -> List[PipelineConfig]:
        """Generate configurations using intelligent sampling."""
        configs = []
        
        # 1. Include recommended combinations
        for denoise_method, edge_method, circle_method in self.recommended_combinations:
            denoise_params = self.param_grid['denoising'].get(denoise_method, {})
            edge_params = self.param_grid['edge_detection'].get(edge_method, {})
            circle_params = self.param_grid['circle_detection'].get(circle_method, {})
            
            # Sample 2 parameter combinations for each method
            for dp in self._sample_parameters(denoise_params, 2):
                for ep in self._sample_parameters(edge_params, 2):
                    for cp in self._sample_parameters(circle_params, 2):
                        # FIX: Create DEEP copies
                        config = PipelineConfig(
                            denoising=DenoisingConfig(
                                method=denoise_method, 
                                params=copy.deepcopy(dp)
                            ),
                            contour_detection=ContourConfig(
                                method=edge_method, 
                                params=copy.deepcopy(ep)
                            ),
                            circle_detection=CircleDetectionConfig(
                                method=circle_method, 
                                params=copy.deepcopy(cp)
                            )
                        )
                        configs.append(config)
        
        # 2. Add some random combinations for exploration
        additional_configs = self._generate_random_configs(20)
        configs.extend(additional_configs)
        
        print(f"Generated {len(configs)} smart configurations")
        return configs
    
    def _generate_recommended_configs(self) -> List[PipelineConfig]:
        """Generate only recommended method combinations."""
        configs = []
        
        for denoise_method, edge_method, circle_method in self.recommended_combinations:
            # FIX: Create new objects each time
            config = PipelineConfig(
                denoising=DenoisingConfig(
                    method=denoise_method, 
                    params={}  # Empty dict, no need to copy
                ),
                contour_detection=ContourConfig(
                    method=edge_method, 
                    params={}  # Empty dict, no need to copy
                ),
                circle_detection=CircleDetectionConfig(
                    method=circle_method, 
                    params={}  # Empty dict, no need to copy
                )
            )
            configs.append(config)
        
        print(f"Generated {len(configs)} recommended configurations")
        return configs
    
    def _generate_random_configs(self, num_configs: int) -> List[PipelineConfig]:
        """Generate random configurations for exploration."""
        configs = []
        
        denoise_methods = list(self.param_grid['denoising'].keys())
        edge_methods = list(self.param_grid['edge_detection'].keys())
        circle_methods = list(self.param_grid['circle_detection'].keys())
        
        for _ in range(num_configs):
            denoise_method = np.random.choice(denoise_methods)
            edge_method = np.random.choice(edge_methods)
            circle_method = np.random.choice(circle_methods)
            
            denoise_params = self._sample_parameters(self.param_grid['denoising'][denoise_method], 1)[0]
            edge_params = self._sample_parameters(self.param_grid['edge_detection'][edge_method], 1)[0]
            circle_params = self._sample_parameters(self.param_grid['circle_detection'][circle_method], 1)[0]
            
            # FIX: Create DEEP copies
            config = PipelineConfig(
                denoising=DenoisingConfig(
                    method=denoise_method, 
                    params=copy.deepcopy(denoise_params)
                ),
                contour_detection=ContourConfig(
                    method=edge_method, 
                    params=copy.deepcopy(edge_params)
                ),
                circle_detection=CircleDetectionConfig(
                    method=circle_method, 
                    params=copy.deepcopy(circle_params)
                )
            )
            configs.append(config)
        
        return configs
    
    def _sample_parameters(self, param_dict: Dict, num_samples: int) -> List[Dict]:
        """Sample parameter combinations from a parameter dictionary."""
        if not param_dict:
            return [{}]
        
        all_combinations = self._generate_param_combinations(param_dict)
        
        if len(all_combinations) <= num_samples:
            return all_combinations
        else:
            # Sample evenly spaced combinations
            indices = np.linspace(0, len(all_combinations) - 1, num_samples, dtype=int)
            # FIX: Return deep copies to avoid reference issues
            return [copy.deepcopy(all_combinations[i]) for i in indices]
    
    def _generate_param_combinations(self, param_dict: Dict) -> List[Dict]:
        """Generate all combinations of parameters."""
        if not param_dict:
            return [{}]
        
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        
        combinations = []
        for combination in itertools.product(*values):
            # FIX: Create new dict for each combination
            param_dict = {}
            for key, value in zip(keys, combination):
                param_dict[key] = value
            combinations.append(param_dict)
        
        return combinations
    
    def run_grid_search(self, image: np.ndarray, max_configs: int = 50, 
                    strategy: str = "smart", shuffle: bool = True) -> List[Dict]:
        """Run grid search on the image.
        
        Args:
            image: Input image for testing configurations
            max_configs: Maximum number of configurations to test
            strategy: Search strategy ('exhaustive', 'smart', 'recommended')
            shuffle: Whether to shuffle configurations before testing
        
        Returns:
            List of results sorted by score
        """
        configs = self.generate_configs(strategy)
        
        # Shuffle configurations if requested
        if shuffle and configs:
            random.shuffle(configs)
            print(f"Shuffled {len(configs)} configurations")
        
        # Limit number of configs to test
        if len(configs) > max_configs:
            # Prioritize recommended combinations
            recommended_indices = []
            other_indices = []
            
            for i, config in enumerate(configs):
                combo = (config.denoising.method, config.contour_detection.method, config.circle_detection.method)
                if combo in self.recommended_combinations:
                    recommended_indices.append(i)
                else:
                    other_indices.append(i)
            
            # Keep all recommended + sample from others
            if len(recommended_indices) < max_configs:
                needed = max_configs - len(recommended_indices)
                sampled_other = np.random.choice(other_indices, min(needed, len(other_indices)), replace=False)
                selected_indices = recommended_indices + list(sampled_other)
            else:
                selected_indices = recommended_indices[:max_configs]
            
            test_configs = [configs[i] for i in selected_indices]
        else:
            test_configs = configs
        
        print(f"Testing {len(test_configs)} configurations...")
        if shuffle:
            print("Configurations are shuffled")
        
        results = []
        for i, config in enumerate(test_configs):
            print(f"Config {i+1}/{len(test_configs)}: "
                f"{config.denoising.method} + {config.contour_detection.method} + {config.circle_detection.method}")
            print(f"  Denoise params: {config.denoising.params}")
            print(f"  Edge params: {config.contour_detection.params}")
            print(f"  Circle params: {config.circle_detection.params}")
            
            try:
                pipeline = TankAnalysisPipeline(config)
                result = pipeline.process_tank(image)
                
                # Enhanced scoring
                score = self._score_result_enhanced(result, image.shape)
                
                results.append({
                    'config': config,
                    'result': result,
                    'score': score,
                    'config_id': i
                })
                
                print(f"  → Score: {score:.3f}, Circles: {len(result.get('circles', []))}, Identified: {len(result.get('identified_circles', {}))}")
                print("-"*50)
                print() 
            except Exception as e:
                print(f"Configuration failed: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
        
    def _score_result_enhanced(self, result: Dict, image_shape: Tuple[int, int]) -> float:
        """Enhanced scoring using quality metrics and tank analysis."""
        if 'error' in result:
            return 0.0
        
        circles = result.get('circles', [])
        identified = result.get('identified_circles', {})
        quality_metrics = result.get('quality_metrics', {})
        tank_analysis = result.get('tank_analysis', {})
        
        height, width = image_shape
        
        # Use quality metrics if available
        if quality_metrics:
            base_score = quality_metrics.get('overall_quality', 0)
        else:
            base_score = self._calculate_basic_score(circles, identified, image_shape)
        
        # Bonus for complete tank detection
        completeness_bonus = 0.0
        if len(identified) >= 2:  # At least bottom and top
            completeness_bonus = 0.2
        if len(identified) >= 3:  # All three components
            completeness_bonus = 0.4
        
        # Bonus for high confidence
        confidence_bonus = 0.0
        detection_confidence = tank_analysis.get('detection_confidence', 0)
        if detection_confidence > 0.7:
            confidence_bonus = 0.2
        
        # Penalty for too many false circles
        false_circle_penalty = 0.0
        if len(circles) > 8:
            false_circle_penalty = min(0.3, (len(circles) - 8) * 0.05)
        
        total_score = base_score + completeness_bonus + confidence_bonus - false_circle_penalty
        
        return min(max(total_score, 0.0), 1.0)
    
    def _calculate_basic_score(self, circles: List[Dict], identified: Dict, image_shape: Tuple[int, int]) -> float:
        """Calculate basic score when quality metrics are not available."""
        if not circles:
            return 0.0
        
        height, width = image_shape
        
        # Circle count score
        circle_count_score = min(len(circles) / 3, 1.0)
        
        # Component score
        component_score = 0.0
        if 'bottom' in identified:
            component_score += 0.3
        if 'top' in identified:
            component_score += 0.3
        if 'floating_roof' in identified:
            component_score += 0.4
        
        # Quality score based on circle properties
        quality_score = 0.0
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            
            # Center alignment (tanks should be centered)
            center_distance = abs(center[0] - width / 2) / (width / 2)
            center_score = 1.0 - center_distance
            
            # Size appropriateness
            ideal_radius = width * 0.4
            size_score = 1.0 - abs(radius - ideal_radius) / ideal_radius
            
            quality_score += (center_score + size_score) / 2
        
        if circles:
            quality_score /= len(circles)
        
        return (circle_count_score * 0.2 + component_score * 0.5 + quality_score * 0.3)
    
    def save_results(self, results: List[Dict], filename: str):
        """Save grid search results to file."""
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename: str) -> List[Dict]:
        """Load grid search results from file."""
        import pickle
        
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"Results loaded from {filename}")
        return results