from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from oil_tank_analyzer.core.base_processor import ProcessorRegistry
from oil_tank_analyzer.core.config import PipelineConfig
from oil_tank_analyzer.preprocessing.denoisers import BaseDenoiser
from oil_tank_analyzer.detection.edge_detectors import BaseEdgeDetector
from oil_tank_analyzer.detection.circle_detectors import BaseCircleDetector

class TankAnalysisPipeline:
    """Main pipeline for oil tank analysis with SAR-specific optimizations."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.denoiser = None
        self.edge_detector = None
        self.circle_detector = None
        self._initialize_processors()
        self.performance_metrics = {}
    
    def _initialize_processors(self):
        """Initialize all processors based on config with error handling."""
        try:
            self.denoiser = ProcessorRegistry.create_processor(
                self.config.denoising.method, self.config
            )
            self.edge_detector = ProcessorRegistry.create_processor(
                self.config.contour_detection.method, self.config
            )
            self.circle_detector = ProcessorRegistry.create_processor(
                self.config.circle_detection.method, self.config
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize processors: {e}")
    
    def preprocess_tank_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhanced preprocessing specifically for oil tank SAR images."""
        # Handle different input formats
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Use first channel for SAR
        
        # Convert to proper type and normalize
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply SAR-specific enhancements
        if kwargs.get('enhance_tank_features', True):
            image = self._enhance_tank_features(image, **kwargs)
        
        return image
    
    def _enhance_tank_features(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance features specific to oil tanks in SAR imagery."""
        enhanced = image.copy()
        
        # 1. Contrast enhancement for tank boundaries
        clahe = cv2.createCLAHE(clipLimit=kwargs.get('clahe_clip_limit', 2.0),
                               tileGridSize=kwargs.get('clahe_grid_size', (8, 8)))
        enhanced = clahe.apply(enhanced)
        
        # 2. Emphasize circular patterns
        if kwargs.get('enhance_circular', True):
            enhanced = self._enhance_circular_patterns(enhanced, **kwargs)
        
        # 3. Reduce background noise while preserving tank structure
        if kwargs.get('background_suppression', True):
            enhanced = self._suppress_background(enhanced, **kwargs)
        
        return enhanced
    
    def _enhance_circular_patterns(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance circular and elliptical patterns for tank detection."""
        # Use difference of Gaussians to enhance circular features
        sigma1 = kwargs.get('dog_sigma1', 1.0)
        sigma2 = kwargs.get('dog_sigma2', 2.0)
        
        gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma1)
        gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)
        dog = gaussian1 - gaussian2
        
        # Enhance the DoG result
        enhanced = cv2.addWeighted(image, 0.7, dog, 0.3, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _suppress_background(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Suppress background noise while preserving tank structures."""
        # Use adaptive thresholding to separate tank from background
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find largest connected component (likely the tank)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            # Find the largest component (skip background)
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            
            # Create mask for the tank region
            tank_mask = (labels == largest_component).astype(np.uint8) * 255
            
            # Apply mask to original image
            masked_image = cv2.bitwise_and(image, image, mask=tank_mask)
            
            # Fill background with median value
            background_value = np.median(image[tank_mask == 0]) if np.any(tank_mask == 0) else 0
            result = np.where(tank_mask == 255, masked_image, background_value)
            
            return result.astype(np.uint8)
        
        return image
    
    def process_tank(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process a single tank image through the pipeline with enhanced analysis."""
        results = {
            'pipeline_stages': {},
            'performance_metrics': {},
            'tank_analysis': {}
        }
        
        try:
            # Enhanced preprocessing
            start_time = cv2.getTickCount()
            preprocessed = self.preprocess_tank_image(image, **kwargs)
            results['pipeline_stages']['preprocessed'] = preprocessed
            results['performance_metrics']['preprocessing_time'] = (
                cv2.getTickCount() - start_time
            ) / cv2.getTickFrequency()
            
            # Step 1: Denoising
            start_time = cv2.getTickCount()
            print(f"Denoising with {self.config.denoising.method}...")
            denoised = self.denoiser.process(preprocessed, **kwargs)
            results['pipeline_stages']['denoised'] = denoised
            results['denoising_results'] = self.denoiser.get_results()
            results['performance_metrics']['denoising_time'] = (
                cv2.getTickCount() - start_time
            ) / cv2.getTickFrequency()
            
            # Step 2: Edge Detection
            start_time = cv2.getTickCount()
            print(f"Edge detection with {self.config.contour_detection.method}...")
            edges = self.edge_detector.process(denoised, **kwargs)
            results['pipeline_stages']['edges'] = edges
            results['edge_detection_results'] = self.edge_detector.get_results()
            results['performance_metrics']['edge_detection_time'] = (
                cv2.getTickCount() - start_time
            ) / cv2.getTickFrequency()
            
            # Step 3: Circle Detection
            start_time = cv2.getTickCount()
            print(f"Circle detection with {self.config.circle_detection.method}...")
            circles = self.circle_detector.process(denoised, edges, **kwargs)
            results['pipeline_stages']['circles'] = circles
            results['circle_detection_results'] = self.circle_detector.get_results()
            results['performance_metrics']['circle_detection_time'] = (
                cv2.getTickCount() - start_time
            ) / cv2.getTickFrequency()
            
            # Step 4: Enhanced tank component identification
            identified, analysis = self._identify_tank_components_enhanced(
                circles, denoised.shape, image, **kwargs
            )
            results['identified_circles'] = identified
            results['tank_analysis'] = analysis
            
            # Calculate overall quality metrics
            results['quality_metrics'] = self._calculate_quality_metrics(
                circles, identified, denoised.shape, **kwargs
            )
            
            # Store performance metrics
            total_time = sum([
                results['performance_metrics']['preprocessing_time'],
                results['performance_metrics']['denoising_time'],
                results['performance_metrics']['edge_detection_time'],
                results['performance_metrics']['circle_detection_time']
            ])
            results['performance_metrics']['total_processing_time'] = total_time
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Pipeline processing failed: {e}")
        
        return results
    
    def _identify_tank_components_enhanced(self, circles: List[Dict], 
                                         image_shape: Tuple[int, int],
                                         original_image: np.ndarray,
                                         **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enhanced identification of tank components using multiple criteria."""
        if not circles:
            return {}, {'status': 'no_circles_detected'}
        
        height, width = image_shape
        analysis = {
            'total_circles': len(circles),
            'detection_confidence': 0.0,
            'component_analysis': {}
        }
        
        # Multiple sorting strategies
        circles_by_vertical = sorted(circles, key=lambda x: x['center'][1])
        circles_by_size = sorted(circles, key=lambda x: x['radius'], reverse=True)
        circles_by_quality = sorted(circles, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        identified = {}
        confidence_scores = {}
        
        # Strategy 1: Size-based identification
        if len(circles_by_size) >= 1:
            identified['bottom_candidate'] = circles_by_size[0]
            confidence_scores['bottom_size'] = 0.8
        
        if len(circles_by_size) >= 2:
            identified['top_candidate'] = circles_by_size[1]
            confidence_scores['top_size'] = 0.7
        
        if len(circles_by_size) >= 3:
            identified['floating_roof_candidate'] = circles_by_size[2]
            confidence_scores['floating_size'] = 0.6
        
        # Strategy 2: Position-based identification
        expected_positions = {
            'bottom': height * 0.7,    # Bottom is lower
            'floating_roof': height * 0.5,  # Middle
            'top': height * 0.3        # Top is higher
        }
        
        for circle in circles_by_vertical:
            y_pos = circle['center'][1]
            closest_component = min(expected_positions.items(), 
                                  key=lambda x: abs(y_pos - x[1]))
            
            component_name = f"{closest_component[0]}_position_candidate"
            if component_name not in identified:
                identified[component_name] = circle
                confidence_scores[f"{closest_component[0]}_position"] = 0.7
        
        # Strategy 3: Intensity-based identification (SAR-specific)
        intensity_analysis = self._analyze_circle_intensities(circles, original_image)
        for component, circle in intensity_analysis.items():
            identified[f"{component}_intensity_candidate"] = circle
            confidence_scores[f"{component}_intensity"] = 0.6
        
        # Final assignment with confidence
        final_identified = {}
        for component in ['bottom', 'top', 'floating_roof']:
            candidates = [(k, v) for k, v in identified.items() if component in k]
            if candidates:
                # Take the candidate with highest average confidence
                best_candidate = max(candidates, key=lambda x: self._calculate_candidate_confidence(x[0], confidence_scores))
                final_identified[component] = best_candidate[1]
                analysis['component_analysis'][component] = {
                    'confidence': self._calculate_candidate_confidence(best_candidate[0], confidence_scores),
                    'source': best_candidate[0]
                }
        
        # Calculate overall detection confidence
        if final_identified:
            analysis['detection_confidence'] = np.mean([
                analysis['component_analysis'][comp]['confidence'] 
                for comp in final_identified.keys()
            ])
        
        return final_identified, analysis
    
    def _analyze_circle_intensities(self, circles: List[Dict], image: np.ndarray) -> Dict[str, Dict]:
        """Analyze circle intensities for SAR-specific tank component identification."""
        intensity_analysis = {}
        
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            
            # Create mask for this circle
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            # Calculate intensity statistics
            mean_intensity = cv2.mean(image, mask=mask)[0]
            std_intensity = np.std(image[mask == 255]) if np.any(mask == 255) else 0
            
            # SAR-specific intensity patterns
            if mean_intensity < 100:  # Dark region - likely bottom
                intensity_analysis['bottom'] = circle
            elif std_intensity > 30:  # High variation - likely floating roof
                intensity_analysis['floating_roof'] = circle
            elif mean_intensity > 150:  # Bright region - likely top
                intensity_analysis['top'] = circle
        
        return intensity_analysis
    
    def _calculate_candidate_confidence(self, candidate_name: str, confidence_scores: Dict) -> float:
        """Calculate confidence score for a candidate circle."""
        for score_name, score in confidence_scores.items():
            if score_name in candidate_name:
                return score
        return 0.5  # Default confidence
    
    def _calculate_quality_metrics(self, circles: List[Dict], identified: Dict, 
                                 image_shape: Tuple[int, int], **kwargs) -> Dict[str, float]:
        """Calculate quality metrics for the detection results."""
        height, width = image_shape
        metrics = {
            'circle_count_score': min(len(circles) / 3, 1.0),  # Ideal: 3 circles
            'component_completeness': len(identified) / 3,  # 0-1 scale
            'spatial_coherence': 0.0,
            'size_consistency': 0.0
        }
        
        if len(circles) >= 2:
            # Check spatial coherence (circles should be vertically aligned)
            centers_x = [circle['center'][0] for circle in circles]
            x_std = np.std(centers_x)
            metrics['spatial_coherence'] = max(0, 1 - (x_std / (width * 0.1)))
            
            # Check size consistency
            radii = [circle['radius'] for circle in circles]
            radius_std = np.std(radii)
            mean_radius = np.mean(radii)
            if mean_radius > 0:
                metrics['size_consistency'] = max(0, 1 - (radius_std / mean_radius))
        
        # Overall quality score
        metrics['overall_quality'] = np.mean([
            metrics['circle_count_score'],
            metrics['component_completeness'],
            metrics['spatial_coherence'],
            metrics['size_consistency']
        ])
        
        return metrics
    
    def batch_process_tanks(self, tank_images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple tank images efficiently."""
        results = []
        
        for i, image in enumerate(tank_images):
            print(f"Processing tank {i+1}/{len(tank_images)}...")
            result = self.process_tank(image, **kwargs)
            result['tank_id'] = i
            results.append(result)
        
        return results
    
    def optimize_parameters(self, reference_images: List[np.ndarray], 
                          target_metrics: Dict[str, float], **kwargs) -> PipelineConfig:
        """Simple parameter optimization based on reference images."""
        # This is a placeholder for more sophisticated optimization
        # In practice, you'd use grid search or Bayesian optimization
        
        best_config = self.config
        best_score = 0
        
        # Test a few parameter variations
        param_variations = [
            {'denoising': {'method': 'enhanced_lee', 'params': {'window_size': 5}}},
            {'denoising': {'method': 'lee_sigma', 'params': {'window_size': 7}}},
            {'circle_detection': {'method': 'ellipse_tank', 'params': {'center_tolerance': 0.15}}},
        ]
        
        for params in param_variations:
            test_config = self._create_test_config(params)
            self.update_config(test_config)
            
            # Evaluate on reference images
            total_score = 0
            for image in reference_images:
                result = self.process_tank(image, **kwargs)
                score = result.get('quality_metrics', {}).get('overall_quality', 0)
                total_score += score
            
            avg_score = total_score / len(reference_images)
            
            if avg_score > best_score:
                best_score = avg_score
                best_config = test_config
        
        self.update_config(best_config)
        print(f"Optimized configuration with score: {best_score:.3f}")
        return best_config
    
    def _create_test_config(self, params: Dict) -> PipelineConfig:
        """Create a test configuration from parameter variations."""
        # Implementation depends on your config structure
        # This is a simplified version
        return self.config
    
    def update_config(self, new_config: PipelineConfig):
        """Update pipeline configuration and reinitialize processors."""
        self.config = new_config
        self._initialize_processors()
    
    def list_available_methods(self):
        """List all available methods for each step."""
        return {
            'denoising': ProcessorRegistry.list_processors_for_class(BaseDenoiser),
            'edge_detection': ProcessorRegistry.list_processors_for_class(BaseEdgeDetector),
            'circle_detection': ProcessorRegistry.list_processors_for_class(BaseCircleDetector)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the pipeline."""
        return self.performance_metrics

# Add this to ProcessorRegistry to filter by base class
def list_processors_for_class(base_class):
    return [name for name, cls in ProcessorRegistry._registry.items() 
            if issubclass(cls, base_class)]

ProcessorRegistry.list_processors_for_class = staticmethod(list_processors_for_class)