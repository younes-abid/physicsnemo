import cv2
import numpy as np
from abc import abstractmethod
from oil_tank_analyzer.core.base_processor import BaseProcessor, ProcessorRegistry

class BaseEdgeDetector(BaseProcessor):
    """Base class for edge detection algorithms optimized for oil tanks."""
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Preprocess for oil tank SAR images
        processed_image = self._preprocess_for_tanks(image, **kwargs)
        edges = self._detect_edges(processed_image, **kwargs)
        return self._postprocess_edges(edges, image.shape, **kwargs)
    
    def _preprocess_for_tanks(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocessing specifically for oil tank SAR images."""
        # Enhance contrast for tank features
        if kwargs.get('enhance_contrast', True):
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
        # Apply mild smoothing to reduce SAR speckle while preserving edges
        if kwargs.get('reduce_speckle', True):
            image = cv2.medianBlur(image, 3)
            
        return image
    
    def _postprocess_edges(self, edges: np.ndarray, image_shape: tuple, **kwargs) -> np.ndarray:
        """Post-process edges for oil tank detection."""
        height, width = image_shape
        
        # Remove small noise
        if kwargs.get('remove_small_edges', True):
            min_edge_length = kwargs.get('min_edge_length', 20)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edges_clean = np.zeros_like(edges)
            for contour in contours:
                if cv2.arcLength(contour, True) > min_edge_length:
                    cv2.drawContours(edges_clean, [contour], 0, 255, 1)
            edges = edges_clean
        
        # Emphasize horizontal edges (tank circles are horizontal ellipses)
        if kwargs.get('enhance_horizontal', True):
            kernel_horizontal = np.array([[-1, -1, -1],
                                        [ 2,  2,  2],
                                        [-1, -1, -1]], dtype=np.float32)
            horizontal_edges = cv2.filter2D(edges, cv2.CV_32F, kernel_horizontal)
            horizontal_edges = cv2.normalize(horizontal_edges, None, 0, 255, cv2.NORM_MINMAX)
            edges = cv2.addWeighted(edges, 0.7, horizontal_edges.astype(np.uint8), 0.3, 0)
        
        return edges
    
    @abstractmethod
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass

@ProcessorRegistry.register("canny_tank")
class CannyTankEdgeDetector(BaseEdgeDetector):
    """Canny edge detector optimized for oil tank circles."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Adaptive thresholds for oil tanks
        median_val = np.median(image)
        threshold1 = kwargs.get('threshold1', max(10, median_val * 0.5))
        threshold2 = kwargs.get('threshold2', max(30, median_val * 1.5))
        
        # Use aperture size 5 for better circle detection
        aperture_size = kwargs.get('aperture_size', 5)
        l2_gradient = kwargs.get('l2_gradient', True)
        
        return cv2.Canny(image, threshold1, threshold2, 
                        apertureSize=aperture_size, L2gradient=l2_gradient)

@ProcessorRegistry.register("adaptive_threshold_tank")
class AdaptiveThresholdTankEdgeDetector(BaseEdgeDetector):
    """Adaptive threshold optimized for oil tank features."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        block_size = kwargs.get('block_size', 15)  # Larger for tank circles
        c = kwargs.get('c', 3)  # Higher for SAR contrast
        
        # Use Gaussian adaptive threshold for smoother transitions
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
        
        # Clean up small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary

@ProcessorRegistry.register("multi_scale_tank")
class MultiScaleTankEdgeDetector(BaseEdgeDetector):
    """Multi-scale edge detection for different tank features."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        scales = kwargs.get('scales', [1.0, 0.75, 1.25])
        combined_edges = np.zeros_like(image, dtype=np.uint8)
        
        for scale in scales:
            # Resize image
            if scale != 1.0:
                h, w = image.shape
                new_size = (int(w * scale), int(h * scale))
                scaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                scaled_img = image
            
            # Detect edges at this scale
            edges_scale = cv2.Canny(scaled_img, 20, 60)
            
            # Resize back to original size
            if scale != 1.0:
                edges_scale = cv2.resize(edges_scale, (image.shape[1], image.shape[0]))
            
            combined_edges = cv2.bitwise_or(combined_edges, edges_scale)
        
        return combined_edges

@ProcessorRegistry.register("gradient_magnitude_tank")
class GradientMagnitudeTankEdgeDetector(BaseEdgeDetector):
    """Gradient magnitude based edge detection for tank circles."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and threshold
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        _, edges = cv2.threshold(magnitude, kwargs.get('magnitude_threshold', 30), 
                               255, cv2.THRESH_BINARY)
        
        return edges.astype(np.uint8)

@ProcessorRegistry.register("phase_congruency_tank")
class PhaseCongruencyTankEdgeDetector(BaseEdgeDetector):
    """Phase congruency based edge detection (good for SAR)."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Simple implementation of phase congruency concept
        scales = 3
        orientations = 6
        
        # Create filter bank
        filters = []
        for scale in range(scales):
            sigma = 1.5 * (scale + 1)
            size = int(6 * sigma) | 1
            gaussian = cv2.getGaussianKernel(size, sigma)
            filters.append(gaussian)
        
        # Apply filters and combine responses
        edge_strength = np.zeros_like(image, dtype=np.float32)
        
        for filter_kernel in filters:
            # Apply in both orientations
            filtered_x = cv2.sepFilter2D(image.astype(np.float32), -1, filter_kernel, filter_kernel.T)
            filtered_y = cv2.sepFilter2D(image.astype(np.float32), -1, filter_kernel.T, filter_kernel)
            
            # Compute magnitude
            magnitude = np.sqrt(filtered_x**2 + filtered_y**2)
            edge_strength = np.maximum(edge_strength, magnitude)
        
        # Normalize and threshold
        edge_strength = cv2.normalize(edge_strength, None, 0, 255, cv2.NORM_MINMAX)
        _, edges = cv2.threshold(edge_strength, kwargs.get('pc_threshold', 25), 
                               255, cv2.THRESH_BINARY)
        
        return edges.astype(np.uint8)

@ProcessorRegistry.register("morphological_tank")
class MorphologicalTankEdgeDetector(BaseEdgeDetector):
    """Morphological edge detection for tank structure."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Use morphological gradients to highlight tank boundaries
        kernel_size = kwargs.get('kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Morphological gradient (dilation - erosion)
        dilated = cv2.dilate(image, kernel)
        eroded = cv2.erode(image, kernel)
        edges = dilated - eroded
        
        # Enhance circular patterns
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, circular_kernel)
        
        # Threshold
        _, edges = cv2.threshold(edges, kwargs.get('morph_threshold', 10), 
                               255, cv2.THRESH_BINARY)
        
        return edges

@ProcessorRegistry.register("intensity_profile_tank")
class IntensityProfileTankEdgeDetector(BaseEdgeDetector):
    """Edge detection based on intensity profile analysis for tanks."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        height, width = image.shape
        
        # Analyze horizontal intensity profiles (across tank width)
        edges = np.zeros_like(image)
        
        for y in range(height):
            profile = image[y, :]
            
            # Compute gradient of profile
            gradient = np.gradient(profile.astype(np.float32))
            
            # Find significant transitions (potential tank edges)
            threshold = np.std(gradient) * kwargs.get('profile_threshold', 1.5)
            edge_positions = np.where(np.abs(gradient) > threshold)[0]
            
            # Mark edges
            for x in edge_positions:
                edges[y, x] = 255
        
        # Connect nearby edge points
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges

@ProcessorRegistry.register("combined_tank")
class CombinedTankEdgeDetector(BaseEdgeDetector):
    """Combined multiple edge detection methods for robust tank detection."""
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Combine multiple edge detection methods
        methods = [
            ('canny', CannyTankEdgeDetector(self.config)),
            ('adaptive', AdaptiveThresholdTankEdgeDetector(self.config)),
            ('morphological', MorphologicalTankEdgeDetector(self.config))
        ]
        
        combined_edges = np.zeros_like(image, dtype=np.uint8)
        
        for method_name, detector in methods:
            try:
                edges = detector._detect_edges(image, **kwargs)
                combined_edges = cv2.bitwise_or(combined_edges, edges)
            except Exception as e:
                print(f"Edge method {method_name} failed: {e}")
                continue
        
        # Emphasize circular patterns
        if kwargs.get('enhance_circular', True):
            combined_edges = self._enhance_circular_patterns(combined_edges)
        
        return combined_edges
    
    def _enhance_circular_patterns(self, edges: np.ndarray) -> np.ndarray:
        """Enhance circular/elliptical patterns for tank detection."""
        # Use Hough transform to find circles and enhance those regions
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, 
                                 minDist=30, param1=50, param2=30,
                                 minRadius=10, maxRadius=100)
        
        if circles is not None:
            # Create mask for circle regions
            circle_mask = np.zeros_like(edges)
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                cv2.circle(circle_mask, (x, y), r, 255, -1)
            
            # Enhance edges in circle regions
            edges = cv2.bitwise_and(edges, circle_mask)
            edges = cv2.dilate(edges, None, iterations=1)
        
        return edges

@ProcessorRegistry.register("deep_learning_tank")
class DeepLearningTankEdgeDetector(BaseEdgeDetector):
    """Edge detection using pre-trained deep learning models."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained edge detection model."""
        try:
            # This would load a pre-trained model like HED or RCF
            # For now, we'll use a placeholder
            pass
        except:
            print("Deep learning model not available, falling back to traditional methods")
    
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Placeholder for deep learning based edge detection
        # In practice, you would use models like:
        # - Holistically-Nested Edge Detection (HED)
        # - Richer Convolutional Features (RCF)
        # - Custom trained models on SAR oil tank data
        
        # Fallback to Canny if model not available
        fallback_detector = CannyTankEdgeDetector(self.config)
        return fallback_detector._detect_edges(image, **kwargs)

@ProcessorRegistry.register("sobel")
class SobelEdgeDetector(BaseEdgeDetector):
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        ksize = kwargs.get('ksize', 3)
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

@ProcessorRegistry.register("laplacian")
class LaplacianEdgeDetector(BaseEdgeDetector):
    def _detect_edges(self, image: np.ndarray, **kwargs) -> np.ndarray:
        ksize = kwargs.get('ksize', 3)
        return cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F, ksize=ksize))