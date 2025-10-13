"""
Preprocessing modules for oil tank SAR images.
"""
import numpy as np
import cv2
from abc import abstractmethod
from oil_tank_analyzer.core.base_processor import BaseProcessor, PreprocessorRegistry


class BasePreprocessor(BaseProcessor):
    """Base class for image preprocessing algorithms."""
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Main processing method with optional dtype preservation."""
        preserve_dtype = kwargs.get('preserve_dtype', False)
        original_dtype = image.dtype
        
        processed = self._preprocess(image, **kwargs)
        
        if preserve_dtype and processed.dtype != original_dtype:
            # Convert back to original dtype while preserving range
            if original_dtype == np.uint8:
                processed = np.clip(processed, 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                processed = np.clip(processed, 0, 65535).astype(np.uint16)
            else:
                processed = processed.astype(original_dtype)
        
        return processed
    
    @abstractmethod
    def _preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass


@PreprocessorRegistry.register("identity_preprocessor")
class IdentityPreprocessor(BasePreprocessor):
    """Identity preprocessor - returns image unchanged."""
    
    def _preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Handle different input formats
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Use first channel
        
        # Convert to proper type
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return image


@PreprocessorRegistry.register("basic_normalization")
class BasicNormalizationPreprocessor(BasePreprocessor):
    """Basic normalization to uint8 with optional contrast enhancement."""
    
    def _preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Handle different input formats
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Use first channel
        
        # Convert to proper type
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Optional contrast enhancement
        if kwargs.get('enhance_contrast', False):
            clip_limit = kwargs.get('clahe_clip_limit', 2.0)
            grid_size = kwargs.get('clahe_grid_size', 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            image = clahe.apply(image)
        
        return image


@PreprocessorRegistry.register("tank_enhancement")
class TankEnhancementPreprocessor(BasePreprocessor):
    """Enhanced preprocessing specifically for oil tank SAR images."""
    
    def _preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Handle different input formats
        if len(image.shape) == 3:
            image = image[:, :, 0]
        
        # Convert to uint8 for processing
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        enhanced = image.copy()
        
        # 1. Contrast enhancement
        if kwargs.get('enhance_contrast', True):
            clip_limit = kwargs.get('clahe_clip_limit', 2.0)
            grid_size = kwargs.get('clahe_grid_size', 8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            enhanced = clahe.apply(enhanced)
        
        # 2. Circular pattern enhancement
        if kwargs.get('enhance_circular', True):
            enhanced = self._enhance_circular_patterns(enhanced, **kwargs)
        
        # 3. Background suppression
        if kwargs.get('suppress_background', False):
            enhanced = self._suppress_background(enhanced, **kwargs)
        
        return enhanced
    
    def _enhance_circular_patterns(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance circular and elliptical patterns."""
        sigma1 = kwargs.get('dog_sigma1', 1.0)
        sigma2 = kwargs.get('dog_sigma2', 2.0)
        
        gaussian1 = cv2.GaussianBlur(image, (0, 0), sigma1)
        gaussian2 = cv2.GaussianBlur(image, (0, 0), sigma2)
        dog = gaussian1 - gaussian2
        
        enhanced = cv2.addWeighted(image, 0.7, dog, 0.3, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def _suppress_background(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Suppress background noise while preserving tank structures."""
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels > 1:
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            tank_mask = (labels == largest_component).astype(np.uint8) * 255
            
            masked_image = cv2.bitwise_and(image, image, mask=tank_mask)
            background_value = np.median(image[tank_mask == 0]) if np.any(tank_mask == 0) else 0
            result = np.where(tank_mask == 255, masked_image, background_value)
            
            return result.astype(np.uint8)
        
        return image


@PreprocessorRegistry.register("multi_scale_enhancement")
class MultiScaleEnhancementPreprocessor(BasePreprocessor):
    """Multi-scale enhancement for different tank features."""
    
    def _preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        scales = kwargs.get('scales', [0.5, 1.0, 2.0])
        weights = kwargs.get('weights', [0.2, 0.6, 0.2])
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        enhanced_scales = []
        
        for scale, weight in zip(scales, weights):
            if scale != 1.0:
                h, w = image.shape
                new_size = (int(w * scale), int(h * scale))
                scaled_img = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            else:
                scaled_img = image.copy()
            
            # Enhance at this scale
            if scale <= 1.0:
                # Stronger enhancement for smaller scales (details)
                enhanced_scale = cv2.GaussianBlur(scaled_img, (3, 3), 0)
            else:
                # Lighter enhancement for larger scales (structures)
                enhanced_scale = cv2.medianBlur(scaled_img, 3)
            
            # Resize back if needed
            if scale != 1.0:
                enhanced_scale = cv2.resize(enhanced_scale, (image.shape[1], image.shape[0]))
            
            enhanced_scales.append(enhanced_scale.astype(np.float32) * weight)
        
        # Combine scales
        enhanced = np.sum(enhanced_scales, axis=0)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return enhanced