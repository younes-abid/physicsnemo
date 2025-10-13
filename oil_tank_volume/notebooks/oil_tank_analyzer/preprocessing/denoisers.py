import cv2
import numpy as np
from scipy import ndimage
from abc import abstractmethod
from oil_tank_analyzer.core.base_processor import BaseProcessor, ProcessorRegistry

class BaseDenoiser(BaseProcessor):
    """Base class for denoising algorithms optimized for SAR oil tank imagery."""
    
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Convert to float32 for better processing
        if image.dtype != np.float32:
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = image.astype(np.float32) / 255.0
        
        denoised = self._denoise(image, **kwargs)
        
        # Post-processing for oil tank features
        denoised = self._postprocess_for_tanks(denoised, **kwargs)
        
        return (denoised * 255).astype(np.uint8)
    
    def _postprocess_for_tanks(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Post-processing specifically for oil tank features."""
        # Enhance circular patterns
        if kwargs.get('enhance_circular', True):
            image = self._enhance_circular_features(image, **kwargs)
        
        # Preserve edges while smoothing
        if kwargs.get('edge_preserving', True):
            image = self._edge_preserving_enhancement(image, **kwargs)
        
        return np.clip(image, 0, 1)
    
    def _enhance_circular_features(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance circular/elliptical patterns for tank detection."""
        # Use morphological operations to enhance circular patterns
        kernel_size = kwargs.get('circular_kernel_size', 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Top-hat transform to enhance bright circular features
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        # Black-hat transform to enhance dark circular features
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine with original image
        enhanced = image + 0.5 * tophat - 0.3 * blackhat
        
        return np.clip(enhanced, 0, 1)
    
    def _edge_preserving_enhancement(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance edges while preserving smooth regions."""
        # Compute gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Create edge mask
        edge_threshold = kwargs.get('edge_threshold', 0.1)
        edge_mask = gradient_mag > edge_threshold
        
        # Enhance edges slightly
        enhanced = image.copy()
        enhanced[edge_mask] = np.clip(enhanced[edge_mask] * 1.1, 0, 1)
        
        return enhanced
    
    @abstractmethod
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass

@ProcessorRegistry.register("identity_denoiser")
class IdentityDenoiser(BaseDenoiser):
    """Identity denoiser - returns the image unchanged."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return image.copy()
    
@ProcessorRegistry.register("lee_sigma")
class LeeSigmaDenoiser(BaseDenoiser):
    """Lee-Sigma filter for SAR imagery - preserves edges while reducing speckle."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 7)
        sigma_range = kwargs.get('sigma_range', 2.0)  # Standard deviations for intensity range
        
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        denoised = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                center_val = window[pad, pad]
                mean_val = np.mean(window)
                std_val = np.std(window)
                
                # Lee-Sigma algorithm
                if std_val > 0:
                    # Compute intensity range
                    intensity_range = sigma_range * std_val
                    lower_bound = mean_val - intensity_range
                    upper_bound = mean_val + intensity_range
                    
                    # Filter pixels within range
                    valid_pixels = window[(window >= lower_bound) & (window <= upper_bound)]
                    if len(valid_pixels) > 0:
                        denoised[i, j] = np.mean(valid_pixels)
                    else:
                        denoised[i, j] = center_val
                else:
                    denoised[i, j] = center_val
        
        return denoised

@ProcessorRegistry.register("enhanced_lee")
class EnhancedLeeDenoiser(BaseDenoiser):
    """Enhanced Lee filter - improved version for SAR with better edge preservation."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 7)
        damping_factor = kwargs.get('damping_factor', 1.0)
        cu = kwargs.get('cu', 0.523)  # Noise variation coefficient
        max_sigma = kwargs.get('max_sigma', 2.0)
        
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        denoised = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                center_val = window[pad, pad]
                mean_val = np.mean(window)
                std_val = np.std(window)
                cv_val = std_val / mean_val if mean_val > 0 else 0
                
                # Enhanced Lee algorithm
                if cv_val <= cu:
                    # Homogeneous region - use mean
                    denoised[i, j] = mean_val
                elif cv_val > cu and cv_val <= max_sigma:
                    # Edge region - weighted average
                    weight = np.exp(-damping_factor * (cv_val - cu))
                    denoised[i, j] = weight * center_val + (1 - weight) * mean_val
                else:
                    # Strong edge - preserve original
                    denoised[i, j] = center_val
        
        return denoised

@ProcessorRegistry.register("frost")
class FrostDenoiser(BaseDenoiser):
    """Frost filter - exponential weighting for SAR despeckling."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 7)
        damping_factor = kwargs.get('damping_factor', 2.0)
        
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        denoised = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                center_val = window[pad, pad]
                mean_val = np.mean(window)
                std_val = np.std(window)
                
                # Frost algorithm
                if std_val > 0:
                    cv_val = std_val / mean_val
                    # Create distance-weighted kernel
                    weights = np.zeros_like(window)
                    for m in range(window_size):
                        for n in range(window_size):
                            dist = np.sqrt((m-pad)**2 + (n-pad)**2)
                            k = damping_factor * cv_val * dist
                            weights[m, n] = np.exp(-k)
                    
                    # Normalize weights and apply
                    weights = weights / np.sum(weights)
                    denoised[i, j] = np.sum(window * weights)
                else:
                    denoised[i, j] = center_val
        
        return denoised

@ProcessorRegistry.register("gamma_map")
class GammaMAPDenoiser(BaseDenoiser):
    """Gamma Maximum A Posteriori filter - statistical approach for SAR."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 7)
        looks = kwargs.get('looks', 1)  # Number of looks in SAR data
        eps = 1e-8
        
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        denoised = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                center_val = window[pad, pad]
                mean_val = np.mean(window)
                
                # Gamma MAP algorithm
                alpha = looks - 1
                if alpha > 0 and mean_val > eps:
                    # Solve quadratic equation for Gamma MAP
                    a = 1
                    b = (2 * alpha - looks - 1) / (looks * mean_val)
                    c = -center_val * (2 * alpha - 1) / (looks * mean_val ** 2)
                    
                    # Take positive root
                    discriminant = b**2 - 4*a*c
                    if discriminant >= 0:
                        root1 = (-b + np.sqrt(discriminant)) / (2*a)
                        root2 = (-b - np.sqrt(discriminant)) / (2*a)
                        denoised[i, j] = max(root1, root2, 0)
                    else:
                        denoised[i, j] = center_val
                else:
                    denoised[i, j] = center_val
        
        return denoised

@ProcessorRegistry.register("kuan")
class KuanDenoiser(BaseDenoiser):
    """Kuan filter - minimum mean square error estimator for SAR."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        window_size = kwargs.get('window_size', 7)
        looks = kwargs.get('looks', 1)
        
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        denoised = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+window_size, j:j+window_size]
                center_val = window[pad, pad]
                mean_val = np.mean(window)
                var_val = np.var(window)
                
                # Kuan algorithm
                if var_val > 0 and mean_val > 0:
                    cu = np.sqrt(1 / looks)  # Noise coefficient
                    ci = np.sqrt(var_val) / mean_val  # Signal coefficient
                    
                    if ci > 0:
                        weight = (1 - cu**2 / ci**2) / (1 + cu**2)
                        weight = max(0, min(1, weight))  # Clamp to [0,1]
                        denoised[i, j] = weight * center_val + (1 - weight) * mean_val
                    else:
                        denoised[i, j] = mean_val
                else:
                    denoised[i, j] = center_val
        
        return denoised

@ProcessorRegistry.register("sar_bilateral")
class SARBilateralDenoiser(BaseDenoiser):
    """Bilateral filter optimized for SAR imagery with intensity and spatial domains."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 0.1)  # Lower for SAR
        sigma_space = kwargs.get('sigma_space', 75)
        
        # Convert to uint8 for OpenCV bilateral filter
        image_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(image_uint8, d, sigma_color * 255, sigma_space)
        
        return denoised.astype(np.float32) / 255.0

@ProcessorRegistry.register("nonlocal_means_sar")
class NonLocalMeansSARDenoiser(BaseDenoiser):
    """Non-local means optimized for SAR with template matching."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        h = kwargs.get('h', 7)  # Lower for SAR
        template_size = kwargs.get('template_size', 5)  # Smaller for tank features
        search_size = kwargs.get('search_size', 15)
        
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(image_uint8, None, h, template_size, search_size)
        
        return denoised.astype(np.float32) / 255.0

@ProcessorRegistry.register("wavelet_sar")
class WaveletSARDenoiser(BaseDenoiser):
    """Wavelet denoising optimized for SAR oil tank imagery."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        level = kwargs.get('level', 3)
        threshold_factor = kwargs.get('threshold_factor', 0.1)
        
        # Pyramidal decomposition
        pyramid = [image]
        current = image
        
        for _ in range(level):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        # Apply soft thresholding to detail coefficients
        for i in range(1, len(pyramid)):
            # Simple thresholding - in practice you'd use proper wavelet transform
            threshold = threshold_factor * np.std(pyramid[i])
            pyramid[i] = self._soft_threshold(pyramid[i], threshold)
        
        # Reconstruction
        denoised = pyramid[-1]
        for i in range(len(pyramid)-2, -1, -1):
            denoised = cv2.pyrUp(denoised)
            if denoised.shape != pyramid[i].shape:
                denoised = cv2.resize(denoised, (pyramid[i].shape[1], pyramid[i].shape[0]))
            # Blend with original at each level
            denoised = 0.7 * denoised + 0.3 * pyramid[i]
        
        return denoised
    
    def _soft_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding to wavelet coefficients."""
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

@ProcessorRegistry.register("median_tank")
class MedianTankDenoiser(BaseDenoiser):
    """Median filter optimized for oil tank circular features."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kernel_size = kwargs.get('kernel_size', 5)
        
        # Use circular kernel to preserve tank shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply median filter with circular kernel
        denoised = cv2.medianBlur((image * 255).astype(np.uint8), kernel_size)
        
        return denoised.astype(np.float32) / 255.0

@ProcessorRegistry.register("adaptive_tank")
class AdaptiveTankDenoiser(BaseDenoiser):
    """Adaptive denoising that preserves tank structural features."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        # Combine multiple denoising strategies
        base_denoiser = EnhancedLeeDenoiser(self.config)
        base_denoised = base_denoiser._denoise(image, **kwargs)
        
        # Enhance tank-specific features
        if kwargs.get('enhance_tank_features', True):
            base_denoised = self._enhance_tank_features(base_denoised, image, **kwargs)
        
        return base_denoised
    
    def _enhance_tank_features(self, denoised: np.ndarray, original: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance features specific to oil tanks."""
        # Compute gradient to find edges
        grad_original = cv2.Laplacian(original, cv2.CV_32F)
        grad_denoised = cv2.Laplacian(denoised, cv2.CV_32F)
        
        # Preserve strong edges from original
        edge_mask = np.abs(grad_original) > kwargs.get('edge_threshold', 0.05)
        enhanced = denoised.copy()
        enhanced[edge_mask] = 0.7 * denoised[edge_mask] + 0.3 * original[edge_mask]
        
        return enhanced

@ProcessorRegistry.register("multi_scale_tank")
class MultiScaleTankDenoiser(BaseDenoiser):
    """Multi-scale denoising for different tank feature sizes."""
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        scales = kwargs.get('scales', [1.0, 0.5, 2.0])
        weights = kwargs.get('weights', [0.6, 0.2, 0.2])
        
        # Store original image range
        original_dtype = image.dtype
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Convert to float for processing
        image_float = image.astype(np.float32)
        denoised_scales = []
        
        for scale, weight in zip(scales, weights):
            if scale != 1.0:
                # Resize image
                h, w = image.shape
                new_size = (int(w * scale), int(h * scale))
                scaled_img = cv2.resize(image_float, new_size, interpolation=cv2.INTER_AREA)
            else:
                scaled_img = image_float.copy()
            
            # Denoise at this scale
            if scale <= 1.0:
                # Use stronger denoising for smaller scales
                denoiser = EnhancedLeeDenoiser(self.config)
            else:
                # Use lighter denoising for larger scales
                denoiser = MedianTankDenoiser(self.config)
            
            # Convert to uint8 for denoising (most denoisers expect uint8)
            scaled_img_uint8 = cv2.normalize(scaled_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            denoised_scale = denoiser._denoise(scaled_img_uint8, **kwargs)
            
            # Convert back to float for weighted combination
            denoised_scale = denoised_scale.astype(np.float32)
            
            # Resize back to original size if needed
            if scale != 1.0:
                denoised_scale = cv2.resize(denoised_scale, (image.shape[1], image.shape[0]))
            
            denoised_scales.append(denoised_scale * weight)
        
        # Combine scales
        denoised = np.sum(denoised_scales, axis=0)
        
        # Convert back to original dtype
        if original_dtype == np.uint8:
            denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            denoised = np.clip(denoised, 0, 65535).astype(np.uint16)
        else:
            denoised = denoised.astype(original_dtype)
        
        return denoised

@ProcessorRegistry.register("gaussian")
class GaussianDenoiser(BaseDenoiser):
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        denoised = cv2.GaussianBlur((image * 255).astype(np.uint8), (kernel_size, kernel_size), sigma)
        return denoised.astype(np.float32) / 255.0
