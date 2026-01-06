"""
Feature Extraction Module for SAR-to-Height Data Processing

This module provides comprehensive feature extraction from SAR intensity data
to create enriched multi-channel input for the cordiff training pipeline.

Author: AI Assistant
Date: 2026-01-06
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import uniform_filter, generic_filter
from skimage import filters

# Handle different scikit-image versions and import paths
GLCM_AVAILABLE = False
try:
    # Try newer scikit-image versions
    from skimage.feature.texture import graycomatrix as greycomatrix
    from skimage.feature.texture import graycoprops as greycoprops
    GLCM_AVAILABLE = True
except ImportError:
    try:
        # Try older scikit-image versions
        from skimage.feature import greycomatrix, greycoprops
        GLCM_AVAILABLE = True
    except ImportError:
        try:
            # Try alternative import paths
            from skimage.feature import graycomatrix as greycomatrix
            from skimage.feature import graycoprops as greycoprops
            GLCM_AVAILABLE = True
        except ImportError:
            # GLCM not available, will use alternative texture features
            GLCM_AVAILABLE = False

# Try to import other skimage components with fallbacks
try:
    from skimage.filters import rank
    from skimage.morphology import disk
    RANK_FILTERS_AVAILABLE = True
except ImportError:
    RANK_FILTERS_AVAILABLE = False

from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SARFeatureExtractor:
    """
    Class to extract multiple features from SAR intensity data.
    
    This class provides methods to:
    - Apply intensity transformations (linear, power, percentile scaling)
    - Apply speckle filtering (Lee filter)
    - Compute texture features (GLCM, local statistics)
    - Compute edge and gradient features
    - Stack all features into multi-channel arrays
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
        self.alpha_power = 0.3  # Power transform parameter
        
    def extract_all_features(self, intensity_db: np.ndarray, 
                           metadata: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from SAR intensity data.
        
        Args:
            intensity_db (np.ndarray): SAR intensity in dB
            metadata (Dict, optional): Metadata that might contain incidence angle
            
        Returns:
            Tuple containing:
            - features_stack: Multi-channel feature array (channels, height, width)
            - feature_names: List of feature names corresponding to channels
        """
        logger.info(f"Extracting features from intensity data: {intensity_db.shape}")
        
        features = []
        feature_names = []
        
        # 1. Original intensity values (dB)
        features.append(intensity_db)
        feature_names.append('intensity_db')
        
        # 2. Percentile rescaled (2-98th percentile)
        percentile_rescaled = self._percentile_rescale(intensity_db)
        features.append(percentile_rescaled)
        feature_names.append('intensity_percentile_rescaled')
        
        # 3. Linear intensity (10^(dB/10))
        linear_intensity = self._db_to_linear(intensity_db)
        features.append(linear_intensity)
        feature_names.append('intensity_linear')
        
        # 4. Square root of linear intensity
        sqrt_linear = np.sqrt(np.maximum(linear_intensity, 0))  # Ensure non-negative
        features.append(sqrt_linear)
        feature_names.append('intensity_sqrt_linear')
        
        # 5. Power transform (α=0.3)
        power_transform = self._power_transform(intensity_db, self.alpha_power)
        features.append(power_transform)
        feature_names.append('intensity_power_transform')
        
        # 6. Speckle-filtered intensity (Lee filter)
        lee_filtered = self._lee_filter(linear_intensity)
        features.append(lee_filtered)
        feature_names.append('intensity_lee_filtered')
        
        # 7. Local mean (5x5)
        local_mean_5x5 = self._local_mean(intensity_db, window_size=5)
        features.append(local_mean_5x5)
        feature_names.append('local_mean_5x5')
        
        # 8. Local std (5x5)
        local_std_5x5 = self._local_std(intensity_db, window_size=5)
        features.append(local_std_5x5)
        feature_names.append('local_std_5x5')
        
        # 9. Sobel magnitude and direction
        sobel_mag, sobel_dir = self._sobel_features(intensity_db)
        features.extend([sobel_mag, sobel_dir])
        feature_names.extend(['sobel_magnitude', 'sobel_direction'])
        
        # 10. GLCM contrast and homogeneity (or alternatives)
        glcm_contrast, glcm_homogeneity = self._glcm_features(intensity_db)
        features.extend([glcm_contrast, glcm_homogeneity])
        feature_names.extend(['texture_contrast', 'texture_homogeneity'])
        
        # 11. Multi-scale mean (3x3, 7x7)
        local_mean_3x3 = self._local_mean(intensity_db, window_size=3)
        local_mean_7x7 = self._local_mean(intensity_db, window_size=7)
        features.extend([local_mean_3x3, local_mean_7x7])
        feature_names.extend(['local_mean_3x3', 'local_mean_7x7'])
        
        # 12. Gradient magnitude
        gradient_mag = self._gradient_magnitude(intensity_db)
        features.append(gradient_mag)
        feature_names.append('gradient_magnitude')
        
        # 13. Local incidence angle (if available) or local range
        if metadata and self._has_incidence_angle(metadata):
            incidence_angle = self._extract_incidence_angle(metadata, intensity_db.shape)
            features.append(incidence_angle)
            feature_names.append('incidence_angle')
        else:
            # Use local range as alternative
            local_range = self._local_range(intensity_db, window_size=5)
            features.append(local_range)
            feature_names.append('local_range')
        
        # Stack all features
        features_stack = np.stack(features, axis=0)
        
        logger.info(f"Extracted {features_stack.shape[0]} features: {features_stack.shape}")
        
        self.feature_names = feature_names
        return features_stack, feature_names
    
    def _percentile_rescale(self, data: np.ndarray, 
                          lower_percentile: float = 2, 
                          upper_percentile: float = 98) -> np.ndarray:
        """Apply percentile-based rescaling."""
        valid_mask = np.isfinite(data) & (data > -100)  # Exclude extreme values
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return np.zeros_like(data)
        
        p_low, p_high = np.percentile(valid_data, [lower_percentile, upper_percentile])
        rescaled = np.clip((data - p_low) / (p_high - p_low), 0, 1)
        
        return rescaled
    
    def _db_to_linear(self, intensity_db: np.ndarray) -> np.ndarray:
        """Convert dB intensity to linear scale."""
        # Handle invalid values
        valid_mask = np.isfinite(intensity_db)
        linear = np.zeros_like(intensity_db)
        
        # Convert valid dB values to linear
        linear[valid_mask] = np.power(10.0, intensity_db[valid_mask] / 10.0)
        
        return linear
    
    def _power_transform(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Apply power transform: sign(x) * |x|^alpha."""
        return np.sign(data) * np.power(np.abs(data), alpha)
    
    def _lee_filter(self, linear_intensity: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply Lee speckle filter to linear intensity data.
        
        The Lee filter reduces speckle while preserving edges.
        """
        # Convert to float for processing
        data = linear_intensity.astype(np.float64)
        
        # Avoid division by zero
        data = np.maximum(data, 1e-10)
        
        # Calculate local mean and variance
        mean = uniform_filter(data, size=window_size)
        sqr_mean = uniform_filter(data**2, size=window_size)
        variance = np.maximum(sqr_mean - mean**2, 0)
        
        # Lee filter coefficient
        # Avoid division by zero
        coefficient = np.where(mean > 1e-10, variance / (mean**2), 0)
        coefficient = np.clip(coefficient, 0, 1)
        
        # Apply Lee filter
        filtered = mean + coefficient * (data - mean)
        
        return filtered.astype(linear_intensity.dtype)
    
    def _local_mean(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local mean using uniform filter."""
        return uniform_filter(data.astype(np.float64), size=window_size).astype(data.dtype)
    
    def _local_std(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local standard deviation."""
        data_float = data.astype(np.float64)
        mean = uniform_filter(data_float, size=window_size)
        sqr_mean = uniform_filter(data_float**2, size=window_size)
        variance = np.maximum(sqr_mean - mean**2, 0)
        std = np.sqrt(variance)
        
        return std.astype(data.dtype)
    
    def _local_range(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate local range (max - min) as alternative to incidence angle."""
        from scipy.ndimage import maximum_filter, minimum_filter
        
        local_max = maximum_filter(data, size=window_size)
        local_min = minimum_filter(data, size=window_size)
        local_range = local_max - local_min
        
        return local_range.astype(data.dtype)
    
    def _sobel_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Sobel edge magnitude and direction."""
        # Apply Sobel operators
        sobel_x = cv2.Sobel(data.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(data.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        return magnitude.astype(data.dtype), direction.astype(data.dtype)
    
    def _glcm_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate texture features using GLCM if available, otherwise use alternatives.
        """
        if GLCM_AVAILABLE:
            try:
                return self._glcm_texture_features(data)
            except Exception as e:
                logger.warning(f"GLCM calculation failed: {e}, using alternative texture features")
                return self._alternative_texture_features(data)
        else:
            logger.info("GLCM not available, using alternative texture features")
            return self._alternative_texture_features(data)
    
    def _glcm_texture_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate GLCM-based texture features."""
        # Normalize data to 8-bit for GLCM computation
        data_normalized = ((data - np.nanmin(data)) / 
                          (np.nanmax(data) - np.nanmin(data)) * 255).astype(np.uint8)
        
        # Initialize output arrays
        contrast = np.zeros_like(data, dtype=np.float32)
        homogeneity = np.zeros_like(data, dtype=np.float32)
        
        # Process in smaller blocks for memory efficiency
        block_size = 32
        window_size = 7
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for i in range(0, data.shape[0], block_size):
            for j in range(0, data.shape[1], block_size):
                i_end = min(i + block_size, data.shape[0])
                j_end = min(j + block_size, data.shape[1])
                
                center_i = (i + i_end) // 2
                center_j = (j + j_end) // 2
                
                # Extract window around center
                win_i_start = max(0, center_i - window_size//2)
                win_i_end = min(data.shape[0], center_i + window_size//2 + 1)
                win_j_start = max(0, center_j - window_size//2)
                win_j_end = min(data.shape[1], center_j + window_size//2 + 1)
                
                window = data_normalized[win_i_start:win_i_end, win_j_start:win_j_end]
                
                if window.size > window_size * window_size // 2:  # Ensure sufficient data
                    try:
                        glcm = greycomatrix(window, distances, angles, levels=64, 
                                          symmetric=True, normed=True)
                        
                        contrast_val = np.mean(greycoprops(glcm, 'contrast'))
                        homogeneity_val = np.mean(greycoprops(glcm, 'homogeneity'))
                        
                        contrast[center_i, center_j] = contrast_val
                        homogeneity[center_i, center_j] = homogeneity_val
                        
                    except Exception:
                        pass  # Skip failed calculations
        
        # Fill missing values using interpolation
        contrast = self._fill_missing_values(contrast)
        homogeneity = self._fill_missing_values(homogeneity)
        
        return contrast, homogeneity
    
    def _alternative_texture_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alternative texture features when GLCM is not available.
        
        Uses local variance and uniformity measures as texture proxies.
        """
        window_size = 7
        
        # Local variance as contrast proxy
        local_variance = uniform_filter(data.astype(np.float64)**2, size=window_size) - \
                        uniform_filter(data.astype(np.float64), size=window_size)**2
        local_variance = np.maximum(local_variance, 0)
        
        # Local coefficient of variation as homogeneity proxy
        local_mean = uniform_filter(data.astype(np.float64), size=window_size)
        local_std = np.sqrt(local_variance)
        
        # Avoid division by zero
        cv = np.where(local_mean > 1e-10, local_std / local_mean, 0)
        homogeneity_proxy = 1.0 / (1.0 + cv)
        
        return local_variance.astype(data.dtype), homogeneity_proxy.astype(data.dtype)
    
    def _gradient_magnitude(self, data: np.ndarray) -> np.ndarray:
        """Calculate gradient magnitude using Sobel operator."""
        grad_x = ndimage.sobel(data.astype(np.float64), axis=1)
        grad_y = ndimage.sobel(data.astype(np.float64), axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude.astype(data.dtype)
    
    def _has_incidence_angle(self, metadata: Dict) -> bool:
        """Check if metadata contains incidence angle information."""
        incidence_fields = ['incidence_angle', 'incident_angle', 'look_angle']
        
        for field in incidence_fields:
            if field in metadata:
                return True
        
        if 'tags' in metadata:
            for tag_name, tag_value in metadata['tags'].items():
                if any(keyword in str(tag_name).lower() for keyword in ['incidence', 'incident', 'look']):
                    return True
        
        return False
    
    def _extract_incidence_angle(self, metadata: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """Extract incidence angle from metadata."""
        incidence_fields = ['incidence_angle', 'incident_angle', 'look_angle']
        
        for field in incidence_fields:
            if field in metadata:
                angle = metadata[field]
                if isinstance(angle, (int, float)):
                    return np.full(shape, angle, dtype=np.float32)
                elif isinstance(angle, np.ndarray):
                    if angle.shape == shape:
                        return angle.astype(np.float32)
                    else:
                        from scipy.ndimage import zoom
                        zoom_factors = (shape[0]/angle.shape[0], shape[1]/angle.shape[1])
                        return zoom(angle, zoom_factors).astype(np.float32)
        
        logger.warning("Could not extract incidence angle from metadata, using default value")
        return np.full(shape, 30.0, dtype=np.float32)
    
    def _fill_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Fill missing (zero) values using nearest neighbor interpolation."""
        from scipy.ndimage import distance_transform_edt
        
        missing_mask = (data == 0) | ~np.isfinite(data)
        
        if not np.any(missing_mask):
            return data
        
        valid_mask = ~missing_mask
        if np.any(valid_mask):
            indices = distance_transform_edt(missing_mask, return_distances=False, return_indices=True)
            filled_data = data[tuple(indices)]
        else:
            filled_data = np.zeros_like(data)
        
        return filled_data
    
    def normalize_features(self, features: np.ndarray, 
                         method: str = 'standard') -> Tuple[np.ndarray, Dict]:
        """
        Normalize features across spatial dimensions.
        
        Args:
            features (np.ndarray): Feature array (channels, height, width)
            method (str): Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple containing normalized features and normalization parameters
        """
        normalized_features = np.zeros_like(features)
        norm_params = {}
        
        for i in range(features.shape[0]):
            feature_data = features[i]
            valid_mask = np.isfinite(feature_data)
            valid_data = feature_data[valid_mask]
            
            if len(valid_data) == 0:
                normalized_features[i] = feature_data
                norm_params[i] = {'mean': 0, 'std': 1, 'min': 0, 'max': 1}
                continue
            
            if method == 'standard':
                mean = np.mean(valid_data)
                std = np.std(valid_data)
                if std > 0:
                    normalized_features[i] = (feature_data - mean) / std
                else:
                    normalized_features[i] = feature_data - mean
                norm_params[i] = {'mean': mean, 'std': std}
                
            elif method == 'minmax':
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                if max_val > min_val:
                    normalized_features[i] = (feature_data - min_val) / (max_val - min_val)
                else:
                    normalized_features[i] = feature_data - min_val
                norm_params[i] = {'min': min_val, 'max': max_val}
                
            elif method == 'robust':
                median = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median))
                if mad > 0:
                    normalized_features[i] = (feature_data - median) / mad
                else:
                    normalized_features[i] = feature_data - median
                norm_params[i] = {'median': median, 'mad': mad}
        
        return normalized_features, norm_params


def extract_sar_features(intensity_db: np.ndarray, 
                        metadata: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to extract all SAR features.
    
    Args:
        intensity_db (np.ndarray): SAR intensity in dB
        metadata (Dict, optional): Metadata dictionary
        
    Returns:
        Tuple containing feature stack and feature names
    """
    extractor = SARFeatureExtractor()
    return extractor.extract_all_features(intensity_db, metadata)


def extract_selected_sar_features(intensity_db: np.ndarray, 
                                 enabled_features: List[str],
                                 metadata: Optional[Dict] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Extract only the selected SAR features from the enabled features list.
    
    Args:
        intensity_db (np.ndarray): SAR intensity in dB
        enabled_features (List[str]): List of feature names to extract
        metadata (Dict, optional): Metadata dictionary
        
    Returns:
        Tuple containing:
        - selected_features: Feature stack with only enabled features (selected_features, height, width)
        - selected_names: List of selected feature names
    """
    # Get all features first
    all_features, all_names = extract_sar_features(intensity_db, metadata)
    
    # Filter to only enabled features
    selected_indices = []
    selected_names = []
    
    for name in enabled_features:
        if name in all_names:
            idx = all_names.index(name)
            selected_indices.append(idx)
            selected_names.append(name)
        else:
            logger.warning(f"Feature '{name}' not found in extracted features. Available: {all_names}")
    
    if not selected_indices:
        raise ValueError(f"No valid features selected! Available features: {all_names}")
    
    # Extract selected features
    selected_features = all_features[selected_indices]
    
    logger.info(f"Selected {len(selected_features)} features from {len(all_features)} total: {selected_names}")
    
    return selected_features, selected_names


if __name__ == "__main__":
    # Example usage
    dummy_intensity = np.random.randn(100, 100) * 10 - 20
    features, names = extract_sar_features(dummy_intensity)
    print(f"Extracted {features.shape[0]} features:")
    for i, name in enumerate(names):
        print(f"  {i}: {name} - shape: {features[i].shape}")