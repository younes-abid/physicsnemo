"""
Feature Extraction Module for SAR-to-Height Data Processing

This module provides comprehensive feature extraction from SAR intensity data
to create enriched multi-channel input for the cordiff training pipeline.

Author: AI Assistant
Date: 2026-01-07
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import uniform_filter, generic_filter
from skimage import filters
import time
from functools import wraps
from typing import List, Tuple, Dict, Optional, Union

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

import logging

logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"⏱️  {func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper


class SARFeatureExtractor:
    """
    Class to extract multiple features from SAR intensity data.
    
    This class provides methods to:
    - Apply intensity transformations (linear, power, percentile scaling)
    - Apply speckle filtering (Lee filter)
    - Compute texture features (GLCM, local statistics)
    - Compute edge and gradient features
    - Extract only selected features for performance optimization
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.alpha_power = 0.3  # Power transform parameter
        
    def extract_selected_features(self, 
                                intensity: np.ndarray, 
                                selected_features: List[str],
                                metadata: Optional[Dict] = None,
                                timing: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Extract only selected features for efficiency.
        
        Args:
            intensity: SAR intensity data
            selected_features: List of feature names to extract
            metadata: Optional metadata
            timing: Whether to monitor and log timing information (for internal logging only)
            
        Returns:
            Tuple containing:
            - features (np.ndarray): Feature array (n_features, height, width)
            - feature_names (List[str]): Names of extracted features
        """
        timing_info = {} if timing else None
        start_total = time.time() if timing else None
        
        logger.info(f"Extracting {len(selected_features)} selected features: {selected_features}")
        
        available_features = self._get_available_features()
        
        # Validate selected features
        invalid_features = [f for f in selected_features if f not in available_features]
        if invalid_features:
            raise ValueError(f"Invalid features requested: {invalid_features}. "
                           f"Available: {list(available_features.keys())}")
        
        # Extract only the selected features
        features = []
        feature_names = []
        
        for feature_name in selected_features:
            if timing:
                start_feature = time.time()
            
            feature_func = available_features[feature_name]
            try:
                if feature_name in ['incidence_angle'] and metadata:
                    feature_data = feature_func(intensity, metadata)
                else:
                    feature_data = feature_func(intensity)
                    
                features.append(feature_data)
                feature_names.append(feature_name)
                
                if timing:
                    feature_time = time.time() - start_feature
                    timing_info[f"feature_{feature_name}"] = feature_time
                    logger.info(f"⏱️  Extracted '{feature_name}' in {feature_time:.2f}s")
                    
            except Exception as e:
                logger.warning(f"Failed to extract feature {feature_name}: {e}")
                if timing:
                    timing_info[f"feature_{feature_name}_failed"] = time.time() - start_feature
        
        if not features:
            raise RuntimeError("No features were successfully extracted")
        
        # Stack features
        features = np.stack(features, axis=0)
        
        if timing:
            timing_info['total_extraction'] = time.time() - start_total
            logger.info(f"✅ Total feature extraction completed in {timing_info['total_extraction']:.2f}s")
        
        logger.info(f"Extracted {len(features)} selected features: {feature_names}")
        return features, feature_names

    def _get_available_features(self) -> Dict[str, callable]:
        """Get dictionary of available feature extraction functions."""
        return {
            'intensity_db': self._extract_db_intensity,
            'intensity_percentile_rescaled': self._extract_percentile_rescaled,
            'intensity_linear': self._extract_linear_intensity,
            'intensity_sqrt_linear': self._extract_sqrt_linear,
            'intensity_power_transform': self._extract_power_transform,
            'intensity_lee_filtered': self._extract_lee_filtered,
            'local_mean_5x5': self._extract_local_mean_5x5,
            'local_std_5x5': self._extract_local_std_5x5,
            'sobel_magnitude': self._extract_sobel_magnitude,
            'sobel_direction': self._extract_sobel_direction,
            'texture_contrast': self._extract_texture_contrast,
            'texture_homogeneity': self._extract_texture_homogeneity,
            'local_mean_3x3': self._extract_local_mean_3x3,
            'local_mean_7x7': self._extract_local_mean_7x7,
            'gradient_magnitude': self._extract_gradient_magnitude,
            'incidence_angle': self._extract_incidence_angle_feature,
            'local_range': self._extract_local_range
        }
    
    # Individual feature extraction methods with timing
    @timing_decorator
    def _extract_db_intensity(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract original intensity values (dB)."""
        return intensity_db
    
    @timing_decorator
    def _extract_percentile_rescaled(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract percentile rescaled (2-98th percentile)."""
        return self._percentile_rescale(intensity_db)
    
    @timing_decorator
    def _extract_linear_intensity(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract linear intensity (10^(dB/10))."""
        return self._db_to_linear(intensity_db)
    
    @timing_decorator
    def _extract_sqrt_linear(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract square root of linear intensity."""
        linear_intensity = self._db_to_linear(intensity_db)
        return np.sqrt(np.maximum(linear_intensity, 0))
    
    @timing_decorator
    def _extract_power_transform(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract power transform (α=0.3)."""
        return self._power_transform(intensity_db, self.alpha_power)
    
    @timing_decorator
    def _extract_lee_filtered(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract speckle-filtered intensity (Lee filter)."""
        linear_intensity = self._db_to_linear(intensity_db)
        return self._lee_filter(linear_intensity)
    
    @timing_decorator
    def _extract_local_mean_5x5(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract local mean (5x5)."""
        return self._local_mean(intensity_db, window_size=5)
    
    @timing_decorator
    def _extract_local_std_5x5(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract local std (5x5)."""
        return self._local_std(intensity_db, window_size=5)
    
    @timing_decorator
    def _extract_sobel_magnitude(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract Sobel magnitude."""
        sobel_mag, _ = self._sobel_features(intensity_db)
        return sobel_mag
    
    @timing_decorator
    def _extract_sobel_direction(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract Sobel direction."""
        _, sobel_dir = self._sobel_features(intensity_db)
        return sobel_dir
    
    @timing_decorator
    def _extract_texture_contrast(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract GLCM contrast (or alternative)."""
        contrast, _ = self._glcm_features(intensity_db)
        return contrast
    
    @timing_decorator
    def _extract_texture_homogeneity(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract GLCM homogeneity (or alternative)."""
        _, homogeneity = self._glcm_features(intensity_db)
        return homogeneity
    
    @timing_decorator
    def _extract_local_mean_3x3(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract local mean (3x3)."""
        return self._local_mean(intensity_db, window_size=3)
    
    @timing_decorator
    def _extract_local_mean_7x7(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract local mean (7x7)."""
        return self._local_mean(intensity_db, window_size=7)
    
    @timing_decorator
    def _extract_gradient_magnitude(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract gradient magnitude."""
        return self._gradient_magnitude(intensity_db)
    
    @timing_decorator
    def _extract_incidence_angle_feature(self, intensity_db: np.ndarray, metadata: Optional[Dict] = None) -> np.ndarray:
        """Extract local incidence angle (if available)."""
        if metadata and self._has_incidence_angle(metadata):
            return self._extract_incidence_angle(metadata, intensity_db.shape)
        else:
            logger.warning("No incidence angle metadata available, using default value")
            return np.full(intensity_db.shape, 30.0, dtype=np.float32)
    
    @timing_decorator
    def _extract_local_range(self, intensity_db: np.ndarray) -> np.ndarray:
        """Extract local range."""
        return self._local_range(intensity_db, window_size=5)

    # ...existing code for helper methods...
    
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


# Public API Functions

def get_available_feature_names() -> List[str]:
    """
    Get list of all available feature names.
    
    Returns:
        List[str]: Available feature names that can be used with extract_selected_sar_features()
    """
    return [
        'intensity_db',
        'intensity_percentile_rescaled', 
        'intensity_linear',
        'intensity_sqrt_linear',
        'intensity_power_transform',
        'intensity_lee_filtered',
        'local_mean_5x5',
        'local_std_5x5',
        'sobel_magnitude',
        'sobel_direction',
        'texture_contrast',
        'texture_homogeneity',
        'local_mean_3x3',
        'local_mean_7x7',
        'gradient_magnitude',
        'incidence_angle',
        'local_range'
    ]


def extract_selected_sar_features(intensity_data: np.ndarray,
                                selected_features: List[str],
                                metadata: Optional[Dict] = None,
                                normalize: bool = False,
                                normalization_method: str = 'standard',
                                timing: bool = True) -> Union[Tuple[np.ndarray, List[str]], 
                                                             Tuple[np.ndarray, List[str], Dict]]:
    """
    Extract only selected SAR features from intensity data for efficiency.
    
    This function only computes the features that are actually needed,
    saving significant computation time compared to extracting all features.
    
    Args:
        intensity_data (np.ndarray): SAR intensity data in dB (2D array)
        selected_features (List[str]): List of feature names to extract
        metadata (Dict, optional): Metadata for feature extraction
        normalize (bool): Whether to normalize features
        normalization_method (str): Normalization method ('standard', 'minmax', 'robust')
        timing (bool): Whether to monitor and log timing information
    
    Returns:
        Tuple containing:
        - features (np.ndarray): Selected feature array (n_features, height, width)
        - feature_names (List[str]): Names of extracted features
    
    Raises:
        ValueError: If invalid feature names are provided
    """
    logger.info(f"🔄 Starting feature extraction for {len(selected_features)} features")
    
    extractor = SARFeatureExtractor()
    
    if timing:
        features, feature_names = extractor.extract_selected_features(
            intensity_data, selected_features, metadata, timing=True
        )
    else:
        features, feature_names = extractor.extract_selected_features(
            intensity_data, selected_features, metadata, timing=False
        )
    
    if normalize:
        if timing:
            norm_start = time.time()
        features, _ = extractor.normalize_features(features, method=normalization_method)
        if timing:
            timing_info['normalization'] = time.time() - norm_start
            logger.info(f"⏱️  Feature normalization took {timing_info['normalization']:.2f}s")
    
    # Log timing summary
    if timing:
        logger.info(f"📊 Feature extraction timing summary:")
        feature_times = [(k.replace('feature_', ''), v) for k, v in timing_info.items() 
                        if k.startswith('feature_') and not k.endswith('_failed')]
        feature_times.sort(key=lambda x: x[1], reverse=True)
        
        for feature_name, timing_val in feature_times:
            logger.info(f"   {feature_name}: {timing_val:.2f}s")
        logger.info(f"   ✅ Total: {timing_info.get('total_extraction', 0):.2f}s")
        
        return features, feature_names, timing_info
    
    return features, feature_names


def get_default_feature_set() -> List[str]:
    """
    Get a default set of features that provide good performance/quality balance.
    
    Returns:
        List[str]: Default feature names optimized for most SAR-to-height tasks
    """
    return [
        'intensity_db',
        'intensity_percentile_rescaled', 
        'intensity_linear',
        'intensity_sqrt_linear',
        'intensity_power_transform',
        'intensity_lee_filtered',
        'local_mean_5x5',
        'local_std_5x5'
    ]


def get_texture_feature_set() -> List[str]:
    """
    Get texture-focused feature set.
    
    Returns:
        List[str]: Feature names focused on texture analysis
    """
    return [
        'intensity_db',
        'local_mean_3x3',
        'local_mean_5x5',
        'local_mean_7x7',
        'local_std_5x5',
        'texture_contrast',
        'texture_homogeneity',
        'local_range'
    ]


def get_edge_feature_set() -> List[str]:
    """
    Get edge and gradient focused feature set.
    
    Returns:
        List[str]: Feature names focused on edge detection
    """
    return [
        'intensity_db',
        'sobel_magnitude',
        'sobel_direction', 
        'gradient_magnitude',
        'intensity_lee_filtered'
    ]


def get_minimal_feature_set() -> List[str]:
    """
    Get minimal feature set for fastest processing.
    
    Returns:
        List[str]: Minimal feature set for quick processing
    """
    return [
        'intensity_db',
        'intensity_linear',
        'local_mean_5x5',
        'local_std_5x5'
    ]


if __name__ == "__main__":
    # Example usage demonstrating the refactored approach
    print("🧪 Testing optimized feature extraction...")
    dummy_intensity = np.random.randn(100, 100) * 10 - 20
    
    # Get all available features
    all_features = get_available_feature_names()
    print(f"📋 Available features ({len(all_features)}): {all_features}")
    
    # Test with default feature set
    print("\n🔧 Testing default feature set...")
    default_features = get_default_feature_set()
    features, names, timing_info = extract_selected_sar_features(
        dummy_intensity, default_features, timing=True
    )
    print(f"✅ Extracted {features.shape[0]} default features in {timing_info['total_extraction']:.2f}s")
    
    # Test with all features (equivalent to old extract_all_features)
    print("\n🔧 Testing all features extraction...")
    all_features_data, all_names, all_timing = extract_selected_sar_features(
        dummy_intensity, all_features, timing=True
    )
    print(f"✅ Extracted ALL {all_features_data.shape[0]} features in {all_timing['total_extraction']:.2f}s")
    
    # Demonstrate feature set helpers
    print(f"\n📊 Feature set options:")
    print(f"   Default ({len(get_default_feature_set())}): {get_default_feature_set()}")
    print(f"   Texture ({len(get_texture_feature_set())}): {get_texture_feature_set()}")
    print(f"   Edge ({len(get_edge_feature_set())}): {get_edge_feature_set()}")
    print(f"   Minimal ({len(get_minimal_feature_set())}): {get_minimal_feature_set()}")