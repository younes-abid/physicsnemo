import cv2
import numpy as np
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from oil_tank_analyzer.core.base_processor import BaseProcessor, CircleDetectorRegistry

class BaseCircleDetector(BaseProcessor):
    """Base class for circle detection algorithms optimized for oil tanks."""
    
    def process(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        raw_circles = self._detect_circles(image, edges, **kwargs)
        filtered_circles = self._filter_oil_tank_circles(raw_circles, image.shape, **kwargs)
        
        # Store both raw and filtered circles for analysis
        # Add metadata to help with visualization
        for circle in filtered_circles:
            circle['filtered'] = False
        
        # Mark filtered out circles
        filtered_out = []
        for raw_circle in raw_circles:
            is_kept = any(
                abs(raw_circle['center'][0] - kept['center'][0]) < 3 and 
                abs(raw_circle['center'][1] - kept['center'][1]) < 3 
                for kept in filtered_circles
            )
            if not is_kept:
                raw_circle['filtered'] = True
                raw_circle['filter_reason'] = self._get_filter_reason(raw_circle, image.shape, **kwargs)
                filtered_out.append(raw_circle)
        
        # Add debug info
        debug = kwargs.get('debug', True)
        if debug and filtered_out:
            print(f"  Filtered out {len(filtered_out)} circles:")
            for i, circle in enumerate(filtered_out):
                reason = circle.get('filter_reason', 'unknown')
                center = circle['center']
                radius = circle['radius']
                print(f"    {i+1}. R={radius}px at ({center[0]}, {center[1]}) - {reason}")
        
        # Store all circles (kept + filtered) for visualization
        all_circles_with_status = filtered_circles + filtered_out
        
        return filtered_circles
    
    def _get_filter_reason(self, circle: Dict, image_shape: Tuple[int, int], **kwargs) -> str:
        """Determine why a circle was filtered out."""
        height, width = image_shape
        x, y = circle['center']
        radius = circle['radius']
        
        # Check each filter condition
        center_tolerance = kwargs.get('center_tolerance', 0.2)
        min_diameter_ratio = kwargs.get('min_diameter_ratio', 0.6)
        max_diameter_ratio = kwargs.get('max_diameter_ratio', 1.1)
        
        center_x = width / 2
        allowed_x_range = (center_x - width * center_tolerance, 
                          center_x + width * center_tolerance)
        
        min_radius = (width * min_diameter_ratio) / 2
        max_radius = (width * max_diameter_ratio) / 2
        
        # Check position filter
        if not (allowed_x_range[0] <= x <= allowed_x_range[1]):
            return f"off-center (x={x}, allowed: {allowed_x_range[0]:.1f}-{allowed_x_range[1]:.1f})"
        
        # Check size filter
        if radius < min_radius:
            return f"too small (R={radius}, min={min_radius:.1f})"
        elif radius > max_radius:
            return f"too large (R={radius}, max={max_radius:.1f})"
        
        # If it passed position and size filters, it was filtered by quality/max_circles limit
        return f"low quality score (excluded by max_circles={kwargs.get('max_circles', 5)} limit)"
    
    @abstractmethod
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        pass
    
    def _filter_oil_tank_circles(self, circles: List[Dict], image_shape: Tuple[int, int], 
                                **kwargs) -> List[Dict]:
        """Specialized filtering for oil tank circles."""
        height, width = image_shape
        
        # Tank-specific filtering parameters
        center_tolerance = kwargs.get('center_tolerance', 0.2)  # 20% of width from center
        min_diameter_ratio = kwargs.get('min_diameter_ratio', 0.6)  # At least 60% of image width
        max_diameter_ratio = kwargs.get('max_diameter_ratio', 1.1)  # Up to 110% of image width
        max_circles = kwargs.get('max_circles', 5)  # Maximum circles to return, -1 means no limit
        
        center_x = width / 2
        allowed_x_range = (center_x - width * center_tolerance, 
                          center_x + width * center_tolerance)
        
        min_radius = (width * min_diameter_ratio) / 2
        max_radius = (width * max_diameter_ratio) / 2
        
        filtered = []
        for circle in circles:
            x, y = circle['center']
            radius = circle['radius']
            
            # Filter by center position (must be near vertical center line)
            if not (allowed_x_range[0] <= x <= allowed_x_range[1]):
                continue
                
            # Filter by size (must be close to full tank width)
            if not (min_radius <= radius <= max_radius):
                continue
            
            # Additional quality metrics for oil tanks
            circle['quality_score'] = self._calculate_circle_quality(circle, image_shape)
            filtered.append(circle)
        
        # Sort by quality score and limit number only if max_circles > 0
        filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Apply max_circles limit only if it's positive
        if max_circles > 0:
            return filtered[:max_circles]
        else:
            # No limit when max_circles is -1 or 0
            return filtered
    
    def _calculate_circle_quality(self, circle: Dict, image_shape: Tuple[int, int]) -> float:
        """Calculate quality score for oil tank circles."""
        height, width = image_shape
        center_x, center_y = circle['center']
        radius = circle['radius']
        
        score = 0.0
        
        # High score for being near vertical center
        center_distance = abs(center_x - width / 2) / (width / 2)
        score += (1.0 - center_distance) * 0.4
        
        # High score for appropriate size (close to image width)
        ideal_radius = width / 2
        size_ratio = min(radius / ideal_radius, ideal_radius / radius)
        score += size_ratio * 0.3
        
        # Additional points for circularity if available
        if 'circularity' in circle:
            score += circle['circularity'] * 0.3
        
        return score

@CircleDetectorRegistry.register("hough_tank")
class HoughTankCircleDetector(BaseCircleDetector):
    """Hough circle detector optimized for oil tanks."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        height, width = image.shape
        
        # Tank-optimized parameters
        dp = kwargs.get('dp', 1.1)  # Higher resolution
        min_dist = kwargs.get('min_dist', int(height * 0.8))  # Large distance for vertical separation
        param1 = kwargs.get('param1', 30)  # Lower for faint edges
        param2 = kwargs.get('param2', 25)  # Lower for partial circles
        min_radius = kwargs.get('min_radius', int(width * 0.25))  # At least 50% of width
        max_radius = kwargs.get('max_radius', int(width * 0.6))   # Up to 120% of width
        
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append({
                    'center': (x, y),
                    'radius': r,
                    'method': 'hough_tank'
                })
        
        return detected_circles

@CircleDetectorRegistry.register("ellipse_tank")
class EllipseTankDetector(BaseCircleDetector):
    """Ellipse detector for handling sensor angle distortions."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        min_area = kwargs.get('min_area', 1000)
        max_area = kwargs.get('max_area', 10000)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_ellipses = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
                
            # Fit ellipse
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (major_axis, minor_axis), angle = ellipse
                    
                    # For oil tanks, we expect near-vertical ellipses
                    # Major axis should be close to horizontal due to sensor angle
                    is_horizontal = (abs(angle) < 30 or abs(angle - 180) < 30)
                    
                    if is_horizontal:
                        # Use average of major/minor as equivalent radius
                        equivalent_radius = (major_axis + minor_axis) / 4
                        
                        # Calculate ellipse circularity
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            detected_ellipses.append({
                                'center': (int(x), int(y)),
                                'radius': int(equivalent_radius),
                                'major_axis': major_axis,
                                'minor_axis': minor_axis,
                                'angle': angle,
                                'circularity': circularity,
                                'area': area,
                                'ellipse_ratio': major_axis / minor_axis,
                                'method': 'ellipse'
                            })
                except:
                    continue
        
        return detected_ellipses

@CircleDetectorRegistry.register("contour_tank")
class ContourTankCircleDetector(BaseCircleDetector):
    """Contour-based detector optimized for oil tank features."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        # Use both external and internal contours to catch rings
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = kwargs.get('min_area', 500)
        max_area = kwargs.get('max_area', 15000)
        min_circularity = kwargs.get('min_circularity', 0.5)  # Lower for ellipses
        
        detected_circles = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            
            # Get enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check if this is likely a ring (has child contour)
            is_ring = False
            if hierarchy is not None and hierarchy[0][i][2] != -1:
                is_ring = True
            
            detected_circles.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'circularity': circularity,
                'area': area,
                'is_ring': is_ring,
                'method': 'contour_tank'
            })
        
        return detected_circles

@CircleDetectorRegistry.register("multi_scale_tank")
class MultiScaleTankDetector(BaseCircleDetector):
    """Multi-scale detector for handling different circle types in oil tanks."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        height, width = image.shape
        debug = kwargs.get('debug', True)  # Enable debugging by default
        
        # Detect at multiple scales for different circle types
        scales = [
            {'min_radius': int(width * 0.4), 'max_radius': int(width * 0.6), 'name': 'large'},
            {'min_radius': int(width * 0.3), 'max_radius': int(width * 0.5), 'name': 'medium'},
            {'min_radius': int(width * 0.2), 'max_radius': int(width * 0.4), 'name': 'small'},
        ]
        
        all_circles = []
        
        if debug:
            print(f"  Multi-scale detection on {width}x{height} image:")
        
        for scale in scales:
            # Hough transform for this scale
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT, dp=1.2, 
                minDist=int(height * 0.5),
                param1=30, param2=20,
                minRadius=scale['min_radius'],
                maxRadius=scale['max_radius']
            )
            
            scale_count = 0
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                scale_count = len(circles)
                for (x, y, r) in circles:
                    all_circles.append({
                        'center': (x, y),
                        'radius': r,
                        'scale': scale['name'],
                        'method': 'multi_scale'
                    })
            
            if debug:
                print(f"    {scale['name']} scale (R: {scale['min_radius']}-{scale['max_radius']}): {scale_count} circles")
        
        if debug:
            print(f"  Total before filtering: {len(all_circles)} circles")
        
        return all_circles

@CircleDetectorRegistry.register("template_tank")
class TemplateTankDetector(BaseCircleDetector):
    """Template-based detector using known oil tank structure."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        height, width = image.shape
        
        # We expect 3 circles: bottom, floating roof, top
        expected_circles = 3
        vertical_spacing = height / (expected_circles + 1)
        
        # Search regions for each expected circle
        search_regions = []
        for i in range(expected_circles):
            y_center = vertical_spacing * (i + 1)
            search_regions.append({
                'y_min': max(0, int(y_center - vertical_spacing * 0.3)),
                'y_max': min(height, int(y_center + vertical_spacing * 0.3)),
                'type': ['bottom', 'floating_roof', 'top'][i]
            })
        
        # Find circles in each region
        all_circles = []
        
        for region in search_regions:
            # Extract region from edges
            region_edges = edges[region['y_min']:region['y_max'], :]
            
            if np.any(region_edges > 0):
                # Find contours in this region
                contours, _ = cv2.findContours(region_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:
                        continue
                    
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    # Adjust y to full image coordinates
                    y_full = y + region['y_min']
                    
                    all_circles.append({
                        'center': (int(x), int(y_full)),
                        'radius': int(radius),
                        'region_type': region['type'],
                        'method': 'template'
                    })
        
        return all_circles

@CircleDetectorRegistry.register("intensity_tank")
class IntensityTankDetector(BaseCircleDetector):
    """Intensity-based detector for dark bottom circles and bright rings."""
    
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        height, width = image.shape
        
        # Analyze intensity profiles
        intensity_threshold = kwargs.get('intensity_threshold', 0.3)
        
        # Create intensity-based edges
        _, thresh_dark = cv2.threshold(image, int(255 * intensity_threshold), 255, cv2.THRESH_BINARY_INV)
        _, thresh_bright = cv2.threshold(image, int(255 * (1 - intensity_threshold)), 255, cv2.THRESH_BINARY)
        
        # Combine dark and bright regions
        intensity_edges = cv2.bitwise_or(thresh_dark, thresh_bright)
        
        # Find contours in intensity-based edges
        contours, _ = cv2.findContours(intensity_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
                
            # Check if contour is dark or bright
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_intensity = cv2.mean(image, mask=mask)[0]
            
            contour_type = 'dark' if mean_intensity < 128 else 'bright'
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            detected_circles.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'area': area,
                'intensity_type': contour_type,
                'mean_intensity': mean_intensity,
                'method': 'intensity'
            })
        
        return detected_circles

@CircleDetectorRegistry.register("blob_detector")
class BlobCircleDetector(BaseCircleDetector):
    def _detect_circles(self, image: np.ndarray, edges: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        detected_circles = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = int(centroids[i, 0])
            y = int(centroids[i, 1])
            radius = int(np.sqrt(area / np.pi))
            
            detected_circles.append({
                'center': (x, y),
                'radius': radius,
                'area': area,
                'method': 'blob'
            })
        
        return detected_circles