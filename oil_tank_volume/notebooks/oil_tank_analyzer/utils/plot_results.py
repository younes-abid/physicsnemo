import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any
import cv2

class GridSearchPlotter:
    """Class for visualizing grid search results."""
    
    def __init__(self):
        self.colors = {
            'bottom': 'red',
            'top': 'blue', 
            'floating_roof': 'green',
            'bottom_candidate': 'darkred',
            'top_candidate': 'darkblue',
            'floating_roof_candidate': 'darkgreen'
        }
    
    def plot_top_configurations(self, results: List[Dict], original_image: np.ndarray, 
                              top_k: int = 10, save_path: str = None):
        """Plot the top K configurations from grid search."""
        top_results = results[:top_k]
        
        for i, result in enumerate(top_results):
            self._plot_single_configuration(result, original_image, i, top_k)
            
            if save_path:
                plt.savefig(f"{save_path}/config_rank_{i+1}.png", dpi=150, bbox_inches='tight')
            
            plt.show()
    
    def _plot_single_configuration(self, result: Dict, original_image: np.ndarray, 
                                 rank: int, total: int):
        """Plot a single configuration result with enhanced 4-row layout."""
        config = result['config']
        pipeline_result = result['result']
        score = result['score']
        
        # Store pipeline result for access in drawing methods
        self._current_pipeline_result = pipeline_result
        
        # Create figure with 4 rows, 4 columns
        fig = plt.figure(figsize=(24, 22))
        gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.8], hspace=0.35, wspace=0.25)
        
        # Get pipeline stages and data
        pipeline_stages = pipeline_result.get('pipeline_stages', {})
        circles = pipeline_result.get('circles', [])
        identified = pipeline_result.get('identified_circles', {})
        quality_metrics = pipeline_result.get('quality_metrics', {})
        performance = pipeline_result.get('performance_metrics', {})
        filtered_circles = pipeline_result.get('filtered_circles', [])  # NEW: Get filtered circles
        
        # Get images for each stage
        original = original_image
        preprocessed = pipeline_stages.get('preprocessed', original_image)
        denoised = pipeline_stages.get('denoised', original_image)
        edges = pipeline_stages.get('edges', np.zeros_like(original_image))
        
        # === FIRST ROW: Processing stages without circles ===
        stages = [
            (original, 'Original', 'Raw SAR Image'),
            (preprocessed, 'Preprocessed', config.preprocessing.method if hasattr(config, 'preprocessing') else 'N/A'),
            (denoised, 'Denoised', config.denoising.method),
            (edges, 'Edges', config.contour_detection.method)
        ]
        
        timing_info = [
            'Input',
            f"{performance.get('preprocessing_time', 0):.3f}s",
            f"{performance.get('denoising_time', 0):.3f}s",
            f"{performance.get('edge_detection_time', 0):.3f}s"
        ]
        
        for i, ((image, stage_name, method), timing) in enumerate(zip(stages, timing_info)):
            ax = fig.add_subplot(gs[0, i])
            
            if stage_name == 'Edges':
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image, cmap='gray')
            
            # Add intensity range info
            img_min, img_max = image.min(), image.max()
            range_text = f"Range: [{img_min}, {img_max}]"
            
            ax.set_title(f'{stage_name}\n{method}\n{timing}\n{range_text}', 
                        fontsize=11, pad=10)
            ax.axis('off')
        
        # === SECOND ROW: Same images but with ONLY identified circles overlaid ===
        for i, (image, stage_name, method) in enumerate(stages):
            ax = fig.add_subplot(gs[1, i])
            
            if stage_name == 'Edges':
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image, cmap='gray')
            
            # Only draw identified circles (no yellow background circles)
            self._draw_identified_circles_only(ax, identified, stage_name)
            
            circles_text = f"Circles: {len(circles)}"
            identified_text = f"Identified: {len(identified)}"
            ax.set_title(f'{stage_name} + Identified\n{circles_text} | {identified_text}', 
                        fontsize=11, pad=10)
            ax.axis('off')
        
        # === THIRD ROW: All circles analysis - UPDATED ===
        for i, (image, stage_name, method) in enumerate(stages):
            ax = fig.add_subplot(gs[2, i])
            
            if stage_name == 'Edges':
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image, cmap='gray')
            
            # Draw ALL circles with selection analysis (now includes filtered circles)
            self._draw_all_circles_analysis(ax, circles, identified, stage_name)
            
            # Add circle analysis info - UPDATED to include filtered circles
            total_detected = len(circles) + len(filtered_circles)
            selected_circles = len(identified)
            unselected_circles = len(circles) - selected_circles
            filtered_count = len(filtered_circles)
            
            ax.set_title(f'{stage_name} - All Circles\nDetected: {total_detected} | Kept: {len(circles)} | Selected: {selected_circles} | Filtered: {filtered_count}', 
                        fontsize=10, pad=10)
            ax.axis('off')
        
        # === FOURTH ROW: Method details and performance ===
        
        # 1. Method Configuration
        ax_config = fig.add_subplot(gs[3, 0])
        ax_config.axis('off')
        
        config_text = "METHOD CONFIGURATION:\n\n"
        if hasattr(config, 'preprocessing'):
            config_text += f"[PREPROCESS]\n{config.preprocessing.method}\n"
            if config.preprocessing.params:
                config_text += f"Params: {self._format_params(config.preprocessing.params)}\n\n"
            else:
                config_text += "\n"
        
        config_text += f"[DENOISE]\n{config.denoising.method}\n"
        if config.denoising.params:
            config_text += f"Params: {self._format_params(config.denoising.params)}\n\n"
        else:
            config_text += "\n"
        
        config_text += f"[EDGES]\n{config.contour_detection.method}\n"
        if config.contour_detection.params:
            config_text += f"Params: {self._format_params(config.contour_detection.params)}\n\n"
        else:
            config_text += "\n"
        
        config_text += f"[CIRCLES]\n{config.circle_detection.method}\n"
        if config.circle_detection.params:
            config_text += f"Params: {self._format_params(config.circle_detection.params)}"
        
        ax_config.text(0.05, 0.95, config_text, transform=ax_config.transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
                      fontfamily='monospace')
        
        # 2. Performance Metrics
        ax_perf = fig.add_subplot(gs[3, 1])
        ax_perf.axis('off')
        
        perf_text = "PERFORMANCE METRICS:\n\n"
        if performance:
            total_time = performance.get('total_processing_time', 0)
            perf_text += f"Total Time: {total_time:.3f}s\n\n"
            
            perf_text += f"Preprocessing: {performance.get('preprocessing_time', 0):.3f}s\n"
            perf_text += f"Denoising: {performance.get('denoising_time', 0):.3f}s\n"
            perf_text += f"Edge Detection: {performance.get('edge_detection_time', 0):.3f}s\n"
            perf_text += f"Circle Detection: {performance.get('circle_detection_time', 0):.3f}s\n\n"
            
            # Calculate percentages
            if total_time > 0:
                preproc_pct = (performance.get('preprocessing_time', 0) / total_time) * 100
                denoise_pct = (performance.get('denoising_time', 0) / total_time) * 100
                edge_pct = (performance.get('edge_detection_time', 0) / total_time) * 100
                circle_pct = (performance.get('circle_detection_time', 0) / total_time) * 100
                
                perf_text += "Time Distribution:\n"
                perf_text += f"  Preproc: {preproc_pct:.1f}%\n"
                perf_text += f"  Denoise: {denoise_pct:.1f}%\n"
                perf_text += f"  Edges: {edge_pct:.1f}%\n"
                perf_text += f"  Circles: {circle_pct:.1f}%"
        else:
            perf_text += "No performance data available"
        
        ax_perf.text(0.05, 0.95, perf_text, transform=ax_perf.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
                    fontfamily='monospace')
        
        # 3. Quality Metrics and Score
        ax_quality = fig.add_subplot(gs[3, 2])
        ax_quality.axis('off')
        
        quality_text = f"SCORE & QUALITY:\n\n"
        quality_text += f"Final Score: {score:.3f}\n"
        quality_text += f"Rank: {rank+1}/{total}\n\n"
        
        if quality_metrics:
            quality_text += "Quality Breakdown:\n"
            for metric, value in quality_metrics.items():
                quality_text += f"  {metric}: {value:.3f}\n"
        
        # Tank analysis
        tank_analysis = pipeline_result.get('tank_analysis', {})
        if tank_analysis:
            quality_text += f"\nTank Analysis:\n"
            quality_text += f"Detection Confidence: {tank_analysis.get('detection_confidence', 0):.3f}\n"
            quality_text += f"Total Circles: {tank_analysis.get('total_circles', 0)}\n"
            
            if 'component_analysis' in tank_analysis:
                quality_text += "\nComponent Confidence:\n"
                for comp, analysis in tank_analysis['component_analysis'].items():
                    conf = analysis.get('confidence', 0)
                    quality_text += f"  {comp}: {conf:.3f}\n"
        
        ax_quality.text(0.05, 0.95, quality_text, transform=ax_quality.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                       fontfamily='monospace')
        
        # 4. Circle Selection Analysis - UPDATED with filtered circle info
        ax_selection = fig.add_subplot(gs[3, 3])
        ax_selection.axis('off')
        
        selection_text = "CIRCLE SELECTION ANALYSIS:\n\n"
        
        total_detected = len(circles) + len(filtered_circles)
        if total_detected > 0:
            selection_text += f"Total Detected: {total_detected}\n"
            selection_text += f"Kept After Filter: {len(circles)}\n"
            selection_text += f"Selected: {len(identified)}\n"
            selection_text += f"Unselected: {len(circles) - len(identified)}\n"
            selection_text += f"Filtered Out: {len(filtered_circles)}\n\n"
            
            # Show filtered circles by reason
            if filtered_circles:
                filter_reasons = {}
                for circle in filtered_circles:
                    reason = circle.get('filter_reason', 'unknown')
                    if reason not in filter_reasons:
                        filter_reasons[reason] = []
                    filter_reasons[reason].append(circle)
                
                selection_text += "Filtered Circles by Reason:\n"
                for reason, circles_list in filter_reasons.items():
                    selection_text += f"  {reason}: {len(circles_list)} circles\n"
                    for circle in circles_list[:2]:  # Show first 2
                        center = circle['center']
                        radius = circle['radius']
                        selection_text += f"    R={radius}px at ({center[0]}, {center[1]})\n"
                selection_text += "\n"
            
            # Analyze circle qualities
            selection_text += "Circle Quality Analysis:\n"
            
            # Get identified circle centers for comparison
            identified_centers = []
            for circle_type, circle in identified.items():
                identified_centers.append(circle['center'])
            
            # Analyze all circles and categorize them
            selected_circles = []
            unselected_circles = []
            
            for circle in circles:
                center = circle['center']
                radius = circle['radius']
                is_selected = any(
                    abs(center[0] - id_center[0]) < 5 and abs(center[1] - id_center[1]) < 5 
                    for id_center in identified_centers
                )
                
                circle_info = {
                    'center': center,
                    'radius': radius,
                    'quality_score': circle.get('quality_score', 0),
                    'area': np.pi * radius * radius
                }
                
                if is_selected:
                    selected_circles.append(circle_info)
                else:
                    unselected_circles.append(circle_info)
            
            # Show top unselected circles
            if unselected_circles:
                unselected_circles.sort(key=lambda x: x['radius'], reverse=True)
                selection_text += "\nTop Unselected Circles:\n"
                for i, circle in enumerate(unselected_circles[:3]):
                    selection_text += f"  {i+1}. R={circle['radius']:.1f}px, "
                    selection_text += f"Center=({circle['center'][0]}, {circle['center'][1]})\n"
            
            # Show selection quality
            if selected_circles:
                avg_selected_radius = np.mean([c['radius'] for c in selected_circles])
                selection_text += f"\nSelected Avg Radius: {avg_selected_radius:.1f}px\n"
            
            if unselected_circles:
                avg_unselected_radius = np.mean([c['radius'] for c in unselected_circles])
                selection_text += f"Unselected Avg Radius: {avg_unselected_radius:.1f}px\n"
            
        else:
            selection_text += "No circles detected\n"
        
        # Add updated legend for colors
        selection_text += "\nLegend:\n"
        selection_text += "RED/GREEN/BLUE: Selected (identified)\n"
        selection_text += "PINK: Unselected (kept but not selected)\n"
        selection_text += "ORANGE: Filtered (too small)\n"
        selection_text += "CYAN: Filtered (too large)\n"
        selection_text += "PURPLE: Filtered (off-center)\n"
        selection_text += "YELLOW: All kept circles (faint)"
        
        ax_selection.text(0.05, 0.95, selection_text, transform=ax_selection.transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                         fontfamily='monospace')
        
        # Add overall title with proper spacing after it
        fig.suptitle(f'Oil Tank Analysis Pipeline - Configuration Rank {rank+1}/{total} (Score: {score:.3f})', 
                    fontsize=16, fontweight='bold', y=1)  
        
        # Use subplots_adjust with proper margins
        plt.subplots_adjust(top=0.92, bottom=0.04, left=0.05, right=0.95, hspace=0.4, wspace=0.25)
    
    def _draw_identified_circles_only(self, ax, identified, stage_name):
        """Draw only identified circles (no yellow background circles)."""
        # Draw identified components with specific colors only
        for circle_type, circle in identified.items():
            base_type = circle_type.split('_')[0]
            color = self.colors.get(base_type, 'yellow')
            center = circle['center']
            radius = circle['radius']
            
            circle_plot = plt.Circle(center, radius, color=color, fill=False, 
                                   linewidth=2, alpha=0.9)
            ax.add_patch(circle_plot)
            
            # Add label only for non-edge images to avoid clutter
            if stage_name != 'Edges':
                ax.text(center[0], center[1], base_type[0].upper(), 
                       color=color, fontsize=8, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    def _draw_circles_on_axis(self, ax, circles, identified, stage_name):
        """Draw circles on a given axis (DEPRECATED - keeping for compatibility)."""
        # This method is now only used by old code, new code uses specific methods
        self._draw_identified_circles_only(ax, identified, stage_name)
    
    def _draw_all_circles_analysis(self, ax, circles, identified, stage_name):
        """Draw all circles with selection analysis highlighting."""
        # Get pipeline result data for filtered circles
        pipeline_result = getattr(self, '_current_pipeline_result', {})
        filtered_circles = pipeline_result.get('filtered_circles', [])
        
        # Get identified circle centers for comparison
        identified_centers = []
        for circle_type, circle in identified.items():
            identified_centers.append(circle['center'])
        
        # Draw all kept circles first (faint yellow)
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            circle_plot = plt.Circle(center, radius, color='yellow', fill=False, 
                                   linewidth=1, alpha=0.3, linestyle=':')
            ax.add_patch(circle_plot)
        
        # Draw filtered out circles in orange/cyan
        for circle in filtered_circles:
            center = circle['center']
            radius = circle['radius']
            reason = circle.get('filter_reason', 'unknown')
            
            # Color by filter reason
            if 'too small' in reason:
                color = 'orange'
                label = 'S'  # Small
            elif 'too large' in reason:
                color = 'cyan'
                label = 'L'  # Large
            elif 'off-center' in reason:
                color = 'purple'
                label = 'C'  # Center
            else:
                color = 'gray'
                label = 'F'  # Filtered
            
            # Draw filtered circles with dotted lines
            circle_plot = plt.Circle(center, radius, color=color, fill=False, 
                                   linewidth=1.5, alpha=0.7, linestyle=':')
            ax.add_patch(circle_plot)
            
            # Add small label for filtered circles
            if stage_name != 'Edges':
                ax.text(center[0], center[1], label, 
                       color=color, fontsize=5, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6))
        
        # Draw unselected circles in pink (from kept circles)
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            
            # Check if this circle is selected (identified)
            is_selected = any(
                abs(center[0] - id_center[0]) < 5 and abs(center[1] - id_center[1]) < 5 
                for id_center in identified_centers
            )
            
            if not is_selected:
                # Draw unselected circles in bright pink
                circle_plot = plt.Circle(center, radius, color='hotpink', fill=False, 
                                       linewidth=2, alpha=0.8, linestyle='--')
                ax.add_patch(circle_plot)
                
                # Add small text label for unselected circles
                if stage_name != 'Edges':
                    ax.text(center[0], center[1], 'U', 
                           color='hotpink', fontsize=6, ha='center', va='center', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        
        # Draw identified components with specific colors (on top)
        for circle_type, circle in identified.items():
            base_type = circle_type.split('_')[0]
            color = self.colors.get(base_type, 'yellow')
            center = circle['center']
            radius = circle['radius']
            
            # Draw selected circles with thick borders
            circle_plot = plt.Circle(center, radius, color=color, fill=False, 
                                   linewidth=3, alpha=1.0)
            ax.add_patch(circle_plot)
            
            # Add label for selected circles
            if stage_name != 'Edges':
                ax.text(center[0], center[1], base_type[0].upper(), 
                       color=color, fontsize=10, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    def _format_params(self, params_dict, max_items=3):
        """Format parameters dictionary for display."""
        if not params_dict:
            return "None"
        
        items = list(params_dict.items())[:max_items]
        formatted = []
        for key, value in items:
            if isinstance(value, float):
                formatted.append(f"{key}={value:.3f}")
            elif isinstance(value, list) and len(value) > 3:
                formatted.append(f"{key}=[{len(value)} items]")
            else:
                formatted.append(f"{key}={value}")
        
        result = ", ".join(formatted)
        if len(params_dict) > max_items:
            result += f" (+{len(params_dict) - max_items} more)"
        
        return result
    
    def _get_metric_emoji(self, metric_name):
        """Get appropriate text prefix for metric (no emojis)."""
        prefix_map = {
            'circle_count_score': 'CIRCLES',
            'component_completeness': 'COMPLETE',
            'spatial_coherence': 'SPATIAL',
            'size_consistency': 'SIZE',
            'overall_quality': 'OVERALL'
        }
        return prefix_map.get(metric_name, 'METRIC')
    
    def _get_intensity_label(self, intensity):
        """Get intensity label based on value."""
        if intensity < 85:
            return "Dark"
        elif intensity < 170:
            return "Medium"
        else:
            return "Bright"
    
    def plot_score_distribution(self, results: List[Dict], save_path: str = None):
        """Plot distribution of scores from grid search."""
        scores = [r['score'] for r in results]
        methods = [(r['config'].denoising.method, 
                   r['config'].contour_detection.method,
                   r['config'].circle_detection.method) for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Score distribution
        ax1.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Pipeline Scores')
        ax1.grid(True, alpha=0.3)
        
        # Top methods
        top_indices = np.argsort(scores)[-10:]