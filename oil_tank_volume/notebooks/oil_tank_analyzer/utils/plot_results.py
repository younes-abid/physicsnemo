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
        """Plot a single configuration result."""
        config = result['config']
        pipeline_result = result['result']
        score = result['score']
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Get pipeline stages
        pipeline_stages = pipeline_result.get('pipeline_stages', {})
        circles = pipeline_result.get('circles', [])
        identified = pipeline_result.get('identified_circles', {})
        quality_metrics = pipeline_result.get('quality_metrics', {})
        
        # 1. Original image with circles
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image, cmap='gray')
        
        # Plot all circles
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            circle_plot = plt.Circle(center, radius, color='yellow', fill=False, 
                                   linewidth=1, alpha=0.5, linestyle='--')
            ax1.add_patch(circle_plot)
        
        # Plot identified components
        for circle_type, circle in identified.items():
            base_type = circle_type.split('_')[0]
            color = self.colors.get(base_type, 'yellow')
            center = circle['center']
            radius = circle['radius']
            
            circle_plot = plt.Circle(center, radius, color=color, fill=False, 
                                   linewidth=3, alpha=0.9)
            ax1.add_patch(circle_plot)
            
            ax1.text(center[0], center[1], base_type.upper(), 
                    color=color, fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax1.set_title(f'Rank {rank+1}/{total} - Score: {score:.3f}\n'
                     f'Circles: {len(circles)}, Identified: {len(identified)}')
        ax1.axis('off')
        
        # 2. Denoised image
        ax2 = fig.add_subplot(gs[0, 1])
        denoised = pipeline_stages.get('denoised', original_image)
        ax2.imshow(denoised, cmap='gray')
        ax2.set_title(f'Denoised: {config.denoising.method}')
        ax2.axis('off')
        
        # 3. Edge detection
        ax3 = fig.add_subplot(gs[0, 2])
        edges = pipeline_stages.get('edges', np.zeros_like(original_image))
        ax3.imshow(edges, cmap='gray')
        ax3.set_title(f'Edges: {config.contour_detection.method}')
        ax3.axis('off')
        
        # 4. Method configuration
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        config_text = "METHOD CONFIGURATION:\n\n"
        config_text += f"Denoising: {config.denoising.method}\n"
        config_text += f"Params: {config.denoising.params}\n\n"
        config_text += f"Edge Detection: {config.contour_detection.method}\n"
        config_text += f"Params: {config.contour_detection.params}\n\n"
        config_text += f"Circle Detection: {config.circle_detection.method}\n"
        config_text += f"Params: {config.circle_detection.params}"
        
        ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightgray"),
                fontfamily='monospace')
        
        # 5. Quality metrics
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.axis('off')
        
        if quality_metrics:
            metrics_text = "QUALITY METRICS:\n\n"
            for metric, value in quality_metrics.items():
                metrics_text += f"{metric}: {value:.3f}\n"
        else:
            metrics_text = "No quality metrics available"
        
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightblue"),
                fontfamily='monospace')
        
        # 6. Component analysis
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.axis('off')
        
        tank_analysis = pipeline_result.get('tank_analysis', {})
        if 'component_analysis' in tank_analysis:
            comp_text = "COMPONENT ANALYSIS:\n\n"
            for comp, analysis in tank_analysis['component_analysis'].items():
                confidence = analysis.get('confidence', 0)
                comp_text += f"{comp}: {confidence:.3f}\n"
        else:
            comp_text = "No component analysis"
        
        ax6.text(0.05, 0.95, comp_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightgreen"),
                fontfamily='monospace')
        
        # 7. Performance metrics
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.axis('off')
        
        performance = pipeline_result.get('performance_metrics', {})
        if performance:
            perf_text = "PERFORMANCE:\n\n"
            for metric, value in performance.items():
                if 'time' in metric:
                    perf_text += f"{metric}: {value:.3f}s\n"
        else:
            perf_text = "No performance data"
        
        ax7.text(0.05, 0.95, perf_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightyellow"),
                fontfamily='monospace')
        
        # 8. Score breakdown
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        
        score_text = "SCORE BREAKDOWN:\n\n"
        score_text += f"Total Score: {score:.3f}\n"
        score_text += f"Circles Found: {len(circles)}\n"
        score_text += f"Components ID: {len(identified)}\n"
        score_text += f"Detection Conf: {tank_analysis.get('detection_confidence', 0):.3f}"
        
        ax8.text(0.05, 0.95, score_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lightcoral"),
                fontfamily='monospace')
        
        # 9. SAR intensity analysis
        ax9 = fig.add_subplot(gs[2, :])
        if identified and 'denoised' in pipeline_stages:
            denoised = pipeline_stages['denoised']
            ax9.imshow(denoised, cmap='gray', alpha=0.7)
            
            for circle_type, circle in identified.items():
                if any(base_type in circle_type for base_type in ['bottom', 'top', 'floating_roof']):
                    center = circle['center']
                    radius = circle['radius']
                    
                    # Intensity analysis
                    mask = np.zeros_like(denoised, dtype=np.uint8)
                    cv2.circle(mask, center, radius, 255, -1)
                    mean_intensity = cv2.mean(denoised, mask=mask)[0]
                    
                    # Color by intensity
                    if mean_intensity < 85:
                        color = 'red'
                        label = "DARK"
                    elif mean_intensity < 170:
                        color = 'yellow'
                        label = "MEDIUM"
                    else:
                        color = 'cyan'
                        label = "BRIGHT"
                    
                    circle_plot = plt.Circle(center, radius, color=color, fill=False, 
                                           linewidth=2, linestyle=':')
                    ax9.add_patch(circle_plot)
                    ax9.text(center[0], center[1] - radius - 5, f"{label} ({mean_intensity:.1f})",
                            color=color, fontsize=8, ha='center', weight='bold')
            
            ax9.set_title('SAR Intensity Analysis (Dark=Bottom, Medium=Floating, Bright=Top)')
        else:
            ax9.text(0.5, 0.5, "No components for intensity analysis", 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('SAR Intensity Analysis')
        
        ax9.axis('off')
        
        plt.tight_layout()
    
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