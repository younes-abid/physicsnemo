import geopandas as gpd
from pathlib import Path
from oil_tank_analyzer.utils.geospatial import TankProcessor
from oil_tank_analyzer.core.types import ProcessedTank
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


import os
import json
from pathlib import Path

class AnnotationEnumerator:
    """Class to enumerate annotations in GeoJSON files."""
    def enumerate_annotations(self, input_geojson_path, output_geojson_path):
        """
        Enumerate annotations in a GeoJSON file by adding a unique number to each feature.

        Args:
            input_geojson_path (str): Path to the input GeoJSON file.
            output_geojson_path (str): Path to save the enumerated GeoJSON file.
        """
        # Load the GeoJSON file
        with open(input_geojson_path, 'r') as f:
            geojson_data = json.load(f)

        # Enumerate features
        for idx, feature in enumerate(geojson_data.get("features", []), start=1):
            feature["properties"]["annotation_id"] = idx

        # Save the updated GeoJSON
        output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=4)

        print(f"Enumerated GeoJSON saved to: {output_geojson_path}")

    def process_all_geojsons_recursive(self, input_dir, output_dir, suffix=""):
        """
        Recursively process all GeoJSON files in a directory, enumerating annotations and saving them.

        Args:
            input_dir (str): Directory containing input GeoJSON files.
            output_dir (str): Directory to save the enumerated GeoJSON files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        for geojson_file in input_dir.rglob("*.geojson"):  # Recursively find all .geojson files
            relative_path = geojson_file.relative_to(input_dir)
            output_file = output_dir / relative_path.parent / f"{geojson_file.stem}{suffix}.geojson"
            self.enumerate_annotations(geojson_file, output_file)


def load_annotations(geojson_path: Path, selected_ids: Optional[List[int]] = None) -> gpd.GeoDataFrame:
    """Load and filter annotations."""
    annotations = gpd.read_file(geojson_path)
    
    if selected_ids is not None:
        annotations = annotations[annotations["annotation_id"].isin(selected_ids)]
    
    print(f"Loaded {len(annotations)} annotations")
    return annotations


def convert_to_oriented_bounding_boxes(annotations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert annotations to oriented bounding boxes."""
    obb_annotations = annotations.copy()
    obb_annotations["geometry"] = obb_annotations["geometry"].apply(
        lambda geom: geom.minimum_rotated_rectangle
    )
    return obb_annotations


def process_tank_batch(raster_path: Path, annotation_path: Path, 
                      selected_ids: Optional[List[int]] = None,
                      verbose: bool = True) -> List[ProcessedTank]:
    """
    Process a batch of oil tanks.
    
    Args:
        raster_path: Path to raster file
        annotation_path: Path to GeoJSON annotations
        selected_ids: List of annotation IDs to process (None for all)
        verbose: Whether to print progress information
    
    Returns:
        List of processed tanks
    """
    # Load annotations
    annotations = load_annotations(annotation_path, selected_ids)
    obb_annotations = convert_to_oriented_bounding_boxes(annotations)
    
    # Initialize processor
    processor = TankProcessor(verbose=verbose)
    
    print(f"Processing {len(obb_annotations)} oil tanks...")
    
    # Process each tank
    processed_tanks = []
    for _, annotation in obb_annotations.iterrows():
        tank = processor.process_single_tank(
            str(raster_path),
            annotation["geometry"],
            annotation["annotation_id"]
        )
        processed_tanks.append(tank)
    
    return processed_tanks


def create_results_dataframe(processed_tanks: List[ProcessedTank]) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from processed tanks."""
    records = []
    for tank in processed_tanks:
        records.append({
            'annotation_id': tank.annotation_id,
            'geometry': tank.geometry,
            'height_m': tank.measurements.height_m,
            'diameter_m': tank.measurements.diameter_m,
            'rotation_angle': tank.rotation_angle,
            'vertical_direction': tank.vertical_direction,
            'aspect_ratio': tank.measurements.aspect_ratio,
            'pixel_resolution': tank.pixel_resolution,
            'measurement_type': tank.measurements.measurement_type
        })
    
    return gpd.GeoDataFrame(records, geometry='geometry')


def display_processing_results(processed_tanks: List[ProcessedTank], max_display: int = 10):
    """Display processing results."""
    print("\n" + "="*50)
    print("PROCESSING RESULTS SUMMARY")
    print("="*50)
    
    for i, tank in enumerate(processed_tanks[:max_display]):
        print(f"{i+1}. {tank}")
    
    if len(processed_tanks) > max_display:
        print(f"... and {len(processed_tanks) - max_display} more tanks")


def visualize_tank_comparison(processed_tanks: List[ProcessedTank], max_display: int = 6):
    """Visualize original vs rotated crops."""
    n_tanks = min(len(processed_tanks), max_display)
    
    if n_tanks == 0:
        print("No tanks to visualize")
        return
    
    fig, axes = plt.subplots(n_tanks, 2, figsize=(12, 4 * n_tanks))
    
    if n_tanks == 1:
        axes = [axes]
    
    for idx, tank in enumerate(processed_tanks[:n_tanks]):
        # Original crop
        if tank.non_rotated_crop.shape[0] == 1:
            axes[idx][0].imshow(tank.non_rotated_crop[0], cmap="gray")
        else:
            axes[idx][0].imshow(np.transpose(tank.non_rotated_crop, (1, 2, 0)))
        
        axes[idx][0].set_title(
            f"Tank {tank.annotation_id} - Original\n"
            f"Shape: {tank.non_rotated_crop.shape}"
        )
        axes[idx][0].axis('off')
        
        # Rotated crop
        if tank.rotated_crop.shape[0] == 1:
            axes[idx][1].imshow(tank.rotated_crop[0], cmap="gray")
        else:
            axes[idx][1].imshow(np.transpose(tank.rotated_crop, (1, 2, 0)))
        
        axes[idx][1].set_title(
            f"Tank {tank.annotation_id} - Rotated\n"
            f"Shape: {tank.rotated_crop.shape}\n"
            f"Angle: {tank.rotation_angle:.1f}°"
        )
        axes[idx][1].axis('off')
    
    plt.tight_layout()
    plt.show()

