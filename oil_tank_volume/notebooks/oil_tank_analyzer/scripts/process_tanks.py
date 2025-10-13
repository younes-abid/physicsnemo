#!/usr/bin/env python3
"""
Script for processing oil tank annotations from raster data.
"""
from oil_tank_analyzer.utils.annotation_processor import process_tank_batch, create_results_dataframe, visualize_tank_comparison, display_processing_results

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    DATA_PATH = Path("/app/data/oil_tank_volume/oiltank")
    RAW_DATA_PATH = DATA_PATH / "raw_data"
    PROCESSED_DATA_PATH = DATA_PATH / "processed_data"
    
    selected_image = "ICEYE_ARCHIVE_SLH_184358_20211201T092638"
    selected_ids = [130, 131, 132, 133]
    
    raster_path = RAW_DATA_PATH / selected_image / f"{selected_image}.tif"
    annotation_path = PROCESSED_DATA_PATH / selected_image / f"{selected_image}.geojson"
    
    # Process tanks
    processed_tanks = process_tank_batch(
        raster_path=raster_path,
        annotation_path=annotation_path,
        selected_ids=selected_ids,
        verbose=True
    )
    
    # Create results dataframe
    results_df = create_results_dataframe(processed_tanks)
    
    # Display results
    display_processing_results(processed_tanks)
    visualize_tank_comparison(processed_tanks)
    
    # Save results
    output_path = "oil_tanks_processed_results.geojson"
    results_df.to_file(output_path, driver='GeoJSON')
    print(f"\nSaved results to {output_path}")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(results_df[['annotation_id', 'height_m', 'diameter_m', 'vertical_direction', 'measurement_type']])