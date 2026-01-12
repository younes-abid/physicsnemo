import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import time
from scipy.stats import skew, kurtosis
import os
from glob import glob
from netCDF4 import Dataset
import shutil

def analyze_rain_rate(file_path, sample_indices=None):
    """
    Analyze and visualize Rain_rate variable
    
    Parameters:
    -----------
    file_path : str
        Path to the netCDF file
    sample_indices : list, optional
        List of sample indices to visualize (default: [0, 100, 200])
    """
    
    if sample_indices is None:
        sample_indices = [0, 100, 200]
    
    print(f"Opening file: {file_path}")
    
    # Open the output group containing Rain_rate
    with xr.open_dataset(file_path, group="output") as output_ds:
        rain_rate = output_ds['Rain_rate']
        
        print(f"Rain_rate data shape: {rain_rate.shape}")
        print(f"Rain_rate data type: {rain_rate.dtype}")
        #print(f"Dimensions: {dict(rain_rate.dims)}")
        
        # Basic statistics
        print("\n=== Rain Rate Statistics ===")
        print(f"Min value: {float(rain_rate.min()):.6f}")
        print(f"Max value: {float(rain_rate.max()):.6f}")
        print(f"Mean value: {float(rain_rate.mean()):.6f}")
        print(f"Std deviation: {float(rain_rate.std()):.6f}")
        print(f"Median value: {float(rain_rate.median()):.6f}")
        
        # Percentiles
        print(f"\nPercentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = float(rain_rate.quantile(p/100))
            print(f"  {p}th percentile: {val:.6f}")
        
        # Count non-zero values
        non_zero_count = (rain_rate > 0).sum()
        total_count = rain_rate.size
        non_zero_percentage = (non_zero_count / total_count) * 100
        print(f"\nNon-zero values: {non_zero_count} / {total_count} ({non_zero_percentage:.2f}%)")
        
        # Create visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Sample maps
        n_samples = min(len(sample_indices), rain_rate.shape[0])
        actual_indices = sample_indices[:n_samples]
        
        for i, idx in enumerate(actual_indices):
            if idx < rain_rate.shape[0]:
                ax = plt.subplot(3, 3, i+1)
                im = ax.imshow(rain_rate[idx], cmap='Blues', origin='lower')
                ax.set_title(f'Rain Rate - Sample {idx}')
                ax.set_xlabel('X (grid points)')
                ax.set_ylabel('Y (grid points)')
                plt.colorbar(im, ax=ax, label='Rain Rate')
        
        # Plot 2: Overall statistics histogram
        ax = plt.subplot(3, 3, 4)
        rain_data_flat = rain_rate.values.flatten()
        
        # Remove zeros for better histogram visualization
        rain_nonzero = rain_data_flat[rain_data_flat > 0]
        if len(rain_nonzero) > 0:
            ax.hist(rain_nonzero, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Rain Rate')
            ax.set_ylabel('Frequency')
            ax.set_title('Rain Rate Distribution (Non-zero values)')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No non-zero rain values', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title('Rain Rate Distribution')
        
        # Plot 3: Log-scale histogram for better visualization
        ax = plt.subplot(3, 3, 5)
        if len(rain_nonzero) > 0:
            ax.hist(np.log10(rain_nonzero + 1e-10), bins=50, alpha=0.7, 
                   color='lightgreen', edgecolor='black')
            ax.set_xlabel('Log10(Rain Rate)')
            ax.set_ylabel('Frequency')
            ax.set_title('Log-scale Rain Rate Distribution')
        
        # Plot 4: Time series of spatial mean
        ax = plt.subplot(3, 3, 6)
        spatial_mean = rain_rate.mean(dim=['y_hr', 'x_hr'])
        ax.plot(spatial_mean, marker='o', markersize=2, alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Spatial Mean Rain Rate')
        ax.set_title('Rain Rate Time Series (Spatial Mean)')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Spatial mean map
        ax = plt.subplot(3, 3, 7)
        temporal_mean = rain_rate.mean(dim='sample')
        im = ax.imshow(temporal_mean, cmap='Blues', origin='lower')
        ax.set_title('Temporal Mean Rain Rate')
        ax.set_xlabel('X (grid points)')
        ax.set_ylabel('Y (grid points)')
        plt.colorbar(im, ax=ax, label='Mean Rain Rate')
        
        # Plot 6: Spatial standard deviation
        ax = plt.subplot(3, 3, 8)
        temporal_std = rain_rate.std(dim='sample')
        im = ax.imshow(temporal_std, cmap='Reds', origin='lower')
        ax.set_title('Temporal Std Rain Rate')
        ax.set_xlabel('X (grid points)')
        ax.set_ylabel('Y (grid points)')
        plt.colorbar(im, ax=ax, label='Std Rain Rate')
        
        # Plot 7: Box plot of rain rate by sample (every 10th sample)
        ax = plt.subplot(3, 3, 9)
        if rain_rate.shape[0] > 10:
            sample_subset = rain_rate[::max(1, rain_rate.shape[0]//10)]
            rain_subset = [sample_subset[i].values.flatten() for i in range(sample_subset.shape[0])]
            ax.boxplot(rain_subset, showfliers=False)
            ax.set_xlabel('Sample Index (subset)')
            ax.set_ylabel('Rain Rate')
            ax.set_title('Rain Rate Distribution by Sample')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('rain_rate_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as 'rain_rate_analysis.png'")
        plt.show()
        
        return rain_rate
    
    
def preprocess_rain_rate(rain_rate, method='asinh', alpha=0.25, threshold=0.1):
    """Preprocess rain rate data for better training"""
    data = rain_rate.values  # Convert to numpy array
    
    if method == 'log_normal':
        # Traditional log transformation with epsilon for zeros
        return np.log(data + 1e-6)
    elif method == 'asinh':
        # Inverse hyperbolic sine - handles zeros naturally
        return np.arcsinh(data)
    elif method == 'sqrt':
        # Square root transformation
        return np.sqrt(data)
    elif method == 'log1p':
        # log(1 + x) - NumPy optimized, handles zeros naturally
        return np.log1p(data)
    elif method == 'power_transform':
        # Tunable power transformation: x^alpha
        return np.power(data + 1e-8, alpha)
    elif method == 'hybrid_log':
        # Hybrid: asinh for small values, log1p for large values
        return np.where(data < threshold, 
                       np.arcsinh(data), 
                       np.log1p(data - threshold) + np.arcsinh(threshold))
    elif method == 'quantile_uniform':
        # Quantile normalization to uniform distribution
        flat_data = data.flatten()
        sorted_indices = np.argsort(flat_data)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(flat_data))
        uniform = ranks / (len(flat_data) - 1)
        return uniform.reshape(data.shape)
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_metrics(original, processed, processed_nonzero):
    """Compute comprehensive metrics for preprocessing comparison"""
    metrics = {}
    
    # Basic statistics
    metrics['range'] = (processed.min(), processed.max())
    metrics['mean'] = processed.mean()
    metrics['std'] = processed.std()
    
    # Distribution shape metrics
    metrics['skewness'] = skew(processed.flatten())
    metrics['kurtosis'] = kurtosis(processed.flatten())
    
    # Spread metrics
    if len(processed_nonzero) > 0:
        # Coefficient of variation (lower is more concentrated)
        metrics['cv_nonzero'] = processed_nonzero.std() / (abs(processed_nonzero.mean()) + 1e-8)
        # Range spread for non-zero values
        metrics['range_spread'] = processed_nonzero.max() - processed_nonzero.min()
        # Inter-quartile range
        metrics['iqr'] = np.percentile(processed_nonzero, 75) - np.percentile(processed_nonzero, 25)
    else:
        metrics['cv_nonzero'] = 0
        metrics['range_spread'] = 0
        metrics['iqr'] = 0
    
    # Normality test (closer to 0 skew and 3 kurtosis = more normal)
    metrics['normality_score'] = abs(metrics['skewness']) + abs(metrics['kurtosis'] - 3)
    
    return metrics

def compare_preprocessing_methods(rain_rate_data, alpha=0.25, threshold=0.1):
    """Compare preprocessing methods with comprehensive analysis"""
    
    # Methods to compare (removed yeo_johnson for speed)
    methods = ['asinh', 'log_normal', 'log1p', 'sqrt', 'power_transform', 'hybrid_log', 'quantile_uniform']
    
    original = rain_rate_data.values.flatten()
    original_nonzero = original[original > 0]
    
    print("=== PREPROCESSING COMPARISON ===")
    print(f"Original data: {len(original_nonzero):,}/{len(original):,} non-zero ({100*len(original_nonzero)/len(original):.1f}%)")
    print(f"Original range: [{original.min():.6f}, {original.max():.6f}]")
    print(f"Parameters: alpha={alpha}, threshold={threshold}")
    print("-" * 80)
    
    results = {}
    timing_results = {}
    
    for method in methods:
        print(f"Processing {method.upper()}...")
        
        # Time the preprocessing
        start_time = time.time()
        try:
            processed = preprocess_rain_rate(rain_rate_data, method=method, alpha=alpha, threshold=threshold)
            processing_time = time.time() - start_time
            timing_results[method] = processing_time
            
            processed_flat = processed.flatten()
            processed_nonzero = processed_flat[original > 0]
            
            # Compute comprehensive metrics
            metrics = compute_metrics(original, processed_flat, processed_nonzero)
            results[method] = {
                'processed': processed,
                'metrics': metrics,
                'time': processing_time
            }
            
            # Print results
            print(f"  Time: {processing_time:.4f}s")
            print(f"  Range: [{metrics['range'][0]:.3f}, {metrics['range'][1]:.3f}]")
            print(f"  Mean: {metrics['mean']:.3f}, Std: {metrics['std']:.3f}")
            print(f"  Skewness: {metrics['skewness']:.3f}, Kurtosis: {metrics['kurtosis']:.3f}")
            print(f"  CV (non-zero): {metrics['cv_nonzero']:.3f}")
            print(f"  Range spread: {metrics['range_spread']:.3f}")
            print(f"  Normality score: {metrics['normality_score']:.3f} (lower=better)")
            print()
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results[method] = None
            timing_results[method] = float('inf')
            print()
    
    # Create visualization
    valid_methods = [m for m in methods if results[m] is not None]
    n_methods = len(valid_methods)
    
    if n_methods == 0:
        print("No valid methods to plot!")
        return results
    
    fig, axes = plt.subplots(3, n_methods, figsize=(4*n_methods, 12))
    if n_methods == 1:
        axes = axes.reshape(3, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(valid_methods):
        result = results[method]
        processed = result['processed']
        processed_flat = processed.flatten()
        processed_nonzero = processed_flat[original > 0]
        
        # Row 1: Histogram of all processed data
        axes[0, i].hist(processed_flat, bins=50, alpha=0.7, density=True, color=colors[i])
        axes[0, i].set_title(f'{method.upper()}\nTime: {result["time"]:.3f}s')
        axes[0, i].set_xlabel('Processed Values')
        axes[0, i].set_ylabel('Density')
        axes[0, i].grid(True, alpha=0.3)
        
        # Row 2: Histogram of non-zero processed data
        if len(processed_nonzero) > 0:
            axes[1, i].hist(processed_nonzero, bins=50, alpha=0.7, density=True, color=colors[i])
            axes[1, i].set_title(f'Non-zero Only\nCV: {result["metrics"]["cv_nonzero"]:.2f}')
        else:
            axes[1, i].text(0.5, 0.5, 'No non-zero values', ha='center', va='center')
        axes[1, i].set_xlabel('Processed Values (Non-zero)')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
        
        # Row 3: Sample visualization
        sample_data = processed[0] if processed.ndim > 2 else processed
        im = axes[2, i].imshow(sample_data, cmap='viridis', origin='lower')
        axes[2, i].set_title(f'Sample\nNorm Score: {result["metrics"]["normality_score"]:.2f}')
        plt.colorbar(im, ax=axes[2, i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('rain_rate_preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Rank methods by different criteria
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Speed ranking (faster is better)
        speed_ranking = sorted(valid_results.keys(), key=lambda x: timing_results[x])
        
        # Normality ranking (lower normality_score is better)
        normality_ranking = sorted(valid_results.keys(), 
                                 key=lambda x: valid_results[x]['metrics']['normality_score'])
        
        # Spread ranking (higher CV for non-zero values might be better for diffusion)
        spread_ranking = sorted(valid_results.keys(), 
                              key=lambda x: valid_results[x]['metrics']['cv_nonzero'], reverse=True)
        
        print(f"📊 SPEED RANKING (fastest first): {' > '.join(speed_ranking)}")
        print(f"📈 NORMALITY RANKING (most normal first): {' > '.join(normality_ranking)}")
        print(f"🎯 SPREAD RANKING (highest spread first): {' > '.join(spread_ranking)}")
        
        print(f"\n🏆 TOP RECOMMENDATION: {normality_ranking[0].upper()}")
    
    return results

def process_and_save_rain_rate_variables(data_dir, alpha_pt=0.05):
    """
    Process Rain_rate variable and add preprocessed versions to NetCDF files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing NetCDF files with Rain_rate data
    alpha_pt : float
        Alpha parameter for power transform (default: 0.05)
    """
    
    print(f"Processing Rain_rate variables in directory: {data_dir}")
    print(f"Using power transform alpha: {alpha_pt}")
    print("-" * 80)
    
    # Get all NetCDF files in the directory
    nc_files = sorted(glob(os.path.join(data_dir, "*.nc")))
    
    if not nc_files:
        print("No NetCDF files found in the directory!")
        return
    
    print(f"Found {len(nc_files)} NetCDF files to process")
    
    for i, file_path in enumerate(nc_files):
        print(f"\nProcessing file {i+1}/{len(nc_files)}: {os.path.basename(file_path)}")
        
        try:
            # Check if processed variables already exist
            with xr.open_dataset(file_path, group="output") as ds:
                existing_vars = list(ds.data_vars.keys())
                
                if 'Rain_rate_PT' in existing_vars and 'Rain_rate_LN' in existing_vars:
                    print(f"  ✓ Preprocessed variables already exist, skipping...")
                    continue
                
                # Load Rain_rate data
                rain_rate = ds['Rain_rate']
                print(f"  Original Rain_rate shape: {rain_rate.shape}")
                print(f"  Original Rain_rate range: [{float(rain_rate.min()):.6f}, {float(rain_rate.max()):.6f}]")
                
                # Apply power transform (x^alpha)
                rain_rate_pt = np.power(rain_rate.values + 1e-8, alpha_pt)
                print(f"  Power Transform range: [{rain_rate_pt.min():.6f}, {rain_rate_pt.max():.6f}]")
                
                # Apply log normal (log(x + epsilon))
                rain_rate_ln = np.log(rain_rate.values + 1e-6)
                print(f"  Log Normal range: [{rain_rate_ln.min():.6f}, {rain_rate_ln.max():.6f}]")
            
            # Create backup of original file
            # backup_path = file_path + ".backup"
            # if not os.path.exists(backup_path):
            #     print(f"  Creating backup: {os.path.basename(backup_path)}")
            #     shutil.copy2(file_path, backup_path)
            
            # Open file in append mode and add new variables
            print(f"  Adding preprocessed variables to file...")
            
            with Dataset(file_path, 'a') as nc_file:
                # Access the output group
                output_group = nc_file.groups['output']
                
                # Create Rain_rate_PT variable
                if 'Rain_rate_PT' not in output_group.variables:
                    var_pt = output_group.createVariable(
                        'Rain_rate_PT', 'f4', 
                        ('sample', 'y_hr', 'x_hr'),
                        compression='zlib', complevel=4
                    )
                    var_pt.long_name = f"Rain rate power transform (alpha={alpha_pt})"
                    var_pt.units = "transformed"
                    var_pt.method = "power_transform"
                    var_pt.alpha = alpha_pt
                    var_pt[:] = rain_rate_pt
                    print(f"    ✓ Added Rain_rate_PT")
                
                # Create Rain_rate_LN variable
                if 'Rain_rate_LN' not in output_group.variables:
                    var_ln = output_group.createVariable(
                        'Rain_rate_LN', 'f4', 
                        ('sample', 'y_hr', 'x_hr'),
                        compression='zlib', complevel=4
                    )
                    var_ln.long_name = "Rain rate log normal transform"
                    var_ln.units = "transformed"
                    var_ln.method = "log_normal"
                    var_ln.epsilon = 1e-6
                    var_ln[:] = rain_rate_ln
                    print(f"    ✓ Added Rain_rate_LN")
            
            print(f"  ✅ Successfully processed {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(file_path)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def verify_processed_variables(data_dir):
    """
    Verify that processed variables have been added to all files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed NetCDF files
    """
    
    print(f"Verifying processed variables in: {data_dir}")
    print("-" * 50)
    
    nc_files = sorted(glob(os.path.join(data_dir, "*.nc")))
    
    for file_path in nc_files:
        if file_path.endswith('.backup'):
            continue
            
        filename = os.path.basename(file_path)
        
        try:
            with xr.open_dataset(file_path, group="output") as ds:
                vars_present = []
                if 'Rain_rate' in ds.data_vars:
                    vars_present.append('Rain_rate')
                if 'Rain_rate_PT' in ds.data_vars:
                    vars_present.append('Rain_rate_PT')
                if 'Rain_rate_LN' in ds.data_vars:
                    vars_present.append('Rain_rate_LN')
                
                status = "✅" if len(vars_present) == 3 else "⚠️"
                print(f"{status} {filename}: {', '.join(vars_present)}")
                
                if 'Rain_rate_PT' in ds.data_vars:
                    pt_data = ds['Rain_rate_PT']
                    print(f"    Rain_rate_PT range: [{float(pt_data.min()):.3f}, {float(pt_data.max()):.3f}]")
                
                if 'Rain_rate_LN' in ds.data_vars:
                    ln_data = ds['Rain_rate_LN']
                    print(f"    Rain_rate_LN range: [{float(ln_data.min()):.3f}, {float(ln_data.max()):.3f}]")
                    
        except Exception as e:
            print(f"❌ {filename}: Error - {str(e)}")

def get_preprocessing_stats(data_dir, sample_file_idx=0):
    """
    Get statistics for all rain rate preprocessing methods from a sample file
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed NetCDF files
    sample_file_idx : int
        Index of file to sample for statistics (default: 0)
    """
    
    nc_files = sorted([f for f in glob(os.path.join(data_dir, "*.nc")) if not f.endswith('.backup')])
    
    if not nc_files:
        print("No NetCDF files found!")
        return
        
    sample_file = nc_files[sample_file_idx]
    print(f"Getting preprocessing statistics from: {os.path.basename(sample_file)}")
    print("-" * 60)
    
    with xr.open_dataset(sample_file, group="output") as ds:
        if 'Rain_rate' not in ds.data_vars:
            print("Rain_rate variable not found!")
            return
            
        # Get original data
        rain_rate = ds['Rain_rate'].isel(sample=slice(0, 1000))  # Sample for speed
        original = rain_rate.values.flatten()
        original_nonzero = original[original > 0]
        
        print(f"Original Rain_rate:")
        print(f"  Non-zero: {len(original_nonzero):,}/{len(original):,} ({100*len(original_nonzero)/len(original):.1f}%)")
        print(f"  Range: [{original.min():.6f}, {original.max():.6f}]")
        print(f"  Mean: {original.mean():.6f}, Std: {original.std():.6f}")
        
        # Check processed variables
        if 'Rain_rate_PT' in ds.data_vars:
            pt_data = ds['Rain_rate_PT'].isel(sample=slice(0, 1000)).values.flatten()
            print(f"\nRain_rate_PT (Power Transform):")
            print(f"  Range: [{pt_data.min():.6f}, {pt_data.max():.6f}]")
            print(f"  Mean: {pt_data.mean():.6f}, Std: {pt_data.std():.6f}")
            
        if 'Rain_rate_LN' in ds.data_vars:
            ln_data = ds['Rain_rate_LN'].isel(sample=slice(0, 1000)).values.flatten()
            print(f"\nRain_rate_LN (Log Normal):")
            print(f"  Range: [{ln_data.min():.6f}, {ln_data.max():.6f}]")
            print(f"  Mean: {ln_data.mean():.6f}, Std: {ln_data.std():.6f}")



