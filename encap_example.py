import pystac_client
import planetary_computer
import odc.stac
import odc.geo
import xarray as xr
from shapely.geometry import Polygon
import shapely
from typing import Dict, Any, List, Tuple, Optional, Union
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import scipy.ndimage as ndimage

def process_sentinel1_sar_complete(
    geometry: shapely.geometry.Polygon,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    analysis_type: str = "water_flood",  # Options: "water_only", "flood_only", "water_flood"
    measurements: List[str] = ["vv", "vh"],
    resolution: int = 10,
    platform: Optional[str] = None,
    orbit_direction: Optional[str] = None,
    water_threshold: float = -2.0,
    flood_change_threshold: float = -7.0,
    persistence_threshold: float = 0.8,
    filter_size: int = 5,
    use_log10: bool = True,
    save_outputs: bool = True,
    output_prefix: str = "sentinel1_analysis",
    show_plots: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete Sentinel-1 SAR data processing function for water mapping and flood detection.
    This is a single comprehensive function that handles the entire workflow without nested functions.
    
    Args:
        geometry: AOI polygon for data extraction
        start_date: Start date for data query
        end_date: End date for data query
        analysis_type: Type of analysis ("water_only", "flood_only", "water_flood")
        measurements: List of SAR bands to retrieve
        resolution: Spatial resolution in meters
        platform: Specific platform ('sentinel-1a', 'sentinel-1b', or None)
        orbit_direction: Orbit direction ('ascending', 'descending', or None)
        water_threshold: Threshold for water detection (log10 space if use_log10=True)
        flood_change_threshold: Threshold for flood detection (change in backscatter)
        persistence_threshold: Fraction of time for persistent water mapping
        filter_size: Size of speckle filter window
        use_log10: Whether to apply log10 transformation for thresholds
        save_outputs: Whether to save visualization outputs
        output_prefix: Prefix for saved files
        show_plots: Whether to display plots
        verbose: Whether to print processing updates
        
    Returns:
        Dict containing processed dataset, water masks, flood results, and statistics
    """
    
    # Initialize results dictionary
    results = {
        'dataset': None,
        'water_masks': {},
        'flood_results': {},
        'statistics': {}
    }
    
    try:
        # =================================================================
        # STEP 1: LOAD SENTINEL-1 DATA
        # =================================================================
        if verbose:
            print("Loading Sentinel-1 data...")
        
        # Get STAC catalog
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        # Prepare query parameters
        datetime_range = f"{start_date.isoformat()}Z/{end_date.isoformat()}Z"
        query_params = {}
        if platform:
            query_params["platform"] = {"eq": platform}
        if orbit_direction:
            query_params["sat:orbit_state"] = {"eq": orbit_direction}
        query_params["sar:instrument_mode"] = {"eq": "IW"}
        query_params["sar:product_type"] = {"eq": "GRD"}
        
        # Search for scenes
        search = client.search(
            collections=["sentinel-1-rtc"],
            intersects=geometry,
            datetime=datetime_range,
            query=query_params,
        )
        
        items = search.item_collection()
        if len(items) == 0:
            raise ValueError("No matching scenes found. Try adjusting date range or geometry.")
        
        if verbose:
            print(f"Found {len(items)} Sentinel-1 scenes")
        
        # Load data using odc.stac
        dataset = odc.stac.load(
            items=items,
            bands=measurements,
            geopolygon=geometry,
            groupby="solar_day",
            resolution=resolution,
            fail_on_error=False,
        )
        
        # Store metadata from STAC items
        metadata_list = []
        for item in items:
            metadata = {
                "platform": item.properties.get("platform", "Unknown"),
                "orbit_state": item.properties.get("sat:orbit_state", "Unknown"),
                "datetime": item.properties.get("datetime", "Unknown"),
                "orbit_number": item.properties.get("sat:absolute_orbit", "Unknown"),
                "relative_orbit": item.properties.get("sat:relative_orbit", "Unknown"),
                "instrument_mode": item.properties.get("sar:instrument_mode", "Unknown"),
                "product_type": item.properties.get("sar:product_type", "Unknown"),
            }
            metadata_list.append(metadata)
        
        dataset.attrs['metadata'] = metadata_list
        dataset = dataset.compute()
        
        if verbose:
            print(f"Dataset loaded with {len(dataset.time)} time steps")
            print(f"Available variables: {list(dataset.data_vars)}")
        
        # =================================================================
        # STEP 2: SELECT BEST ORBIT GROUP FOR CONSISTENCY
        # =================================================================
        if verbose:
            print("Selecting optimal orbit group...")
        
        if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
            metadata_list = dataset.attrs['metadata']
            n_times = len(dataset.time)
            n_metadata = len(metadata_list)
            max_index = min(n_times, n_metadata)
            
            # Try different grouping strategies to find best orbit consistency
            strategies = [
                ("platform_orbit_relative", lambda m: f"{m.get('platform', 'unknown')}_{m.get('orbit_state', 'unknown')}_{m.get('relative_orbit', 'unknown')}"),
                ("orbit_relative", lambda m: f"{m.get('orbit_state', 'unknown')}_{m.get('relative_orbit', 'unknown')}"),
                ("relative_only", lambda m: f"rel_{m.get('relative_orbit', 'unknown')}")
            ]
            
            best_result = None
            best_count = 0
            best_strategy = None
            
            for strategy_name, key_func in strategies:
                grouped_indices = defaultdict(list)
                
                for i in range(max_index):
                    try:
                        meta = metadata_list[i]
                        key = key_func(meta)
                        grouped_indices[key].append(i)
                    except Exception:
                        continue
                
                # Find best group for this strategy
                for key, indices in grouped_indices.items():
                    if len(indices) > best_count and len(indices) > 1:
                        best_result = (key, indices)
                        best_count = len(indices)
                        best_strategy = strategy_name
            
            if best_result is not None:
                key, indices = best_result
                if verbose:
                    print(f"Selected orbit group '{key}' with {len(indices)} acquisitions using {best_strategy}")
                
                dataset = dataset.isel(time=indices)
                subset_metadata = [metadata_list[i] for i in indices if i < n_metadata]
                dataset.attrs['metadata'] = subset_metadata
            else:
                if verbose:
                    print("No suitable orbit groups found, using original dataset")
        else:
            if verbose:
                print("No metadata available for orbit grouping, using original dataset")
        
        if len(dataset.time) == 0:
            raise ValueError("No valid time steps available after orbit selection")
        
        # =================================================================
        # STEP 3: APPLY SPECKLE FILTERING
        # =================================================================
        if verbose:
            print(f"Applying speckle filter (size={filter_size}) to reduce noise...")
        
        # Ensure filter size is odd
        if filter_size % 2 == 0:
            filter_size += 1
        
        # Fill null values with zeros
        dataset_filled = dataset.where(~dataset.isnull(), 0)
        
        # Apply speckle filtering to each measurement band
        for band in measurements:
            if band not in dataset:
                continue
                
            try:
                filtered_band_name = f"filtered_{band}"
                if verbose:
                    print(f"  Filtering {band} -> {filtered_band_name}")
                
                # Convert to power, apply filter, convert back to dB
                filtered_data = []
                for i in range(len(dataset.time)):
                    # Get time slice and convert dB to power
                    time_slice = dataset_filled[band].isel(time=i)
                    power_data = 10**(time_slice.values/10)
                    
                    # Apply uniform filter
                    filtered_power = ndimage.uniform_filter(power_data, size=filter_size)
                    
                    # Convert back to dB
                    filtered_db = 10 * np.log10(np.maximum(filtered_power, 1e-10))
                    
                    # Create DataArray with same coordinates
                    filtered_slice = xr.DataArray(
                        filtered_db,
                        coords=time_slice.coords,
                        dims=time_slice.dims,
                        attrs=time_slice.attrs
                    )
                    filtered_data.append(filtered_slice)
                
                # Combine time slices
                if len(filtered_data) > 1:
                    filtered_stack = xr.concat(filtered_data, dim='time')
                    filtered_stack = filtered_stack.assign_coords(time=dataset.time)
                else:
                    filtered_stack = filtered_data[0].expand_dims(dim='time')
                    filtered_stack = filtered_stack.assign_coords(time=dataset.time)
                
                dataset[filtered_band_name] = filtered_stack
                
            except Exception as e:
                if verbose:
                    print(f"  Error filtering {band}: {e}")
        
        results['dataset'] = dataset
        
        # =================================================================
        # STEP 4: WATER ANALYSIS
        # =================================================================
        if analysis_type in ["water_only", "water_flood"]:
            if verbose:
                print("Performing water detection analysis...")
            
            # Determine best band for water detection (prefer filtered if available)
            water_band = 'filtered_vh' if 'filtered_vh' in dataset else 'vh'
            if water_band not in dataset:
                # Fallback to any available band
                available_bands = list(dataset.data_vars)
                water_band = available_bands[0] if available_bands else None
                
            if water_band is None:
                raise ValueError("No suitable bands available for water detection")
            
            if verbose:
                print(f"Using {water_band} for water detection with threshold {water_threshold}")
            
            # Create water masks for each time step
            for i in range(len(dataset.time)):
                date_str = np.datetime_as_string(dataset.time.values[i], unit='D')
                
                # Get data for this time step
                data = dataset.isel(time=i)[water_band]
                
                # Apply water detection threshold
                if use_log10:
                    # Safe log10 transformation
                    data_values = data.values
                    data_values = np.where((data_values <= 0) | (~np.isfinite(data_values)), 1e-10, data_values)
                    data_log = np.log10(data_values)
                    data_log = np.where(~np.isfinite(data_log), -10, data_log)
                    water_mask = data_log < water_threshold
                else:
                    data_values = np.where(~np.isfinite(data.values), 0, data.values)
                    water_mask = data_values < water_threshold
                
                # Store water mask
                mask_key = f'water_mask_{i}_{date_str}'
                results['water_masks'][mask_key] = water_mask
                
                # Calculate water area statistics
                water_pixels = np.sum(water_mask)
                water_area_sq_km = water_pixels * (resolution * resolution) / 1_000_000
                results['statistics'][f'water_area_km2_{i}'] = water_area_sq_km
                
                if verbose:
                    print(f"  {date_str}: {water_area_sq_km:.2f} sq km of water detected")
            
            # Create persistent water mask if multiple dates available
            if len(dataset.time) > 1:
                if verbose:
                    print(f"Creating persistent water mask (threshold: {persistence_threshold})")
                
                # Stack all water masks and calculate frequency
                all_masks = []
                for i in range(len(dataset.time)):
                    date_str = np.datetime_as_string(dataset.time.values[i], unit='D')
                    mask_key = f'water_mask_{i}_{date_str}'
                    all_masks.append(results['water_masks'][mask_key])
                
                water_stack = np.stack(all_masks, axis=0)
                water_frequency = np.mean(water_stack, axis=0)
                persistent_water = water_frequency > persistence_threshold
                
                results['water_masks']['persistent_water'] = persistent_water
                
                # Calculate persistent water statistics
                persistent_area = np.sum(persistent_water) * (resolution * resolution) / 1_000_000
                results['statistics']['persistent_water_area_km2'] = persistent_area
                
                if verbose:
                    print(f"  Persistent water area: {persistent_area:.2f} sq km")
            
            # Create water extent visualization
            if len(dataset.time) > 0:
                if verbose:
                    print("Creating water extent visualization...")
                
                # Use first time step for visualization
                first_date = np.datetime_as_string(dataset.time.values[0], unit='D')
                water_mask_viz = results['water_masks'][f'water_mask_0_{first_date}']
                
                # Create RGB visualization with water overlay
                scene = dataset.isel(time=0)
                
                # Get band data for visualization
                if water_band in scene:
                    band_data = np.array(scene[water_band].values, dtype=np.float64)
                else:
                    band_data = np.array(scene[list(scene.data_vars)[0]].values, dtype=np.float64)
                
                band_data[~np.isfinite(band_data)] = np.nan
                
                # Apply contrast stretching
                non_nan = band_data[~np.isnan(band_data)]
                if len(non_nan) > 0:
                    low, high = np.percentile(non_nan, [2, 98])
                    band_data = np.clip((band_data - low) / (high - low), 0, 1)
                
                band_data = np.nan_to_num(band_data)
                rgb_data = np.dstack([band_data, band_data, band_data])
                
                # Create plot
                plt.figure(figsize=(12, 12))
                plt.imshow(rgb_data, cmap='gray')
                
                # Add water mask overlay
                if water_mask_viz.shape == rgb_data.shape[:2]:
                    water_overlay = np.zeros((water_mask_viz.shape[0], water_mask_viz.shape[1], 4))
                    water_overlay[:,:,2] = 1.0  # Blue channel
                    water_overlay[:,:,3] = water_mask_viz * 0.7  # Alpha transparency
                    plt.imshow(water_overlay)
                
                plt.title(f"Water Extent Detection - {first_date}\n(Blue = Water)", fontsize=14)
                plt.axis('off')
                
                if save_outputs:
                    plt.savefig(f"{output_prefix}_water_extent.png", dpi=300, bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
        
        # =================================================================
        # STEP 5: FLOOD DETECTION ANALYSIS
        # =================================================================
        if analysis_type in ["flood_only", "water_flood"] and len(dataset.time) > 1:
            if verbose:
                print("Performing flood detection analysis...")
            
            # Use first and last acquisitions for flood comparison
            first_idx = 0
            last_idx = len(dataset.time) - 1
            
            first_date = np.datetime_as_string(dataset.time.values[first_idx], unit='D')
            last_date = np.datetime_as_string(dataset.time.values[last_idx], unit='D')
            
            if verbose:
                print(f"Comparing {first_date} (baseline) with {last_date} (target)")
            
            # Determine band for flood detection
            flood_band = 'filtered_vh' if 'filtered_vh' in dataset else 'vh'
            
            # Get data for both time steps
            first_data = dataset.isel(time=first_idx)[flood_band].values
            last_data = dataset.isel(time=last_idx)[flood_band].values
            
            # Calculate backscatter change
            if use_log10:
                # Safe log10 transformation for both datasets
                first_clean = np.where((first_data <= 0) | (~np.isfinite(first_data)), 1e-10, first_data)
                last_clean = np.where((last_data <= 0) | (~np.isfinite(last_data)), 1e-10, last_data)
                
                first_log = np.log10(first_clean)
                last_log = np.log10(last_clean)
                
                first_log = np.where(~np.isfinite(first_log), -10, first_log)
                last_log = np.where(~np.isfinite(last_log), -10, last_log)
                
                change = last_log - first_log
                baseline_water_mask = first_log < water_threshold
            else:
                first_clean = np.where(~np.isfinite(first_data), 0, first_data)
                last_clean = np.where(~np.isfinite(last_data), 0, last_data)
                change = last_clean - first_clean
                baseline_water_mask = first_clean < water_threshold
            
            # Identify potential flooding (significant backscatter decrease)
            flood_mask = change < flood_change_threshold
            
            # Store flood detection results
            results['flood_results'] = {
                'flood_mask': flood_mask,
                'baseline_water_mask': baseline_water_mask,
                'change_array': change,
                'first_date': first_date,
                'last_date': last_date,
                'first_idx': first_idx,
                'last_idx': last_idx
            }
            
            # Calculate flood statistics
            flood_pixels = np.sum(flood_mask)
            baseline_water_pixels = np.sum(baseline_water_mask)
            total_pixels = flood_mask.size
            
            flood_area_sq_km = flood_pixels * (resolution * resolution) / 1_000_000
            baseline_water_area_sq_km = baseline_water_pixels * (resolution * resolution) / 1_000_000
            
            flood_percent = (flood_pixels / total_pixels) * 100
            baseline_water_percent = (baseline_water_pixels / total_pixels) * 100
            
            results['statistics'].update({
                'flood_area_km2': flood_area_sq_km,
                'baseline_water_area_km2': baseline_water_area_sq_km,
                'flood_pixels': flood_pixels,
                'baseline_water_pixels': baseline_water_pixels,
                'flood_percent': flood_percent,
                'baseline_water_percent': baseline_water_percent
            })
            
            if verbose:
                print(f"  Potential new flooding: {flood_area_sq_km:.2f} sq km ({flood_percent:.2f}%)")
                print(f"  Baseline water area: {baseline_water_area_sq_km:.2f} sq km ({baseline_water_percent:.2f}%)")
            
            # Create flood detection visualization
            if verbose:
                print("Creating flood detection visualization...")
            
            # Create RGB visualization with flood and water overlays
            scene = dataset.isel(time=first_idx)
            
            # Get band data for background
            if flood_band in scene:
                band_data = np.array(scene[flood_band].values, dtype=np.float64)
            else:
                band_data = np.array(scene[list(scene.data_vars)[0]].values, dtype=np.float64)
            
            band_data[~np.isfinite(band_data)] = np.nan
            
            # Apply contrast stretching
            non_nan = band_data[~np.isnan(band_data)]
            if len(non_nan) > 0:
                low, high = np.percentile(non_nan, [2, 98])
                band_data = np.clip((band_data - low) / (high - low), 0, 1)
            
            band_data = np.nan_to_num(band_data)
            rgb_data = np.dstack([band_data, band_data, band_data])
            
            # Create plot
            plt.figure(figsize=(12, 12))
            plt.imshow(rgb_data, cmap='gray')
            
            # Add flood mask overlay (red)
            if flood_mask.shape == rgb_data.shape[:2]:
                flood_overlay = np.zeros((flood_mask.shape[0], flood_mask.shape[1], 4))
                flood_overlay[:,:,0] = 1.0  # Red channel
                flood_overlay[:,:,3] = flood_mask * 0.7  # Alpha transparency
                plt.imshow(flood_overlay)
            
            # Add baseline water mask overlay (blue)
            if baseline_water_mask.shape == rgb_data.shape[:2]:
                water_overlay = np.zeros((baseline_water_mask.shape[0], baseline_water_mask.shape[1], 4))
                water_overlay[:,:,2] = 1.0  # Blue channel
                water_overlay[:,:,3] = baseline_water_mask * 0.7  # Alpha transparency
                plt.imshow(water_overlay)
            
            plt.title(f"Flood Detection: {first_date} to {last_date}\n(Red = Potential flooding, Blue = Baseline water)", fontsize=14)
            plt.axis('off')
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_flood_detection.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # =================================================================
        # STEP 6: TIME SERIES ANALYSIS
        # =================================================================
        if len(dataset.time) > 1 and analysis_type in ["water_only", "water_flood"]:
            if verbose:
                print("Creating water extent time series...")
            
            # Extract dates and water areas
            dates = [np.datetime64(dt) for dt in dataset.time.values]
            water_areas = []
            
            for i in range(len(dataset.time)):
                date_str = np.datetime_as_string(dataset.time.values[i], unit='D')
                water_area_key = f'water_area_km2_{i}'
                if water_area_key in results['statistics']:
                    water_areas.append(results['statistics'][water_area_key])
                else:
                    water_areas.append(0.0)
            
            # Create time series plot
            plt.figure(figsize=(12, 8))
            plt.plot(dates, water_areas, 'o-', linewidth=2, markersize=8, color='blue')
            
            # Add data labels
            for i, (date, area) in enumerate(zip(dates, water_areas)):
                plt.annotate(f'{area:.1f}', 
                           (date, area),
                           textcoords="offset points", 
                           xytext=(0, 10), 
                           ha='center',
                           fontsize=10)
            
            plt.title('Water Surface Area Time Series', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Water Surface Area (sq km)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate()
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_water_timeseries.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        if verbose:
            print("\n" + "="*60)
            print("PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Dataset dimensions: {dict(dataset.dims)}")
            print(f"Available bands: {list(dataset.data_vars)}")
            print(f"Time range: {np.datetime_as_string(dataset.time.values[0], unit='D')} to {np.datetime_as_string(dataset.time.values[-1], unit='D')}")
            print(f"Water masks created: {len(results['water_masks'])}")
            print(f"Flood analysis: {'Yes' if results['flood_results'] else 'No'}")
            
            if results['statistics']:
                print("\nKey Statistics:")
                for key, value in results['statistics'].items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"ERROR during processing: {e}")
            import traceback
            traceback.print_exc()
        return results


# Example usage:
if __name__ == "__main__":
    # Define area of interest (example coordinates)
    geometry = Polygon([
        (-92.695, 17.813),
        (-92.495, 17.813),
        (-92.495, 17.613),
        (-92.695, 17.613),
        (-92.695, 17.813)
    ])
    
    # Run complete analysis
    results = process_sentinel1_sar_complete(
        geometry=geometry,
        start_date=datetime.datetime(2020, 10, 25),
        end_date=datetime.datetime(2020, 12, 5),
        analysis_type="water_flood",  # Full analysis including water detection and flood detection
        water_threshold=-0.7,         # Water detection threshold (log10 space)
        flood_change_threshold=-0.5,  # Flood detection threshold (backscatter change)
        persistence_threshold=0.8,    # Persistent water threshold (80% of time)
        use_log10=True,              # Use log10 transformation
        save_outputs=True,           # Save visualization files
        output_prefix="flood_analysis", # Prefix for output files
        verbose=True                 # Print progress updates
    )
    
    # Print final summary
    print("\n" + "="*50)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*50)
    if results['dataset'] is not None:
        print(f"✓ Dataset loaded: {len(results['dataset'].time)} time steps")
        print(f"✓ Variables: {list(results['dataset'].data_vars)}")
    
    if results['water_masks']:
        print(f"✓ Water masks: {len(results['water_masks'])} created")
    
    if results['flood_results']:
        print(f"✓ Flood detection: Completed")
        flood_area = results['statistics'].get('flood_area_km2', 0)
        print(f"  → Potential flooding area: {flood_area:.2f} sq km")
    
    print(f"✓ Statistics calculated: {len(results['statistics'])} metrics")
