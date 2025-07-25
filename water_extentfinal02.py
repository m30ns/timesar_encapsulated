#water_extentfinal02.py

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
import warnings
import logging

# Suppress rasterio/GDAL warnings about blob storage
warnings.filterwarnings('ignore', message='.*GET Request.*')
warnings.filterwarnings('ignore', message='.*ignoring read failure.*')
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('rasterio._io').setLevel(logging.ERROR)

def process_sentinel1_water_extent_complete(
    geometry: shapely.geometry.Polygon,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    measurements: List[str] = ["vv", "vh"],
    resolution: int = 10,
    platform: Optional[str] = None,
    orbit_direction: Optional[str] = None,
    instrument_mode: str = "IW",
    product_type: str = "GRD",
    filter_type: str = 'mean',
    filter_size: int = 5,
    water_threshold: float = -2.0,
    persistence_threshold: float = 0.8,
    use_log10: bool = True,
    flood_analysis: bool = True,
    first_acq_ind: Optional[int] = None,
    second_acq_ind: Optional[int] = None,
    change_threshold: float = -0.5,
    save_outputs: bool = True,
    output_prefix: str = "sentinel1_analysis",
    show_plots: bool = True,
    verbose: bool = True,
    grouping_criteria: str = "platform_orbit_relative",
    min_acquisitions: int = 2
) -> Dict[str, Any]:
    """
    Complete Sentinel-1 SAR data processing function for water extent mapping and flood detection.
    This is a single comprehensive function that handles the entire workflow without nested functions.
    
    Args:
        geometry: AOI polygon for data extraction
        start_date: Start date for data query
        end_date: End date for data query
        measurements: List of bands to retrieve (e.g., ["vv", "vh"])
        resolution: Spatial resolution in meters
        platform: Satellite platform ('sentinel-1a', 'sentinel-1b', or None for all)
        orbit_direction: Orbit direction ('ascending', 'descending', or None)
        instrument_mode: SAR instrument mode
        product_type: Product type
        filter_type: Type of speckle filter ('mean', 'median', 'min', 'max')
        filter_size: Size of the filter window
        water_threshold: Threshold for water detection (log10 space if use_log10=True)
        persistence_threshold: Fraction of time for persistent water mapping
        use_log10: Whether to apply log10 transformation
        flood_analysis: Whether to perform flood detection analysis
        first_acq_ind: Index of first acquisition for flood detection
        second_acq_ind: Index of second acquisition for flood detection
        change_threshold: Threshold for flood detection (change in backscatter)
        save_outputs: Whether to save visualization outputs
        output_prefix: Prefix for saved files
        show_plots: Whether to display plots
        verbose: Whether to print processing updates
        grouping_criteria: How to group acquisitions by orbit
        min_acquisitions: Minimum number of acquisitions required for a group
        
    Returns:
        Dict containing processed dataset, water masks, flood results, and statistics
    """
    
    # Initialize results dictionary
    results = {
        'dataset': None,
        'water_masks': {},
        'flood_results': {},
        'statistics': {},
        'visualizations': []
    }
    
    try:
        # =================================================================
        # STEP 1: GET STAC CATALOG AND LOAD SENTINEL-1 DATA
        # =================================================================
        if verbose:
            print("Getting STAC catalog and loading Sentinel-1 data...")
        
        # Get the STAC catalog client for Planetary Computer
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        # Format the datetime range for STAC query
        datetime_range = f"{start_date.isoformat()}Z/{end_date.isoformat()}Z"
        
        # Prepare query parameters
        query_params = {}
        if platform:
            query_params["platform"] = {"eq": platform}
        if instrument_mode:
            query_params["sar:instrument_mode"] = {"eq": instrument_mode}
        if product_type:
            query_params["sar:product_type"] = {"eq": product_type}
        if orbit_direction:
            query_params["sat:orbit_state"] = {"eq": orbit_direction}
        
        # Search for Sentinel-1 scenes
        search = client.search(
            collections=["sentinel-1-rtc"],  # Radiometrically terrain corrected data
            intersects=geometry,
            datetime=datetime_range,
            query=query_params,
        )
        
        # Get the search results as a collection of STAC items
        items = search.item_collection()
        
        # Check if any items were found
        if len(items) == 0:
            raise ValueError(
                "No matching scenes found. Please refine your search criteria,"
                " such as the timeframe or location, and try again."
            )
        
        if verbose:
            print(f"Found {len(items)} Sentinel-1 scenes")
        
        # Extract and print metadata information about platform and orbit
        if verbose:
            print("\nMetadata Information:")
            print("---------------------")
            for i, item in enumerate(items):
                platform_id = item.properties.get("platform", "Unknown")
                orbit_state = item.properties.get("sat:orbit_state", "Unknown")
                acquisition_date = item.properties.get("datetime", "Unknown")
                print(f"Scene {i+1}: Platform: {platform_id}, Orbit: {orbit_state}, Date: {acquisition_date}")
        
        # Load with odc.stac.load
        dataset = odc.stac.load(
            items=items,
            bands=measurements,
            geopolygon=geometry,
            groupby="solar_day",
            resolution=resolution,
            fail_on_error=False,
        )
        
        # Store metadata for each time step in the dataset attributes
        metadata_list = []
        for item in items:
            metadata = {
                "platform": item.properties.get("platform", "Unknown"),
                "orbit_state": item.properties.get("sat:orbit_state", "Unknown"),
                "datetime": item.properties.get("datetime", "Unknown"),
                # Add additional metadata fields as needed
                "orbit_number": item.properties.get("sat:absolute_orbit", "Unknown"),
                "relative_orbit": item.properties.get("sat:relative_orbit", "Unknown"),
                "instrument_mode": item.properties.get("sar:instrument_mode", "Unknown"),
                "product_type": item.properties.get("sar:product_type", "Unknown"),
            }
            metadata_list.append(metadata)
        
        # Add metadata to dataset attributes
        if hasattr(dataset, 'attrs'):
            dataset.attrs['metadata'] = metadata_list
        
        # Compute the dataset to load it into memory
        if verbose:
            print("Computing dataset...")
        dataset = dataset.compute()
        
        if verbose:
            print(f"Dataset loaded successfully with {len(dataset.time)} time steps")
            print(f"Available variables: {list(dataset.data_vars)}")
        
        # =================================================================
        # STEP 2: FILTER DATASET BY METADATA IF SPECIFIED
        # =================================================================
        if platform or orbit_direction:
            if verbose:
                print(f"Filtering dataset by platform={platform}, orbit_direction={orbit_direction}")
            
            if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
                metadata_list = dataset.attrs['metadata']
                
                # Create a boolean mask for filtering time steps
                keep_indices = []
                for i, metadata in enumerate(metadata_list):
                    keep = True
                    
                    if platform and metadata['platform'].lower() != platform.lower():
                        keep = False
                    
                    if orbit_direction and metadata['orbit_state'].lower() != orbit_direction.lower():
                        keep = False
                    
                    if keep:
                        keep_indices.append(i)
                
                # Check if any time steps match the criteria
                if not keep_indices:
                    raise ValueError(
                        f"No matching scenes found with platform={platform} and orbit_direction={orbit_direction}. "
                        "Try different filter criteria."
                    )
                
                # Filter the dataset by time indices
                dataset = dataset.isel(time=keep_indices)
                
                # Update metadata in filtered dataset
                filtered_metadata = [metadata_list[i] for i in keep_indices]
                dataset.attrs['metadata'] = filtered_metadata
                
                if verbose:
                    print(f"Filtered dataset to {len(dataset.time)} time steps")
        
        # =================================================================
        # STEP 3: GROUP ACQUISITIONS BY ORBIT FOR CONSISTENCY
        # =================================================================
        if verbose:
            print(f"Grouping acquisitions by orbit using criteria: {grouping_criteria}")
        
        if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
            metadata_list = dataset.attrs['metadata']
            n_times = len(dataset.time)
            n_metadata = len(metadata_list)
            
            if verbose:
                print(f"Dataset has {n_times} time steps and {n_metadata} metadata entries")
            
            # Handle mismatch between time steps and metadata
            if n_metadata != n_times:
                if verbose:
                    print(f"Warning: Metadata count ({n_metadata}) doesn't match time count ({n_times})")
                max_index = min(n_times, n_metadata)
                if verbose:
                    print(f"Will only process first {max_index} entries")
            else:
                max_index = n_times
            
            # Try different grouping strategies to find best orbit consistency
            strategies = [
                ("platform_orbit_relative", lambda m: f"{m.get('platform', 'unknown')}_{m.get('orbit_state', 'unknown')}_{m.get('relative_orbit', 'unknown')}"),
                ("orbit_relative", lambda m: f"{m.get('orbit_state', 'unknown')}_{m.get('relative_orbit', 'unknown')}"),
                ("relative_only", lambda m: f"rel_{m.get('relative_orbit', 'unknown')}"),
                ("platform_orbit", lambda m: f"{m.get('platform', 'unknown')}_{m.get('orbit_state', 'unknown')}")
            ]
            
            best_result = None
            best_count = 0
            best_strategy = None
            
            # Try the preferred strategy first
            for strategy_name, key_func in strategies:
                if strategy_name == grouping_criteria or (best_result is None and grouping_criteria not in [s[0] for s in strategies]):
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
                        if len(indices) >= min_acquisitions and len(indices) > best_count:
                            best_result = (key, indices)
                            best_count = len(indices)
                            best_strategy = strategy_name
                    
                    if best_result and strategy_name == grouping_criteria:
                        break
            
            # If preferred strategy failed, try all strategies
            if best_result is None:
                if verbose:
                    print("Preferred strategy failed, trying all strategies...")
                
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
                        if len(indices) >= min_acquisitions and len(indices) > best_count:
                            best_result = (key, indices)
                            best_count = len(indices)
                            best_strategy = strategy_name
            
            if best_result is not None:
                key, indices = best_result
                if verbose:
                    print(f"Selected orbit group '{key}' with {len(indices)} acquisitions using {best_strategy}")
                
                # Ensure all indices are valid
                valid_indices = [idx for idx in indices if idx < n_times]
                if len(valid_indices) >= min_acquisitions:
                    dataset = dataset.isel(time=valid_indices)
                    subset_metadata = [metadata_list[i] for i in valid_indices if i < n_metadata]
                    dataset.attrs['metadata'] = subset_metadata
                else:
                    if verbose:
                        print("Not enough valid indices after filtering, using original dataset")
            else:
                if verbose:
                    print("No suitable orbit groups found, using original dataset")
        
        if len(dataset.time) == 0:
            raise ValueError("No valid time steps available after orbit selection")
        
        results['dataset'] = dataset
        
        # =================================================================
        # STEP 4: DISPLAY AVAILABLE ACQUISITIONS
        # =================================================================
        if verbose:
            print("\nAvailable acquisition times:")
            print("----------------------------------------------------------")
            print("Index  |  Date        |  Platform    |  Orbit Direction  |  Orbit Number  |  Relative Orbit")
            print("----------------------------------------------------------")
            
            if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
                metadata_list = dataset.attrs['metadata']
                
                for i, time in enumerate(dataset.time.values):
                    time_str = np.datetime_as_string(time, unit='D')
                    
                    if i < len(metadata_list):
                        metadata = metadata_list[i]
                        platform_str = metadata.get('platform', 'Unknown')
                        orbit_state = metadata.get('orbit_state', 'Unknown')
                        orbit_number = metadata.get('orbit_number', 'Unknown')
                        relative_orbit = metadata.get('relative_orbit', 'Unknown')
                        
                        print(f"{i:<7}|  {time_str}  |  {platform_str:<11}  |  {orbit_state:<16}  |  {orbit_number:<13}  |  {relative_orbit}")
                    else:
                        print(f"{i:<7}|  {time_str}  |  Unknown      |  Unknown          |  Unknown       |  Unknown")
            else:
                for i, time in enumerate(dataset.time.values):
                    time_str = np.datetime_as_string(time, unit='D')
                    print(f"{i:<7}|  {time_str}  |  Unknown      |  Unknown          |  Unknown       |  Unknown")
        
        # =================================================================
        # STEP 5: APPLY SPECKLE FILTERING
        # =================================================================
        if verbose:
            print(f"\nApplying {filter_type} filter with size {filter_size} to bands: {measurements}")
        
        # Ensure filter size is odd
        if filter_size % 2 == 0:
            filter_size += 1
        
        # Fill null values to prevent issues with filtering
        # Note: This operation is lazy until we actually access the data
        dataset_filled = dataset.where(~dataset.isnull(), 0)
        
        # Apply filter to each band
        for band in measurements:
            if band not in dataset:
                continue
                
            if verbose:
                print(f"Filtering band: {band}")
            
            try:
                filtered_band_name = f"block_{filter_type}_filter_{band}"
                
                # Convert to power, apply filter, convert back to dB
                filtered_data = []
                for i in range(len(dataset.time)):
                    if verbose and i % 5 == 0:
                        print(f"  Processing time step {i}/{len(dataset.time)}")
                    
                    # Get time slice
                    time_slice = dataset_filled[band].isel(time=i)
                    
                    # Convert dB to power: power = 10^(dB/10)
                    power_data = 10**(time_slice.values/10)
                    
                    # Apply selected statistical filter
                    if filter_type == 'mean':
                        filtered_power = ndimage.uniform_filter(power_data, size=filter_size)
                    elif filter_type == 'median':
                        filtered_power = ndimage.median_filter(power_data, size=filter_size)
                    elif filter_type == 'min':
                        filtered_power = ndimage.minimum_filter(power_data, size=filter_size)
                    elif filter_type == 'max':
                        filtered_power = ndimage.maximum_filter(power_data, size=filter_size)
                    else:
                        raise ValueError(f"Unsupported statistic: {filter_type}")
                    
                    # Convert back to dB: dB = 10*log10(power)
                    filtered_db = 10 * np.log10(np.maximum(filtered_power, 1e-10))
                    
                    # Create DataArray with same coordinates
                    filtered_slice = xr.DataArray(
                        filtered_db,
                        coords=time_slice.coords,
                        dims=time_slice.dims,
                        attrs=time_slice.attrs
                    )
                    filtered_data.append(filtered_slice)
                
                # Stack the filtered time slices back together
                if len(filtered_data) > 1:
                    filtered_stack = xr.concat(filtered_data, dim='time')
                    filtered_stack = filtered_stack.assign_coords(time=dataset.time)
                else:
                    filtered_stack = filtered_data[0].expand_dims(dim='time')
                    filtered_stack = filtered_stack.assign_coords(time=dataset.time)
                
                # Add filtered band to dataset
                dataset[filtered_band_name] = filtered_stack
                
                if verbose:
                    print(f"Successfully created filtered band: {filtered_band_name}")
                
            except Exception as e:
                if verbose:
                    print(f"Error filtering band {band}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # =================================================================
        # STEP 6: CALCULATE BACKSCATTER AMPLITUDES
        # =================================================================
        if verbose:
            print("\nCalculating backscatter amplitudes...")
        
        try:
            # Check if filtered bands exist
            if 'block_mean_filter_vv' in dataset and 'block_mean_filter_vh' in dataset:
                # Apply backscatter scaling optimized for block-filtered data
                # VV band range is 0dB to -16dB which is DN=1.00 to DN=0.158
                # VH band range is -5dB to -27dB which is DN=0.562 to DN=0.045
                vv_convert = (10**(dataset.block_mean_filter_vv/20)-0.158)*303
                vh_convert = (10**(dataset.block_mean_filter_vh/20)-0.045)*493
                
                dataset['vv_amp'] = vv_convert
                dataset['vh_amp'] = vh_convert
                dataset['vvvh_amp'] = (vv_convert / vh_convert) * 20
                
                if verbose:
                    print("Backscatter amplitudes calculated successfully")
            else:
                if verbose:
                    print("Filtered bands not available for amplitude calculation")
        except Exception as e:
            if verbose:
                print(f"Error calculating backscatter amplitudes: {e}")
        
        # =================================================================
        # STEP 7: CREATE BACKSCATTER RGB VISUALIZATION
        # =================================================================
        if show_plots or save_outputs:
            if all(band in dataset for band in ['vv_amp', 'vh_amp', 'vvvh_amp']):
                if verbose:
                    print("\nCreating backscatter RGB visualization...")
                
                plt.figure(figsize=(12, 12))
                
                # Get the three bands for RGB
                r = np.array(dataset.isel(time=0)['vv_amp'].values, dtype=np.float64)
                g = np.array(dataset.isel(time=0)['vh_amp'].values, dtype=np.float64)
                b = np.array(dataset.isel(time=0)['vvvh_amp'].values, dtype=np.float64)
                
                # Replace infinity values with NaN
                r[~np.isfinite(r)] = np.nan
                g[~np.isfinite(g)] = np.nan
                b[~np.isfinite(b)] = np.nan
                
                # Stack the bands into an RGB array
                rgb_data = np.dstack((r, g, b))
                
                # Apply contrast stretching
                p_low, p_high = (2, 98)
                for i in range(3):
                    band = rgb_data[:,:,i]
                    non_nan = band[~np.isnan(band)]
                    if len(non_nan) > 0:
                        low, high = np.percentile(non_nan, [p_low, p_high])
                        rgb_data[:,:,i] = np.clip((band - low) / (high - low), 0, 1)
                
                # Replace NaN values with zeros
                rgb_data = np.nan_to_num(rgb_data)
                
                # Plot the RGB image
                plt.imshow(rgb_data)
                
                # Add metadata information to title
                title = 'Backscatter RGB: VV, VH, VV/VH'
                if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
                    metadata = dataset.attrs['metadata'][0]
                    platform = metadata.get('platform', 'Unknown')
                    orbit_state = metadata.get('orbit_state', 'Unknown')
                    date_str = np.datetime_as_string(dataset.time.values[0], unit='D')
                    title += f'\n{platform.upper()}, {orbit_state.capitalize()} orbit, {date_str}'
                
                plt.title(title, fontsize=14)
                plt.axis('off')
                
                if save_outputs:
                    plt.savefig(f"{output_prefix}_backscatter_rgb.png", dpi=300, bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                results['visualizations'].append('backscatter_rgb')
        
        # =================================================================
        # STEP 8: CREATE BACKSCATTER HISTOGRAM
        # =================================================================
        if show_plots or save_outputs:
            if verbose:
                print("\nCreating backscatter histogram...")
            
            plt.figure(figsize=(15, 5))
            
            try:
                # Check if filtered bands exist
                vv_band = 'block_mean_filter_vv' if 'block_mean_filter_vv' in dataset else 'vv'
                vh_band = 'block_mean_filter_vh' if 'block_mean_filter_vh' in dataset else 'vh'
                
                if vv_band in dataset:
                    vv_data = dataset.isel(time=0)[vv_band].values
                    # Safe log10 transformation
                    vv_clean = np.where((vv_data <= 0) | (~np.isfinite(vv_data)), 1e-10, vv_data)
                    vv_log = np.log10(vv_clean)
                    vv_log = vv_log[np.isfinite(vv_log)]
                    
                    if len(vv_log) > 0:
                        plt.hist(vv_log, bins=200, alpha=0.7, label=f"{vv_band} (log10)", color='blue')
                
                if vh_band in dataset:
                    vh_data = dataset.isel(time=0)[vh_band].values
                    # Safe log10 transformation
                    vh_clean = np.where((vh_data <= 0) | (~np.isfinite(vh_data)), 1e-10, vh_data)
                    vh_log = np.log10(vh_clean)
                    vh_log = vh_log[np.isfinite(vh_log)]
                    
                    if len(vh_log) > 0:
                        plt.hist(vh_log, bins=200, alpha=0.7, label=f"{vh_band} (log10)", color='red')
                
                plt.legend()
                plt.xlabel("Backscatter Intensity (log10 dB)")
                plt.ylabel("Number of Pixels")
                plt.title("Histogram Comparison of Backscatter Values (log10 transformed)")
                plt.grid(True, alpha=0.3)
                
                if save_outputs:
                    plt.savefig(f"{output_prefix}_backscatter_histogram.png", dpi=300, bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                results['visualizations'].append('backscatter_histogram')
                
            except Exception as e:
                if verbose:
                    print(f"Error creating histogram: {e}")
        
        # =================================================================
        # STEP 9: WATER EXTENT DETECTION
        # =================================================================
        if verbose:
            print(f"\nPerforming water extent detection...")
            print(f"Water threshold: {water_threshold} (log10 space: {use_log10})")
        
        # Determine best band for water detection
        water_band = 'block_mean_filter_vh' if 'block_mean_filter_vh' in dataset else 'vh'
        if water_band not in dataset:
            available_bands = list(dataset.data_vars)
            water_band = available_bands[0] if available_bands else None
        
        if water_band is None:
            raise ValueError("No suitable bands available for water detection")
        
        if verbose:
            print(f"Using {water_band} for water detection")
        
        # Create water masks for each time step
        water_areas = []
        for i in range(len(dataset.time)):
            date_str = np.datetime_as_string(dataset.time.values[i], unit='D')
            
            # Get data for this time step
            data = dataset.isel(time=i)[water_band]
            
            # Apply water detection threshold
            if use_log10:
                # Safe log10 transformation
                data_values = data.values
                # Replace zero, negative, and infinite values with small positive number
                data_values = np.where((data_values <= 0) | (~np.isfinite(data_values)), 1e-10, data_values)
                data_log = np.log10(data_values)
                # Handle any remaining infinite or NaN values
                data_log = np.where(~np.isfinite(data_log), -10, data_log)
                water_mask = data_log < water_threshold
            else:
                data_values = data.values
                data_values = np.where(~np.isfinite(data_values), 0, data_values)
                water_mask = data_values < water_threshold
            
            # Store water mask
            mask_key = f'water_mask_{i}_{date_str}'
            results['water_masks'][mask_key] = water_mask
            
            # Calculate water area statistics
            water_pixels = np.sum(water_mask)
            water_area_sq_km = water_pixels * 100 / 1_000_000  # Matches original calculation
            water_areas.append(water_area_sq_km)
            results['statistics'][f'water_area_km2_{i}'] = water_area_sq_km
            
            if verbose:
                print(f"  {date_str}: {water_pixels} water pixels = {water_area_sq_km:.2f} sq km")
        
        # =================================================================
        # STEP 10: CREATE WATER EXTENT VISUALIZATION
        # =================================================================
        if (show_plots or save_outputs) and len(dataset.time) > 0:
            if verbose:
                print("\nCreating water extent visualization...")
            
            # Use first time step for visualization
            first_date = np.datetime_as_string(dataset.time.values[0], unit='D')
            water_mask_viz = results['water_masks'][f'water_mask_0_{first_date}']
            
            # Create water extent visualization
            plt.figure(figsize=(12, 12))
            
            # Get VH band data for background
            scene = dataset.isel(time=0)
            vh_data = np.array(scene['vh'].values, dtype=np.float64)
            vh_data[~np.isfinite(vh_data)] = np.nan
            
            # Apply contrast stretching
            non_nan = vh_data[~np.isnan(vh_data)]
            if len(non_nan) > 0:
                low, high = np.percentile(non_nan, [2, 98])
                vh_data = np.clip((vh_data - low) / (high - low), 0, 1)
            
            vh_data = np.nan_to_num(vh_data)
            
            # Apply minimum intensity
            min_inten = 0.6
            intensity = vh_data
            intensity_safe = np.maximum(intensity, 1e-10)
            multiplier = np.maximum(min_inten, intensity_safe) / intensity_safe
            vh_data = np.minimum(vh_data * multiplier, 1.0)
            
            # Create RGB from single band
            rgb_data = np.dstack([vh_data, vh_data, vh_data])
            
            # Plot the background
            plt.imshow(rgb_data, cmap='gray')
            
            # Add water mask overlay (blue)
            if water_mask_viz.shape == rgb_data.shape[:2]:
                water_overlay = np.zeros((water_mask_viz.shape[0], water_mask_viz.shape[1], 4))
                water_overlay[:,:,2] = 1.0  # Blue channel
                water_overlay[:,:,3] = water_mask_viz * 0.7  # Alpha transparency
                plt.imshow(water_overlay)
            
            # Add title with metadata
            title = 'VH-Band Threshold Water Extent'
            if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
                metadata = dataset.attrs['metadata'][0]
                platform = metadata.get('platform', 'Unknown')
                orbit_state = metadata.get('orbit_state', 'Unknown')
                title += f'\n{platform.upper()}, {orbit_state.capitalize()} orbit, {first_date}'
            
            plt.title(title, fontsize=14)
            plt.axis('off')
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_water_extent.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            results['visualizations'].append('water_extent')
        
        # =================================================================
        # STEP 11: CREATE PERSISTENT WATER MASK
        # =================================================================
        if len(dataset.time) > 1:
            if verbose:
                print(f"\nCreating persistent water mask (threshold: {persistence_threshold})...")
            
            # Stack all water masks
            all_masks = []
            for i in range(len(dataset.time)):
                date_str = np.datetime_as_string(dataset.time.values[i], unit='D')
                mask_key = f'water_mask_{i}_{date_str}'
                all_masks.append(results['water_masks'][mask_key])
            
            # Calculate water frequency across time
            water_stack = np.stack(all_masks, axis=0)
            water_frequency = np.mean(water_stack, axis=0)
            
            # Create persistent water mask
            persistent_water = water_frequency > persistence_threshold
            results['water_masks']['persistent_water'] = persistent_water
            
            # Calculate statistics
            persistent_pixels = np.sum(persistent_water)
            persistent_area_sq_km = persistent_pixels * 100 / 1_000_000
            results['statistics']['persistent_water_area_km2'] = persistent_area_sq_km
            results['statistics']['persistent_water_pixels'] = persistent_pixels
            results['statistics']['persistent_water_percent'] = (persistent_pixels / persistent_water.size) * 100
            
            if verbose:
                print(f"Persistent water covers {persistent_pixels} pixels ({results['statistics']['persistent_water_percent']:.2f}% of scene)")
                print(f"Persistent water area: {persistent_area_sq_km:.2f} sq km")
            
            # Create persistent water visualization
            if show_plots or save_outputs:
                if verbose:
                    print("Creating persistent water visualization...")
                
                plt.figure(figsize=(12, 12))
                
                # Use first time step as background
                scene = dataset.isel(time=0)
                vh_data = np.array(scene['vh'].values, dtype=np.float64)
                vh_data[~np.isfinite(vh_data)] = np.nan
                
                # Apply contrast stretching
                non_nan = vh_data[~np.isnan(vh_data)]
                if len(non_nan) > 0:
                    low, high = np.percentile(non_nan, [2, 98])
                    vh_data = np.clip((vh_data - low) / (high - low), 0, 1)
                
                vh_data = np.nan_to_num(vh_data)
                
                # Apply minimum intensity
                min_inten = 0.6
                intensity = vh_data
                intensity_safe = np.maximum(intensity, 1e-10)
                multiplier = np.maximum(min_inten, intensity_safe) / intensity_safe
                vh_data = np.minimum(vh_data * multiplier, 1.0)
                
                # Create RGB from single band
                rgb_data = np.dstack([vh_data, vh_data, vh_data])
                
                # Plot the background
                plt.imshow(rgb_data, cmap='gray')
                
                # Add persistent water mask overlay (cyan)
                if persistent_water.shape == rgb_data.shape[:2]:
                    water_overlay = np.zeros((persistent_water.shape[0], persistent_water.shape[1], 4))
                    water_overlay[:,:,0] = 0.0  # Red channel
                    water_overlay[:,:,1] = 1.0  # Green channel
                    water_overlay[:,:,2] = 1.0  # Blue channel
                    water_overlay[:,:,3] = persistent_water * 0.7  # Alpha transparency
                    plt.imshow(water_overlay)
                
                plt.title('Persistent Water Areas (>80% of time)', fontsize=14)
                plt.axis('off')
                
                if save_outputs:
                    plt.savefig(f"{output_prefix}_persistent_water.png", dpi=300, bbox_inches='tight')
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                
                results['visualizations'].append('persistent_water')
        
        # =================================================================
        # STEP 12: CREATE WATER EXTENT TIME SERIES
        # =================================================================
        if len(dataset.time) > 1 and (show_plots or save_outputs):
            if verbose:
                print("\nCreating water extent time series...")
            
            # Get the dates for the x-axis
            dates = [np.datetime64(dt) for dt in dataset.time.values]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            plt.plot(dates, water_areas, c='black', marker='o', mfc='blue', markersize=10, linewidth=1)
            
            # Format x-axis as dates
            plt.gcf().autofmt_xdate()
            
            # Add labels and title
            title = f'Water Surface Area versus Time\n(Using {water_band}, threshold={water_threshold}'
            if use_log10:
                title += ', log10 transformed)'
            else:
                title += ')'
            plt.title(title, fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Water Surface Area (square kilometers)', fontsize=12)
            
            # Add grid lines
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add data labels
            for i, (date, area) in enumerate(zip(dates, water_areas)):
                plt.annotate(f'{area:.1f}', 
                             (date, area),
                             textcoords="offset points", 
                             xytext=(0, 10), 
                             ha='center')
            
            plt.tight_layout()
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_water_timeseries.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            results['visualizations'].append('water_timeseries')
        
        # =================================================================
        # STEP 13: FLOOD DETECTION ANALYSIS
        # =================================================================
        if flood_analysis and len(dataset.time) > 1:
            if verbose:
                print("\nPerforming flood detection analysis...")
            
            # Determine acquisition indices for flood detection
            if first_acq_ind is None:
                first_acq_ind = 0
            if second_acq_ind is None:
                second_acq_ind = len(dataset.time) - 1
            
            # Validate indices
            if first_acq_ind < 0 or first_acq_ind >= len(dataset.time):
                if verbose:
                    print(f"Invalid first_acq_ind: {first_acq_ind}. Skipping flood detection.")
                flood_analysis = False
            elif second_acq_ind < 0 or second_acq_ind >= len(dataset.time):
                if verbose:
                    print(f"Invalid second_acq_ind: {second_acq_ind}. Skipping flood detection.")
                flood_analysis = False
            elif first_acq_ind == second_acq_ind:
                if verbose:
                    print("First and second acquisition indices are the same. Skipping flood detection.")
                flood_analysis = False
            
            if flood_analysis:
                # Get the acquisition dates
                first_date = np.datetime_as_string(dataset.time.values[first_acq_ind], unit='D')
                second_date = np.datetime_as_string(dataset.time.values[second_acq_ind], unit='D')
                
                transform_note = " (log10 transformed)" if use_log10 else ""
                if verbose:
                    print(f"Analyzing flooding between {first_date} (index {first_acq_ind}) and {second_date} (index {second_acq_ind})")
                    print(f"Using {water_band}{transform_note} with change threshold of {change_threshold}")
                    print(f"Using water threshold of {water_threshold}{transform_note}")
                
                # Extract the two acquisitions
                first_acq = dataset.isel(time=first_acq_ind)
                second_acq = dataset.isel(time=second_acq_ind)
                
                # Get the data for both acquisitions
                first_data = first_acq[water_band]
                second_data = second_acq[water_band]
                
                # Apply log10 transformation if requested
                if use_log10:
                    first_values = first_data.values
                    second_values = second_data.values
                    
                    # Handle negative values and zeros
                    first_values = np.where((first_values <= 0) | (~np.isfinite(first_values)), 1e-10, first_values)
                    second_values = np.where((second_values <= 0) | (~np.isfinite(second_values)), 1e-10, second_values)
                    
                    first_log = np.log10(first_values)
                    second_log = np.log10(second_values)
                    
                    # Handle any remaining infinite or NaN values
                    first_log = np.where(~np.isfinite(first_log), -10, first_log)
                    second_log = np.where(~np.isfinite(second_log), -10, second_log)
                    
                    # Calculate the difference in log space
                    change_values = second_log - first_log
                    
                    # Create water mask for the first acquisition
                    baseline_water_mask = first_log < water_threshold
                    
                else:
                    # Original approach without log10
                    change_values = second_data.values - first_data.values
                    first_values = first_data.values
                    first_values = np.nan_to_num(first_values, nan=0)
                    baseline_water_mask = first_values < water_threshold
                
                # Handle NaN values
                change_values = np.nan_to_num(change_values, nan=0)
                
                # Find pixels with significant backscatter decrease (potential flooding)
                flooding_mask = change_values < change_threshold
                
                # Store flood detection results
                results['flood_results'] = {
                    'flooding_mask': flooding_mask,
                    'baseline_water_mask': baseline_water_mask,
                    'change_array': change_values,
                    'first_date': first_date,
                    'second_date': second_date,
                    'first_acq_ind': first_acq_ind,
                    'second_acq_ind': second_acq_ind
                }
                
                # Calculate flooding statistics
                total_pixels = flooding_mask.size
                flooded_pixels = np.sum(flooding_mask)
                water_pixels = np.sum(baseline_water_mask)
                flooded_percent = (flooded_pixels / total_pixels) * 100
                water_percent = (water_pixels / total_pixels) * 100
                
                # Calculate areas
                flooded_area_sq_km = flooded_pixels * 100 / 1_000_000
                water_area_sq_km = water_pixels * 100 / 1_000_000
                
                results['statistics'].update({
                    'flood_pixels': flooded_pixels,
                    'flood_percent': flooded_percent,
                    'flood_area_km2': flooded_area_sq_km,
                    'baseline_water_pixels': water_pixels,
                    'baseline_water_percent': water_percent,
                    'baseline_water_area_km2': water_area_sq_km
                })
                
                if verbose:
                    print("\nFlooding Analysis Results:")
                    print(f"Total pixels in the scene: {total_pixels}")
                    print(f"Pixels identified as new flooding: {flooded_pixels} ({flooded_percent:.2f}%)")
                    print(f"Pixels identified as existing water: {water_pixels} ({water_percent:.2f}%)")
                    print(f"Estimated area of new flooding: {flooded_area_sq_km:.2f} sq km")
                    print(f"Estimated area of existing water: {water_area_sq_km:.2f} sq km")
                
                # Create flooding visualization
                if show_plots or save_outputs:
                    if verbose:
                        print("\nCreating flood detection visualization...")
                    
                    plt.figure(figsize=(12, 12))
                    
                    # Get VH band data for background
                    vh_data = np.array(first_acq['vh'].values, dtype=np.float64)
                    vh_filled = np.nan_to_num(vh_data, nan=np.nanmean(vh_data[~np.isnan(vh_data)]))
                    
                    # Apply contrast stretching
                    non_nan = vh_filled[np.isfinite(vh_filled)]
                    if len(non_nan) > 0:
                        low, high = np.percentile(non_nan, [2, 98])
                        vh_filled = np.clip((vh_filled - low) / (high - low), 0, 1)
                    
                    # Apply minimum intensity
                    min_inten = 0.6
                    intensity = vh_filled
                    intensity_safe = np.maximum(intensity, 1e-10)
                    multiplier = np.maximum(min_inten, intensity_safe) / intensity_safe
                    vh_filled = np.minimum(vh_filled * multiplier, 1.0)
                    
                    # Create RGB from single band
                    rgb_data = np.dstack([vh_filled, vh_filled, vh_filled])
                    
                    # Plot the background
                    plt.imshow(rgb_data, cmap='gray')
                    
                    # Ensure masks have correct shape
                    expected_shape = rgb_data.shape[:2]
                    
                    # Add flooding mask overlay (red)
                    if flooding_mask.shape == expected_shape:
                        flood_overlay = np.zeros((flooding_mask.shape[0], flooding_mask.shape[1], 4))
                        flood_overlay[:,:,0] = 1.0  # Red channel
                        flood_overlay[:,:,3] = flooding_mask.astype(bool) * 0.7
                        plt.imshow(flood_overlay)
                    
                    # Add baseline water mask overlay (blue)
                    if baseline_water_mask.shape == expected_shape:
                        water_overlay = np.zeros((baseline_water_mask.shape[0], baseline_water_mask.shape[1], 4))
                        water_overlay[:,:,2] = 1.0  # Blue channel
                        water_overlay[:,:,3] = baseline_water_mask.astype(bool) * 0.7
                        plt.imshow(water_overlay)
                    
                    plt.title('Multi-Date Flooding Detection\nRED: New flooding, BLUE: Existing water', fontsize=14)
                    plt.axis('off')
                    
                    if save_outputs:
                        plt.savefig(f"{output_prefix}_flood_detection.png", dpi=300, bbox_inches='tight')
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                    
                    results['visualizations'].append('flood_detection')
        
        # =================================================================
        # STEP 14: CREATE FACETED PLOTS
        # =================================================================
        if (show_plots or save_outputs) and len(dataset.time) > 1:
            if verbose:
                print("\nCreating faceted plots...")
            
            # Create faceted plot for VH band
            max_cols = 3
            n_times = len(dataset.time)
            n_cols = min(max_cols, n_times)
            n_rows = (n_times + n_cols - 1) // n_cols
            
            # VH Band faceted plot
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            
            if n_times == 1:
                axes = np.array([axes])
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            
            for i, time_value in enumerate(dataset.time.values):
                if i < len(axes):
                    date_str = np.datetime_as_string(time_value, unit='D')
                    
                    try:
                        scene = dataset.isel(time=i)
                        vh_data = np.array(scene['vh'].values, dtype=np.float64)
                        vh_data[~np.isfinite(vh_data)] = np.nan
                        
                        # Apply contrast stretching
                        non_nan = vh_data[~np.isnan(vh_data)]
                        if len(non_nan) > 0:
                            low, high = np.percentile(non_nan, [2, 98])
                            vh_data = np.clip((vh_data - low) / (high - low), 0, 1)
                        
                        vh_data = np.nan_to_num(vh_data)
                        
                        # Apply minimum intensity
                        min_inten = 0.6
                        intensity = vh_data
                        intensity_safe = np.maximum(intensity, 1e-10)
                        multiplier = np.maximum(min_inten, intensity_safe) / intensity_safe
                        vh_data = np.minimum(vh_data * multiplier, 1.0)
                        
                        axes[i].imshow(vh_data, cmap='gray')
                        axes[i].set_title(f"{date_str}", fontsize=12)
                        axes[i].axis('off')
                    except Exception as e:
                        if verbose:
                            print(f"Error plotting time slice {i} ({date_str}): {e}")
                        axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform=axes[i].transAxes)
            
            # Hide unused subplots
            for j in range(n_times, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle('Sentinel-1 VH Band Time Series', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_vh_facet.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            results['visualizations'].append('vh_facet')
        
        # =================================================================
        # STEP 15: CREATE THRESHOLD DISTRIBUTION VISUALIZATION
        # =================================================================
        if (show_plots or save_outputs) and verbose:
            if verbose:
                print("\nCreating threshold distribution visualization...")
            
            data = dataset.isel(time=0)[water_band].values
            
            # Apply log10 transformation if requested
            if use_log10:
                # Handle negative values and zeros
                data = np.where((data <= 0) | (~np.isfinite(data)), 1e-10, data)
                data = np.log10(data)
                # Handle any remaining infinite or NaN values
                data = np.where(~np.isfinite(data), -10, data)
                xlabel = f"{water_band} values (log10 transformed)"
                title_suffix = " (log10 transformed)"
            else:
                xlabel = f"{water_band} values (dB)"
                title_suffix = ""
            
            # Remove infinite and NaN values
            data = data[np.isfinite(data)]
            
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=200, color='gray', alpha=0.8, label=f"{water_band} values")
            plt.axvline(water_threshold, color='blue', linestyle='--', 
                        label=f"Water Threshold ({water_threshold}{' log10' if use_log10 else ' dB'})")
            if flood_analysis:
                plt.axvline(change_threshold, color='red', linestyle='--', 
                           label=f"Flood Threshold ({change_threshold}{' log10' if use_log10 else ' dB'})")
            
            plt.title(f"Histogram of {water_band} at time index 0{title_suffix}")
            plt.xlabel(xlabel)
            plt.ylabel("Pixel Count")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            if save_outputs:
                plt.savefig(f"{output_prefix}_threshold_distribution.png", dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            results['visualizations'].append('threshold_distribution')
        
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
            print(f"Visualizations created: {results['visualizations']}")
            
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
    # Define area of interest
    geometry = Polygon([
        (-92.695, 17.813),
        (-92.495, 17.813),
        (-92.495, 17.613),
        (-92.695, 17.613),
        (-92.695, 17.813)
    ])
    
    # Set date range
    start_date = datetime.datetime(2020, 9, 1)
    end_date = datetime.datetime(2021, 1, 1)
    
    # Run complete analysis
    results = process_sentinel1_water_extent_complete(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        measurements=["vv", "vh"],
        resolution=10,
        platform=None,
        orbit_direction=None,
        filter_type='mean',
        filter_size=5,
        water_threshold=-2.0,
        persistence_threshold=0.8,
        use_log10=True,
        flood_analysis=True,
        first_acq_ind=None,  # Will use first acquisition
        second_acq_ind=None,  # Will use last acquisition
        change_threshold=-0.5,
        save_outputs=True,
        output_prefix="sentinel1_analysis",
        show_plots=True,
        verbose=True,
        grouping_criteria="platform_orbit_relative",
        min_acquisitions=2
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
    print(f"✓ Visualizations created: {len(results['visualizations'])} plots")
