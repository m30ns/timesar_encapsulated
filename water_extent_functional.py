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

# Function to get the STAC catalog
def get_catalogue():
    """
    Get the STAC catalog client for Planetary Computer.
    Returns:
        pystac_client.Client: A STAC client for the Planetary Computer catalog.
    """
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return client

# Function to load Sentinel-1 data
def load_sentinel1(
    geometry: shapely.geometry.Polygon,
    measurements: List[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    resolution: int = 10,
    platform: Optional[str] = None,
    orbit_direction: Optional[str] = None,
    instrument_mode: str = "IW",
    product_type: str = "GRD"
) -> xr.Dataset:
    """
    Uses odc.stac.load to retrieve raw Sentinel-1 SAR data for a given geometry, time range, and band list.
    
    Args:
        geometry: AOI polygon
        measurements: List of bands to retrieve (e.g., ["vv", "vh"])
        start_date: Start date for the query
        end_date: End date for the query
        resolution: Spatial resolution in meters
        platform: Satellite platform ('sentinel-1a', 'sentinel-1b', or None for all S1 satellites)
        orbit_direction: Orbit direction ('ascending' or 'descending')
        instrument_mode: SAR instrument mode
        product_type: Product type
        
    Returns:
        xr.Dataset: Dataset containing the requested SAR data
    """
    # Get the STAC catalog
    catalog = get_catalogue()
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
    search = catalog.search(
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
    
    print(f"Found {len(items)} Sentinel-1 scenes")
    
    # Extract and print metadata information about platform and orbit
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
    try:
        print("Computing dataset...")
        loaded_dataset = dataset.compute()
        print(f"Dataset loaded successfully with {len(loaded_dataset.time)} time steps")
        print(f"Available variables: {list(loaded_dataset.data_vars)}")
        
        # Make sure metadata is transferred to computed dataset
        if hasattr(loaded_dataset, 'attrs') and 'metadata' not in loaded_dataset.attrs:
            loaded_dataset.attrs['metadata'] = metadata_list
            
        return loaded_dataset
    except Exception as e:
        print(f"Error computing dataset: {e}")
        # Return the dataset without computing as a fallback
        print("Returning dataset without computing as a fallback")
        return dataset

# Function to filter dataset by platform and orbit direction
def filter_dataset_by_metadata(
    dataset: xr.Dataset,
    platform: Optional[str] = None,
    orbit_direction: Optional[str] = None
) -> xr.Dataset:
    """
    Filters a dataset based on platform and orbit direction metadata.
    
    Args:
        dataset: Input xarray Dataset
        platform: Platform to filter by (e.g., 'sentinel-1a', 'sentinel-1b')
        orbit_direction: Orbit direction to filter by ('ascending' or 'descending')
        
    Returns:
        xr.Dataset: Filtered dataset
    """
    if not hasattr(dataset, 'attrs') or 'metadata' not in dataset.attrs:
        print("Warning: Dataset does not have metadata attributes for filtering")
        return dataset
    
    # Get metadata list
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
    filtered_dataset = dataset.isel(time=keep_indices)
    
    # Update metadata in filtered dataset
    filtered_metadata = [metadata_list[i] for i in keep_indices]
    filtered_dataset.attrs['metadata'] = filtered_metadata
    
    print(f"Filtered dataset to {len(filtered_dataset.time)} time steps")
    return filtered_dataset

# UPDATED ORBIT GROUPING FUNCTIONS WITH FIXES

def group_acquisitions_by_orbit(dataset: xr.Dataset, 
                               grouping_criteria: str = "platform_orbit_relative") -> Dict[str, xr.Dataset]:
    """
    Groups dataset acquisitions by orbit configuration with improved error handling.
    
    Args:
        dataset: Input xarray Dataset with metadata
        grouping_criteria: How to group acquisitions:
            - "platform_orbit_relative": Groups by platform, orbit_state, and relative_orbit (default)
            - "orbit_relative": Groups by orbit_state and relative_orbit only
            - "relative_only": Groups by relative_orbit only
            - "platform_orbit": Groups by platform and orbit_state only
    
    Returns:
        Dict[str, xr.Dataset]: Dictionary of grouped datasets with key as orbit signature.
    """
    if not hasattr(dataset, 'attrs') or 'metadata' not in dataset.attrs:
        raise ValueError("Dataset missing metadata for grouping by orbit")
    
    metadata_list = dataset.attrs['metadata']
    n_times = len(dataset.time)
    n_metadata = len(metadata_list)
    
    print(f"Dataset has {n_times} time steps and {n_metadata} metadata entries")
    
    # Handle mismatch between time steps and metadata
    if n_metadata != n_times:
        print(f"Warning: Metadata count ({n_metadata}) doesn't match time count ({n_times})")
        # Use the minimum to avoid index errors
        max_index = min(n_times, n_metadata)
        print(f"Will only process first {max_index} entries")
    else:
        max_index = n_times
    
    grouped_indices = defaultdict(list)
    
    # Group acquisitions based on the selected criteria
    for i in range(max_index):
        try:
            meta = metadata_list[i]
            
            # Create grouping key based on criteria
            if grouping_criteria == "platform_orbit_relative":
                key = f"{meta.get('platform', 'unknown')}_{meta.get('orbit_state', 'unknown')}_{meta.get('relative_orbit', 'unknown')}"
            elif grouping_criteria == "orbit_relative":
                key = f"{meta.get('orbit_state', 'unknown')}_{meta.get('relative_orbit', 'unknown')}"
            elif grouping_criteria == "relative_only":
                key = f"rel_{meta.get('relative_orbit', 'unknown')}"
            elif grouping_criteria == "platform_orbit":
                key = f"{meta.get('platform', 'unknown')}_{meta.get('orbit_state', 'unknown')}"
            else:
                raise ValueError(f"Unknown grouping criteria: {grouping_criteria}")
                
            grouped_indices[key].append(i)
            
        except (IndexError, KeyError) as e:
            print(f"Error processing metadata at index {i}: {e}")
            continue
    
    # Create datasets for groups with multiple acquisitions
    grouped_datasets = {}
    for key, indices in grouped_indices.items():
        if len(indices) > 1:  # Only keep groups with multiple acquisitions
            print(f"Group '{key}': {len(indices)} acquisitions at indices {indices}")
            try:
                # Ensure all indices are valid
                valid_indices = [idx for idx in indices if idx < n_times]
                if len(valid_indices) > 1:
                    subset = dataset.isel(time=valid_indices)
                    # Update metadata to match the subset
                    subset_metadata = [metadata_list[i] for i in valid_indices if i < n_metadata]
                    subset.attrs['metadata'] = subset_metadata
                    grouped_datasets[key] = subset
                else:
                    print(f"Skipping group '{key}': not enough valid indices")
            except Exception as e:
                print(f"Error creating subset for group '{key}': {e}")
                continue
        else:
            print(f"Skipping group '{key}': only {len(indices)} acquisition")
    
    print(f"Found {len(grouped_datasets)} valid orbit groups with multiple acquisitions")
    return grouped_datasets

def select_most_common_orbit_group(dataset: xr.Dataset, 
                                  grouping_criteria: str = "platform_orbit_relative",
                                  min_acquisitions: int = 2) -> Tuple[str, xr.Dataset]:
    """
    Select the orbit group with the most acquisitions from the dataset.
    
    Args:
        dataset: xarray Dataset with metadata
        grouping_criteria: How to group acquisitions (see group_acquisitions_by_orbit)
        min_acquisitions: Minimum number of acquisitions required for a group
    
    Returns:
        Tuple[str, xr.Dataset]: Key and dataset of the group with most acquisitions
    """
    groups = group_acquisitions_by_orbit(dataset, grouping_criteria)
    
    if not groups:
        raise ValueError(f"No orbit groups found with at least {min_acquisitions} acquisitions using criteria '{grouping_criteria}'")
    
    # Filter groups by minimum acquisitions
    valid_groups = {k: v for k, v in groups.items() if len(v.time) >= min_acquisitions}
    
    if not valid_groups:
        raise ValueError(f"No orbit groups found with at least {min_acquisitions} acquisitions")
    
    # Sort by number of acquisitions (descending)
    sorted_groups = sorted(valid_groups.items(), key=lambda kv: len(kv[1].time), reverse=True)
    best_key, best_dataset = sorted_groups[0]
    
    print(f"Selected orbit group '{best_key}' with {len(best_dataset.time)} acquisitions")
    return best_key, best_dataset

def display_orbit_groups(dataset: xr.Dataset, 
                        grouping_criteria: str = "platform_orbit_relative") -> None:
    """
    Display all available orbit groups and their acquisition counts.
    
    Args:
        dataset: xarray Dataset with metadata
        grouping_criteria: How to group acquisitions
    """
    print(f"\nOrbit groups using criteria '{grouping_criteria}':")
    print("-" * 80)
    
    try:
        groups = group_acquisitions_by_orbit(dataset, grouping_criteria)
        
        if not groups:
            print("No orbit groups with multiple acquisitions found.")
            return
        
        # Sort by number of acquisitions
        sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1].time), reverse=True)
        
        for i, (key, group_dataset) in enumerate(sorted_groups):
            n_acq = len(group_dataset.time)
            print(f"{i+1}. {key}: {n_acq} acquisitions")
            
            # Show dates for this group
            dates = [np.datetime_as_string(t, unit='D') for t in group_dataset.time.values]
            print(f"   Dates: {', '.join(dates)}")
            print()
            
    except Exception as e:
        print(f"Error displaying orbit groups: {e}")

def select_orbit_group_interactively(dataset: xr.Dataset,
                                   grouping_criteria: str = "platform_orbit_relative") -> Tuple[str, xr.Dataset]:
    """
    Allow user to interactively select an orbit group.
    
    Args:
        dataset: xarray Dataset with metadata
        grouping_criteria: How to group acquisitions
        
    Returns:
        Tuple[str, xr.Dataset]: Selected group key and dataset
    """
    groups = group_acquisitions_by_orbit(dataset, grouping_criteria)
    
    if not groups:
        raise ValueError("No orbit groups with multiple acquisitions found")
    
    # Display options
    display_orbit_groups(dataset, grouping_criteria)
    
    # Get user selection
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1].time), reverse=True)
    
    while True:
        try:
            choice = input(f"Select orbit group (1-{len(sorted_groups)}) or press Enter for the largest group: ")
            
            if choice.strip() == "":
                # Default to largest group
                selected_key, selected_dataset = sorted_groups[0]
                break
            else:
                choice_num = int(choice) - 1
                if 0 <= choice_num < len(sorted_groups):
                    selected_key, selected_dataset = sorted_groups[choice_num]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(sorted_groups)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"Selected: {selected_key} with {len(selected_dataset.time)} acquisitions")
    return selected_key, selected_dataset

def try_different_grouping_strategies(dataset: xr.Dataset) -> Tuple[str, xr.Dataset]:
    """
    Try different grouping strategies to find the best one for the dataset.
    
    Args:
        dataset: xarray Dataset with metadata
        
    Returns:
        Tuple[str, xr.Dataset]: Best group key and dataset found
    """
    strategies = [
        "platform_orbit_relative",
        "orbit_relative", 
        "relative_only",
        "platform_orbit"
    ]
    
    print("Trying different grouping strategies...")
    
    best_result = None
    best_count = 0
    best_strategy = None
    
    for strategy in strategies:
        try:
            print(f"\nTrying strategy: {strategy}")
            groups = group_acquisitions_by_orbit(dataset, strategy)
            
            if groups:
                # Find the group with most acquisitions
                best_group = max(groups.items(), key=lambda kv: len(kv[1].time))
                group_count = len(best_group[1].time)
                
                print(f"  Best group: {best_group[0]} with {group_count} acquisitions")
                
                if group_count > best_count:
                    best_result = best_group
                    best_count = group_count
                    best_strategy = strategy
            else:
                print(f"  No valid groups found")
                
        except Exception as e:
            print(f"  Error with strategy {strategy}: {e}")
    
    if best_result is None:
        raise ValueError("No valid orbit groups found with any strategy")
    
    print(f"\nBest strategy: {best_strategy}")
    print(f"Selected group: {best_result[0]} with {best_count} acquisitions")
    
    return best_result

def robust_orbit_selection(dataset: xr.Dataset, 
                          preferred_strategy: str = "platform_orbit_relative",
                          min_acquisitions: int = 2) -> Tuple[str, xr.Dataset]:
    """
    Robust orbit selection that tries multiple strategies if the preferred one fails.
    
    Args:
        dataset: xarray Dataset with metadata
        preferred_strategy: Preferred grouping strategy
        min_acquisitions: Minimum number of acquisitions required
        
    Returns:
        Tuple[str, xr.Dataset]: Selected group key and dataset
    """
    try:
        # Try preferred strategy first
        print(f"Trying preferred strategy: {preferred_strategy}")
        return select_most_common_orbit_group(dataset, preferred_strategy, min_acquisitions)
        
    except ValueError as e:
        print(f"Preferred strategy failed: {e}")
        print("Trying alternative strategies...")
        
        # Try different strategies
        return try_different_grouping_strategies(dataset)

# Define conversion functions for SAR data
def to_pwr(x):
    """Convert from dB to power."""
    return 10**(x/10)

def to_db(x):
    """Convert from power to dB."""
    return 10*np.log10(x)

def fill_null_values(dataset: xr.Dataset, fill_value: float = 0) -> xr.Dataset:
    """
    Fill null values in a dataset with a specified value.
    
    Args:
        dataset: Input xarray Dataset
        fill_value: Value to use for null values
        
    Returns:
        xr.Dataset: Dataset with null values filled
    """
    return dataset.where(~dataset.isnull(), fill_value)

# Define the 2D statistics filter function for speckle reduction
def stats_filter_2d(data_array, statistic='mean', filter_size=5):
    """
    Apply a statistical filter to a 2D xarray.DataArray to reduce speckle noise.
    
    Args:
        data_array (xarray.DataArray): Input data array
        statistic (str): Statistical operation ('mean', 'median', 'min', 'max')
        filter_size (int): Size of the filter window (must be odd)
    
    Returns:
        xarray.DataArray: Filtered data array
    """
    import scipy.ndimage as ndimage
    
    # Ensure filter size is odd
    if filter_size % 2 == 0:
        filter_size += 1
    
    # Select the appropriate statistical function
    if statistic == 'mean':
        filtered_data = ndimage.uniform_filter(data_array.values, size=filter_size)
    elif statistic == 'median':
        filtered_data = ndimage.median_filter(data_array.values, size=filter_size)
    elif statistic == 'min':
        filtered_data = ndimage.minimum_filter(data_array.values, size=filter_size)
    elif statistic == 'max':
        filtered_data = ndimage.maximum_filter(data_array.values, size=filter_size)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    
    # Create a new DataArray with the same coordinates and attributes
    result = xr.DataArray(
        filtered_data,
        coords=data_array.coords,
        dims=data_array.dims,
        attrs=data_array.attrs
    )
    
    return result

def apply_speckle_filter(dataset: xr.Dataset, bands: List[str], 
                         filter_type: str = 'mean', filter_size: int = 5) -> xr.Dataset:
    """
    Apply speckle filtering to specified bands in a dataset.
    
    Args:
        dataset: Input xarray Dataset
        bands: List of bands to filter
        filter_type: Type of filter ('mean', 'median', 'min', 'max')
        filter_size: Size of the filter window
        
    Returns:
        xr.Dataset: Dataset with filtered bands added
    """
    print(f"Applying {filter_type} filter with size {filter_size} to bands: {bands}")
    
    # First fill null values to prevent issues with filtering
    dataset_filled = fill_null_values(dataset)
    
    # Create a new dataset with original data
    result_dataset = dataset.copy()
    
    # Apply filter to each band
    for band in bands:
        print(f"Filtering band: {band}")
        try:
            filtered_band_name = f"block_{filter_type}_filter_{band}"
            
            # Print some diagnostics about the band
            print(f"Band shape: {dataset_filled[band].shape}")
            print(f"Band has NaN: {np.isnan(dataset_filled[band].values).any()}")
            
            # Apply filtering using pipe operations
            filtered_band = dataset_filled[band].pipe(to_pwr)
            
            # Apply filtering to each time step separately
            filtered_data = []
            for i in range(len(dataset.time)):
                print(f"  Processing time step {i}")
                time_slice = filtered_band.isel(time=i)
                filtered_slice = stats_filter_2d(time_slice, 
                                               statistic=filter_type, 
                                               filter_size=filter_size)
                filtered_data.append(filtered_slice)
            
            # Stack the filtered time slices back together
            if len(filtered_data) > 1:
                # Create a new DataArray with the filtered data
                coords = dataset[band].coords
                filtered_stack = xr.concat(filtered_data, dim='time')
                filtered_stack = filtered_stack.assign_coords(time=dataset.time)
            else:
                # Single time step case
                filtered_stack = xr.DataArray(
                    filtered_data[0],
                    coords=dataset.isel(time=0)[band].coords,
                    dims=dataset.isel(time=0)[band].dims
                )
                filtered_stack = filtered_stack.expand_dims(dim='time')
                filtered_stack = filtered_stack.assign_coords(time=dataset.time)
            
            # Convert back to dB
            result_dataset[filtered_band_name] = filtered_stack.pipe(to_db)
            print(f"Successfully created filtered band: {filtered_band_name}")
            
        except Exception as e:
            print(f"Error filtering band {band}: {e}")
            import traceback
            traceback.print_exc()
    
    return result_dataset

def calculate_backscatter_amplitudes(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate backscatter amplitudes for visualization from filtered bands.
    
    Args:
        dataset: Input xarray Dataset containing filtered bands
        
    Returns:
        xr.Dataset: Dataset with amplitude bands added
    """
    # Create a new dataset with original data
    result_dataset = dataset.copy()
    
    # Apply backscatter scaling optimized for block-filtered data
    # VV band range is 0dB to -16dB which is DN=1.00 to DN=0.158
    # VH band range is -5dB to -27dB which is DN=0.562 to DN=0.045
    # VV/VH range is 0.0 to 1.0. This data is scaled by 20 for improved color contrast
    vv_convert = (10**(dataset.block_mean_filter_vv/20)-0.158)*303
    vh_convert = (10**(dataset.block_mean_filter_vh/20)-0.045)*493
    
    result_dataset['vv_amp'] = vv_convert
    result_dataset['vh_amp'] = vh_convert
    result_dataset['vvvh_amp'] = (vv_convert / vh_convert) * 20
    
    return result_dataset

# Updated RGB visualization function with improved error handling
def rgb(dataset, bands, width=10, fig=None, ax=None, percentile_stretch=(2, 98), 
        paint_on_mask=None, min_inten=None):
    """
    Create an RGB plot from three bands in a dataset with optional mask overlay.
    UPDATED to match debugging code approach.
    
    Args:
        dataset (xarray.Dataset): Dataset containing the bands to plot
        bands (list): List of three band names to plot as RGB
        width (int): Width of the plot in inches
        fig (matplotlib.figure.Figure): Optional existing figure to plot on
        ax (matplotlib.axes.Axes): Optional existing axes to plot on
        percentile_stretch (tuple): Lower and upper percentiles for contrast stretching
        paint_on_mask (list): List of tuples (mask, color) for overlaying masks
        min_inten (float): Minimum intensity multiplier for dark areas
        
    Returns:
        matplotlib.figure.Figure: The figure containing the RGB plot
    """
    # Get the three bands and handle potential issues
    try:
        r = np.array(dataset[bands[0]].values, dtype=np.float64)
        g = np.array(dataset[bands[1]].values, dtype=np.float64)
        b = np.array(dataset[bands[2]].values, dtype=np.float64)
    except Exception as e:
        print(f"Error accessing bands: {e}")
        print(f"Available bands: {list(dataset.data_vars)}")
        # Try to recover if possible by using the first band for all channels
        if bands[0] in dataset:
            r = g = b = np.array(dataset[bands[0]].values, dtype=np.float64)
        else:
            # Find any band we can use
            available_band = list(dataset.data_vars)[0]
            print(f"Using {available_band} for all channels as fallback")
            r = g = b = np.array(dataset[available_band].values, dtype=np.float64)
    
    # Replace infinity values with NaN
    r[~np.isfinite(r)] = np.nan
    g[~np.isfinite(g)] = np.nan
    b[~np.isfinite(b)] = np.nan
    
    # Stack the bands into an RGB array
    rgb_data = np.dstack((r, g, b))
    
    # Apply contrast stretching to each band
    p_low, p_high = percentile_stretch
    for i in range(3):
        band = rgb_data[:,:,i]
        non_nan = band[~np.isnan(band)]
        if len(non_nan) > 0:  # Check if band has non-NaN values
            low, high = np.percentile(non_nan, [p_low, p_high])
            rgb_data[:,:,i] = np.clip((band - low) / (high - low), 0, 1)
    
    # Replace NaN values with zeros
    rgb_data = np.nan_to_num(rgb_data)
    
    # Apply minimum intensity if specified
    if min_inten is not None:
        # Calculate intensity as average of RGB channels
        intensity = np.mean(rgb_data, axis=2)
        # Create a multiplier that increases darker areas
        # Avoid division by zero
        intensity_safe = np.maximum(intensity, 1e-10)
        multiplier = np.maximum(min_inten, intensity_safe) / intensity_safe
        # Apply multiplier to each channel
        for i in range(3):
            rgb_data[:,:,i] = np.minimum(rgb_data[:,:,i] * multiplier, 1.0)
    
    # Create the plot if axes not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(width, width * r.shape[0] / r.shape[1]))
    
    # Plot the RGB image
    ax.imshow(rgb_data, cmap='gray' if bands[0] == bands[1] == bands[2] else None)
    
    # Apply masks if provided
    if paint_on_mask is not None:
        for mask, color in paint_on_mask:
            # Ensure mask has the right shape
            if mask.shape != r.shape:
                print(f"Warning: Mask shape {mask.shape} doesn't match image shape {r.shape}")
                continue
                
            try:
                # Create a color overlay with the mask
                mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
                for i in range(3):
                    mask_overlay[:,:,i] = color[i] / 255.0
                mask_overlay[:,:,3] = mask * 0.7  # Alpha channel (transparency)
                
                # Display the mask overlay
                ax.imshow(mask_overlay)
                print(f"Applied mask overlay with color {color}")
            except Exception as e:
                print(f"Error applying mask: {e}")
    
    return fig

# UPDATED water mask functions based on debugging code
def create_water_mask(dataset: xr.Dataset, band: str, threshold: float, 
                      time_index: Optional[int] = None, use_log10: bool = True) -> np.ndarray:
    """
    Create a water mask using a threshold on the specified band.
    UPDATED with improved log10 transformation handling.
    
    Args:
        dataset: Input xarray Dataset
        band: Band name to use for thresholding
        threshold: Threshold value (values below this are classified as water)
        time_index: Time index to use (None for all time slices)
        use_log10: Whether to apply log10 transformation (matches debugging approach)
        
    Returns:
        np.ndarray: Boolean mask where True indicates water
    """
    if time_index is not None:
        scene = dataset.isel(time=time_index)
        data = scene[band]
    else:
        data = dataset[band]
    
    # Apply log10 transformation if requested (matches debugging code)
    if use_log10:
        # Handle negative values and zeros before log10 using safe transformation
        data_values = data.values
        # Replace zero, negative, and infinite values with small positive number
        data_values = np.where((data_values <= 0) | (~np.isfinite(data_values)), 1e-10, data_values)
        data_transformed = np.log10(data_values)
        # Handle any remaining infinite or NaN values
        data_transformed = np.where(~np.isfinite(data_transformed), -10, data_transformed)
        water_mask = data_transformed < threshold
    else:
        data_values = data.values
        data_values = np.where(~np.isfinite(data_values), 0, data_values)
        water_mask = data_values < threshold
    
    return water_mask

def create_combined_water_mask(dataset: xr.Dataset, vv_band: str = 'vv', vh_band: str = 'vh', 
                              threshold: float = 4.05, time_index: Optional[int] = None) -> np.ndarray:
    """
    Create a water mask using the combined VV*VH approach with improved error handling.
    This matches: (np.log10(vv) * np.log10(vh)) > 4.05
    
    Args:
        dataset: Input xarray Dataset
        vv_band: VV band name
        vh_band: VH band name  
        threshold: Threshold for the combined product
        time_index: Time index to use (None for all time slices)
        
    Returns:
        np.ndarray: Boolean mask where True indicates water
    """
    if time_index is not None:
        scene = dataset.isel(time=time_index)
        vv_data = scene[vv_band]
        vh_data = scene[vh_band]
    else:
        vv_data = dataset[vv_band]
        vh_data = dataset[vh_band]
    
    # Apply log10 transformation and handle negative/zero values safely
    vv_values = vv_data.values
    vh_values = vh_data.values
    
    # Safe handling of problematic values
    vv_values = np.where((vv_values <= 0) | (~np.isfinite(vv_values)), 1e-10, vv_values)
    vh_values = np.where((vh_values <= 0) | (~np.isfinite(vh_values)), 1e-10, vh_values)
    
    # Calculate log10 safely
    vv_log = np.log10(vv_values)
    vh_log = np.log10(vh_values)
    
    # Handle any remaining infinite or NaN values
    vv_log = np.where(~np.isfinite(vv_log), -10, vv_log)
    vh_log = np.where(~np.isfinite(vh_log), -10, vh_log)
    
    # Calculate the product as in debugging code
    log_product = vv_log * vh_log
    water_mask = log_product > threshold
    
    return water_mask

def create_persistent_water_mask(dataset: xr.Dataset, band: str = 'block_mean_filter_vh', 
                                threshold: float = -2.0, persistence_threshold: float = 0.8,
                                use_log10: bool = True) -> np.ndarray:
    """
    Create a persistent water mask based on water frequency across time.
    This matches the debugging approach: water_stats.where(water_stats > .8)
    
    Args:
        dataset: Input xarray Dataset with multiple time slices
        band: Band name to use for water detection
        threshold: Threshold for water detection in individual scenes
        persistence_threshold: Fraction of time a pixel must be water to be considered persistent water
        use_log10: Whether to apply log10 transformation
        
    Returns:
        np.ndarray: Boolean mask where True indicates persistent water
    """
    print(f"Creating persistent water mask using {band} with threshold {threshold}")
    print(f"Persistence threshold: {persistence_threshold}")
    
    # Create water masks for all time slices
    water_masks = []
    for i in range(len(dataset.time)):
        water_mask = create_water_mask(dataset, band, threshold, time_index=i, use_log10=use_log10)
        water_masks.append(water_mask)
    
    # Stack masks and calculate mean across time
    water_stack = np.stack(water_masks, axis=0)
    water_frequency = np.mean(water_stack, axis=0)
    
    # Create persistent water mask
    persistent_water = water_frequency > persistence_threshold
    
    print(f"Persistent water covers {np.sum(persistent_water)} pixels ({np.mean(persistent_water)*100:.2f}% of scene)")
    
    return persistent_water

# UPDATED water extent time series function with corrected calculation
def create_water_extent_timeseries(dataset: xr.Dataset, 
                                  threshold_variable: str = 'block_mean_filter_vh',
                                  water_threshold: float = -2.0,  # Updated default threshold
                                  pixel_size_m: float = 10.0,
                                  use_log10: bool = True,  # New parameter
                                  save_path: Optional[str] = None, 
                                  show: bool = True) -> None:
    """
    Create a time series plot of water surface area over time.
    UPDATED to use log10 transformation and corrected thresholds.
    
    Args:
        dataset: Input xarray Dataset with multiple time slices
        threshold_variable: Band name to use for water detection
        water_threshold: Threshold for water detection (default updated to -2.0 for log10 space)
        pixel_size_m: Pixel size in meters
        use_log10: Whether to apply log10 transformation to the data
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    # Check if the threshold variable exists
    if threshold_variable not in dataset:
        available_vars = list(dataset.data_vars)
        print(f"Warning: {threshold_variable} not found. Available variables: {available_vars}")
        # Try to use a fallback variable
        if 'vh' in dataset:
            threshold_variable = 'vh'
            print(f"Using 'vh' instead of {threshold_variable}")
        elif 'block_mean_filter_vh' in dataset:
            threshold_variable = 'block_mean_filter_vh'
            print(f"Using 'block_mean_filter_vh' as fallback")
        elif len(available_vars) > 0:
            threshold_variable = available_vars[0]
            print(f"Using '{threshold_variable}' as fallback")
        else:
            raise ValueError("No suitable variables found for water detection")

    print(f"Using threshold variable: {threshold_variable}")
    print(f"Water threshold: {water_threshold}")
    print(f"Using log10 transformation: {use_log10}")

    # Calculate water area for each time slice using the updated approach
    water_areas = []
    for i in range(len(dataset.time)):
        # Create water mask for this time slice
        water_mask = create_water_mask(dataset, threshold_variable, water_threshold, 
                                     time_index=i, use_log10=use_log10)
        
        # Count water pixels
        water_pixels = np.sum(water_mask)
        
        # Convert to area in square kilometers (matching the original calculation)
        water_area_sq_km = water_pixels * 100 / 1_000_000  # Matches colab: *100/1000/1000
        water_areas.append(water_area_sq_km)
        
        print(f"Time {i}: {water_pixels} water pixels = {water_area_sq_km:.2f} sq km")

    # Get the dates for the x-axis
    dates = [np.datetime64(dt) for dt in dataset.time.values]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(dates, water_areas, c='black', marker='o', mfc='blue', markersize=10, linewidth=1)

    # Format x-axis as dates
    plt.gcf().autofmt_xdate()

    # Add labels and title
    title = f'Water Surface Area versus Time\n(Using {threshold_variable}, threshold={water_threshold}'
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

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

# Updated visualization functions
def visualize_water_extent(dataset: xr.Dataset, water_mask: np.ndarray, time_index: int = 0,
                          save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a visualization of water extent using a water mask.
    
    Args:
        dataset: Input xarray Dataset
        water_mask: Boolean mask where True indicates water
        time_index: Time index to visualize
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    # Define color for water (BLUE)
    color_blue = np.array([0, 0, 255])
    
    # Create water extent visualization
    plt.figure(figsize=(12, 12))
    rgb(dataset.isel(time=time_index), bands=['vh', 'vh', 'vh'], 
        paint_on_mask=[(water_mask, color_blue)], width=8, min_inten=0.6)
    
    # Add metadata information to the title if available
    title = 'VH-Band Threshold Water Extent'
    if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
        metadata = dataset.attrs['metadata'][time_index]
        platform = metadata.get('platform', 'Unknown')
        orbit_state = metadata.get('orbit_state', 'Unknown')
        date_str = np.datetime_as_string(dataset.time.values[time_index], unit='D')
        title += f'\n{platform.upper()}, {orbit_state.capitalize()} orbit, {date_str}'
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_persistent_water(dataset: xr.Dataset, persistent_water_mask: np.ndarray,
                              reference_time_index: int = 0, save_path: Optional[str] = None, 
                              show: bool = True) -> None:
    """
    Create a visualization of persistent water areas.
    
    Args:
        dataset: Input xarray Dataset
        persistent_water_mask: Boolean mask where True indicates persistent water
        reference_time_index: Time index to use as background
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    # Define color for persistent water (CYAN)
    color_cyan = np.array([0, 255, 255])
    
    # Create persistent water visualization
    plt.figure(figsize=(12, 12))
    rgb(dataset.isel(time=reference_time_index), bands=['vh', 'vh', 'vh'], 
        paint_on_mask=[(persistent_water_mask, color_cyan)], width=8, min_inten=0.6)
    
    plt.title('Persistent Water Areas (>80% of time)', fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_backscatter(dataset: xr.Dataset, time_index: int = 0, 
                          save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create an RGB visualization of SAR backscatter.
    
    Args:
        dataset: Input xarray Dataset with amplitude bands
        time_index: Time index to visualize
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    plt.figure(figsize=(12, 12))
    rgb(dataset.isel(time=time_index), bands=['vv_amp', 'vh_amp', 'vvvh_amp'], width=10)
    
    # Add metadata information to the title if available
    title = 'Backscatter RGB: VV, VH, VV/VH'
    if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
        metadata = dataset.attrs['metadata'][time_index]
        platform = metadata.get('platform', 'Unknown')
        orbit_state = metadata.get('orbit_state', 'Unknown')
        date_str = np.datetime_as_string(dataset.time.values[time_index], unit='D')
        title += f'\n{platform.upper()}, {orbit_state.capitalize()} orbit, {date_str}'
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def safe_log10_transform(data: np.ndarray, min_value: float = 1e-10) -> np.ndarray:
    """
    Safely apply log10 transformation handling zero, negative, and infinite values.
    
    Args:
        data: Input numpy array
        min_value: Minimum value to replace zeros/negatives with
        
    Returns:
        np.ndarray: Log10 transformed data with finite values only
    """
    # Create a copy to avoid modifying original data
    data_clean = data.copy()
    
    # Replace zero, negative, and infinite values
    data_clean = np.where((data_clean <= 0) | (~np.isfinite(data_clean)), min_value, data_clean)
    
    # Apply log10 transformation
    log_data = np.log10(data_clean)
    
    # Remove any remaining infinite or NaN values
    log_data = log_data[np.isfinite(log_data)]
    
    return log_data

def visualize_backscatter_histogram(dataset: xr.Dataset, time_index: int = 0, 
                                   save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Create a histogram of VV and VH backscatter values with improved error handling.
    
    Args:
        dataset: Input xarray Dataset with filtered bands
        time_index: Time index to visualize
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    fig = plt.figure(figsize=(15, 5))
    
    try:
        # Check if filtered bands exist
        vv_band = 'block_mean_filter_vv'
        vh_band = 'block_mean_filter_vh'
        
        if vv_band not in dataset:
            print(f"Warning: {vv_band} not found. Available bands: {list(dataset.data_vars)}")
            # Try fallback to original bands
            vv_band = 'vv' if 'vv' in dataset else None
            vh_band = 'vh' if 'vh' in dataset else None
        
        if vv_band and vv_band in dataset:
            # Get VV data and safely transform
            vv_data = dataset.isel(time=time_index)[vv_band].values
            vv_log = safe_log10_transform(vv_data)
            
            if len(vv_log) > 0:
                plt.hist(vv_log, bins=200, alpha=0.7, label=f"{vv_band} (log10)", color='blue')
            else:
                print(f"Warning: No valid data for {vv_band} after cleaning")
        
        if vh_band and vh_band in dataset:
            # Get VH data and safely transform
            vh_data = dataset.isel(time=time_index)[vh_band].values
            vh_log = safe_log10_transform(vh_data)
            
            if len(vh_log) > 0:
                plt.hist(vh_log, bins=200, alpha=0.7, label=f"{vh_band} (log10)", color='red')
            else:
                print(f"Warning: No valid data for {vh_band} after cleaning")
        
        plt.legend()
        plt.xlabel("Backscatter Intensity (log10 dB)")
        plt.ylabel("Number of Pixels")
        
        # Add metadata information to the title if available
        title = "Histogram Comparison of Backscatter Values (log10 transformed)"
        if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs and time_index < len(dataset.attrs['metadata']):
            metadata = dataset.attrs['metadata'][time_index]
            platform = metadata.get('platform', 'Unknown')
            orbit_state = metadata.get('orbit_state', 'Unknown')
            date_str = np.datetime_as_string(dataset.time.values[time_index], unit='D')
            title += f'\n{platform.upper()}, {orbit_state.capitalize()} orbit, {date_str}'
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"Error creating histogram: {e}")
        plt.text(0.5, 0.5, f"Error creating histogram:\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

# NEW FUNCTIONS FOR MULTI-DATE FLOODING DETECTION

def display_available_acquisitions(dataset: xr.Dataset) -> None:
    """
    Display a table of available acquisition times for the dataset.
    
    Args:
        dataset: Input xarray Dataset with time dimension
    """
    print("\nAvailable acquisition times:")
    print("----------------------------------------------------------")
    print("Index  |  Date        |  Platform    |  Orbit Direction  |  Orbit Number  |  Relative Orbit")
    print("----------------------------------------------------------")
    
    # Get metadata information if available
    if hasattr(dataset, 'attrs') and 'metadata' in dataset.attrs:
        metadata_list = dataset.attrs['metadata']
        
        for i, time in enumerate(dataset.time.values):
            # Format the time for display
            time_str = np.datetime_as_string(time, unit='D')
            
            # Get metadata for this time step
            if i < len(metadata_list):
                metadata = metadata_list[i]
                platform = metadata.get('platform', 'Unknown')
                orbit_state = metadata.get('orbit_state', 'Unknown')
                orbit_number = metadata.get('orbit_number', 'Unknown')
                relative_orbit = metadata.get('relative_orbit', 'Unknown')
                
                print(f"{i:<7}|  {time_str}  |  {platform:<11}  |  {orbit_state:<16}  |  {orbit_number:<13}  |  {relative_orbit}")
            else:
                print(f"{i:<7}|  {time_str}  |  Unknown      |  Unknown          |  Unknown       |  Unknown")
    else:
        # Fallback if metadata not available
        for i, time in enumerate(dataset.time.values):
            time_str = np.datetime_as_string(time, unit='D')
            print(f"{i:<7}|  {time_str}  |  Unknown      |  Unknown          |  Unknown       |  Unknown")
    
    print("\nNote: When comparing acquisitions for flooding detection, try to use the same orbit number")
    print("and pass direction for best results, as they provide similar viewing angles.")

def calculate_flooding_mask(dataset: xr.Dataset, 
                            first_acq_ind: int, 
                            second_acq_ind: int,
                            threshold_variable: str = 'block_mean_filter_vh',
                            change_threshold: float = -0.5,
                            water_threshold: float = -2.0,  # Updated default
                            use_log10: bool = True) -> Tuple[np.ndarray, np.ndarray, xr.DataArray]:
    """
    Calculate a flooding mask by comparing two acquisitions and finding areas 
    where backscatter has significantly decreased. Enhanced to match updated water detection.
    
    Args:
        dataset: Input xarray Dataset
        first_acq_ind: Index of the first (earlier) acquisition
        second_acq_ind: Index of the second (later) acquisition
        threshold_variable: Band name to use for thresholding
        change_threshold: Threshold value for backscatter change (negative for decrease)
        water_threshold: Threshold value for identifying water in the first acquisition
        use_log10: Whether to apply log10 transformation
        
    Returns:
        Tuple[np.ndarray, np.ndarray, xr.DataArray]: 
            - Boolean mask where True indicates likely new flooding
            - Boolean mask where True indicates water in the first acquisition
            - DataArray with the change product
    """
    # Extract the two acquisitions
    first_acq = dataset.isel(time=first_acq_ind)
    second_acq = dataset.isel(time=second_acq_ind)
    
    # Check if threshold_variable exists in both acquisitions
    if threshold_variable not in first_acq or threshold_variable not in second_acq:
        print(f"Warning: {threshold_variable} not found in dataset. Available variables:")
        print(list(first_acq.data_vars))
        # Try to use a fallback if the specific filtered band isn't available
        if 'vh' in first_acq and 'vh' in second_acq:
            print(f"Using 'vh' band instead of {threshold_variable}")
            threshold_variable = 'vh'
        elif 'block_mean_filter_vh' in first_acq and 'block_mean_filter_vh' in second_acq:
            print(f"Using 'block_mean_filter_vh' band instead")
            threshold_variable = 'block_mean_filter_vh'
    
    # Get the data for both acquisitions
    first_data = first_acq[threshold_variable]
    second_data = second_acq[threshold_variable]
    
    # Apply log10 transformation if requested
    if use_log10:
        first_values = first_data.values
        second_values = second_data.values
        
        # Handle negative values and zeros
        first_values = np.where(first_values <= 0, 1e-10, first_values)
        second_values = np.where(second_values <= 0, 1e-10, second_values)
        
        first_log = np.log10(first_values)
        second_log = np.log10(second_values)
        
        # Calculate the difference in log space
        change_product = xr.DataArray(
            second_log - first_log,
            coords=first_data.coords,
            dims=first_data.dims,
            attrs=first_data.attrs
        )
        
        # Create water mask for the first acquisition (baseline water)
        water_mask = first_log < water_threshold
        
    else:
        # Original approach without log10
        change_product = second_data - first_data
        first_values = first_data.values
        first_values = np.nan_to_num(first_values, nan=0)
        water_mask = first_values < water_threshold
        
        # Handle NaN values in the change product
        change_values = change_product.values
        change_values = np.nan_to_num(change_values, nan=0)
    
    # Find pixels with significant backscatter decrease (potential flooding)
    if use_log10:
        change_values = change_product.values
        change_values = np.nan_to_num(change_values, nan=0)
    
    flooding_mask = change_values < change_threshold
    
    return flooding_mask, water_mask, change_product

def visualize_flooding_detection(dataset: xr.Dataset, 
                                flooding_mask: np.ndarray, 
                                water_mask: np.ndarray,
                                first_acq_ind: int,
                                save_path: Optional[str] = None, 
                                show: bool = True) -> None:
    """
    Create a visualization of flooding detection. Enhanced to match Colab implementation.
    
    Args:
        dataset: Input xarray Dataset
        flooding_mask: Boolean mask where True indicates likely new flooding
        water_mask: Boolean mask where True indicates water in the first acquisition
        first_acq_ind: Index of the first acquisition to use as background
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    # Set the overlay colors (matching Colab)
    color_loss = np.array([255, 0, 0])    # backscatter decrease (RED) - new flooding
    color_blue = np.array([0, 0, 255])    # existing water (BLUE)
    
    # Create flooding visualization
    plt.figure(figsize=(12, 12))
    
    # Make sure masks have the correct shape and are boolean arrays
    # Get the shape from the VH band
    expected_shape = dataset.isel(time=first_acq_ind).vh.shape
    
    # Ensure flooding_mask is the right shape
    if flooding_mask.shape != expected_shape:
        print(f"Warning: Reshaping flooding mask from {flooding_mask.shape} to {expected_shape}")
        # Try to reshape or recreate the mask if needed
        if flooding_mask.size == expected_shape[0] * expected_shape[1]:
            flooding_mask = flooding_mask.reshape(expected_shape)
    
    # Ensure water_mask is the right shape
    if water_mask.shape != expected_shape:
        print(f"Warning: Reshaping water mask from {water_mask.shape} to {expected_shape}")
        # Try to reshape or recreate the mask if needed
        if water_mask.size == expected_shape[0] * expected_shape[1]:
            water_mask = water_mask.reshape(expected_shape)
    
    # Convert to boolean to be safe
    flooding_mask = flooding_mask.astype(bool)
    water_mask = water_mask.astype(bool)
    
    # Make sure we don't have NaN values in our VH band for visualization
    vh_data = dataset.isel(time=first_acq_ind).vh.values
    vh_filled = np.nan_to_num(vh_data, nan=np.nanmean(vh_data))
    
    # Create a modified dataset for visualization with the filled VH band
    vis_dataset = dataset.isel(time=first_acq_ind).copy()
    vis_dataset['vh'] = (('y', 'x'), vh_filled)
    
    # Apply RGB function with masks
    rgb(vis_dataset, bands=['vh', 'vh', 'vh'], 
        paint_on_mask=[(flooding_mask, color_loss), (water_mask, color_blue)], 
        width=10, min_inten=0.6)
    
    plt.title('Multi-Date Flooding Detection\nRED: New flooding, BLUE: Existing water', fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def create_faceted_plot(dataset: xr.Dataset, 
                       band: str = 'vh', 
                       max_cols: int = 3,
                       save_path: Optional[str] = None, 
                       show: bool = True) -> None:
    """
    Create a faceted plot showing all acquisition dates for a given band.
    
    Args:
        dataset: Input xarray Dataset with multiple time slices
        band: Band name to visualize
        max_cols: Maximum number of columns in the faceted plot
        save_path: Path to save the figure (None to skip saving)
        show: Whether to display the figure
    """
    # Get the number of time slices
    n_times = len(dataset.time)
    
    # Calculate the number of rows and columns for the faceted plot
    n_cols = min(max_cols, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # If there's only one plot, make axes iterable
    if n_times == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    
    # Plot each time slice
    for i, time_value in enumerate(dataset.time.values):
        if i < len(axes):
            # Format the date for display
            date_str = np.datetime_as_string(time_value, unit='D')
            
            # Plot the specified band for this time slice
            try:
                # Create RGB visualization
                rgb(dataset.isel(time=i), bands=[band, band, band], 
                    width=5, fig=fig, ax=axes[i], min_inten=0.6)
                axes[i].set_title(f"{date_str}", fontsize=12)
                axes[i].axis('off')
            except Exception as e:
                print(f"Error plotting time slice {i} ({date_str}): {e}")
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[i].transAxes)
    
    # Hide any unused subplots
    for j in range(n_times, len(axes)):
        axes[j].axis('off')
    
    # Add a global title
    plt.suptitle(f'Sentinel-1 {band.upper()} Band Time Series', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_all_postprocessed_data(dataset: xr.Dataset,
                                    save_path_prefix: Optional[str] = None,
                                    show: bool = True) -> None:
    """
    Create a comprehensive visualization of all post-processed data.
    
    Args:
        dataset: Input xarray Dataset with multiple time slices
        save_path_prefix: Prefix for saved paths (None to skip saving)
        show: Whether to display figures
    """
    print("Creating faceted plot of VH band...")
    vh_facet_path = f"{save_path_prefix}_vh_facet.png" if save_path_prefix else None
    create_faceted_plot(
        dataset=dataset,
        band='vh',
        max_cols=3,
        save_path=vh_facet_path,
        show=show
    )
    
    print("Creating faceted plot of VV band...")
    vv_facet_path = f"{save_path_prefix}_vv_facet.png" if save_path_prefix else None
    create_faceted_plot(
        dataset=dataset,
        band='vv',
        max_cols=3,
        save_path=vv_facet_path,
        show=show
    )
    
    # Create faceted plots for filtered bands if they exist
    filtered_bands = [var for var in dataset.data_vars if var.startswith('block_')]
    for filtered_band in filtered_bands:
        band_name = filtered_band.split('_')[-1]  # Extract band name (vv or vh)
        print(f"Creating faceted plot of {filtered_band}...")
        filtered_facet_path = f"{save_path_prefix}_{filtered_band}_facet.png" if save_path_prefix else None
        try:
            create_faceted_plot(
                dataset=dataset,
                band=filtered_band,
                max_cols=3,
                save_path=filtered_facet_path,
                show=show
            )
        except Exception as e:
            print(f"Error creating faceted plot for {filtered_band}: {e}")
    
    # Create water extent time series with updated parameters
    print("Creating water extent time series...")
    water_timeseries_path = f"{save_path_prefix}_water_timeseries.png" if save_path_prefix else None
    try:
        create_water_extent_timeseries(
            dataset=dataset,
            threshold_variable='block_mean_filter_vh',
            water_threshold=-2.0,  # Updated threshold
            use_log10=True,  # Enable log10 transformation
            save_path=water_timeseries_path,
            show=show
        )
    except Exception as e:
        print(f"Error creating water extent time series: {e}")
        try:
            # Try with unfiltered VH band as fallback
            create_water_extent_timeseries(
                dataset=dataset,
                threshold_variable='vh',
                water_threshold=-2.0,  # Updated threshold
                use_log10=True,  # Enable log10 transformation
                save_path=water_timeseries_path,
                show=show
            )
        except Exception as e2:
            print(f"Error creating water extent time series with fallback: {e2}")

def run_flood_detection(dataset: xr.Dataset,
                       first_acq_ind: int,
                       second_acq_ind: int,
                       threshold_variable: str = 'block_mean_filter_vh',
                       change_threshold: float = -0.5,
                       water_threshold: float = -2.0,  # Updated default
                       use_log10: bool = True,  # New parameter
                       save_path_prefix: Optional[str] = None,
                       show: bool = True) -> None:
    """
    Run the complete flood detection workflow and generate visualizations.
    Enhanced to match updated water detection approach.
    
    Args:
        dataset: Input xarray Dataset
        first_acq_ind: Index of the first (earlier) acquisition
        second_acq_ind: Index of the second (later) acquisition
        threshold_variable: Band name to use for thresholding
        change_threshold: Threshold value for backscatter change (negative for decrease)
        water_threshold: Threshold value for identifying water
        use_log10: Whether to apply log10 transformation
        save_path_prefix: Prefix for saved paths (None to skip saving)
        show: Whether to display figures
    """
    # Get the acquisition dates for display
    first_date = np.datetime_as_string(dataset.time.values[first_acq_ind], unit='D')
    second_date = np.datetime_as_string(dataset.time.values[second_acq_ind], unit='D')
    
    transform_note = " (log10 transformed)" if use_log10 else ""
    print(f"Analyzing flooding between {first_date} (index {first_acq_ind}) and {second_date} (index {second_acq_ind})")
    print(f"Using {threshold_variable}{transform_note} with change threshold of {change_threshold}")
    print(f"Using water threshold of {water_threshold}{transform_note}")
    
    # Calculate the flooding mask
    flooding_mask, water_mask, change_product = calculate_flooding_mask(
        dataset=dataset,
        first_acq_ind=first_acq_ind,
        second_acq_ind=second_acq_ind,
        threshold_variable=threshold_variable,
        change_threshold=change_threshold,
        water_threshold=water_threshold,
        use_log10=use_log10
    )
    
    # Generate flooding visualization
    save_path_flood = f"{save_path_prefix}_flooding.png" if save_path_prefix else None
    visualize_flooding_detection(
        dataset=dataset,
        flooding_mask=flooding_mask,
        water_mask=water_mask,
        first_acq_ind=first_acq_ind,
        save_path=save_path_flood,
        show=show
    )
    
    # Calculate and display flooding statistics
    total_pixels = flooding_mask.size
    flooded_pixels = np.sum(flooding_mask)
    water_pixels = np.sum(water_mask)
    flooded_percent = (flooded_pixels / total_pixels) * 100
    water_percent = (water_pixels / total_pixels) * 100
    
    print("\nFlooding Analysis Results:")
    print(f"Total pixels in the scene: {total_pixels}")
    print(f"Pixels identified as new flooding: {flooded_pixels} ({flooded_percent:.2f}%)")
    print(f"Pixels identified as existing water: {water_pixels} ({water_percent:.2f}%)")
    
    # Calculate area (assuming 10m resolution by default)
    pixel_area_sq_m = 10 * 10  # 10m x 10m = 100 sq meters per pixel
    # Using the same conversion as in the water extent time series
    flooded_area_sq_km = flooded_pixels * 100 / 1_000_000
    water_area_sq_km = water_pixels * 100 / 1_000_000
    
    print(f"Estimated area of new flooding: {flooded_area_sq_km:.2f} sq km")
    print(f"Estimated area of existing water: {water_area_sq_km:.2f} sq km")

def visualize_threshold_distribution(dataset: xr.Dataset, 
                                     band: str = 'block_mean_filter_vh', 
                                     time_index: int = 0, 
                                     water_threshold: float = -2.0,  # Updated default
                                     change_threshold: float = -0.5,
                                     use_log10: bool = True) -> None:  # New parameter
    """
    Visualize histogram of selected band and mark water and flood thresholds.
    Updated to include log10 transformation option.
    
    Args:
        dataset: xarray Dataset with the band of interest
        band: Band name to visualize
        time_index: Index of acquisition to visualize
        water_threshold: Threshold below which pixels are considered water
        change_threshold: Backscatter change threshold indicating flooding
        use_log10: Whether to apply log10 transformation
    """
    data = dataset.isel(time=time_index)[band].values
    
    # Apply log10 transformation if requested
    if use_log10:
        # Handle negative values and zeros
        data = np.where(data <= 0, 1e-10, data)
        data = np.log10(data)
        xlabel = f"{band} values (log10 transformed)"
        title_suffix = " (log10 transformed)"
    else:
        xlabel = f"{band} values (dB)"
        title_suffix = ""
    
    # Remove infinite and NaN values
    data = data[np.isfinite(data)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=200, color='gray', alpha=0.8, label=f"{band} values")
    plt.axvline(water_threshold, color='blue', linestyle='--', 
                label=f"Water Threshold ({water_threshold}{' log10' if use_log10 else ' dB'})")
    plt.axvline(change_threshold, color='red', linestyle='--', 
                label=f"Flood Threshold ({change_threshold}{' log10' if use_log10 else ' dB'})")
    
    plt.title(f"Histogram of {band} at time index {time_index}{title_suffix}")
    plt.xlabel(xlabel)
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage after loading:
# sar_dataset = load_sentinel1(...)
# best_orbit_key, aligned_dataset = robust_orbit_selection(sar_dataset)
# visualize_threshold_distribution(aligned_dataset, time_index=0, use_log10=True)

if __name__ == "__main__":
    geometry = Polygon([
        (-92.695, 17.813),
        (-92.495, 17.813),
        (-92.495, 17.613),
        (-92.695, 17.613),
        (-92.695, 17.813)
    ])
#    geometry = Polygon([
#        (28.11823, 37.50479),
#        (28.13823, 37.50479),
#        (28.13823, 37.48479),
#        (28.11823, 37.48479),
#        (28.11823, 37.50479),

#    ])
    
    start_date = datetime.datetime(2020, 9, 1)
    end_date = datetime.datetime(2021, 1, 1)
    
    print("Loading Sentinel-1 data...")
    try:
        sar_dataset = load_sentinel1(
            geometry=geometry,
            measurements=["vv", "vh"],
            start_date=start_date,
            end_date=end_date,
            resolution=10,
            platform=None
        )
        
        print(f"Initial dataset loaded with {len(sar_dataset.time)} time steps")
        print(f"Data variables: {list(sar_dataset.data_vars)}")
        
        print("Selecting most consistent orbit group...")
        try:
            orbit_key, sar_dataset = robust_orbit_selection(sar_dataset)
            print(f"Using orbit group: {orbit_key} with {len(sar_dataset.time)} acquisitions")
        except ValueError as e:
            print(f"Robust orbit selection failed: {e}")
            # Try less strict grouping
            print("Trying less strict grouping...")
            try:
                orbit_key, sar_dataset = select_most_common_orbit_group(
                    sar_dataset, 
                    grouping_criteria="relative_only",
                    min_acquisitions=2
                )
                print(f"Using fallback orbit group: {orbit_key} with {len(sar_dataset.time)} acquisitions")
            except ValueError as e2:
                print(f"All orbit selection strategies failed: {e2}")
                print("Proceeding with original dataset...")
                orbit_key = "original_dataset"
        
        if len(sar_dataset.time) == 0:
            print("Error: No time steps available after orbit selection.")
            exit(1)
        
        print("Applying speckle filtering...")
        try:
            sar_dataset = apply_speckle_filter(sar_dataset, ["vv", "vh"], "mean", 5)
            print("Speckle filtering completed successfully")
        except Exception as e:
            print(f"Error during speckle filtering: {e}")
            print("Continuing without speckle filtering...")
        
        print("Calculating backscatter amplitudes...")
        try:
            sar_dataset = calculate_backscatter_amplitudes(sar_dataset)
            print("Backscatter amplitudes calculated successfully")
        except Exception as e:
            print(f"Error calculating backscatter amplitudes: {e}")
            print("Continuing without amplitude calculations...")
        
        print("Creating visualizations...")
        display_available_acquisitions(sar_dataset)
        acq_ind_to_show = 0
        
        # Visualize backscatter RGB if amplitude bands exist
        try:
            if all(band in sar_dataset for band in ['vv_amp', 'vh_amp', 'vvvh_amp']):
                visualize_backscatter(sar_dataset, acq_ind_to_show, save_path='sentinel1_rgb_composite.png')
            else:
                print("Amplitude bands not available, skipping RGB visualization")
        except Exception as e:
            print(f"Error creating RGB visualization: {e}")
        
        # Create histogram with error handling
        try:
            visualize_backscatter_histogram(sar_dataset, acq_ind_to_show, save_path='sentinel1_backscatter_histogram.png')
        except Exception as e:
            print(f"Error creating histogram: {e}")
        
        # Visualize threshold distribution with error handling
        try:
            print("Visualizing threshold distribution...")
            visualize_threshold_distribution(sar_dataset, time_index=acq_ind_to_show, use_log10=True)
        except Exception as e:
            print(f"Error creating threshold distribution: {e}")
        
        # Water extent mapping with error handling
        try:
            print("Creating water extent map using updated VH band threshold...")
            water_threshold = -2.0
            
            # Determine which band to use for water detection
            water_band = 'block_mean_filter_vh' if 'block_mean_filter_vh' in sar_dataset else 'vh'
            print(f"Using {water_band} for water detection")
            
            water_mask = create_water_mask(sar_dataset, water_band, water_threshold, 
                                         time_index=acq_ind_to_show, use_log10=True)
            visualize_water_extent(sar_dataset, water_mask, acq_ind_to_show, save_path='sentinel1_water_extent.png')
        except Exception as e:
            print(f"Error creating water extent map: {e}")
        
        # Test combined water detection approach
        try:
            print("Testing combined VV*VH water detection approach...")
            combined_water_mask = create_combined_water_mask(sar_dataset, threshold=4.05, time_index=acq_ind_to_show)
            visualize_water_extent(sar_dataset, combined_water_mask, acq_ind_to_show, save_path='sentinel1_combined_water_extent.png')
        except Exception as e:
            print(f"Error with combined water detection: {e}")
        
        # Persistent water detection for multi-temporal datasets
        if len(sar_dataset.time) > 1:
            try:
                print("Creating persistent water mask...")
                persistent_water = create_persistent_water_mask(sar_dataset, 
                                                              threshold=-2.0, 
                                                              persistence_threshold=0.8,
                                                              use_log10=True)
                visualize_persistent_water(sar_dataset, persistent_water, save_path='sentinel1_persistent_water.png')
            except Exception as e:
                print(f"Error creating persistent water mask: {e}")
        
        # Flood detection for multi-temporal datasets
        if len(sar_dataset.time) > 1:
            try:
                print("\nPerforming multi-date flooding detection with updated parameters...")
                first_acq_ind = 0
                second_acq_ind = len(sar_dataset.time) - 1
                run_flood_detection(
                    dataset=sar_dataset,
                    first_acq_ind=first_acq_ind,
                    second_acq_ind=second_acq_ind,
                    threshold_variable='block_mean_filter_vh' if 'block_mean_filter_vh' in sar_dataset else 'vh',
                    change_threshold=-0.5,
                    water_threshold=-2.0,
                    use_log10=True,
                    save_path_prefix='sentinel1_flood_detection'
                )
            except Exception as e:
                print(f"Error during flood detection: {e}")
            
            try:
                print("\nCreating comprehensive visualizations of all post-processed data...")
                visualize_all_postprocessed_data(
                    dataset=sar_dataset,
                    save_path_prefix='sentinel1_postprocessed',
                    show=True
                )
            except Exception as e:
                print(f"Error creating comprehensive visualizations: {e}")
        else:
            print("\nMulti-date flooding detection requires at least two acquisitions.")
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
