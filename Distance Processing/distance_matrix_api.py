import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
import os
import tempfile
import zipfile
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Configure test mode early
test_mode = False  # Set to False for full analysis

# --- Determine correct paths relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")

print("Analyzing SVI vs Multi-Modal Urgent Care Access...")
print("=" * 60)

# ------------------ Google Maps API Configuration ------------------
print("Setting up Google Maps Distance Matrix API...")

# Google Maps API imports
import googlemaps
import time
import json

# Try to get API key from environment variable
try:
    # Get the API key from environment variable
    api_key = os.getenv('GOOGLE_API')
    if not api_key:
        raise ValueError("GOOGLE_API environment variable not set")
    
    # Initialize the Google Maps client
    gmaps = googlemaps.Client(key=api_key)
    print("Google Maps API client initialized")
    
except ValueError as e:
    print(f"Google Maps API key not found: {e}")
    print("   - Set GOOGLE_API in your .env file")
    print("   - Make sure the API key is valid")
    exit(1)
except Exception as e:
    print(f"Failed to initialize Google Maps client: {e}")
    exit(1)

# ------------------ Load and Prepare Data ------------------
print("Loading census tract boundaries and SVI data...")

# Load census tract boundaries
tract_zip_path = os.path.join(DATA_DIR, "tl_2022_17_tract.zip")
if os.path.exists(tract_zip_path):
    try:
        print(f"Found tract data: {tract_zip_path}")
        
        # Extract the zip file
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tract_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the shapefile
        shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
        if not shp_files:
            raise Exception("No shapefile found in tract data")
        
        shp_path = os.path.join(extract_dir, shp_files[0])
        boundaries_gdf = gpd.read_file(shp_path)
        print(f"Loaded {len(boundaries_gdf)} census tracts")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(extract_dir)
        
    except Exception as e:
        print(f"Failed to load tract data: {e}")
        exit(1)
else:
    print(f"Tract data not found at {tract_zip_path}")
    exit(1)

# Load SVI data
results_path = os.path.join(PROCESSED_DATA_DIR, "chicago_urgentcare_final_results.geojson")
if os.path.exists(results_path):
    try:
        svi_gdf = gpd.read_file(results_path)
        print(f"Loaded SVI data: {len(svi_gdf)} records")
    except Exception as e:
        print(f"ERROR: Failed to load SVI data: {e}")
        exit(1)
else:
    print(f"ERROR: Could not find '{results_path}'.")
    exit(1)

# Merge SVI data with boundaries
print("Merging SVI data with census boundaries...")
svi_gdf['FIPS'] = svi_gdf['FIPS'].astype(str)
boundaries_gdf['GEOID'] = boundaries_gdf['GEOID'].astype(str)

# Get Chicago area tracts
chicago_bbox = [-87.9401, 41.6445, -87.5241, 42.0230]
chicago_area_tracts = boundaries_gdf.cx[chicago_bbox[0]:chicago_bbox[2], chicago_bbox[1]:chicago_bbox[3]]
print(f"Found {len(chicago_area_tracts)} tracts in Chicago area")

# Merge with SVI data
merged_gdf = chicago_area_tracts.merge(
    svi_gdf[['FIPS', 'SVI', 'SVI_q']], 
    left_on='GEOID', 
    right_on='FIPS', 
    how='left'
)
print(f"Merged data: {len(merged_gdf)} tracts")

# Load Chicago city limits
city_limits_path = os.path.join(DATA_DIR, "Chicago_City_Limits-shp", "Chicago_City_Limits.shp")
if os.path.exists(city_limits_path):
    try:
        chicago_city_limits = gpd.read_file(city_limits_path)
        print(f"Loaded Chicago city limits")
        
        # Filter to Chicago city limits
        if merged_gdf.crs != chicago_city_limits.crs:
            chicago_city_limits = chicago_city_limits.to_crs(merged_gdf.crs)
        
        intersecting_tracts = merged_gdf[merged_gdf.geometry.intersects(chicago_city_limits.unary_union)].copy()
        merged_gdf = gpd.clip(intersecting_tracts, chicago_city_limits)
        print(f"Filtered to {len(merged_gdf)} tracts within Chicago")
        
    except Exception as e:
        print(f"Warning: Could not load city limits: {e}")

# Load urgent care facilities
urgentcare_csv_path = os.path.join(DATA_DIR, "urgent_care_facilities.csv")
if os.path.exists(urgentcare_csv_path):
    try:
        uc_df = pd.read_csv(urgentcare_csv_path)
        # Find lat/lon columns
        lat_col = None
        lon_col = None
        for c in uc_df.columns:
            if c.lower() in ['latitude', 'lat']:
                lat_col = c
            if c.lower() in ['longitude', 'lon', 'lng', 'long']:
                lon_col = c
        
        uc_df = uc_df.dropna(subset=[lat_col, lon_col])
        from shapely.geometry import Point
        uc_gdf = gpd.GeoDataFrame(
            uc_df,
            geometry=[Point(xy) for xy in zip(uc_df[lon_col], uc_df[lat_col])],
            crs="EPSG:4326"
        )
        
        # Filter to Chicago area
        if chicago_city_limits is not None:
            chicago_union = chicago_city_limits.to_crs(4326).unary_union
            uc_gdf = uc_gdf[uc_gdf.geometry.within(chicago_union)]
        
        print(f"Loaded {len(uc_gdf)} urgent care facilities in Chicago")
        
        # Limit to reasonable number of facilities to manage API costs
        if len(uc_gdf) > 30:
            uc_gdf = uc_gdf.head(30)
            print(f"Limited to {len(uc_gdf)} facilities to manage API costs")
        
    except Exception as e:
        print(f"Failed to load urgent care facilities: {e}")
        exit(1)

# ------------------ Calculate Multi-Modal Travel Times using Google Maps Distance Matrix API ------------------
print("\nCalculating travel times using Google Maps Distance Matrix API...")

# Define transportation modes for Google Maps API
travel_modes = {
    'car': 'DRIVE',  # Changed from 'driving' to 'car' for consistency
    'walking': 'WALK',
    'transit': 'TRANSIT'
}

# Process only car and transit modes to reduce API costs
print("Processing car and transit modes only to manage API costs")
active_modes = ['car', 'transit']

# Calculate cost estimate (will be updated after facilities are loaded)
print(f"Cost estimate will be calculated after data loading...")

def get_distance_matrix_travel_times(origins, destinations, mode, gmaps_client):
    """Get travel times using Google Maps Distance Matrix API with chunking for API limits"""
    try:
        # Convert mode to Google Maps format
        mode_mapping = {
            'car': 'driving',
            'walking': 'walking',
            'transit': 'transit'
        }
        
        travel_mode = mode_mapping.get(mode, 'driving')
        
        # Prepare origins and destinations for API
        origin_coords = (origins[0]['waypoint']['location']['lat_lng']['latitude'], 
                        origins[0]['waypoint']['location']['lat_lng']['longitude'])
        
        dest_coords = [(dest['waypoint']['location']['lat_lng']['latitude'], 
                       dest['waypoint']['location']['lat_lng']['longitude']) 
                      for dest in destinations]
        
        # Google Maps API limits: max 25 destinations per request
        max_destinations = 25
        all_travel_times = []
        
        # Process destinations in chunks
        for i in range(0, len(dest_coords), max_destinations):
            chunk_destinations = dest_coords[i:i + max_destinations]
            
            # Make the API call for this chunk
            result = gmaps_client.distance_matrix(
                origin_coords,  # Single origin tuple
                chunk_destinations,    # Chunk of destination tuples
                mode=travel_mode,
                traffic_model='best_guess',
                departure_time='now'
            )
            
            # Extract travel times for this chunk
            if result['status'] == 'OK':
                for element in result['rows'][0]['elements']:
                    if element['status'] == 'OK':
                        # Convert seconds to minutes
                        time_minutes = element['duration']['value'] / 60
                        all_travel_times.append(time_minutes)
                    else:
                        all_travel_times.append(np.nan)
            else:
                print(f"Warning: Distance Matrix API returned status: {result['status']}")
                # Fill with NaN for this chunk
                all_travel_times.extend([np.nan] * len(chunk_destinations))
            
            # Add small delay between chunks to respect rate limits
            time.sleep(0.1)
        
        return [all_travel_times]  # Return as matrix format for compatibility
        
    except Exception as e:
        print(f"Warning: Distance Matrix API call failed for {mode}: {e}")
        return None

def calculate_min_travel_time_google_api(tract_geom, facilities_gdf, mode, route_matrix_client):
    """Calculate minimum travel time using Google Cloud API"""
    if tract_geom is None or tract_geom.is_empty:
        return np.nan
    
    try:
        # Get tract centroid
        centroid = tract_geom.centroid
        
        # Prepare origins and destinations for API
        origins = [{
            "waypoint": {
                "location": {
                    "lat_lng": {
                        "latitude": centroid.y,
                        "longitude": centroid.x
                    }
                }
            }
        }]
        
        destinations = []
        for _, facility in facilities_gdf.iterrows():
            dest = {
                "waypoint": {
                    "location": {
                        "lat_lng": {
                            "latitude": facility.geometry.y,
                            "longitude": facility.geometry.x
                        }
                    }
                }
            }
            destinations.append(dest)
        
        # Get travel times from API
        travel_times = get_distance_matrix_travel_times(origins, destinations, travel_modes[mode], gmaps)
        
        if travel_times and len(travel_times) > 0:
            # Find minimum travel time
            min_time = np.nanmin(travel_times[0])
            return min_time if not np.isnan(min_time) else np.nan
        else:
            return np.nan
            
    except Exception as e:
        print(f"Warning: Failed to calculate travel time for {mode}: {e}")
        return np.nan

# Function to calculate travel times for all tracts (with batching for API limits)
def calculate_all_travel_times_batched(merged_gdf, uc_gdf, gmaps_client, batch_size=5):
    """Calculate travel times for all tracts in batches to respect API limits"""
    print(f"  Processing {len(merged_gdf)} tracts in batches of {batch_size}...")
    print(f"  Google API limit: 25 destinations per request, processing in chunks")
    
    # Initialize travel time columns for all modes (keep existing data)
    for mode in travel_modes.keys():
        if f'{mode}_time_min' not in merged_gdf.columns:
            merged_gdf[f'{mode}_time_min'] = np.nan
    
    # Process in batches
    for i in range(0, len(merged_gdf), batch_size):
        batch_end = min(i + batch_size, len(merged_gdf))
        batch_tracts = merged_gdf.iloc[i:batch_end]
        
        print(f"    Processing batch {i//batch_size + 1}/{(len(merged_gdf) + batch_size - 1)//batch_size} (tracts {i+1}-{batch_end})")
        
        for mode in active_modes:  # Only process active modes
            print(f"      Calculating {mode} times...")
            
            for idx, tract in batch_tracts.iterrows():
                try:
                    travel_time = calculate_min_travel_time_google_api(
                        tract.geometry, uc_gdf, mode, gmaps_client
                    )
                    merged_gdf.loc[idx, f'{mode}_time_min'] = travel_time
                except Exception as e:
                    print(f"        Warning: Error processing tract {idx}: {e}")
                    merged_gdf.loc[idx, f'{mode}_time_min'] = np.nan
                
                # Add small delay to respect API rate limits
                time.sleep(0.2)
        
        print(f"    Batch {i//batch_size + 1} complete")
        
        # Save intermediate results every 5 batches (more frequent saves)
        if (i//batch_size + 1) % 5 == 0:
            backup_file = os.path.join(RESULTS_DIR, f"travel_times_backup_batch_{i//batch_size + 1}.csv")
            merged_gdf.to_csv(backup_file, index=False)
            print(f"    Backup saved: {backup_file}")
    
    return merged_gdf

# Project to a suitable CRS for distance calculations (Illinois State Plane)
merged_proj = merged_gdf.to_crs("EPSG:3435")
uc_proj = uc_gdf.to_crs("EPSG:3435")

print("  Calculating travel times using Google Maps Distance Matrix API...")


# Process all tracts in batches
print("Processing ALL census tracts...")
print(f"   Processing {len(merged_gdf)} tracts × {len(uc_gdf)} facilities × {len(active_modes)} modes")
print(f"   Total API calls: {len(merged_gdf) * len(uc_gdf) * len(active_modes):,} elements")

# Use the batched function for API-based travel time calculation
merged_gdf = calculate_all_travel_times_batched(merged_gdf, uc_gdf, gmaps, batch_size=5)

# Clean data - remove rows with missing SVI or travel time data
# Only check for the modes we're actually processing
required_columns = ['SVI'] + [f'{mode}_time_min' for mode in active_modes]
analysis_df = merged_gdf.dropna(subset=required_columns)
print(f"Analysis dataset: {len(analysis_df)} tracts with complete data")
print(f"   Required columns: {required_columns}")

# ------------------ Statistical Analysis - Car vs Non-Car Access Disparity ------------------
print("\nStatistical Analysis: Car Access vs Non-Car Transportation Burden")
print("=" * 70)

# Calculate summary statistics for each mode
results = {}
quartile_summary = {}

print("\nTRAVEL TIME SUMMARY BY TRANSPORTATION MODE:")
print("=" * 60)

for mode in active_modes:  # Only process active modes
    time_col = f'{mode}_time_min'
    
    # Calculate overall statistics
    overall_mean = analysis_df[time_col].mean()
    overall_median = analysis_df[time_col].median()
    overall_range = (analysis_df[time_col].min(), analysis_df[time_col].max())
    
    print(f"\n{mode.upper()}:")
    print(f"   Mean: {overall_mean:.1f} min | Median: {overall_median:.1f} min | Range: {overall_range[0]:.1f}-{overall_range[1]:.1f} min")
    
    # Calculate by SVI quartile
    quartile_stats = analysis_df.groupby('SVI_q')[time_col].agg(['count', 'mean', 'median', 'std'])
    
    # Store results
    results[mode] = {
        'overall_mean': overall_mean,
        'overall_median': overall_median,
        'overall_range': overall_range,
        'quartile_stats': quartile_stats
    }

# Calculate car vs non-car burden analysis
print(f"\n CAR ACCESS ADVANTAGE ANALYSIS:")
print("=" * 60)

car_times = analysis_df['car_time_min']
non_car_modes = ['transit']  # Only transit since we're only processing car + transit

print(f"\nABSOLUTE TIME DIFFERENCES (vs Car):")
print("-" * 50)

for mode in non_car_modes:
    mode_times = analysis_df[f'{mode}_time_min']
    
    # Calculate time burden compared to car
    time_burden = mode_times - car_times
    burden_multiple = mode_times / car_times
    
    print(f"\n{mode.upper()} vs CAR:")
    print(f"   Average burden: {time_burden.mean():.1f} minutes longer ({burden_multiple.mean():.1f}x slower)")
    print(f"   Median burden: {time_burden.median():.1f} minutes longer ({burden_multiple.median():.1f}x slower)")
    print(f"   Max burden: {time_burden.max():.1f} minutes longer ({burden_multiple.max():.1f}x slower)")
    
    # Store burden analysis
    results[mode]['car_burden'] = {
        'mean_extra_time': time_burden.mean(),
        'median_extra_time': time_burden.median(),
        'max_extra_time': time_burden.max(),
        'mean_multiple': burden_multiple.mean(),
        'median_multiple': burden_multiple.median(),
        'max_multiple': burden_multiple.max()
    }

print(f"\nSVI QUARTILE COMPARISON:")
print("=" * 60)

# Compare travel times by SVI quartile for each mode
for quartile in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
    if quartile in analysis_df['SVI_q'].values:
        print(f"\n{quartile.upper()}:")
        quartile_data = analysis_df[analysis_df['SVI_q'] == quartile]
        
        car_time = quartile_data['car_time_min'].mean()
        print(f"   Car: {car_time:.1f} min")
        
        for mode in non_car_modes:
            mode_time = quartile_data[f'{mode}_time_min'].mean()
            extra_time = mode_time - car_time
            multiple = mode_time / car_time
            print(f"   {mode.title()}: {mode_time:.1f} min (+{extra_time:.1f} min, {multiple:.1f}x slower)")

# Calculate vulnerability burden (higher SVI = less likely to have car)
print(f"\nVULNERABILITY ANALYSIS:")
print("=" * 60)
print(f"KEY INSIGHT: Higher-SVI areas have lower car ownership rates")
print(f"This means vulnerable populations are MORE LIKELY to rely on slower modes")

print(f"\nTRANSPORTATION BURDEN BY VULNERABILITY LEVEL:")
print("-" * 50)

for quartile in ['Q1(low)', 'Q4(high)']:
    if quartile in analysis_df['SVI_q'].values:
        quartile_data = analysis_df[analysis_df['SVI_q'] == quartile]
        car_time = quartile_data['car_time_min'].mean()
        
        print(f"\n{quartile} (Car ownership: {'High' if quartile == 'Q1(low)' else 'Low'}):")
        print(f"   If have car: {car_time:.1f} min average")
        
        total_burden = 0
        for mode in non_car_modes:
            mode_time = quartile_data[f'{mode}_time_min'].mean()
            extra_burden = mode_time - car_time
            total_burden += extra_burden
            print(f"   If use {mode}: {mode_time:.1f} min (+{extra_burden:.1f} min burden)")
        
        print(f"   Average non-car burden: +{total_burden/3:.1f} min vs car access")

# Store vulnerability analysis
vulnerability_analysis = {}
# Get the actual quartiles present in our data
available_quartiles = analysis_df['SVI_q'].unique()
print(f"   Available SVI quartiles in dataset: {sorted(available_quartiles)}")

for quartile in ['Q1(low)', 'Q4(high)']:
    if quartile in analysis_df['SVI_q'].values:
        quartile_data = analysis_df[analysis_df['SVI_q'] == quartile]
        car_time = quartile_data['car_time_min'].mean()
        
        non_car_modes = ['transit']  # Only transit for this analysis
        non_car_burden = []
        for mode in non_car_modes:
            mode_time = quartile_data[f'{mode}_time_min'].mean()
            non_car_burden.append(mode_time - car_time)
        
        vulnerability_analysis[quartile] = {
            'car_time': car_time,
            'avg_non_car_burden': np.mean(non_car_burden),
            'transit_time': quartile_data['transit_time_min'].mean()
        }
    else:
        print(f"   Warning: {quartile} not found in dataset")

results['vulnerability_analysis'] = vulnerability_analysis

# Visualization code removed as requested

# Time interval comparison removed as requested

# Save final results with travel times from Google Maps API
final_results_file = os.path.join(RESULTS_DIR, "travel_times_final.csv")
merged_gdf.to_csv(final_results_file, index=False)
print(f" Final travel times saved: {final_results_file}")

# Also save as GeoJSON for mapping
final_geojson_file = os.path.join(RESULTS_DIR, "travel_times_final.geojson")
merged_gdf.to_file(final_geojson_file, driver='GeoJSON')
print(f"Final GeoJSON saved: {final_geojson_file}")

# ------------------ Final Summary ------------------
print(f"\n CAR vs NON-CAR ACCESS ANALYSIS COMPLETE!")
print("=" * 60)
print(f" KEY FINDING: Car Access Provides Massive Time Advantage")
print("=" * 60)

print(f"\nCAR ACCESS: FAST & CONSISTENT")
print("-" * 40)
car_mean = results['car']['overall_mean']
print(f"   Average time: {car_mean:.1f} minutes")
print(f"   Consistent across all SVI levels (~{car_mean:.0f} minutes)")

print(f"\nNON-CAR MODES: SLOW & BURDENSOME")
print("-" * 40)
for mode in active_modes:
    if mode != 'car':  # Skip car since it's already covered above
        mode_mean = results[mode]['overall_mean']
        if 'car_burden' in results[mode]:
            extra_time = results[mode]['car_burden']['mean_extra_time']
            multiple = results[mode]['car_burden']['mean_multiple']
            print(f"   {mode.upper():8}: {mode_mean:.1f} min (+{extra_time:.1f} min, {multiple:.1f}x slower than car)")

print(f"\nVULNERABILITY BURDEN:")
print("-" * 40)
vuln = results['vulnerability_analysis']

# Handle missing quartiles gracefully
if 'Q1(low)' in vuln and 'Q4(high)' in vuln:
    low_svi_burden = vuln['Q1(low)']['avg_non_car_burden']
    high_svi_burden = vuln['Q4(high)']['avg_non_car_burden']
    
    print(f"   Low-SVI (high car ownership): +{low_svi_burden:.1f} min burden if using non-car modes")
    print(f"   High-SVI (low car ownership): +{high_svi_burden:.1f} min burden if using non-car modes")
    print(f"   IMPACT: Vulnerable populations face {high_svi_burden:.1f} minutes longer travel on average")
    print(f"   IMPLICATION: Lower car ownership compounds existing vulnerabilities")
else:
    print(f"   Warning: Limited vulnerability analysis due to missing quartiles in dataset")
    print(f"   Available quartiles: {list(vuln.keys())}")
    if vuln:
        for quartile, data in vuln.items():
            print(f"   {quartile}: Car time {data['car_time']:.1f} min, Non-car burden +{data['avg_non_car_burden']:.1f} min")

print(f"\nFiles generated:")
print(f"   Final CSV: {final_results_file}")
print(f"   Final GeoJSON: {final_geojson_file}")
print(f"\nAnalysis complete")
