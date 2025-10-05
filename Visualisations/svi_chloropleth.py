# Download census tract boundaries and create a proper SVI choropleth map

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import sys
import os
import zipfile
import tempfile
from shapely.geometry import Point
import numpy as np

warnings.filterwarnings('ignore')

print("Creating Chicago SVI Visualization with Census Tract Boundaries...")
print("=" * 60)

# ------------------ Load Census Tract Boundaries from Data Folder ------------------
print("Loading census tract boundaries from data folder...")

# Use the local tract data file
tract_zip_path = "data/tl_2022_17_tract.zip"
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
        print(f"Extracted: {shp_files[0]}")
        
        # Load the boundaries
        boundaries_gdf = gpd.read_file(shp_path)
        print(f"Loaded {len(boundaries_gdf)} census tracts")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(extract_dir)
        
    except Exception as e:
        print(f"Failed to load tract data: {e}")
        boundaries_gdf = None
else:
    print(f"Tract data not found at {tract_zip_path}")
    boundaries_gdf = None

# ------------------ Load SVI Data ------------------
print("Loading SVI data...")

# Try to load the existing results file first
results_path = "./processed_data/chicago_urgentcare_final_results.geojson"
if os.path.exists(results_path):
    try:
        svi_gdf = gpd.read_file(results_path)
        print(f"Loaded SVI data: {len(svi_gdf)} records")
        
        # Check if we have the boundaries from the data folder
        if boundaries_gdf is not None:
            # Merge SVI data with boundaries using FIPS
            print("Merging SVI data with census boundaries...")
            
            # Ensure FIPS columns are the same type
            svi_gdf['FIPS'] = svi_gdf['FIPS'].astype(str)
            boundaries_gdf['GEOID'] = boundaries_gdf['GEOID'].astype(str)
            
            # First, get all tracts that intersect with Chicago area
            chicago_bbox = [-87.9401, 41.6445, -87.5241, 42.0230]  # [min_lon, min_lat, max_lon, max_lat]
            chicago_area_tracts = boundaries_gdf.cx[chicago_bbox[0]:chicago_bbox[2], chicago_bbox[1]:chicago_bbox[3]]
            print(f"Found {len(chicago_area_tracts)} tracts in Chicago area")
            
            # Merge with SVI data using LEFT join to keep ALL tracts
            merged_gdf = chicago_area_tracts.merge(
                svi_gdf[['FIPS', 'SVI', 'SVI_q']], 
                left_on='GEOID', 
                right_on='FIPS', 
                how='left'
            )
            
            print(f"Merged data: {len(merged_gdf)} tracts")
            
            # Check for tracts without SVI data
            missing_svi = merged_gdf[merged_gdf['SVI'].isna()]
            if len(missing_svi) > 0:
                print(f"Found {len(missing_svi)} tracts without SVI data - will filter these out later")
                
                # Don't fill missing SVI data here - we'll handle it after filtering to city limits
                # This prevents tracts outside Chicago from getting incorrect Q2 values
            
            print(f"Final merged data: {len(merged_gdf)} tracts")
            
        else:
            # Use existing data but convert points to small circles for visualization
            print("Using existing point data (will show as circles)")
            merged_gdf = svi_gdf.copy()
            
    except Exception as e:
        print(f"ERROR: Failed to load SVI data: {e}")
        sys.exit(1)
else:
    print(f"ERROR: Could not find '{results_path}'.")
    sys.exit(1)

# ------------------ Load Chicago City Limits ------------------
city_limits_path = "data/Chicago_City_Limits-shp/Chicago_City_Limits.shp"
if os.path.exists(city_limits_path):
    print("Loading Chicago city limits...")
    try:
        chicago_city_limits = gpd.read_file(city_limits_path)
        print(f"Loaded Chicago city limits with {len(chicago_city_limits)} features")
        
        # Ensure both datasets are in the same CRS
        if merged_gdf.crs != chicago_city_limits.crs:
            chicago_city_limits = chicago_city_limits.to_crs(merged_gdf.crs)
            print("Reprojected city limits to match tracts CRS")
        
        # Filter tracts to only include those that intersect with Chicago city limits
        print("Filtering tracts to Chicago city limits...")
        # Keep tracts that intersect with city limits (includes border tracts)
        intersecting_tracts = merged_gdf[merged_gdf.geometry.intersects(chicago_city_limits.unary_union)].copy()
        print(f"Tracts intersecting city limits: {len(intersecting_tracts)}")
        
        # Instead of clipping, use intersection to maintain tract boundaries
        # This prevents gaps between adjacent tracts
        print("Intersecting tracts with city boundaries...")
        try:
            # Ensure both datasets are in the same CRS for intersection
            if intersecting_tracts.crs != chicago_city_limits.crs:
                chicago_city_limits_intersect = chicago_city_limits.to_crs(intersecting_tracts.crs)
            else:
                chicago_city_limits_intersect = chicago_city_limits
            
            # Use intersection instead of clipping to maintain tract boundaries
            # This keeps the original tract shapes while filtering to city limits
            merged_gdf = intersecting_tracts.copy()
            
            # Apply a small buffer to fill gaps without affecting fill coverage
            print("Applying small buffer to eliminate gaps...")
            try:
                # Use a small buffer that fills gaps but doesn't cause rendering issues
                merged_gdf['geometry'] = merged_gdf.geometry.buffer(0.00005)  # ~5 meters
                print("Applied small buffer to fill gaps between tracts")
            except Exception as e:
                print(f"Warning: Could not apply buffer: {e}")
            
            print(f"Using intersecting tracts without clipping: {len(merged_gdf)}")
        except Exception as e:
            print(f"Warning: Could not process tracts, using intersecting tracts: {e}")
            merged_gdf = intersecting_tracts
        
    except Exception as e:
        print(f"Warning: Could not load city limits: {e}")
        chicago_city_limits = None
else:
    print("Warning: Chicago city limits not found, using all tracts")
    chicago_city_limits = None

# ------------------ Handle Missing SVI Data After City Limits Filtering ------------------
# Now that we have filtered to city limits, we can safely fill missing SVI data
# This prevents tracts outside Chicago from getting incorrect Q2 values
if chicago_city_limits is not None and len(merged_gdf) > 0:
    # Check for tracts without SVI data within city limits
    missing_svi_in_city = merged_gdf[merged_gdf['SVI'].isna()]
    if len(missing_svi_in_city) > 0:
        print(f"Found {len(missing_svi_in_city)} tracts within city limits without SVI data")
        
        # Instead of filling all missing data with Q2, only fill data for tracts that:
        # 1. Have some SVI data in nearby tracts, or
        # 2. Are residential areas (not airport/industrial areas)
        
        # First, identify and exclude airport/industrial areas that shouldn't have SVI data
        # O'Hare Airport is roughly at coordinates (41.9786, -87.9048)
        o_hare_center = Point(-87.9048, 41.9786)
        
        # Convert to WGS84 for distance calculations
        merged_gdf_wgs84 = merged_gdf.to_crs(4326)
        
        # Mark tracts that are likely airport areas (within ~2km of O'Hare center)
        airport_buffer_distance = 0.02  # roughly 2km in degrees
        airport_tracts = []
        
        for idx, row in merged_gdf_wgs84.iterrows():
            if row.geometry is not None:
                # Check if tract is near O'Hare
                if row.geometry.distance(o_hare_center) < airport_buffer_distance:
                    # If it's a missing SVI tract near O'Hare, mark it as airport area
                    if pd.isna(row['SVI']):
                        airport_tracts.append(idx)
                        print(f"Marked tract {idx} as airport area near O'Hare - will not fill with SVI data")
        
        # Remove airport tracts from consideration for filling
        tracts_to_fill = merged_gdf[merged_gdf['SVI'].isna() & ~merged_gdf.index.isin(airport_tracts)]
        print(f"Excluding {len(airport_tracts)} airport tracts from SVI data filling")
        print(f"Will attempt to fill {len(tracts_to_fill)} non-airport tracts")
        
        # Get tracts that have SVI data
        tracts_with_svi = merged_gdf[merged_gdf['SVI'].notna()].copy()
        
        if len(tracts_with_svi) > 0:
            # Calculate distances from missing tracts to tracts with SVI data
            # Only fill tracts that are close to tracts with actual SVI data
            from shapely.geometry import Point
            import numpy as np
            
            # Convert to WGS84 for distance calculations
            merged_gdf_wgs84 = merged_gdf.to_crs(4326)
            tracts_with_svi_wgs84 = tracts_with_svi.to_crs(4326)
            
            # For each missing tract (excluding airport areas), find the closest tract with SVI data
            for idx, row in merged_gdf_wgs84.loc[tracts_to_fill.index].iterrows():
                if row.geometry is not None:
                    # Calculate distance to nearest tract with SVI data
                    distances = []
                    for _, svi_row in tracts_with_svi_wgs84.iterrows():
                        if svi_row.geometry is not None:
                            dist = row.geometry.distance(svi_row.geometry)
                            distances.append(dist)
                    
                    if distances:
                        min_distance = min(distances)
                        # Only fill if very close (within ~0.01 degrees, roughly 1km)
                        if min_distance < 0.01:
                            # Fill with the closest tract's SVI data
                            closest_idx = distances.index(min_distance)
                            closest_svi_row = tracts_with_svi.iloc[closest_idx]
                            merged_gdf.loc[idx, 'SVI_q'] = closest_svi_row['SVI_q']
                            merged_gdf.loc[idx, 'SVI'] = closest_svi_row['SVI']
                            print(f"Filled tract {idx} with SVI data from nearby tract (distance: {min_distance:.4f})")
                        else:
                            print(f"Tract {idx} is too far from SVI data (distance: {min_distance:.4f}) - leaving unfilled")
                    else:
                        print(f"Tract {idx} has no nearby SVI data - leaving unfilled")
            
            # Count how many were actually filled
            filled_count = len(merged_gdf[merged_gdf['SVI'].notna()]) - len(tracts_with_svi)
            print(f"Filled SVI data for {filled_count} tracts that were close to existing data")
        else:
            print("No tracts with SVI data found - cannot fill missing data")
else:
    # If no city limits, only fill missing data for tracts that have some SVI data
    # This prevents creating fake data for completely unknown areas
    missing_svi = merged_gdf[merged_gdf['SVI'].isna()]
    if len(missing_svi) > 0:
        print(f"Found {len(missing_svi)} tracts without SVI data - filling gaps...")
        
        # Fill missing SVI data with median values from nearby tracts
        svi_quartiles = svi_gdf['SVI_q'].value_counts().sort_index()
        svi_medians = svi_gdf.groupby('SVI_q')['SVI'].median()
        
        # Fill missing SVI_q with most common quartile, missing SVI with median
        merged_gdf['SVI_q'] = merged_gdf['SVI_q'].fillna('Q2')  # Default to middle quartile
        merged_gdf['SVI'] = merged_gdf['SVI'].fillna(svi_medians.get('Q2', merged_gdf['SVI'].median()))
        
        print(f"Filled missing SVI data for {len(missing_svi)} tracts")

# ------------------ Create SVI Choropleth Map with Matplotlib ------------------
print("\nCreating SVI choropleth map with matplotlib...")

# Color scheme for SVI quartiles using cividis palette
svi_colors = {
    'Q1(low)': '#b7d4ea',  
    'Q2':      '#6aaed6',
    'Q3':      '#2e7ebc',
    'Q4(high)': '#084a91'   
}

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Ensure we have the data in the right CRS for plotting
if boundaries_gdf is not None and merged_gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
    print("Creating choropleth with polygon boundaries...")
    
    # Convert to a suitable CRS for plotting (Web Mercator for better display)
    plot_gdf = merged_gdf.to_crs('EPSG:3857')
    
    # Clip the tracts to Chicago city limits to ensure coloring stays within city boundaries
    if chicago_city_limits is not None:
        print("Clipping census tracts to Chicago city limits...")
        city_limits_clip = chicago_city_limits.to_crs('EPSG:3857')
        
        # Clip the tracts to city boundaries
        plot_gdf = gpd.clip(plot_gdf, city_limits_clip)
        print(f"Clipped to {len(plot_gdf)} tracts within city limits")
    
    # Create a color column based on SVI quartiles
    plot_gdf['color'] = plot_gdf['SVI_q'].map(svi_colors)
    plot_gdf['color'] = plot_gdf['color'].fillna('#f5f5f5')  # Very light grey for missing data
    
    # Plot the choropleth with no gaps
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf['color'],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8
    )
    
    # Chicago city limits boundary removed - coloring is already clipped to city limits

else:
    print("Creating point-based visualization...")
    # Convert points to circles
    plot_gdf = merged_gdf.to_crs('EPSG:3857')
    
    # Filter points to only those within Chicago city limits
    if chicago_city_limits is not None:
        print("Filtering points to Chicago city limits...")
        city_limits_filter = chicago_city_limits.to_crs('EPSG:3857')
        city_union = city_limits_filter.unary_union
        plot_gdf = plot_gdf[plot_gdf.geometry.within(city_union)]
        print(f"Filtered to {len(plot_gdf)} points within city limits")
    
    plot_gdf['color'] = plot_gdf['SVI_q'].map(svi_colors)
    plot_gdf['color'] = plot_gdf['color'].fillna('#f5f5f5')  # Very light grey for missing data
    
    # Plot as scatter points
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf['color'],
        markersize=20,
        alpha=0.8
    )

# Add urgent care facilities
print("Adding urgent care facilities...")
urgentcare_csv_path = "data/urgent_care_facilities.csv"
if os.path.exists(urgentcare_csv_path):
    try:
        uc_df = pd.read_csv(urgentcare_csv_path)
        # Find latitude and longitude columns
        lat_col = None
        lon_col = None
        for c in uc_df.columns:
            if c.lower() in ['latitude', 'lat']:
                lat_col = c
            if c.lower() in ['longitude', 'lon', 'lng', 'long']:
                lon_col = c
        
        if lat_col and lon_col:
            # Drop rows with missing coordinates
            uc_df = uc_df.dropna(subset=[lat_col, lon_col])
            # Create geometry
            uc_gdf = gpd.GeoDataFrame(
                uc_df,
                geometry=[Point(xy) for xy in zip(uc_df[lon_col], uc_df[lat_col])],
                crs="EPSG:4326"
            )
            # Convert to plot CRS
            uc_gdf = uc_gdf.to_crs('EPSG:3857')
            
            # Filter to city limits if available
            if chicago_city_limits is not None:
                city_union = chicago_city_limits.to_crs('EPSG:3857').unary_union
                uc_gdf = uc_gdf[uc_gdf.geometry.within(city_union)]
            
            # Plot urgent care facilities as red dots
            for idx, row in uc_gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                ax.scatter(x, y, s=188, c='#78040f', alpha=0.9, marker='o', zorder=5)
            print(f"Added {len(uc_gdf)} urgent care facilities")
    except Exception as e:
        print(f"Failed to add urgent care facilities: {e}")

# Customize the plot
ax.set_title('Chicago SVI (Social Vulnerability Index) by Census Tract', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Remove axis ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Scale bar removed

# Create legend
from matplotlib.lines import Line2D
legend_elements = [
    mpatches.Patch(color='#b7d4ea', label='Q1 (Low Vulnerability)'),
    mpatches.Patch(color='#6aaed6', label='Q2'),
    mpatches.Patch(color='#2e7ebc', label='Q3'),
    mpatches.Patch(color='#084a91', label='Q4 (High Vulnerability)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#78040f', 
           markersize=10, label='Urgent Care Facility')
]

ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
          fontsize=10, frameon=True, fancybox=True, shadow=True)

# Set equal aspect ratio and tight layout
ax.set_aspect('equal')
plt.tight_layout()

# Save the plot
print("\nSaving map...")
output_file = "results/chicago_svi_choropleth.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Map saved: {output_file}")

print("\nSVI choropleth visualization complete!")
print(f"\nView the saved image: {output_file}") 
