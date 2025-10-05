# chicago_svi_with_boundaries.py
# Download census tract boundaries and create a proper SVI choropleth map

import pandas as pd
import geopandas as gpd
import folium
import warnings
import sys
import os
import requests
import zipfile
import tempfile
import ssl
import numpy as np
from shapely.geometry import Point, box

# --- Determine correct paths relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")

# Try to import matplotlib for the multiple choropleth function
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

print("  Creating Chicago SVI Visualization with Census Tract Boundaries...")
print("=" * 60)

# ------------------ Load Census Tract Boundaries from Data Folder ------------------
print(" Loading census tract boundaries from data folder...")

tract_zip_path = os.path.join(DATA_DIR, "tl_2022_17_tract.zip")
if os.path.exists(tract_zip_path):
    try:
        print(f" Found tract data: {tract_zip_path}")
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tract_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
        if not shp_files:
            raise Exception("No shapefile found in tract data")
        shp_path = os.path.join(extract_dir, shp_files[0])
        print(f" Extracted: {shp_files[0]}")
        boundaries_gdf = gpd.read_file(shp_path)
        print(f" Loaded {len(boundaries_gdf)} census tracts")
        import shutil
        shutil.rmtree(extract_dir)
    except Exception as e:
        print(f"ERROR: Failed to load tract data: {e}")
        boundaries_gdf = None
else:
    print(f"ERROR: Tract data not found at {tract_zip_path}")
    boundaries_gdf = None

# ------------------ Load SVI Data ------------------
print(" Loading SVI data...")

results_path = os.path.join(PROCESSED_DATA_DIR, "chicago_urgentcare_final_results.geojson")
if os.path.exists(results_path):
    try:
        svi_gdf = gpd.read_file(results_path)
        print(f" Loaded SVI data: {len(svi_gdf)} records")
        if boundaries_gdf is not None:
            print(" Merging SVI data with census boundaries...")
            svi_gdf['FIPS'] = svi_gdf['FIPS'].astype(str)
            boundaries_gdf['GEOID'] = boundaries_gdf['GEOID'].astype(str)
            chicago_bbox = [-87.9401, 41.6445, -87.5241, 42.0230]
            chicago_area_tracts = boundaries_gdf.cx[chicago_bbox[0]:chicago_bbox[2], chicago_bbox[1]:chicago_bbox[3]]
            print(f" Found {len(chicago_area_tracts)} tracts in Chicago area")
            # Only merge SVI score, not quartile
            merged_gdf = chicago_area_tracts.merge(
                svi_gdf[['FIPS', 'SVI']],
                left_on='GEOID',
                right_on='FIPS',
                how='left'
            )
            print(f" Merged data: {len(merged_gdf)} tracts")
            missing_svi = merged_gdf[merged_gdf['SVI'].isna()]
            if len(missing_svi) > 0:
                print(f"Warning:  Found {len(missing_svi)} tracts without SVI data - will filter these out later")
            print(f" Final merged data: {len(merged_gdf)} tracts")
        else:
            print("Warning:  Using existing point data (will show as circles)")
            merged_gdf = svi_gdf.copy()
    except Exception as e:
        print(f"ERROR: ERROR: Failed to load SVI data: {e}")
        sys.exit(1)
else:
    print(f"ERROR: ERROR: Could not find '{results_path}'.")
    sys.exit(1)

# ------------------ Load Chicago City Limits ------------------
city_limits_path = os.path.join(DATA_DIR, "Chicago_City_Limits-shp", "Chicago_City_Limits.shp")
if os.path.exists(city_limits_path):
    print("  Loading Chicago city limits...")
    try:
        chicago_city_limits = gpd.read_file(city_limits_path)
        print(f" Loaded Chicago city limits with {len(chicago_city_limits)} features")
        if merged_gdf.crs != chicago_city_limits.crs:
            chicago_city_limits = chicago_city_limits.to_crs(merged_gdf.crs)
            print(" Reprojected city limits to match tracts CRS")
        print(" Filtering tracts to Chicago city limits...")
        intersecting_tracts = merged_gdf[merged_gdf.geometry.intersects(chicago_city_limits.unary_union)].copy()
        print(f" Tracts intersecting city limits: {len(intersecting_tracts)}")
        print("  Clipping tracts to city boundaries...")
        try:
            if intersecting_tracts.crs != chicago_city_limits.crs:
                chicago_city_limits_clip = chicago_city_limits.to_crs(intersecting_tracts.crs)
            else:
                chicago_city_limits_clip = chicago_city_limits
            merged_gdf = gpd.clip(intersecting_tracts, chicago_city_limits_clip)
            print(f" Clipped tracts to city boundaries: {len(merged_gdf)}")
        except Exception as e:
            print(f"Warning:  Warning: Could not clip tracts, using intersecting tracts: {e}")
            merged_gdf = intersecting_tracts
    except Exception as e:
        print(f"Warning:  Warning: Could not load city limits: {e}")
        chicago_city_limits = None
else:
    print("Warning:  Warning: Chicago city limits not found, using all tracts")
    chicago_city_limits = None

# ------------------ Handle Missing SVI Data After City Limits Filtering ------------------
if chicago_city_limits is not None and len(merged_gdf) > 0:
    missing_svi_in_city = merged_gdf[merged_gdf['SVI'].isna()]
    if len(missing_svi_in_city) > 0:
        print(f"Warning:  Found {len(missing_svi_in_city)} tracts within city limits without SVI data")
        o_hare_center = Point(-87.9048, 41.9786)
        merged_gdf_wgs84 = merged_gdf.to_crs(4326)
        airport_buffer_distance = 0.02
        airport_tracts = []
        for idx, row in merged_gdf_wgs84.iterrows():
            if row.geometry is not None:
                if row.geometry.distance(o_hare_center) < airport_buffer_distance:
                    if pd.isna(row['SVI']):
                        airport_tracts.append(idx)
                        print(f"Warning:  Marked tract {idx} as airport area near O'Hare - will not fill with SVI data")
        tracts_to_fill = merged_gdf[merged_gdf['SVI'].isna() & ~merged_gdf.index.isin(airport_tracts)]
        print(f"Warning:  Excluding {len(airport_tracts)} airport tracts from SVI data filling")
        print(f"Warning:  Will attempt to fill {len(tracts_to_fill)} non-airport tracts")
        tracts_with_svi = merged_gdf[merged_gdf['SVI'].notna()].copy()
        if len(tracts_with_svi) > 0:
            import numpy as np
            merged_gdf_wgs84 = merged_gdf.to_crs(4326)
            tracts_with_svi_wgs84 = tracts_with_svi.to_crs(4326)
            for idx, row in merged_gdf_wgs84.loc[tracts_to_fill.index].iterrows():
                if row.geometry is not None:
                    distances = []
                    for _, svi_row in tracts_with_svi_wgs84.iterrows():
                        if svi_row.geometry is not None:
                            dist = row.geometry.distance(svi_row.geometry)
                            distances.append(dist)
                    if distances:
                        min_distance = min(distances)
                        if min_distance < 0.01:
                            closest_idx = distances.index(min_distance)
                            closest_svi_row = tracts_with_svi.iloc[closest_idx]
                            merged_gdf.loc[idx, 'SVI'] = closest_svi_row['SVI']
                            print(f" Filled tract {idx} with SVI data from nearby tract (distance: {min_distance:.4f})")
                        else:
                            print(f"Warning:  Tract {idx} is too far from SVI data (distance: {min_distance:.4f}) - leaving unfilled")
                    else:
                        print(f"Warning:  Tract {idx} has no nearby SVI data - leaving unfilled")
            filled_count = len(merged_gdf[merged_gdf['SVI'].notna()]) - len(tracts_with_svi)
            print(f" Filled SVI data for {filled_count} tracts that were close to existing data")
        else:
            print("Warning:  No tracts with SVI data found - cannot fill missing data")
else:
    missing_svi = merged_gdf[merged_gdf['SVI'].isna()]
    if len(missing_svi) > 0:
        print(f"Warning:  Found {len(missing_svi)} tracts without SVI data - filling gaps...")
        merged_gdf['SVI'] = merged_gdf['SVI'].fillna(merged_gdf['SVI'].median())
        print(f" Filled missing SVI data for {len(missing_svi)} tracts")

# ------------------ Create SVI Choropleth Map ------------------
print("\n  Creating SVI choropleth map...")

chicago_center = [41.8781, -87.6298]
m = folium.Map(location=chicago_center, zoom_start=10, tiles='cartodbpositron')

if chicago_city_limits is not None:
    import numpy as np
    city_limits_wgs84 = chicago_city_limits.to_crs(4326)
    minx, miny, maxx, maxy = city_limits_wgs84.total_bounds
    world = box(-180, -90, 180, 90)
    try:
        city_union = city_limits_wgs84.unary_union
        mask_geom = world.difference(city_union)
        mask_gdf = gpd.GeoDataFrame(geometry=[mask_geom], crs="EPSG:4326")
        folium.GeoJson(
            mask_gdf,
            name='Mask',
            style_function=lambda x: {
                'fillColor': 'white',
                'color': 'white',
                'weight': 0,
                'fillOpacity': 1
            },
            tooltip=None
        ).add_to(m)
    except Exception as e:
        print(f"Warning:  Could not add white mask outside city limits: {e}")

# Color scheme for SVI (continuous) - use a blue-red gradient
from matplotlib import cm, colors
def svi_color_scale(svi_value, min_svi, max_svi):
    # Use a blue-red colormap (low=blue, high=red)
    norm = colors.Normalize(vmin=min_svi, vmax=max_svi)
    cmap = cm.get_cmap('RdYlBu_r')
    rgba = cmap(norm(svi_value))
    return colors.to_hex(rgba)

# Get min/max SVI for color scaling
if 'SVI' in merged_gdf.columns:
    min_svi = merged_gdf['SVI'].min()
    max_svi = merged_gdf['SVI'].max()
else:
    min_svi = 0
    max_svi = 1

def choropleth_style_function(feature):
    svi = feature['properties'].get('SVI', None)
    if svi is None or pd.isna(svi):
        return {
            'fillColor': 'transparent',
            'color': 'lightgray',
            'weight': 0.5,
            'fillOpacity': 0
        }
    color = svi_color_scale(float(svi), min_svi, max_svi)
    return {
        'fillColor': color,
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.8
    }

if boundaries_gdf is not None and merged_gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
    print(" Creating choropleth with polygon boundaries...")
    folium.GeoJson(
        merged_gdf.to_crs(4326),
        name='SVI Choropleth',
        style_function=choropleth_style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['SVI'],
            aliases=['SVI Score:'],
            localize=True,
            sticky=True,
            labels=True,
            toLocaleString=True,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 5px;")
        ),
        popup=folium.GeoJsonPopup(
            fields=['SVI'],
            aliases=['SVI Score'],
            localize=True,
            labels=True,
            style="background-color: white;"
        )
    ).add_to(m)
else:
    print("Warning:  Creating point-based visualization (not a true choropleth)...")
    if chicago_city_limits is not None:
        if merged_gdf.crs != chicago_city_limits.crs:
            merged_gdf = merged_gdf.to_crs(chicago_city_limits.crs)
        merged_gdf = merged_gdf[merged_gdf.geometry.intersects(chicago_city_limits.unary_union)].copy()
    for idx, row in merged_gdf.iterrows():
        if pd.notna(row.geometry):
            svi = row.get('SVI', None)
            if svi is not None and not pd.isna(svi):
                color = svi_color_scale(float(svi), min_svi, max_svi)
            else:
                color = 'gray'
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=1,
                popup=f"SVI Score: {row.get('SVI', 'N/A'):.2f}"
            ).add_to(m)

# ------------------ Add Urgent Care Facilities as Dots with Crosses ------------------
print(" Adding urgent care facilities as dots with crosses from CSV...")

urgentcare_csv_path = os.path.join(DATA_DIR, "urgent_care_facilities.csv")
if os.path.exists(urgentcare_csv_path):
    try:
        uc_df = pd.read_csv(urgentcare_csv_path)
        lat_col = None
        lon_col = None
        for c in uc_df.columns:
            if c.lower() in ['latitude', 'lat']:
                lat_col = c
            if c.lower() in ['longitude', 'lon', 'lng', 'long']:
                lon_col = c
        if lat_col is None or lon_col is None:
            raise Exception("Could not find latitude/longitude columns in urgent care CSV")
        uc_df = uc_df.dropna(subset=[lat_col, lon_col])
        uc_gdf = gpd.GeoDataFrame(
            uc_df,
            geometry=[Point(xy) for xy in zip(uc_df[lon_col], uc_df[lat_col])],
            crs="EPSG:4326"
        )
        if chicago_city_limits is not None:
            if uc_gdf.crs != chicago_city_limits.to_crs(4326).crs:
                uc_gdf = uc_gdf.to_crs(chicago_city_limits.to_crs(4326).crs)
            chicago_union = chicago_city_limits.to_crs(4326).unary_union
            uc_gdf = uc_gdf[uc_gdf.geometry.within(chicago_union)]
        for idx, row in uc_gdf.iterrows():
            lat = row.geometry.y
            lon = row.geometry.x
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color='red',
                fill=True,
                fillColor='white',
                fillOpacity=1,
                weight=2,
                popup=row.get('name', 'Urgent Care')
            ).add_to(m)
            cross_size = 0.0015
            folium.PolyLine(
                locations=[
                    [lat - cross_size, lon],
                    [lat + cross_size, lon]
                ],
                color='red',
                weight=2,
                opacity=1
            ).add_to(m)
            folium.PolyLine(
                locations=[
                    [lat, lon - cross_size],
                    [lat, lon + cross_size]
                ],
                color='red',
                weight=2,
                opacity=1
            ).add_to(m)
    except Exception as e:
        print(f"ERROR: Failed to plot urgent care facilities from CSV: {e}")
else:
    print(f"Warning:  Urgent care facilities CSV not found at {urgentcare_csv_path}")

# Add a custom legend for SVI (continuous) and urgent care
legend_html_svi = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 270px; height: 120px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 14px 12px 10px 12px; box-sizing: border-box;">
            <p style="margin:0 0 8px 0;"><b>SVI Score (Continuous)</b></p>
            <div style="height:18px;width:180px;background:linear-gradient(to right,#053061,#2166ac,#ef8a62,#67001f);margin-bottom:8px;border:1px solid #888"></div>
            <div style="display:flex;justify-content:space-between;font-size:12px;">
                <span>Low</span>
                <span>High</span>
            </div>
            <p style="margin:10px 0 0 0;">
                <span style="display:inline-block;vertical-align:middle;width:18px;height:18px;position:relative;">
                    <span style="display:block;width:18px;height:18px;border-radius:9px;background:white;border:2px solid red;position:absolute;top:0;left:0;"></span>
                    <span style="display:block;width:2px;height:12px;background:red;position:absolute;top:3px;left:8px;"></span>
                    <span style="display:block;width:12px;height:2px;background:red;position:absolute;top:8px;left:3px;"></span>
                </span>
                <span style="margin-left:8px;">Urgent Care Facility</span>
            </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html_svi))

folium.LayerControl().add_to(m)

print("\n Saving map...")
output_file = os.path.join(RESULTS_DIR, "chicago_svi_choropleth.html")
m.save(output_file)
print(f" Map saved: {output_file}")
print("\n SVI choropleth visualization complete!")
print(f"\n Open '{output_file}' in your browser to explore the SVI map.") 

# ------------------ New Function: Create Multiple SVI Variable Choropleths ------------------
def create_multiple_svi_choropleths():
    """
    Create 5 choropleth maps for different SVI variables:
    1. Age 65+
    2. Single Parent
    3. Disability
    4. Age 17 & Under
    5. Housing Burden
    All variables are normalized to 0-1.
    """
    print("\n  Creating Multiple SVI Variable Choropleths...")
    print("=" * 60)
    
    svi_path = os.path.join(DATA_DIR, "SVI", "Illinois.csv")
    if not os.path.exists(svi_path):
        print(f"ERROR: SVI data not found at {svi_path}")
        return
    
    try:
        svi_data = pd.read_csv(svi_path, dtype=str)
        print(f" Loaded SVI data: {len(svi_data)} records")
        svi_chi = svi_data[svi_data['FIPS'].str.startswith('17031')].copy()
        print(f" Filtered to Chicago tracts: {len(svi_chi)} records")
        numeric_columns = ['E_AGE65', 'E_SNGPNT', 'E_DISABL', 'E_AGE17', 'E_HBURD', 'E_TOTPOP']
        for col in numeric_columns:
            if col in svi_chi.columns:
                svi_chi[col] = pd.to_numeric(svi_chi[col], errors='coerce')
        if 'E_TOTPOP' in svi_chi.columns and 'E_AGE65' in svi_chi.columns:
            svi_chi['AGE65_PCT'] = (svi_chi['E_AGE65'] / svi_chi['E_TOTPOP'] * 100).fillna(0)
        if 'E_TOTPOP' in svi_chi.columns and 'E_SNGPNT' in svi_chi.columns:
            svi_chi['SINGLE_PARENT_PCT'] = (svi_chi['E_SNGPNT'] / svi_chi['E_TOTPOP'] * 100).fillna(0)
        if 'E_TOTPOP' in svi_chi.columns and 'E_DISABL' in svi_chi.columns:
            svi_chi['DISABILITY_PCT'] = (svi_chi['E_DISABL'] / svi_chi['E_TOTPOP'] * 100).fillna(0)
        if 'E_TOTPOP' in svi_chi.columns and 'E_AGE17' in svi_chi.columns:
            svi_chi['AGE17_UNDER_PCT'] = (svi_chi['E_AGE17'] / svi_chi['E_TOTPOP'] * 100).fillna(0)
        if 'E_TOTPOP' in svi_chi.columns and 'E_HBURD' in svi_chi.columns:
            svi_chi['HOUSING_BURDEN_PCT'] = (svi_chi['E_HBURD'] / svi_chi['E_TOTPOP'] * 100).fillna(0)
        if 'EP_AGE65' in svi_chi.columns:
            svi_chi['AGE65_PCT'] = pd.to_numeric(svi_chi['EP_AGE65'], errors='coerce').fillna(0)
        if 'EP_SNGPNT' in svi_chi.columns:
            svi_chi['SINGLE_PARENT_PCT'] = pd.to_numeric(svi_chi['EP_SNGPNT'], errors='coerce').fillna(0)
        if 'EP_DISABL' in svi_chi.columns:
            svi_chi['DISABILITY_PCT'] = pd.to_numeric(svi_chi['EP_DISABL'], errors='coerce').fillna(0)
        if 'EP_AGE17' in svi_chi.columns:
            svi_chi['AGE17_UNDER_PCT'] = pd.to_numeric(svi_chi['EP_AGE17'], errors='coerce').fillna(0)
        if 'EP_HBURD' in svi_chi.columns:
            svi_chi['HOUSING_BURDEN_PCT'] = pd.to_numeric(svi_chi['EP_HBURD'], errors='coerce').fillna(0)
        print(" Normalizing continuous variables across all graphs...")
        continuous_vars = ['AGE65_PCT', 'SINGLE_PARENT_PCT', 'DISABILITY_PCT', 'AGE17_UNDER_PCT', 'HOUSING_BURDEN_PCT']
        continuous_data = svi_chi[continuous_vars].values
        global_min = np.nanmin(continuous_data)
        global_max = np.nanmax(continuous_data)
        print(f"Global range across all variables: {global_min:.2f} to {global_max:.2f}")
        for var in continuous_vars:
            if var in svi_chi.columns:
                normalized_col = var.replace('_PCT', '_NORM')
                svi_chi[normalized_col] = (svi_chi[var] - global_min) / (global_max - global_min)
                print(f" Normalized {var} to {normalized_col} (0-1 scale)")
        svi_chi['AGE65_NORM'] = svi_chi['AGE65_PCT'].apply(lambda x: (x - global_min) / (global_max - global_min) if pd.notna(x) else np.nan)
        svi_chi['SINGLE_PARENT_NORM'] = svi_chi['SINGLE_PARENT_PCT'].apply(lambda x: (x - global_min) / (global_max - global_min) if pd.notna(x) else np.nan)
        svi_chi['DISABILITY_NORM'] = svi_chi['DISABILITY_PCT'].apply(lambda x: (x - global_min) / (global_max - global_min) if pd.notna(x) else np.nan)
        svi_chi['AGE17_UNDER_NORM'] = svi_chi['AGE17_UNDER_PCT'].apply(lambda x: (x - global_min) / (global_max - global_min) if pd.notna(x) else np.nan)
        svi_chi['HOUSING_BURDEN_NORM'] = svi_chi['HOUSING_BURDEN_PCT'].apply(lambda x: (x - global_min) / (global_max - global_min) if pd.notna(x) else np.nan)
        tract_zip_path = os.path.join(DATA_DIR, "tl_2022_17_tract.zip")
        if not os.path.exists(tract_zip_path):
            print(f"ERROR: Tract boundaries not found at {tract_zip_path}")
            return
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tract_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
        if not shp_files:
            print("ERROR: No shapefile found in tract data")
            return
        shp_path = os.path.join(extract_dir, shp_files[0])
        boundaries_gdf = gpd.read_file(shp_path)
        import shutil
        shutil.rmtree(extract_dir)
        chicago_bbox = [-87.9401, 41.6445, -87.5241, 42.0230]
        chicago_area_tracts = boundaries_gdf.cx[chicago_bbox[0]:chicago_bbox[2], chicago_bbox[1]:chicago_bbox[3]]
        svi_chi['FIPS'] = svi_chi['FIPS'].astype(str)
        chicago_area_tracts['GEOID'] = chicago_area_tracts['GEOID'].astype(str)
        merged_gdf = chicago_area_tracts.merge(
            svi_chi[['FIPS', 'AGE65_PCT', 'SINGLE_PARENT_PCT', 
                     'DISABILITY_PCT', 'AGE17_UNDER_PCT', 'HOUSING_BURDEN_PCT',
                     'AGE65_NORM', 'SINGLE_PARENT_NORM', 'DISABILITY_NORM', 
                     'AGE17_UNDER_NORM', 'HOUSING_BURDEN_NORM']],
            left_on='GEOID',
            right_on='FIPS',
            how='left'
        )
        city_limits_path = os.path.join(DATA_DIR, "Chicago_City_Limits-shp", "Chicago_City_Limits.shp")
        if os.path.exists(city_limits_path):
            chicago_city_limits = gpd.read_file(city_limits_path)
            if merged_gdf.crs != chicago_city_limits.crs:
                chicago_city_limits = chicago_city_limits.to_crs(merged_gdf.crs)
            intersecting_tracts = merged_gdf[merged_gdf.geometry.intersects(chicago_city_limits.unary_union)].copy()
            merged_gdf = intersecting_tracts
        consistent_colors = ["#FDE333", "#BDBF55", "#7F9973", "#496C84", "#26456E", "#00224E"]
        variables = {
            'Age 65+ (Normalized)': {
                'column': 'AGE65_NORM',
                'title': 'Population Age 65+ (Normalized 0-1)',
                'type': 'continuous',
                'colors': consistent_colors,
                'bins': None
            },
            'Single Parent (Normalized)': {
                'column': 'SINGLE_PARENT_NORM',
                'title': 'Single Parent Households (Normalized 0-1)',
                'type': 'continuous',
                'colors': consistent_colors,
                'bins': None
            },
            'Disability (Normalized)': {
                'column': 'DISABILITY_NORM',
                'title': 'Population with Disability (Normalized 0-1)',
                'type': 'continuous',
                'colors': consistent_colors,
                'bins': None
            },
            'Age 17 & Under (Normalized)': {
                'column': 'AGE17_UNDER_NORM',
                'title': 'Population Age 17 & Under (Normalized 0-1)',
                'type': 'continuous',
                'colors': consistent_colors,
                'bins': None
            },
            'Housing Burden (Normalized)': {
                'column': 'HOUSING_BURDEN_NORM',
                'title': 'Households with Housing Cost > 30% of Income (Normalized 0-1)',
                'type': 'continuous',
                'colors': consistent_colors,
                'bins': None
            }
        }
        fig, axes = plt.subplots(2, 3, figsize=(20, 16))
        axes_flat = axes.flatten()
        for idx, (var_name, var_info) in enumerate(variables.items()):
            ax = axes_flat[idx]
            plot_data = merged_gdf[merged_gdf[var_info['column']].notna()].copy()
            
            # Create a custom colormap with exact color mapping
            # Use 6 colors for 6 quantile bins to ensure exact color matching
            custom_cmap = LinearSegmentedColormap.from_list('custom', consistent_colors, N=6)
            
            plot_data.plot(
                column=var_info['column'],
                ax=ax,
                legend=False,  # Remove individual legends
                missing_kwds={'color': 'lightgray'},
                cmap=custom_cmap,
                scheme='quantiles',
                k=6  # Use 6 bins to match our 6 colors exactly
            )
            ax.set_title(var_info['title'], fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')
        # Hide the unused 6th subplot
        if len(variables) < len(axes_flat):
            axes_flat[-1].axis('off')
        
        # Add a single legend at the bottom right
        # Create a custom legend showing the color scale
        from matplotlib.patches import Rectangle
        
        # Position the legend at the bottom right
        legend_ax = fig.add_axes([0.85, 0.05, 0.12, 0.15])
        legend_ax.axis('off')
        
        # Create color patches for the legend
        legend_colors = consistent_colors
        legend_labels = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
        
        for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
            # Reverse the order so 0.0 is at top and 1.0 is at bottom
            y_pos = (5 - i) * 0.12
            rect = Rectangle((0, y_pos), 0.3, 0.1, facecolor=color, edgecolor='black', linewidth=0.5)
            legend_ax.add_patch(rect)
            legend_ax.text(0.35, y_pos + 0.05, label, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        output_file = os.path.join(RESULTS_DIR, "chicago_multiple_svi_choropleths_normalized.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Combined choropleth maps saved: {output_file}")
        plt.show()
        print(f"\n All multiple SVI choropleth visualizations complete!")
        print(f" Check the 'results' folder for all generated maps.")
        print(f" Note: Continuous variables are normalized to 0-1 scale across all graphs for fair comparison.")
    except Exception as e:
        print(f"ERROR: Error creating multiple SVI choropleths: {e}")
        import traceback
        traceback.print_exc()

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    print("  Creating Chicago SVI Visualization with Census Tract Boundaries...")
    print("=" * 60)
    # ... existing code for original visualization ...
    print("\n" + "=" * 60)
    print("Creating Additional SVI Variable Choropleths (Normalized)...")
    print("=" * 60)
    if MATPLOTLIB_AVAILABLE:
        create_multiple_svi_choropleths()
    else:
        print("Warning:  Matplotlib not available for multiple choropleths")
        print(" Install matplotlib to create the multiple variable choropleths:")
        print("   pip install matplotlib")
        print(" Note: The normalized choropleths will show continuous variables on a 0-1 scale for fair comparison.") 