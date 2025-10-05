# ------------------ Load Data from CSV ------------------
print("Loading travel time data from CSV...")
print("=" * 60)

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.wkt import loads
import warnings
import os
warnings.filterwarnings('ignore')

# --- Determine correct paths relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")

# Load the travel times data from CSV
try:
    # Read the CSV file
    analysis_df = pd.read_csv(os.path.join(RESULTS_DIR, "travel_times_final.csv"))
    print(f"Loaded {len(analysis_df)} census tracts from CSV")
    
    # Debug: Show what columns are actually in the CSV
    print(f" CSV columns: {list(analysis_df.columns)}")
    
    
    # Convert geometry column from WKT string to geometry objects
    print(" Converting geometry data...")
    
    # Filter out rows with NaN geometry values
    valid_geometry_mask = analysis_df['geometry'].notna() & (analysis_df['geometry'] != '')
    analysis_df = analysis_df[valid_geometry_mask].copy()
    print(f"   Filtered to {len(analysis_df)} tracts with valid geometry")
    
    # Convert WKT strings to geometry objects
    def safe_load_geometry(wkt_string):
        try:
            if pd.isna(wkt_string) or wkt_string == '':
                return None
            return loads(str(wkt_string))
        except Exception as e:
            print(f"   Warning:  Warning: Could not parse geometry: {e}")
            return None
    
    analysis_df['geometry'] = analysis_df['geometry'].apply(safe_load_geometry)
    
    # Remove rows where geometry conversion failed
    analysis_df = analysis_df[analysis_df['geometry'].notna()].copy()
    print(f"   Final dataset: {len(analysis_df)} tracts with valid geometry")
    
    # Convert to GeoDataFrame
    analysis_df = gpd.GeoDataFrame(analysis_df, geometry='geometry', crs=4326)
    print(" Converted to GeoDataFrame")
    
    
    # Check what modes are available
    print(f"\n Checking available transportation modes in CSV...")
    print(f"   CSV columns: {[col for col in analysis_df.columns if 'time' in col.lower()]}")
    
    # Show data types for time columns
    for col in [col for col in analysis_df.columns if 'time' in col.lower()]:
        print(f"    {col}: dtype={analysis_df[col].dtype}, non-null={analysis_df[col].notna().sum()}")
    
    available_modes = []
    for mode in ['car', 'transit', 'walking']:
        time_col = f'{mode}_time_min'
        if time_col in analysis_df.columns:
            data_count = analysis_df[time_col].notna().sum()
            total_count = len(analysis_df)
            
            # Show some sample values to debug
            non_null_data = analysis_df[time_col].dropna()
            if len(non_null_data) > 0:
                sample_values = non_null_data.head(3).tolist()
                print(f"    {mode}: {data_count}/{total_count} tracts have data ({data_count/total_count*100:.1f}%)")
                print(f"      Sample values: {sample_values}")
                available_modes.append(mode)
            else:
                print(f"   Warning:  {mode}: Column exists but all values are NaN")
        else:
            print(f"   ERROR: {mode}: Column '{time_col}' not found in CSV")
    
    # Include all available modes
    active_modes = available_modes
    print(f"\n Analyzing modes: {', '.join(active_modes)}")
    
    if len(active_modes) == 0:
        print("ERROR: ERROR: No transportation modes found with data!")
        print("   Please check your CSV file for columns like:")
        print("   - car_time_min")
        print("   - transit_time_min") 
        print("   - walking_time_min")
        exit(1)
    
    # Check data quality
    print(f"\n Data Quality Check:")
    for mode in active_modes:
        time_col = f'{mode}_time_min'
        if time_col in analysis_df.columns:
            valid_data = analysis_df[time_col].notna().sum()
            total_data = len(analysis_df)
            print(f"   {mode}: {valid_data}/{total_data} tracts ({valid_data/total_data*100:.1f}%)")
            
            if valid_data > 0:
                mean_time = analysis_df[time_col].mean()
                print(f"     Average time: {mean_time:.1f} minutes")
    
    # Keep all tracts for mapping, but identify data availability for statistics
    print(f"\n Identifying data availability for statistics...")
    mode_columns = [f'{mode}_time_min' for mode in active_modes if f'{mode}_time_min' in analysis_df.columns]
    
    # Count how many modes have data for each tract
    analysis_df['n_modes'] = analysis_df[mode_columns].notna().sum(axis=1)
    
    # Keep ALL tracts for mapping (including South Deering and tracts with partial data)
    map_df = analysis_df.copy()  # Use this for maps (no drop)
    
    # For statistics, include tracts with ANY transportation data (not requiring 2+ modes)
    tracts_with_any_data = analysis_df['n_modes'] >= 1
    complete_data = analysis_df[tracts_with_any_data].copy()
    
    print(f"   Total tracts for mapping: {len(map_df)} tracts")
    print(f"   Tracts with any transportation data (for statistics): {len(complete_data)} tracts")
    print(f"   Tracts with 2+ modes: {len(analysis_df[analysis_df['n_modes'] >= 2])} tracts")
    print(f"   Tracts with only 1 mode: {len(analysis_df[analysis_df['n_modes'] == 1])} tracts")
    
    # Show data availability for each mode
    print(f"\n Data availability per mode:")
    for mode in active_modes:
        time_col = f'{mode}_time_min'
        if time_col in analysis_df.columns:
            # Count in the complete dataset (tracts with any data)
            available_count = complete_data[time_col].notna().sum()
            total_count = len(complete_data)
            print(f"   {mode}: {available_count}/{total_count} tracts ({available_count/total_count*100:.1f}%)")
        else:
            print(f"   {mode}: Column not found")
    
    # Use complete_data for statistics (includes tracts with any transportation data)
    analysis_df = complete_data.copy()
    
except Exception as e:
    print(f"ERROR: Error loading CSV: {e}")
    exit(1)

# ------------------ Create Geographical Maps: All Transportation Modes ------------------
print(f"\n  Creating geographical maps comparing all transportation modes...")
print("=" * 60)

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# Create a large figure with subplots for all modes comparison
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Transportation Mode Access to Urgent Care - Chicago Census Tracts', fontsize=20, fontweight='bold')

# Define time intervals and colors for accessibility mapping
# Use a red-to-blue color scale with increasing darkness for better black & white distinction
time_intervals = [10, 20, 30, 40, 50, 60]  # minutes
interval_colors = ["#FDE333", "#BDBF55", "#7F9973", "#496C84", "#26456E", "#00224E"]

# Create accessibility categories for each mode and time interval
def categorize_accessibility(time_series, interval):
    """Categorize tracts based on whether they can reach urgent care within the time interval"""
    return (time_series <= interval).astype(int)

# Calculate accessibility for all modes
accessibility_data = {}
for mode in active_modes:
    accessibility_data[mode] = {}
    time_col = f'{mode}_time_min'
    
    for interval in time_intervals:
        accessibility_data[mode][interval] = categorize_accessibility(analysis_df[time_col], interval)

# Create the geographical side-by-side comparison figure
print(" Creating geographical accessibility maps...")

# Mode configurations for the subplots
mode_configs = [
    ('car', 'Car Access', 0, 0),
    ('transit', 'Public Transit Access', 0, 1),
    ('walking', 'Walking Access', 1, 0)
]

# Create each subplot
for mode, mode_name, row, col in mode_configs:
    ax = axes[row, col]
    
    if mode in active_modes:
        # Create a copy of the map dataframe for this mode (includes all tracts)
        plot_gdf = map_df.copy()
        
        # Add accessibility categories for this mode
        for interval in time_intervals:
            plot_gdf[f'accessible_{interval}min'] = accessibility_data[mode][interval]
        
        # Create a color mapping based on accessibility
        def get_accessibility_color(row):
            """Determine color based on accessibility time intervals"""
            # Check if this tract has data for this mode
            time_col = f'{mode}_time_min'
            if pd.isna(row[time_col]):
                return '#F5F5F5'  # Very light gray - no data available
            
            if row[f'accessible_10min'] == 1:
                return interval_colors[0]  # Dark Red - 10 min
            elif row[f'accessible_20min'] == 1:
                return interval_colors[1]  # Light Red - 20 min
            elif row[f'accessible_30min'] == 1:
                return interval_colors[2]  # Orange - 30 min
            elif row[f'accessible_40min'] == 1:
                return interval_colors[3]  # Light Blue - 40 min
            elif row[f'accessible_50min'] == 1:
                return interval_colors[4]  # Dark Blue - 50 min
            elif row[f'accessible_60min'] == 1:
                return interval_colors[5]  # Darker Blue - 60 min
            else:
                return '#E0E0E0'  # Light Gray - not accessible within 60 min
        
        # Apply color mapping
        plot_gdf['accessibility_color'] = plot_gdf.apply(get_accessibility_color, axis=1)
        
        # Plot the census tracts
        plot_gdf.plot(ax=ax, color=plot_gdf['accessibility_color'], edgecolor='white', linewidth=0.3)
        
        # Customize the plot
        ax.set_title(f'{mode_name} by Time Intervals', fontsize=16, fontweight='bold', pad=20)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Show subplot but indicate no data available
        ax.set_title(f'{mode_name} - No Data Available', fontsize=16, fontweight='bold', pad=20, color='gray')
        ax.text(0.5, 0.5, 'No travel time data\navailable for this mode', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, color='gray', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])

# Create a single legend for all time intervals
legend_elements = []
legend_labels = []

for i, interval in enumerate(time_intervals):
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=interval_colors[i], edgecolor='white', linewidth=2))
    legend_labels.append(f'{interval} min')

# Add light gray for not accessible within 60 min
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='#E0E0E0', edgecolor='white', linewidth=2))
legend_labels.append('>60 min')

# Hide the unused subplot (bottom right)
axes[1, 1].axis('off')

# Add the single legend to the figure
fig.legend(legend_elements, legend_labels, title='Accessibility Time Intervals', 
                loc='center right', bbox_to_anchor=(1.02, 0.5), framealpha=0.9, 
                fontsize=12, ncol=1, title_fontsize=14)

# Adjust layout to accommodate the single legend
plt.tight_layout()
plt.subplots_adjust(top=0.94, right=0.85)

# Save the geographical comparison
geographical_file = os.path.join(RESULTS_DIR, "all_modes_geographical_comparison.png")
plt.savefig(geographical_file, dpi=300, bbox_inches='tight')
print(f" Geographical all modes comparison saved: {geographical_file}")

# ------------------ Create Statistical Summary ------------------
print("\n Creating statistical summary...")

# Calculate summary statistics
print(f"\n TRAVEL TIME SUMMARY - ALL TRANSPORTATION MODES:")
print("=" * 60)

# Calculate statistics for each mode
mode_stats = {}
for mode in active_modes:
    time_col = f'{mode}_time_min'
    times = analysis_df[time_col]
    
    mode_stats[mode] = {
        'mean': times.mean(),
        'median': times.median(),
        'min': times.min(),
        'max': times.max(),
        'std': times.std()
    }
    
    print(f"\n{mode.upper()} ACCESS:")
    print(f"   Mean: {mode_stats[mode]['mean']:.1f} min")
    print(f"   Median: {mode_stats[mode]['median']:.1f} min")
    print(f"   Range: {mode_stats[mode]['min']:.1f}-{mode_stats[mode]['max']:.1f} min")
    print(f"   Std Dev: {mode_stats[mode]['std']:.1f} min")

# Show which modes are missing data
missing_modes = ['car', 'transit', 'walking']
for mode in missing_modes:
    if mode not in active_modes:
        print(f"\n{mode.upper()} ACCESS:")
        print(f"   ERROR: No data available")
        print(f"   This mode was not found in the CSV file")

# Calculate mode comparison analysis
print(f"\n TRANSPORTATION MODE COMPARISON ANALYSIS:")
print("=" * 60)

# Compare each mode to car (baseline)
car_mean = mode_stats['car']['mean']
print(f"\n CAR (Baseline): {car_mean:.1f} minutes average")

for mode in active_modes:
    if mode != 'car':
        mode_mean = mode_stats[mode]['mean']
        time_difference = mode_mean - car_mean
        time_multiple = mode_mean / car_mean
        
        print(f"\n{mode.upper()} vs CAR:")
        print(f"   Average time: {mode_mean:.1f} min")
        print(f"   Time difference: {time_difference:+.1f} min ({time_multiple:.1f}x slower)")

# Calculate accessibility by time intervals
print(f"\n ACCESSIBILITY BY TIME INTERVALS:")
print("=" * 60)

for interval in time_intervals:
    print(f"\n{interval} minutes:")
    for mode in active_modes:
        accessible = accessibility_data[mode][interval].sum()
        total_tracts = len(analysis_df)
        percentage = (accessible / total_tracts) * 100
        
        print(f"   {mode.title()}: {accessible}/{total_tracts} tracts ({percentage:.1f}%)")

# ------------------ Final Summary ------------------
print(f"\n ALL TRANSPORTATION MODES ANALYSIS COMPLETE!")
print("=" * 60)
print(f" KEY FINDING: Transportation Mode Accessibility Varies Significantly")
print("=" * 60)

print(f"\n  MAPPING SUMMARY:")
print("-" * 40)
print(f"   Total tracts mapped: {len(map_df)} (includes all tracts with geometry)")
print(f"   Tracts analyzed: {len(analysis_df)} (tracts with any transportation data)")
print(f"   Tracts with 2+ modes: {len(map_df[map_df['n_modes'] >= 2])} tracts")
print(f"   Tracts with only 1 mode: {len(map_df[map_df['n_modes'] == 1])} tracts")

print(f"\n CAR ACCESS:")
print("-" * 40)
print(f"   Average time: {mode_stats['car']['mean']:.1f} minutes")

if 'transit' in active_modes:
    print(f"\n TRANSIT ACCESS:")
    print("-" * 40)
    transit_ratio = mode_stats['transit']['mean'] / mode_stats['car']['mean']
    print(f"   Average time: {mode_stats['transit']['mean']:.1f} min ({transit_ratio:.1f}x slower than car)")


if 'walking' in active_modes:
    print(f"\n WALKING ACCESS:")
    print("-" * 40)
    walking_ratio = mode_stats['walking']['mean'] / mode_stats['car']['mean']
    print(f"   Average time: {mode_stats['walking']['mean']:.1f} min ({walking_ratio:.1f}x slower than car)")

print(f"\n Files generated:")
print(f"    Geographical comparison: {geographical_file}")

print(f"\n Main Takeaway:")
print(f"   Car access provides the fastest access to urgent care")
print(f"   Walking offers health benefits but longer travel times")
print(f"   Transit provides middle-ground accessibility for those without cars")
print(f"   This multi-modal analysis with 10-minute intervals reveals transportation equity gaps across Chicago")