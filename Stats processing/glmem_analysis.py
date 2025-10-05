import pandas as pd
import geopandas as gpd
import folium
import os
import zipfile
import tempfile
import numpy as np
import scipy.stats  # <-- Import the scipy.stats module itself

# --- Determine correct data directory relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = SCRIPT_DIR  # Save results in the equitability directory

# Load the Illinois census tract DP05 data and SVI data
dp05_path = os.path.join(DATA_DIR, "illinois_census_tract_DP05.csv")
svi_path = os.path.join(DATA_DIR, "SVI", "Illinois.csv")
urgent_care_path = os.path.join(DATA_DIR, "urgent_care_facilities.csv")

if not os.path.exists(dp05_path):
    raise FileNotFoundError(f"DP05 CSV not found at {dp05_path}")

if not os.path.exists(svi_path):
    raise FileNotFoundError(f"SVI CSV not found at {svi_path}")

dp05 = pd.read_csv(dp05_path, dtype=str)
svi = pd.read_csv(svi_path, dtype=str)

# --- Robustly find or construct the GEOID column for DP05 ---
geoid_col = None
for c in dp05.columns:
    c_clean = c.strip().upper()
    if c_clean in ["GEOID", "GEOID10"]:
        geoid_col = c
        break

# If not found, try to construct GEOID from state/county/tract
if geoid_col is None:
    # Check for state, county, tract columns
    state_col = None
    county_col = None
    tract_col = None
    for c in dp05.columns:
        if c.strip().lower() == "state":
            state_col = c
        if c.strip().lower() == "county":
            county_col = c
        if c.strip().lower() == "tract":
            tract_col = c
    if state_col and county_col and tract_col:
        # Pad state (2), county (3), tract (6) to correct width
        dp05["GEOID"] = (
            dp05[state_col].str.zfill(2)
            + dp05[county_col].str.zfill(3)
            + dp05[tract_col].str.zfill(6)
        )
        geoid_col = "GEOID"
    elif "GEO_ID" in dp05.columns:
        # Sometimes GEO_ID is like "1400000US17031010100"
        # Extract the last 11 digits
        dp05["GEOID"] = dp05["GEO_ID"].str[-11:]
        geoid_col = "GEOID"
    else:
        print("Available columns:", list(dp05.columns))
        raise KeyError("No GEOID column found or constructible in DP05 CSV. Please check the column names.")

# Only keep rows with a valid GEOID (should be 11 digits for tracts)
dp05 = dp05[dp05[geoid_col].str.len() == 11].copy()

# Filter to only Chicago tracts (FIPS starting with '17031')
dp05_chi = dp05[dp05[geoid_col].str.startswith('17031')].copy()

# --- Find or construct GEOID for SVI data ---
svi_geoid_col = None
for c in svi.columns:
    c_clean = c.strip().upper()
    if c_clean in ["GEOID", "GEOID10", "FIPS"]:
        svi_geoid_col = c
        break

if svi_geoid_col is None:
    print("Available SVI columns:", list(svi.columns))
    raise KeyError("No GEOID column found in SVI CSV. Please check the column names.")

# Filter SVI to Chicago tracts
svi_chi = svi[svi[svi_geoid_col].str.startswith('17031')].copy()

# Define socioeconomic and vulnerability indicators to analyze
# All 15 SVI factors from CDC
socioeconomic_indicators = {
    "Poverty (150%)": "E_POV150",      # Population below 150% of poverty line
    "Unemployment": "E_UNEMP",          # Unemployed population
    "Housing Burden": "E_HBURD",        # Households with housing cost > 30% of income
    "No High School Diploma": "E_NOHSDP", # Population 25+ without HS diploma
    "Age 65 and Over": "E_AGE65",       # Population age 65+
    "Age 17 and Under": "E_AGE17",      # Population age 17 and under
    "With Disability": "E_DISABL",      # Population with a disability
    "Single Parent": "E_SNGPNT",        # Single parent households
    "Limited English": "E_LIMENG",      # Population with limited English proficiency
    "Minority": "E_MINRTY",             # Minority population
    "Multi-Unit Housing": "E_MUNIT",    # Multi-unit housing structures
    "Mobile Homes": "E_MOBILE",         # Mobile homes
    "Overcrowding": "E_CROWD",          # Homes with more people than rooms
    "No Vehicle": "E_NOVEH",            # Households without vehicle
    "Group Quarters": "E_GROUPQ"        # Population in group quarters
}

svi_themes = {
    "SVI Theme 1 (Socioeconomic)": "RPL_THEME1",
    "SVI Theme 2 (Household Composition)": "RPL_THEME2", 
    "SVI Theme 3 (Minority Status & Language)": "RPL_THEME3",
    "SVI Theme 4 (Housing Type & Transportation)": "RPL_THEME4",
    "SVI Overall": "RPL_THEMES"
}

# SVI quartile analysis
svi_quartile_indicators = {
    "SVI Overall Quartile": "SVI_q"
}

# Load the census tract boundaries (as in svi_chloropleth.py)
tract_zip_path = os.path.join(DATA_DIR, "tl_2022_17_tract.zip")
if os.path.exists(tract_zip_path):
    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(tract_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    shp_files = [f for f in os.listdir(extract_dir) if f.endswith('.shp')]
    if not shp_files:
        raise Exception("No shapefile found in tract data")
    shp_path = os.path.join(extract_dir, shp_files[0])
    boundaries_gdf = gpd.read_file(shp_path)
    boundaries_gdf['GEOID'] = boundaries_gdf['GEOID'].astype(str)
else:
    raise FileNotFoundError("Census tract shapefile not found.")

# Merge DP05, SVI, and boundaries data
gdf = boundaries_gdf.merge(dp05_chi, left_on="GEOID", right_on=geoid_col, how="inner")
gdf = gdf.merge(svi_chi, left_on="GEOID", right_on=svi_geoid_col, how="inner")

# Focus on Chicago area (bounding box)
chicago_bbox = [-87.9401, 41.6445, -87.5241, 42.0230]
gdf = gdf.cx[chicago_bbox[0]:chicago_bbox[2], chicago_bbox[1]:chicago_bbox[3]]

os.makedirs(RESULTS_DIR, exist_ok=True)

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load travel time data from travel_times.csv
traveltimes_path = os.path.join(ROOT_DIR, "results", "travel_times.csv")
if not os.path.exists(traveltimes_path):
    raise FileNotFoundError(f"Could not find travel_times.csv at {traveltimes_path}")

traveltimes_df = pd.read_csv(traveltimes_path, dtype={"GEOID": str})

# Identify all available travel modes
all_modes = ["walking", "biking", "car", "transit"]
modes_to_check = []
mode_columns = {}

for mode in all_modes:
    col = f"{mode}_time_min"
    if col in traveltimes_df.columns:
        modes_to_check.append(mode)
        mode_columns[mode] = col
        print(f"✅ Found {mode} travel times: {col}")
    else:
        print(f"⚠️  {col} not found in travel_times.csv, skipping {mode}")

if not modes_to_check:
    raise ValueError("No travel time columns found in travel_times.csv.")

print(f"\n Analyzing {len(modes_to_check)} travel modes: {', '.join(modes_to_check)}")

# Standardize variables for interpretability
from sklearn.preprocessing import StandardScaler

# Results storage for all modes
all_glmem_results = {}
all_correlation_results = {}
all_summary_stats = {}

# Create comprehensive analysis for each mode
for mode in modes_to_check:
    outcome_var = mode_columns[mode]
    print(f"\n{'='*60}")
    print(f" ANALYZING MODE: {mode.upper()} (using {outcome_var})")
    print(f"{'='*60}")

    # Merge travel times with gdf (boundaries + socioeconomic + SVI data)
    if "GEOID" not in gdf.columns:
        raise KeyError("GEOID column not found in boundaries data.")
    
    # Check if SVI_q exists in traveltimes_df and include it in the merge
    merge_columns = ["GEOID", outcome_var]
    if "SVI_q" in traveltimes_df.columns:
        merge_columns.append("SVI_q")
    
    merged_gdf = gdf.merge(traveltimes_df[merge_columns], on="GEOID", how="left")
    
    # Only use tracts with non-missing outcome
    gdf_model = merged_gdf[~merged_gdf[outcome_var].isna()].copy()
    
    print(f" Data points available for {mode}: {len(gdf_model)} tracts")

    # Convert socioeconomic and SVI columns to numeric
    for indicator, col in socioeconomic_indicators.items():
        if col in gdf_model.columns:
            gdf_model[col] = pd.to_numeric(gdf_model[col], errors='coerce').fillna(0)
    
    for theme, col in svi_themes.items():
        if col in gdf_model.columns:
            gdf_model[col] = pd.to_numeric(gdf_model[col], errors='coerce').fillna(0)

    # Standardize variables
    scaler = StandardScaler()
    for indicator, col in socioeconomic_indicators.items():
        if col in gdf_model.columns:
            gdf_model[f"{col}_z"] = scaler.fit_transform(gdf_model[[col]])
    
    for theme, col in svi_themes.items():
        if col in gdf_model.columns:
            gdf_model[f"{col}_z"] = scaler.fit_transform(gdf_model[[col]])

    # Results storage for this mode
    glmem_results = {}
    correlation_results = {}
    summary_stats = {}

    # Basic statistics for travel times
    travel_times = gdf_model[outcome_var].dropna()
    summary_stats["travel_time_stats"] = {
        "mean": travel_times.mean(),
        "median": travel_times.median(),
        "std": travel_times.std(),
        "min": travel_times.min(),
        "max": travel_times.max(),
        "count": len(travel_times)
    }

    print(f" {mode.capitalize()} travel time statistics:")
    print(f"   Mean: {summary_stats['travel_time_stats']['mean']:.2f} minutes")
    print(f"   Median: {summary_stats['travel_time_stats']['median']:.2f} minutes")
    print(f"   Range: {summary_stats['travel_time_stats']['min']:.2f} - {summary_stats['travel_time_stats']['max']:.2f} minutes")

    # Analyze socioeconomic indicators
    print(f"\n Analyzing socioeconomic indicators for {mode}...")
    for indicator, col in socioeconomic_indicators.items():
        if col not in gdf_model.columns:
            print(f"⚠️  Column {col} not found in data, skipping {indicator}")
            continue

        print(f"\n   {indicator} ({mode})...")

        try:
            # Correlation analysis
            indicator_data = gdf_model[[outcome_var, col]].dropna()
            if len(indicator_data) > 0:
                correlation, p_value = scipy.stats.pearsonr(indicator_data[outcome_var], indicator_data[col])
                correlation_results[f"Socioeconomic_{indicator}"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": len(indicator_data)
                }
                print(f"    Correlation: {correlation:.3f} (p={p_value:.3f})")

            # GLMEM analysis
            model_data = gdf_model[[outcome_var, f"{col}_z", "GEOID"]].dropna()
            if len(model_data) > 0:
                # Fit model
                md = smf.mixedlm(f"{outcome_var} ~ {col}_z", model_data, groups=model_data["GEOID"])
                mdf = md.fit()
                glmem_results[f"Socioeconomic_{indicator}"] = mdf

                print(f"    ✅ GLMEM completed - AIC: {mdf.aic:.2f}")
                
                # Extract key results
                coef = mdf.params.get(f"{col}_z", np.nan)
                pval = mdf.pvalues.get(f"{col}_z", np.nan)
                print(f"    Coefficient: {coef:.3f} (p={pval:.3f})")
            else:
                print(f"    ⚠️  Insufficient data for GLMEM")

        except Exception as e:
            print(f"    ❌ Error analyzing {indicator} ({mode}): {e}")

    # Analyze SVI themes
    print(f"\n Analyzing SVI themes for {mode}...")
    for theme, col in svi_themes.items():
        if col not in gdf_model.columns:
            print(f"⚠️  Column {col} not found in data, skipping {theme}")
            continue

        print(f"\n   {theme} ({mode})...")

        try:
            # Correlation analysis
            theme_data = gdf_model[[outcome_var, col]].dropna()
            if len(theme_data) > 0:
                correlation, p_value = scipy.stats.pearsonr(theme_data[outcome_var], theme_data[col])
                correlation_results[f"SVI_{theme}"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "n": len(theme_data)
                }
                print(f"    Correlation: {correlation:.3f} (p={p_value:.3f})")

            # GLMEM analysis
            model_data = gdf_model[[outcome_var, f"{col}_z", "GEOID"]].dropna()
            if len(model_data) > 0:
                # Fit model
                md = smf.mixedlm(f"{outcome_var} ~ {col}_z", model_data, groups=model_data["GEOID"])
                mdf = md.fit()
                glmem_results[f"SVI_{theme}"] = mdf

                print(f"    ✅ GLMEM completed - AIC: {mdf.aic:.2f}")
                
                # Extract key results
                coef = mdf.params.get(f"{col}_z", np.nan)
                pval = mdf.pvalues.get(f"{col}_z", np.nan)
                print(f"    Coefficient: {coef:.3f} (p={pval:.3f})")
            else:
                print(f"    ⚠️  Insufficient data for GLMEM")

        except Exception as e:
            print(f"    ❌ Error analyzing {theme} ({mode}): {e}")

    # Store results for this mode
    all_glmem_results[mode] = glmem_results
    all_correlation_results[mode] = correlation_results
    all_summary_stats[mode] = summary_stats

    # Analyze SVI quartiles for this mode
    print(f"\n Analyzing SVI quartiles for {mode}...")
    quartile_results = {}
    
    for quartile_name, quartile_col in svi_quartile_indicators.items():
        if quartile_col not in gdf_model.columns:
            print(f"⚠️  Column {quartile_col} not found in data, skipping {quartile_name}")
            continue
            
        print(f"\n   {quartile_name} ({mode})...")
        
        try:
            # Get unique quartile values
            quartile_values = gdf_model[quartile_col].dropna().unique()
            print(f"    Available quartiles: {sorted(quartile_values)}")
            
            # Calculate descriptive statistics for each quartile
            quartile_stats = {}
            for q in sorted(quartile_values):
                if pd.notna(q) and str(q).strip() != '':
                    q_data = gdf_model[gdf_model[quartile_col] == q][outcome_var].dropna()
                    if len(q_data) > 0:
                        quartile_stats[q] = {
                            "count": len(q_data),
                            "mean": q_data.mean(),
                            "median": q_data.median(),
                            "std": q_data.std(),
                            "min": q_data.min(),
                            "max": q_data.max()
                        }
                        print(f"      Q{q}: n={len(q_data)}, Mean={q_data.mean():.2f}min, Median={q_data.median():.2f}min")
            
            # Perform ANOVA test if we have multiple quartiles with sufficient data
            if len(quartile_stats) >= 2:
                # Prepare data for ANOVA
                anova_data = []
                anova_labels = []
                for q, stats in quartile_stats.items():
                    if stats["count"] >= 5:  # Only include quartiles with sufficient data
                        q_data = gdf_model[gdf_model[quartile_col] == q][outcome_var].dropna()
                        anova_data.extend(q_data.tolist())
                        anova_labels.extend([f"Q{q}"] * len(q_data))
                
                if len(anova_data) > 0 and len(set(anova_labels)) >= 2:
                    # Perform one-way ANOVA
                    from scipy.stats import f_oneway
                    unique_labels = list(set(anova_labels))
                    
                    # Create groups properly
                    groups = []
                    for label in unique_labels:
                        group_data = [anova_data[i] for i, label2 in enumerate(anova_labels) if label2 == label]
                        if len(group_data) >= 5:  # Only include groups with sufficient data
                            groups.append(group_data)
                    
                    if len(groups) >= 2 and all(len(g) >= 5 for g in groups):
                        f_stat, p_value = f_oneway(*groups)
                        print(f"      ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
                        
                        # Determine significance
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        if p_value < 0.05:
                            print(f"       Significant differences found between quartiles! {significance}")
                        else:
                            print(f"       No significant differences between quartiles")
                        
                        quartile_results[f"ANOVA_{quartile_name}"] = {
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "quartile_stats": quartile_stats
                        }
                    else:
                        print(f"      ⚠️  Insufficient data for ANOVA")
                else:
                    print(f"      ⚠️  Insufficient data for ANOVA")
            else:
                print(f"      ⚠️  Need at least 2 quartiles for ANOVA")
                
        except Exception as e:
            print(f"      ❌ Error analyzing {quartile_name} ({mode}): {e}")
    
    # Store quartile results
    all_summary_stats[mode]["quartile_analysis"] = quartile_results

# Create comprehensive summary report
print(f"\n{'='*80}")
print(f" COMPREHENSIVE ANALYSIS SUMMARY")
print(f"{'='*80}")

# Summary statistics across modes
print(f"\n TRAVEL TIME SUMMARY ACROSS ALL MODES:")
for mode in modes_to_check:
    stats = all_summary_stats[mode]["travel_time_stats"]
    print(f"  {mode.capitalize():<10}: Mean={stats['mean']:6.2f}min, Median={stats['median']:6.2f}min, Range={stats['min']:5.1f}-{stats['max']:5.1f}min")

# Correlation summary
print(f"\n CORRELATION SUMMARY:")
for mode in modes_to_check:
    print(f"\n  {mode.upper()}:")
    correlations = all_correlation_results[mode]
    for indicator, results in correlations.items():
        corr = results["correlation"]
        pval = results["p_value"]
        significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {indicator:<40}: r={corr:6.3f} (p={pval:6.3f}) {significance}")

# SVI Quartile Analysis Summary
print(f"\n SVI QUARTILE ANALYSIS SUMMARY:")
for mode in modes_to_check:
    print(f"\n  {mode.upper()}:")
    quartile_analysis = all_summary_stats[mode].get("quartile_analysis", {})
    
    if quartile_analysis:
        for anova_key, anova_results in quartile_analysis.items():
            quartile_name = anova_key.replace("ANOVA_", "")
            f_stat = anova_results["f_statistic"]
            p_val = anova_results["p_value"]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            print(f"    {quartile_name:<40}: F={f_stat:6.3f} (p={p_val:6.3f}) {significance}")
            
            # Show quartile means if significant
            if p_val < 0.05:
                quartile_stats = anova_results["quartile_stats"]
                print(f"       Significant differences - Quartile means:")
                for q, stats in sorted(quartile_stats.items()):
                    print(f"        Q{q}: {stats['mean']:.2f} ± {stats['std']:.2f} min (n={stats['count']})")
    else:
        print(f"    No quartile analysis available")

# Save comprehensive results to files
print(f"\n Saving results...")

# Save GLMEM results
results_path = os.path.join(RESULTS_DIR, "multimodal_glmem_socioeconomic_health_equity_results.txt")
with open(results_path, "w") as f:
    f.write("MULTIMODAL SOCIOECONOMIC AND SVI HEALTH EQUITY ANALYSIS\n")
    f.write("=" * 60 + "\n\n")
    
    for mode, glmem_results in all_glmem_results.items():
        f.write(f"MODE: {mode.upper()}\n")
        f.write("-" * 40 + "\n")
        for indicator, mdf in glmem_results.items():
            f.write(f"\n{indicator}:\n")
            f.write(str(mdf.summary()))
            f.write("\n" + "="*60 + "\n\n")

# Save correlation results
correlation_path = os.path.join(RESULTS_DIR, "multimodal_correlation_analysis.csv")
correlation_data = []
for mode in modes_to_check:
    correlations = all_correlation_results[mode]
    for indicator, results in correlations.items():
        correlation_data.append({
            "mode": mode,
            "indicator": indicator,
            "correlation": results["correlation"],
            "p_value": results["p_value"],
            "n": results["n"]
        })

correlation_df = pd.DataFrame(correlation_data)
correlation_df.to_csv(correlation_path, index=False)

# Save summary statistics
summary_path = os.path.join(RESULTS_DIR, "multimodal_summary_statistics.csv")
summary_data = []
for mode in modes_to_check:
    stats = all_summary_stats[mode]["travel_time_stats"]
    summary_data.append({
        "mode": mode,
        "mean_minutes": stats["mean"],
        "median_minutes": stats["median"],
        "std_minutes": stats["std"],
        "min_minutes": stats["min"],
        "max_minutes": stats["max"],
        "n_tracts": stats["count"]
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_path, index=False)

# Save quartile analysis results
quartile_path = os.path.join(RESULTS_DIR, "multimodal_svi_quartile_analysis.csv")
quartile_data = []
for mode in modes_to_check:
    quartile_analysis = all_summary_stats[mode].get("quartile_analysis", {})
    
    for anova_key, anova_results in quartile_analysis.items():
        quartile_name = anova_key.replace("ANOVA_", "")
        f_stat = anova_results["f_statistic"]
        p_val = anova_results["p_value"]
        
        # Add overall ANOVA results
        quartile_data.append({
            "mode": mode,
            "quartile_indicator": quartile_name,
            "anova_f_statistic": f_stat,
            "anova_p_value": p_val,
            "significant": p_val < 0.05,
            "analysis_type": "ANOVA"
        })
        
        # Add individual quartile statistics
        quartile_stats = anova_results["quartile_stats"]
        for q, stats in quartile_stats.items():
            quartile_data.append({
                "mode": mode,
                "quartile_indicator": quartile_name,
                "quartile": q,
                "n_tracts": stats["count"],
                "mean_minutes": stats["mean"],
                "median_minutes": stats["median"],
                "std_minutes": stats["std"],
                "min_minutes": stats["min"],
                "max_minutes": stats["max"],
                "analysis_type": "quartile_stats"
            })

quartile_df = pd.DataFrame(quartile_data)
quartile_df.to_csv(quartile_path, index=False)

print(f"✅ GLMEM results saved to: {results_path}")
print(f"✅ Correlation analysis saved to: {correlation_path}")
print(f"✅ Summary statistics saved to: {summary_path}")
print(f"✅ Quartile analysis results saved to: {quartile_path}")

print(f"\n MULTIMODAL ANALYSIS COMPLETE!")
print(f" Analyzed {len(modes_to_check)} transport modes: {', '.join(modes_to_check)}")
print(f" Compared against {len(socioeconomic_indicators)} socioeconomic indicators and {len(svi_themes)} SVI themes")
print(f" Results saved to {RESULTS_DIR}")


def analyze_glmem_access_times():
    """
    Analyze GLMEM results for car, transit, and walking access times.
    Calculate mean, median, min, max, and standard deviation for each mode.
    """
    print("\n" + "="*80)
    print(" GLMEM ACCESS TIME ANALYSIS")
    print("="*80)
    
    # Try to load the travel times data
    travel_times_path = os.path.join(ROOT_DIR, "results", "travel_times.csv")
    
    if not os.path.exists(travel_times_path):
        print(f"❌ Travel times data not found at: {travel_times_path}")
        print("Looking for alternative data sources...")
        
        # Try to find alternative data files
        alternative_paths = [
            os.path.join(ROOT_DIR, "results", "random", "travel_times_2modes.csv"),
            os.path.join(ROOT_DIR, "results", "random", "travel_times_backup_single.csv"),
            os.path.join(ROOT_DIR, "processed_data", "chicago_urgentcare_final_results.geojson")
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                travel_times_path = alt_path
                print(f"✅ Found alternative data at: {alt_path}")
                break
        else:
            print("❌ No travel times data found. Cannot perform GLMEM analysis.")
            return
    
    try:
        # Load the data based on file type
        if travel_times_path.endswith('.csv'):
            print(f" Loading CSV data from: {travel_times_path}")
            df = pd.read_csv(travel_times_path)
        elif travel_times_path.endswith('.geojson'):
            print(f" Loading GeoJSON data from: {travel_times_path}")
            df = gpd.read_file(travel_times_path)
        else:
            print(f"❌ Unsupported file format: {travel_times_path}")
            return
        
        print(f"✅ Data loaded successfully. Raw shape: {df.shape}")
        print(f" Available columns: {list(df.columns)}")
        
        # Check for duplicate GEOIDs and handle them
        if 'GEOID' in df.columns:
            total_rows = len(df)
            unique_geoids = df['GEOID'].nunique()
            duplicate_rows = total_rows - unique_geoids
            
            if duplicate_rows > 0:
                print(f"⚠️  Found {duplicate_rows} duplicate rows out of {total_rows} total rows")
                print(f" Unique GEOIDs: {unique_geoids}")
                
                # Show duplicate analysis
                dupes = df[df['GEOID'].duplicated(keep=False)]
                if len(dupes) > 0:
                    print(f" Duplicate GEOID analysis:")
                    dupe_counts = dupes['GEOID'].value_counts()
                    print(f"   Top duplicate GEOIDs:")
                    for geoid, count in dupe_counts.head(5).items():
                        print(f"     {geoid}: {count} duplicates")
                
                # Deduplicate by taking the first occurrence of each GEOID
                print(f" Deduplicating data by GEOID...")
                df_deduped = df.drop_duplicates(subset=['GEOID'], keep='first')
                print(f"✅ After deduplication: {len(df_deduped)} rows (was {len(df)})")
                df = df_deduped
            else:
                print(f"✅ No duplicate GEOIDs found. Data is clean.")
        
        # Check for required columns
        required_cols = ['car_time_min', 'transit_time_min', 'walking_time_min']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if not available_cols:
            print("❌ No access time columns found. Available columns:")
            for col in df.columns:
                if 'time' in col.lower() or 'min' in col.lower():
                    print(f"   - {col}")
            return
        
        print(f"✅ Found access time columns: {available_cols}")
        print(f" Final data shape: {df.shape} rows, {df.shape[1]} columns")
        
        # Analyze each available mode
        results = {}
        
        for mode in available_cols:
            print(f"\n Analyzing {mode.replace('_', ' ').title()}...")
            
            # Get the data for this mode
            mode_data = df[mode].dropna()
            
            if len(mode_data) == 0:
                print(f"   ⚠️  No data available for {mode}")
                continue
            
            # Calculate statistics
            stats = {
                'mean': mode_data.mean(),
                'median': mode_data.median(),
                'min': mode_data.min(),
                'max': mode_data.max(),
                'std': mode_data.std(),
                'count': len(mode_data),
                'q25': mode_data.quantile(0.25),
                'q75': mode_data.quantile(0.75)
            }
            
            results[mode] = stats
            
            # Display results
            print(f"    Statistics for {mode.replace('_', ' ').title()}:")
            print(f"      Count: {stats['count']:,}")
            print(f"      Mean: {stats['mean']:.2f} minutes")
            print(f"      Median: {stats['median']:.2f} minutes")
            print(f"      Std Dev: {stats['std']:.2f} minutes")
            print(f"      Min: {stats['min']:.2f} minutes")
            print(f"      Max: {stats['max']:.2f} minutes")
            print(f"      Q25: {stats['q25']:.2f} minutes")
            print(f"      Q75: {stats['q75']:.2f} minutes")
        
        # Summary comparison
        if len(results) > 1:
            print(f"\n SUMMARY COMPARISON:")
            print("-" * 80)
            print(f"{'Mode':<20} {'Mean':<10} {'Median':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
            print("-" * 80)
            
            for mode, stats in results.items():
                mode_name = mode.replace('_', ' ').title()
                print(f"{mode_name:<20} {stats['mean']:<10.2f} {stats['median']:<10.2f} "
                      f"{stats['std']:<10.2f} {stats['min']:<10.2f} {stats['max']:<10.2f}")
        
        # Save detailed results
        results_path = os.path.join(RESULTS_DIR, "glmem_access_time_analysis.csv")
        
        # Prepare data for CSV
        csv_data = []
        for mode, stats in results.items():
            csv_data.append({
                'mode': mode.replace('_', ' ').title(),
                'count': stats['count'],
                'mean_minutes': stats['mean'],
                'median_minutes': stats['median'],
                'std_minutes': stats['std'],
                'min_minutes': stats['min'],
                'max_minutes': stats['max'],
                'q25_minutes': stats['q25'],
                'q75_minutes': stats['q75']
            })
        
        results_df = pd.DataFrame(csv_data)
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Detailed results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing GLMEM access times: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_significant_socioeconomic_predictors():
    """
    Analyze significant socioeconomic predictors for each transit mode based on actual GLMEM results.
    Parse the GLMEM text file to extract real p-values and coefficients.
    """
    print("\n" + "="*80)
    print(" SIGNIFICANT SOCIOECONOMIC PREDICTORS ANALYSIS (GLMEM RESULTS)")
    print("="*80)
    
    # Load the GLMEM results file
    glmem_results_path = os.path.join(RESULTS_DIR, "multimodal_glmem_socioeconomic_health_equity_results.txt")
    
    if not os.path.exists(glmem_results_path):
        print(f"❌ GLMEM results file not found at: {glmem_results_path}")
        return None
    
    try:
        # Read the GLMEM results file
        with open(glmem_results_path, 'r') as f:
            content = f.read()
        
        print(f"✅ Loaded GLMEM results file: {len(content)} characters")
        
        # Parse the results to extract significant predictors
        modes = ['WALKING', 'CAR', 'TRANSIT']
        significant_predictors = {}
        
        for mode in modes:
            print(f"\n Analyzing {mode} mode...")
            
            # Find the section for this mode - completely rewritten parsing logic
            if mode == 'WALKING':
                # For WALKING, take everything between "MODE: WALKING" and "MODE: CAR"
                if "MODE: WALKING" in content and "MODE: CAR" in content:
                    start_idx = content.find("MODE: WALKING")
                    end_idx = content.find("MODE: CAR")
                    mode_section = content[start_idx:end_idx]
                else:
                    mode_section = ""
            elif mode == 'CAR':
                # For CAR, take everything between "MODE: CAR" and "MODE: TRANSIT"
                if "MODE: CAR" in content and "MODE: TRANSIT" in content:
                    start_idx = content.find("MODE: CAR")
                    end_idx = content.find("MODE: TRANSIT")
                    mode_section = content[start_idx:end_idx]
                else:
                    mode_section = ""
            elif mode == 'TRANSIT':
                # For TRANSIT, take everything after "MODE: TRANSIT"
                if "MODE: TRANSIT" in content:
                    start_idx = content.find("MODE: TRANSIT")
                    mode_section = content[start_idx:]
                else:
                    mode_section = ""
            
            print(f"    Mode section length: {len(mode_section)} characters")
            
            # Extract socioeconomic indicators and their GLMEM results
            indicators = [
                "Socioeconomic_Poverty (150%)",
                "Socioeconomic_Unemployment", 
                "Socioeconomic_Housing Burden",
                "Socioeconomic_No High School Diploma",
                "Socioeconomic_Single Parent",
                "Socioeconomic_Limited English",
                "Socioeconomic_Age 65 and Over",
                "Socioeconomic_With Disability"
            ]
            
            mode_predictors = []
            
            for indicator in indicators:
                print(f"\n    Analyzing {indicator.replace('Socioeconomic_', '')}...")
                
                # Look for the indicator in the mode section
                if indicator in mode_section:
                    # Extract the results section for this indicator
                    # Find the start of this indicator's results
                    indicator_start = mode_section.find(indicator)
                    if indicator_start != -1:
                        # Find the next indicator or section boundary
                        remaining_text = mode_section[indicator_start:]
                        next_indicators = [f"Socioeconomic_{ind}" for ind in ["Poverty (150%)", "Unemployment", "Housing Burden", "No High School Diploma", "Single Parent", "Limited English", "Age 65 and Over", "With Disability"] if ind != indicator.replace('Socioeconomic_', '')]
                        
                        # Find the next indicator or section boundary
                        next_pos = len(remaining_text)
                        for next_ind in next_indicators:
                            if next_ind in remaining_text:
                                pos = remaining_text.find(next_ind)
                                if pos != -1 and pos < next_pos:
                                    next_pos = pos
                        
                        # Also check for SVI sections
                        svi_sections = ["SVI_SVI Theme", "SVI_SVI Overall"]
                        for svi_section in svi_sections:
                            if svi_section in remaining_text:
                                pos = remaining_text.find(svi_section)
                                if pos != -1 and pos < next_pos:
                                    next_pos = pos
                        
                        # Extract the indicator section
                        indicator_section = remaining_text[:next_pos]
                        
                        print(f"       Found indicator section: {len(indicator_section)} characters")
                        
                        # Look for the coefficient table
                        if "Coef." in indicator_section and "P>|z|" in indicator_section:
                            # Find the table lines
                            lines = indicator_section.split('\n')
                            table_found = False
                            
                            for line in lines:
                                if 'E_' in line and 'Coef.' not in line and 'Intercept' not in line:
                                    # This is the socioeconomic indicator row
                                    parts = line.split()
                                    if len(parts) >= 5:  # Need at least 5 parts: var_name, coef, std_err, z, p, ci_lower, ci_upper
                                        try:
                                            # Skip the first part (variable name) and get the actual values
                                            coefficient = float(parts[1])  # Coefficient
                                            std_error = float(parts[2])   # Standard Error
                                            z_stat = float(parts[3])     # Z-statistic
                                            p_value = float(parts[4])    # P-value
                                            
                                            print(f"       GLMEM Results:")
                                            print(f"         Coefficient: {coefficient:.3f} minutes per +1 z-score")
                                            print(f"         Std Error: {std_error:.3f}")
                                            print(f"         Z-statistic: {z_stat:.3f}")
                                            print(f"         P-value: {p_value:.6f}")
                                            
                                            # Determine significance level
                                            if p_value < 0.001:
                                                significance = "***"
                                            elif p_value < 0.01:
                                                significance = "**"
                                            elif p_value < 0.05:
                                                significance = "*"
                                            else:
                                                significance = "ns"
                                            
                                            print(f"         Significance: {significance}")
                                            print(f"          Effect: +1 SD increase → +{coefficient:.3f} min access time")
                                            
                                            mode_predictors.append({
                                                'indicator': indicator.replace('Socioeconomic_', ''),
                                                'coefficient': coefficient,
                                                'std_error': std_error,
                                                'z_stat': z_stat,
                                                'p_value': p_value,
                                                'significance': significance
                                            })
                                            
                                            table_found = True
                                            break
                                            
                                        except (ValueError, IndexError) as e:
                                            print(f"         ⚠️  Error parsing line: {e}")
                                            print(f"         Line content: {line}")
                                            print(f"         Parts: {parts}")
                                            continue
                            
                            if not table_found:
                                print(f"      ⚠️  No coefficient table found for {indicator}")
                        else:
                            print(f"      ⚠️  No coefficient table found in indicator section")
                    else:
                        print(f"      ⚠️  Could not find indicator start position")
                else:
                    print(f"      ⚠️  Indicator {indicator} not found in {mode} section")
            
            significant_predictors[mode] = mode_predictors
            
            # Summary of significant predictors
            significant_count = len([p for p in mode_predictors if p['p_value'] < 0.05])
            print(f"\n    Total significant predictors (p<0.05): {significant_count}")
            
            if significant_count > 0:
                print(f"   ✅ Significant predictors:")
                for pred in mode_predictors:
                     if pred['p_value'] < 0.05:
                         print(f"      • {pred['indicator']}: β={pred['coefficient']:.3f} min/+1SD, p={pred['p_value']:.6f} {pred['significance']}")
            else:
                print(f"   ⚠️  No significant socioeconomic predictors found")
        
        # Cross-mode comparison
        print(f"\n CROSS-MODE COMPARISON OF SIGNIFICANT PREDICTORS:")
        print("-" * 80)
        
        # Get all unique indicators
        all_indicators = set()
        for mode_predictors in significant_predictors.values():
            for pred in mode_predictors:
                all_indicators.add(pred['indicator'])
        
        all_indicators = sorted(list(all_indicators))
        
        # Create comparison table
        print(f"{'Indicator':<25} {'Walking':<15} {'Car':<15} {'Transit':<15}")
        print("-" * 80)
        
        for indicator in all_indicators:
            walking_sig = "ns"
            car_sig = "ns"
            transit_sig = "ns"
            
            # Check walking
            for pred in significant_predictors.get('WALKING', []):
                if pred['indicator'] == indicator:
                    if pred['p_value'] < 0.001:
                        walking_sig = "***"
                    elif pred['p_value'] < 0.01:
                        walking_sig = "**"
                    elif pred['p_value'] < 0.05:
                        walking_sig = "*"
                    break
            
            # Check car
            for pred in significant_predictors.get('CAR', []):
                if pred['indicator'] == indicator:
                    if pred['p_value'] < 0.001:
                        car_sig = "***"
                    elif pred['p_value'] < 0.01:
                        car_sig = "**"
                    elif pred['p_value'] < 0.05:
                        car_sig = "*"
                    break
            
            # Check transit
            for pred in significant_predictors.get('TRANSIT', []):
                if pred['indicator'] == indicator:
                    if pred['p_value'] < 0.001:
                        transit_sig = "***"
                    elif pred['p_value'] < 0.01:
                        transit_sig = "**"
                    elif pred['p_value'] < 0.05:
                        transit_sig = "*"
                    break
            
            print(f"{indicator:<25} {walking_sig:<15} {car_sig:<15} {transit_sig:<15}")
        
        print("-" * 80)
        print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
        
        # Summary of findings
        print(f"\n SUMMARY OF SIGNIFICANT FINDINGS:")
        print("-" * 80)
        
        for mode, predictors in significant_predictors.items():
            significant_count = len([p for p in predictors if p['p_value'] < 0.05])
            print(f"\n{mode} Mode:")
            if significant_count > 0:
                for pred in predictors:
                    if pred['p_value'] < 0.05:
                        direction = "positive" if pred['coefficient'] > 0 else "negative"
                        print(f"   ✅ {pred['indicator']}: {direction} relationship (β={pred['coefficient']:.3f} min/+1SD, p={pred['p_value']:.6f})")
            else:
                print(f"   ⚠️  No significant socioeconomic predictors")
        
        # Save results
        results_path = os.path.join(RESULTS_DIR, "glmem_significant_socioeconomic_predictors.csv")
        
        # Prepare data for CSV
        csv_data = []
        for mode, predictors in significant_predictors.items():
            for pred in predictors:
                csv_data.append({
                    'mode': mode.lower(),
                    'indicator': pred['indicator'],
                    'coefficient': pred['coefficient'],
                    'std_error': pred['std_error'],
                    'z_stat': pred['z_stat'],
                    'p_value': pred['p_value'],
                    'significance': pred['significance'],
                    'significant': pred['p_value'] < 0.05
                })
        
        results_df = pd.DataFrame(csv_data)
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Detailed GLMEM results saved to: {results_path}")
        
        return significant_predictors
        
    except Exception as e:
        print(f"❌ Error analyzing GLMEM significant predictors: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Run the GLMEM analysis if this script is executed directly
if __name__ == "__main__":
    print(" Running GLMEM Access Time Analysis...")
    glmem_results = analyze_glmem_access_times()
    
    if glmem_results:
        print(f"\n GLMEM Analysis Complete!")
        print(f" Analyzed {len(glmem_results)} transport modes")
    
    print("\n Running Significant Socioeconomic Predictors Analysis...")
    significant_predictors = analyze_significant_socioeconomic_predictors()
    
    if significant_predictors:
        print(f"\n Significant Predictors Analysis Complete!")
        total_significant = sum(len([p for p in preds if p['p_value'] < 0.05]) for preds in significant_predictors.values())
        print(f" Found {total_significant} significant socioeconomic predictors across all modes")
    else:
        print(f"\n❌ Significant Predictors Analysis Failed")




