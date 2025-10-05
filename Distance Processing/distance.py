# Distance analysis, just travel distance/time 

import pandas as pd
import geopandas as gpd
import folium
import os
import zipfile
import tempfile
import numpy as np
from scipy import stats

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
socioeconomic_indicators = {
    "Poverty (150%)": "E_POV150",  # Population below 150% of poverty line
    "Unemployment": "E_UNEMP",      # Unemployed population
    "Housing Burden": "E_HBURD",    # Households with housing cost > 30% of income
    "No High School Diploma": "E_NOHSDP",  # Population 25+ without HS diploma
    "Single Parent": "E_SNGPNT",    # Single parent households
    "Limited English": "E_LIMENG"   # Population with limited English proficiency
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

# Load travel distance data from travel_times_matrix.csv
travelmatrix_path = os.path.join(ROOT_DIR, "results", "travel_times_matrix.csv")
if not os.path.exists(travelmatrix_path):
    raise FileNotFoundError(f"Could not find travel_times_matrix.csv at {travelmatrix_path}")

travelmatrix_df = pd.read_csv(travelmatrix_path, dtype={"GEOID": str})

# Check for distance_miles column
if "distance_miles" not in travelmatrix_df.columns:
    raise ValueError("No 'distance_miles' column found in travel_times_matrix.csv.")

print(f"\n Analyzing travel distance: distance_miles")

# Standardize variables for interpretability
from sklearn.preprocessing import StandardScaler

# Merge travel distances with gdf (boundaries + socioeconomic + SVI data)
merge_columns = ["GEOID", "distance_miles"]
if "SVI_q" in travelmatrix_df.columns:
    merge_columns.append("SVI_q")

merged_gdf = gdf.merge(travelmatrix_df[merge_columns], on="GEOID", how="left")

# Only use tracts with non-missing distance
gdf_model = merged_gdf[~merged_gdf["distance_miles"].isna()].copy()

print(f" Data points available: {len(gdf_model)} tracts")

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

# Results storage
glmem_results = {}
correlation_results = {}
summary_stats = {}

# Basic statistics for travel distances
travel_distances = gdf_model["distance_miles"].dropna()
summary_stats["travel_distance_stats"] = {
    "mean": travel_distances.mean(),
    "median": travel_distances.median(),
    "std": travel_distances.std(),
    "min": travel_distances.min(),
    "max": travel_distances.max(),
    "count": len(travel_distances)
}

print(f" Travel distance statistics:")
print(f"   Mean: {summary_stats['travel_distance_stats']['mean']:.2f} miles")
print(f"   Median: {summary_stats['travel_distance_stats']['median']:.2f} miles")
print(f"   Range: {summary_stats['travel_distance_stats']['min']:.2f} - {summary_stats['travel_distance_stats']['max']:.2f} miles")

# Analyze socioeconomic indicators
print(f"\n Analyzing socioeconomic indicators for distance_miles...")
for indicator, col in socioeconomic_indicators.items():
    if col not in gdf_model.columns:
        print(f"Warning:  Column {col} not found in data, skipping {indicator}")
        continue

    print(f"\n   {indicator} (distance_miles)...")

    try:
        # Correlation analysis
        indicator_data = gdf_model[["distance_miles", col]].dropna()
        if len(indicator_data) > 0:
            correlation, p_value = stats.pearsonr(indicator_data["distance_miles"], indicator_data[col])
            correlation_results[f"Socioeconomic_{indicator}"] = {
                "correlation": correlation,
                "p_value": p_value,
                "n": len(indicator_data)
            }
            print(f"    Correlation: {correlation:.3f} (p={p_value:.3f})")

        # GLMEM analysis
        model_data = gdf_model[["distance_miles", f"{col}_z", "GEOID"]].dropna()
        if len(model_data) > 0:
            # Fit model
            md = smf.mixedlm("distance_miles ~ " + f"{col}_z", model_data, groups=model_data["GEOID"])
            mdf = md.fit()
            glmem_results[f"Socioeconomic_{indicator}"] = mdf

            print(f"     GLMEM completed - AIC: {mdf.aic:.2f}")
            
            # Extract key results
            coef = mdf.params.get(f"{col}_z", np.nan)
            pval = mdf.pvalues.get(f"{col}_z", np.nan)
            print(f"    Coefficient: {coef:.3f} (p={pval:.3f})")
        else:
            print(f"    Warning:  Insufficient data for GLMEM")

    except Exception as e:
        print(f"    ERROR: Error analyzing {indicator} (distance_miles): {e}")

# Analyze SVI themes
print(f"\n Analyzing SVI themes for distance_miles...")
for theme, col in svi_themes.items():
    if col not in gdf_model.columns:
        print(f"Warning:  Column {col} not found in data, skipping {theme}")
        continue

    print(f"\n   {theme} (distance_miles)...")

    try:
        # Correlation analysis
        theme_data = gdf_model[["distance_miles", col]].dropna()
        if len(theme_data) > 0:
            correlation, p_value = stats.pearsonr(theme_data["distance_miles"], theme_data[col])
            correlation_results[f"SVI_{theme}"] = {
                "correlation": correlation,
                "p_value": p_value,
                "n": len(theme_data)
            }
            print(f"    Correlation: {correlation:.3f} (p={p_value:.3f})")

        # GLMEM analysis
        model_data = gdf_model[["distance_miles", f"{col}_z", "GEOID"]].dropna()
        if len(model_data) > 0:
            # Fit model
            md = smf.mixedlm("distance_miles ~ " + f"{col}_z", model_data, groups=model_data["GEOID"])
            mdf = md.fit()
            glmem_results[f"SVI_{theme}"] = mdf

            print(f"     GLMEM completed - AIC: {mdf.aic:.2f}")
            
            # Extract key results
            coef = mdf.params.get(f"{col}_z", np.nan)
            pval = mdf.pvalues.get(f"{col}_z", np.nan)
            print(f"    Coefficient: {coef:.3f} (p={pval:.3f})")
        else:
            print(f"    Warning:  Insufficient data for GLMEM")

    except Exception as e:
        print(f"    ERROR: Error analyzing {theme} (distance_miles): {e}")

# Analyze SVI quartiles for distance_miles
print(f"\n Analyzing SVI quartiles for distance_miles...")
quartile_results = {}

for quartile_name, quartile_col in svi_quartile_indicators.items():
    if quartile_col not in gdf_model.columns:
        print(f"Warning:  Column {quartile_col} not found in data, skipping {quartile_name}")
        continue
        
    print(f"\n   {quartile_name} (distance_miles)...")
    
    try:
        # Get unique quartile values
        quartile_values = gdf_model[quartile_col].dropna().unique()
        print(f"    Available quartiles: {sorted(quartile_values)}")
        
        # Calculate descriptive statistics for each quartile
        quartile_stats = {}
        for q in sorted(quartile_values):
            if pd.notna(q) and str(q).strip() != '':
                q_data = gdf_model[gdf_model[quartile_col] == q]["distance_miles"].dropna()
                if len(q_data) > 0:
                    quartile_stats[q] = {
                        "count": len(q_data),
                        "mean": q_data.mean(),
                        "median": q_data.median(),
                        "std": q_data.std(),
                        "min": q_data.min(),
                        "max": q_data.max()
                    }
                    print(f"      Q{q}: n={len(q_data)}, Mean={q_data.mean():.2f}mi, Median={q_data.median():.2f}mi")
        
        # Perform ANOVA test if we have multiple quartiles with sufficient data
        if len(quartile_stats) >= 2:
            # Prepare data for ANOVA
            anova_data = []
            anova_labels = []
            for q, stats_ in quartile_stats.items():
                if stats_["count"] >= 5:  # Only include quartiles with sufficient data
                    q_data = gdf_model[gdf_model[quartile_col] == q]["distance_miles"].dropna()
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
                    print(f"      Warning:  Insufficient data for ANOVA")
            else:
                print(f"      Warning:  Insufficient data for ANOVA")
        else:
            print(f"      Warning:  Need at least 2 quartiles for ANOVA")
            
    except Exception as e:
        print(f"      ERROR: Error analyzing {quartile_name} (distance_miles): {e}")

# Store quartile results
summary_stats["quartile_analysis"] = quartile_results

# Create comprehensive summary report
print(f"\n{'='*80}")
print(f" COMPREHENSIVE ANALYSIS SUMMARY")
print(f"{'='*80}")

# Summary statistics
stats = summary_stats["travel_distance_stats"]
print(f"\n TRAVEL DISTANCE SUMMARY:")
print(f"  Mean={stats['mean']:6.2f}mi, Median={stats['median']:6.2f}mi, Range={stats['min']:5.2f}-{stats['max']:5.2f}mi")

# Correlation summary
print(f"\n CORRELATION SUMMARY:")
for indicator, results in correlation_results.items():
    corr = results["correlation"]
    pval = results["p_value"]
    significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"    {indicator:<40}: r={corr:6.3f} (p={pval:6.3f}) {significance}")

# SVI Quartile Analysis Summary
print(f"\n SVI QUARTILE ANALYSIS SUMMARY:")
quartile_analysis = summary_stats.get("quartile_analysis", {})
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
            for q, stats_ in sorted(quartile_stats.items()):
                print(f"        Q{q}: {stats_['mean']:.2f} ± {stats_['std']:.2f} mi (n={stats_['count']})")
else:
    print(f"    No quartile analysis available")

# Save comprehensive results to files
print(f"\n Saving results...")

# Save GLMEM results
results_path = os.path.join(RESULTS_DIR, "distance_glmem_socioeconomic_health_equity_results.txt")
with open(results_path, "w") as f:
    f.write("DISTANCE SOCIOECONOMIC AND SVI HEALTH EQUITY ANALYSIS\n")
    f.write("=" * 60 + "\n\n")
    for indicator, mdf in glmem_results.items():
        f.write(f"\n{indicator}:\n")
        f.write(str(mdf.summary()))
        f.write("\n" + "="*60 + "\n\n")

# Save correlation results
correlation_path = os.path.join(RESULTS_DIR, "distance_correlation_analysis.csv")
correlation_data = []
for indicator, results in correlation_results.items():
    correlation_data.append({
        "indicator": indicator,
        "correlation": results["correlation"],
        "p_value": results["p_value"],
        "n": results["n"]
    })

correlation_df = pd.DataFrame(correlation_data)
correlation_df.to_csv(correlation_path, index=False)

# Save summary statistics
summary_path = os.path.join(RESULTS_DIR, "distance_summary_statistics.csv")
summary_data = [{
    "mean_miles": stats["mean"],
    "median_miles": stats["median"],
    "std_miles": stats["std"],
    "min_miles": stats["min"],
    "max_miles": stats["max"],
    "n_tracts": stats["count"]
}]
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_path, index=False)

# Save quartile analysis results
quartile_path = os.path.join(RESULTS_DIR, "distance_svi_quartile_analysis.csv")
quartile_data = []
for anova_key, anova_results in quartile_analysis.items():
    quartile_name = anova_key.replace("ANOVA_", "")
    f_stat = anova_results["f_statistic"]
    p_val = anova_results["p_value"]
    
    # Add overall ANOVA results
    quartile_data.append({
        "quartile_indicator": quartile_name,
        "anova_f_statistic": f_stat,
        "anova_p_value": p_val,
        "significant": p_val < 0.05,
        "analysis_type": "ANOVA"
    })
    
    # Add individual quartile statistics
    quartile_stats = anova_results["quartile_stats"]
    for q, stats_ in quartile_stats.items():
        quartile_data.append({
            "quartile_indicator": quartile_name,
            "quartile": q,
            "n_tracts": stats_["count"],
            "mean_miles": stats_["mean"],
            "median_miles": stats_["median"],
            "std_miles": stats_["std"],
            "min_miles": stats_["min"],
            "max_miles": stats_["max"],
            "analysis_type": "quartile_stats"
        })

quartile_df = pd.DataFrame(quartile_data)
quartile_df.to_csv(quartile_path, index=False)

print(f" GLMEM results saved to: {results_path}")
print(f" Correlation analysis saved to: {correlation_path}")
print(f" Summary statistics saved to: {summary_path}")
print(f" Quartile analysis results saved to: {quartile_path}")

print(f"\n DISTANCE ANALYSIS COMPLETE!")
print(f" Analyzed travel distance (distance_miles)")
print(f" Compared against {len(socioeconomic_indicators)} socioeconomic indicators and {len(svi_themes)} SVI themes")
print(f" Results saved to {RESULTS_DIR}")
