import pandas as pd
import geopandas as gpd
import os
import zipfile
import tempfile
import numpy as np

# --- Determine correct data directory relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = SCRIPT_DIR  # Save results in the equitability directory

# Load the Illinois census tract DP05 data and SVI data
dp05_path = os.path.join(DATA_DIR, "illinois_census_tract_DP05.csv")
svi_path = os.path.join(DATA_DIR, "SVI", "Illinois.csv")

if not os.path.exists(dp05_path):
    raise FileNotFoundError(f"DP05 CSV not found at {dp05_path}")

if not os.path.exists(svi_path):
    raise FileNotFoundError(f"SVI CSV not found at {svi_path}")

dp05 = pd.read_csv(dp05_path, dtype=str)
svi = pd.read_csv(svi_path, dtype=str)

# We'll get urgent care info from the travel times matrix instead

# --- Robustly find or construct the GEOID column for DP05 ---
geoid_col = None
for c in dp05.columns:
    c_clean = c.strip().upper()
    if c_clean in ["GEOID", "GEOID10"]:
        geoid_col = c
        break

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
        dp05["GEOID"] = (
            dp05[state_col].str.zfill(2)
            + dp05[county_col].str.zfill(3)
            + dp05[tract_col].str.zfill(6)
        )
        geoid_col = "GEOID"
    elif "GEO_ID" in dp05.columns:
        dp05["GEOID"] = dp05["GEO_ID"].str[-11:]
        geoid_col = "GEOID"
    else:
        print("Available columns:", list(dp05.columns))
        raise KeyError("No GEOID column found or constructible in DP05 CSV. Please check the column names.")

dp05 = dp05[dp05[geoid_col].str.len() == 11].copy()
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

svi_chi = svi[svi[svi_geoid_col].str.startswith('17031')].copy()

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

# --- MM-E2SFCA Spatial Access Index Calculation ---

# Load travel time matrix: from each census tract centroid to each urgent care
# Use the combined matrix file that has the proper format
traveltimes_path = os.path.join(ROOT_DIR, "results", "travel_times_matrix.csv")
if not os.path.exists(traveltimes_path):
    raise FileNotFoundError(f"Could not find travel_times_matrix.csv at {traveltimes_path}")

traveltimes_df = pd.read_csv(traveltimes_path, dtype=str)

# Use the exact column names from our matrix file
tract_col = "origin_GEOID"
uc_col = "urgentcare_id"
car_col = "car_time_min"
transit_col = "transit_time_min"

if tract_col is None:
    raise KeyError("Could not find tract (origin) column in travel_times.csv")
if uc_col is None:
    raise KeyError("Could not find urgent care (destination) column in travel_times.csv")
if car_col is None:
    raise KeyError("Could not find car travel time column in travel_times.csv")
if transit_col is None:
    raise KeyError("Could not find transit travel time column in travel_times.csv")

# Convert travel time columns to float
traveltimes_df[car_col] = pd.to_numeric(traveltimes_df[car_col], errors="coerce")
traveltimes_df[transit_col] = pd.to_numeric(traveltimes_df[transit_col], errors="coerce")

print(f"Car time range after conversion: {traveltimes_df[car_col].min()} - {traveltimes_df[car_col].max()}")
print(f"Transit time range after conversion: {traveltimes_df[transit_col].min()} - {traveltimes_df[transit_col].max()}")
print(f"Sample car times: {traveltimes_df[car_col].head().tolist()}")
print(f"Sample transit times: {traveltimes_df[transit_col].head().tolist()}")

# Only keep tracts in our study area
tract_geoids = set(gdf["GEOID"])
print(f"Study area tracts: {len(tract_geoids)}")
print(f"Travel times tracts before filtering: {traveltimes_df[tract_col].nunique()}")

# Filter travel times to only include tracts in our study area
traveltimes_df = traveltimes_df[traveltimes_df[tract_col].isin(tract_geoids)]
print(f"Travel times tracts after filtering: {traveltimes_df[tract_col].nunique()}")

# Check if we have any data left
if len(traveltimes_df) == 0:
    raise ValueError("No travel time data found for tracts in study area. Check GEOID matching.")

# Prepare urgent care supply: get facilities from travel times matrix and assume each is 1 "unit" of supply
urgent_care_facilities = traveltimes_df[uc_col].unique()
urgent_care = pd.DataFrame({'supply': 1}, index=urgent_care_facilities)
print(f"Found {len(urgent_care_facilities)} urgent care facilities in travel times matrix")

# Prepare demand: use total population from DP05
pop_col = None
for c in dp05_chi.columns:
    if c.strip().upper() in ["DP05_0001E", "DP05_0001", "POPULATION", "TOTPOP", "TOTAL_POP"]:
        pop_col = c
        break
if pop_col is None:
    for c in dp05_chi.columns:
        if "total" in c.lower() and "pop" in c.lower():
            pop_col = c
            break
if pop_col is None:
    raise KeyError("Could not find total population column in DP05.")

tract_pop = dp05_chi.set_index("GEOID")[pop_col].astype(float)

# MM-E2SFCA parameters
catchment_car = 20  # minutes
catchment_transit = 20  # minutes

# 1. For each urgent care, compute its catchment area (tracts within 20 min by car or transit)
print("Step 1: Calculating supply-to-demand ratios for each urgent care...")

supply_to_demand = {}
total_catchments = 0
for uc_id, uc_row in urgent_care.iterrows():
    # Find all tracts within 20 min by car or transit to this urgent care
    mask = (
        (traveltimes_df[uc_col] == str(uc_id)) &
        (
            (traveltimes_df[car_col] <= catchment_car) |
            (traveltimes_df[transit_col] <= catchment_transit)
        )
    )
    
    # Debug output for first few facilities
    if uc_id in list(urgent_care.index)[:3]:
        print(f"\nDebug for facility {uc_id}:")
        print(f"  Rows with this facility: {(traveltimes_df[uc_col] == str(uc_id)).sum()}")
        print(f"  Rows within car time limit: {(traveltimes_df[car_col] <= catchment_car).sum()}")
        print(f"  Rows within transit time limit: {(traveltimes_df[transit_col] <= catchment_transit).sum()}")
        print(f"  Rows in final mask: {mask.sum()}")
        if mask.sum() > 0:
            sample_times = traveltimes_df.loc[mask, [car_col, transit_col]].head()
            print(f"  Sample times in mask:\n{sample_times}")
    
    tracts_in_catchment = traveltimes_df.loc[mask, tract_col].unique()
    if len(tracts_in_catchment) == 0:
        supply_to_demand[uc_id] = 0
        continue
    
    total_catchments += len(tracts_in_catchment)
    # Demand is sum of population in these tracts
    demand = tract_pop.loc[[t for t in tracts_in_catchment if t in tract_pop.index]].sum()
    if demand > 0:
        supply_to_demand[uc_id] = uc_row["supply"] / demand
    else:
        supply_to_demand[uc_id] = 0

print(f"Total tracts in catchments: {total_catchments}")
print(f"Facilities with catchments: {sum(1 for v in supply_to_demand.values() if v > 0)}")

# 2. For each tract, compute its spatial access index as the sum of supply-to-demand ratios of all urgent cares within 20 min by car or transit
print("Step 2: Calculating spatial access index for each tract...")

tract_access_index = {}
for tract in tract_geoids:
    # Find all urgent cares within 20 min by car or transit from this tract
    mask = (
        (traveltimes_df[tract_col] == tract) &
        (
            (traveltimes_df[car_col] <= catchment_car) |
            (traveltimes_df[transit_col] <= catchment_transit)
        )
    )
    uc_ids = traveltimes_df.loc[mask, uc_col].unique()
    access_sum = 0
    for uc_id in uc_ids:
        access_sum += supply_to_demand.get(uc_id, 0)
    tract_access_index[tract] = access_sum

# Add spatial access index to gdf
gdf["MME2SFCA_access_index"] = gdf["GEOID"].map(tract_access_index)

# Save results
os.makedirs(RESULTS_DIR, exist_ok=True)
access_index_path = os.path.join(RESULTS_DIR, "mme2sfca_urgentcare_access_index.csv")
gdf[["GEOID", "MME2SFCA_access_index"]].to_csv(access_index_path, index=False)

print(f" MM-E2SFCA spatial access index saved to: {access_index_path}")
print(" MM-E2SFCA spatial access computation complete!")
