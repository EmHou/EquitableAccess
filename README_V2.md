# GeoHealth Analysis Project - Complete Guide

## Project Overview

This project analyzes healthcare accessibility in Chicago by examining the relationship between Social Vulnerability Index (SVI) and urgent care facility access using multiple transportation modes. The analysis uses Google Maps Distance Matrix API for accurate travel time calculations.

## Project Structure

```
GeoHealth Repo Final/
├──  data/                           # Raw data files
│   ├── Chicago_City_Limits-shp/      # Chicago city boundary shapefiles
│   ├── SVI/                          # Social Vulnerability Index data
│   │   ├── Illinois.csv
│   │   ├── NewYork.csv
│   │   └── SVI_2022_US.csv
│   ├── illinois_census_tract_DP05.csv # Census demographic data
│   ├── tl_2022_17_tract.zip          # Illinois census tract boundaries
│   └── urgent_care_facilities.csv    # Urgent care facility locations
│
├──  Distance Processing/            # Google Maps API analysis
│   ├── distance_matrix_api.py        # Main API analysis script
│   ├── distance.py                   # Distance calculation utilities
│   ├── geo_access.py                 # Geographic accessibility functions
│   ├── data update chicago.py        # Fix missing data for 1 specific GEOID
│   └── test_google_cloud_setup.py    # API setup verification
│   
│
├──  Stats processing/               # Statistical analysis
│   ├── glmem_analysis.py             # Main statistical analysis
│   ├── get_travel_times_by_mode.py   # Travel time processing & Friedman test analysis
│   ├── make_glmem_stats_table.py     # Statistics table generation from GLMEM results
│   ├── glmem_statistical_table.csv   # Generated statistics
│   └── population_accessibility_summary.txt
│
├──  Visualisations/                # Data visualization scripts
│   ├── social_visuals.py             # Social vulnerability visualizations
│   ├── svi_chloropleth.py            # SVI choropleth maps
│   └── urgentcare_access_10min_intervals.py # Access interval analysis
│
├──  processed_data/                # Intermediate processed data
│   ├── chicago_urgentcare_detailed_results.csv
│   ├── chicago_urgentcare_final_results.geojson
│   └── chicago_urgentcare_final_summary.csv
│
├──  results/                       # Analysis outputs
│   ├── travel_times_final.csv        # Google Maps API travel times (CSV)
│   ├── travel_times_final.geojson    # Google Maps API travel times (GeoJSON)
│   └── travel_times_backup_batch_X.csv # Backup files during processing
│        
├── data_processing.py                # Initial data processing
├── requirements.txt                  # Python dependencies
└── README_V2.md                     # This file
```

## Quick Start Guide
### 1. Prerequisites

- **Python 3.8+** with pip
- **Google Cloud Account** with billing enabled
- **Google Maps Distance Matrix API** access

### 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Google Cloud API Setup

1. **Enable the Distance Matrix API**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Search for "Distance Matrix API"

2. **Create API Key**

3. **Set Environment Variables**:
   ```bash
   # Create .env file in project root
   echo "GOOGLE_API=your_api_key_here" > .env
   ```

4. **Test Setup**:
   ```bash
   python "Distance Processing/test_google_cloud_setup.py"
   ```

## Analysis Workflow

### Step 1: Data Processing
```bash
python data_processing.py
```
**Purpose**: Processes raw data files and creates initial datasets
**Outputs**: 
- `processed_data/chicago_urgentcare_final_results.geojson`
- `processed_data/chicago_urgentcare_detailed_results.csv`

### Step 2: Distance Matrix Analysis
```bash
python "Distance Processing/distance_matrix_api.py"
```
**Purpose**: Calculates travel times using Google Maps API
**Features**:
- Car vs Transit comparison
- SVI quartile analysis
- Console-based statistical output
- **Saves Google Maps API data** to CSV and GeoJSON files
**Outputs**:
- `results/travel_times_final.csv` - Complete dataset with travel times
- `results/travel_times_final.geojson` - GeoJSON format for mapping
- `results/travel_times_backup_batch_X.csv` - Backup files during processing

### Step 3: Statistical Analysis
```bash
python "Stats processing/glmem_analysis.py"
```
**Purpose**: Comprehensive statistical analysis
**Outputs**:
- Statistical tables
- Population accessibility summaries
- Detailed demographic analysis

### Step 4: Visualization
```bash
Distance Processing/data update chicago.py - Fixes missing transportation data where Google API failed. Updates `travel_times_final.csv` with complete data

python "Visualisations/social_visuals.py"
python "Visualisations/svi_chloropleth.py"
python "Visualisations/urgentcare_access_10min_intervals.py"
```
**Purpose**: Generate maps and charts
**Outputs**: Various visualization files

