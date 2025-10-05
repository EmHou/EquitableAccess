#!/usr/bin/env python3
"""
Test script to re-request Google Distance Matrix API for GEOID 17031838800
to get missing transit and walking times.
"""

import pandas as pd
import googlemaps
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def get_google_maps_client():
    """Initialize Google Maps client with API key."""
    api_key = os.getenv('GOOGLE_API')
    if not api_key:
        raise ValueError("GOOGLE_API environment variable not found. Please set it in your .env file.")
    
    return googlemaps.Client(key=api_key)

def get_urgent_care_centers():
    """Load all Chicago urgent care facilities (excluding Chicago Heights)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(root_dir, "data")
    
    urgent_care_path = os.path.join(data_dir, "urgent_care_facilities.csv")
    df = pd.read_csv(urgent_care_path)
    
    # Filter to only Chicago facilities by city name, but exclude Chicago Heights
    chicago_mask = (
        df['city'].str.contains('CHICAGO', case=False, na=False) & 
        ~df['city'].str.contains('CHICAGO HEIGHTS', case=False, na=False)
    )
    chicago_df = df[chicago_mask].copy()
    
    print(f"Loaded {len(df)} total urgent care facilities")
    print(f"Filtered to {len(chicago_df)} Chicago facilities (excluding Chicago Heights)")
    
    return chicago_df

def make_distance_matrix_request(gmaps_client, origin_coords, urgent_care_coords, mode="transit"):
    """Make Google Distance Matrix API requests with batching for large numbers of destinations."""
    
    # Convert coordinates to strings
    origin_str = f"{origin_coords[0]},{origin_coords[1]}"
    destinations = [f"{row['latitude']},{row['longitude']}" for _, row in urgent_care_coords.iterrows()]
    
    print(f"Origin: {origin_str}")
    print(f"Destinations: {len(destinations)} urgent care centers")
    
    # Google Distance Matrix API has a limit of 25 destinations per request
    max_destinations = 25
    all_results = []
    
    # Process destinations in batches
    for i in range(0, len(destinations), max_destinations):
        batch_destinations = destinations[i:i + max_destinations]
        batch_num = i // max_destinations + 1
        total_batches = (len(destinations) + max_destinations - 1) // max_destinations
        
        print(f"   Processing batch {batch_num}/{total_batches} ({len(batch_destinations)} destinations)")
        
        try:
            if mode == "transit":
                result = gmaps_client.distance_matrix(
                    origins=[origin_str],
                    destinations=batch_destinations,
                    mode="transit",
                    departure_time="now",
                    traffic_model="best_guess"
                )
            else:  # walking
                result = gmaps_client.distance_matrix(
                    origins=[origin_str],
                    destinations=batch_destinations,
                    mode="walking"
                )
            
            all_results.append(result)
            print(f"Batch {batch_num} successful")
            
            # Add a small delay between requests to avoid rate limiting
            if i + max_destinations < len(destinations):
                time.sleep(1)
                
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            return None
    
    print(f"{mode.capitalize()} API requests successful ({len(all_results)} batches)")
    return all_results

def make_walking_request(gmaps_client, origin_coords, urgent_care_coords):
    """Make walking distance matrix requests."""
    return make_distance_matrix_request(gmaps_client, origin_coords, urgent_care_coords, mode="walking")

def make_transit_request(gmaps_client, origin_coords, urgent_care_coords):
    """Make transit distance matrix requests."""
    return make_distance_matrix_request(gmaps_client, origin_coords, urgent_care_coords, mode="transit")

def process_api_results(transit_results, walking_results, urgent_care_coords):
    """Process batched API results and find the closest urgent care center."""
    
    if not transit_results or not walking_results:
        print("Missing API results")
        return None, None
    
    # Combine all transit times from all batches
    transit_times = []
    for result in transit_results:
        if result['rows'][0]['elements']:
            for element in result['rows'][0]['elements']:
                if element['status'] == 'OK':
                    transit_time = element['duration']['value'] / 60  # Convert to minutes
                    transit_times.append(transit_time)
                else:
                    transit_times.append(None)
    
    # Combine all walking times from all batches
    walking_times = []
    for result in walking_results:
        if result['rows'][0]['elements']:
            for element in result['rows'][0]['elements']:
                if element['status'] == 'OK':
                    walking_time = element['duration']['value'] / 60  # Convert to minutes
                    walking_times.append(walking_time)
                else:
                    walking_times.append(None)
    
    print(f"Processed {len(transit_times)} transit times and {len(walking_times)} walking times")
    
    # Find the closest urgent care center (shortest transit time)
    valid_transit_times = [(i, t) for i, t in enumerate(transit_times) if t is not None]
    
    if not valid_transit_times:
        print("No valid transit times found")
        return None, None

    # Get the closest center
    closest_idx, closest_transit_time = min(valid_transit_times, key=lambda x: x[1])
    closest_walking_time = walking_times[closest_idx] if closest_idx < len(walking_times) else None
    
    closest_center = urgent_care_coords.iloc[closest_idx]
    
    print(f"Closest urgent care center:")
    print(f"   Name: {closest_center['name']}")
    print(f"   Address: {closest_center['address']}")
    print(f"   Transit time: {closest_transit_time:.2f} minutes")
    print(f"   Walking time: {closest_walking_time:.2f} minutes" if closest_walking_time else "   Walking time: N/A")
    
    return closest_transit_time, closest_walking_time

def update_csv_file(geoid, transit_time, walking_time):
    """Update the travel_times_final.csv file with new times."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(root_dir, "results")
    csv_path = os.path.join(results_dir, "travel_times_final.csv")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Find the row with the specified GEOID
    geoid_mask = df['GEOID'] == geoid
    matching_rows = df[geoid_mask]
    
    if len(matching_rows) == 0:
        print(f"GEOID {geoid} not found in CSV")
        return False
    
    print(f"Found {len(matching_rows)} rows with GEOID {geoid}")
    
    # Update the times
    df.loc[geoid_mask, 'transit_time_min'] = transit_time
    df.loc[geoid_mask, 'walking_time_min'] = walking_time
    
    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV file with new times for GEOID {geoid}")
    print(f"   Transit time: {transit_time:.2f} minutes")
    print(f"   Walking time: {walking_time:.2f} minutes")
    
    return True

def main():
    """Main function to execute the API request and update."""
    
    print("Starting Google Distance Matrix API request for GEOID 17031838800")
    print("=" * 70)
    
    # Target coordinates for GEOID 17031838800
    target_coords = (41.684030, -87.601876)
    target_geoid = 17031838800
    
    try:
        # Initialize Google Maps client
        gmaps_client = get_google_maps_client()
        print("Google Maps client initialized")
        
        # Load urgent care facilities
        urgent_care_df = get_urgent_care_centers()
        
        # Make API requests
        print("\nMaking Google Distance Matrix API requests...")
        
        # Transit request
        transit_results = make_transit_request(gmaps_client, target_coords, urgent_care_df)
        
        # Add a small delay between requests
        time.sleep(2)
        
        # Walking request
        walking_results = make_walking_request(gmaps_client, target_coords, urgent_care_df)
        
        # Process results
        print("\nProcessing API results...")
        transit_time, walking_time = process_api_results(transit_results, walking_results, urgent_care_df)
        
        if transit_time is not None and walking_time is not None:
            # Update CSV file
            print("\nUpdating CSV file...")
            success = update_csv_file(target_geoid, transit_time, walking_time)
            
            if success:
                print("\SUCCESS! GEOID 17031838800 has been updated with new transit and walking times.")
            else:
                print("\nFailed to update CSV file")
        else:
            print("\nFailed to get valid times from API")
            
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
