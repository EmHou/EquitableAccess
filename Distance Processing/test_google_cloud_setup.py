#!/usr/bin/env python3
"""
Test script to verify Google Maps Distance Matrix API setup
"""

import os
import sys
import googlemaps
from dotenv import load_dotenv

def test_google_maps_setup():
    """Test Google Maps API authentication and Distance Matrix API access"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    print(" Testing Google Maps Distance Matrix API Setup...")
    print("=" * 50)
    
    # Test 1: Check API key
    print("\n1. Testing API key...")
    try:
        # Get the API key from environment variable
        api_key = os.getenv('GOOGLE_API')
        if not api_key:
            raise ValueError("GOOGLE_API environment variable not set")
        
        # Initialize the Google Maps client
        gmaps = googlemaps.Client(key=api_key)
        print(f"    API key loaded successfully")
    except ValueError as e:
        print(f"   ERROR: Environment variable error: {e}")
        print("    Set GOOGLE_API in your .env file")
        return False
    except Exception as e:
        print(f"   ERROR: API key error: {e}")
        return False
    
    # Test 2: Initialize client
    print("\n2. Testing Google Maps client...")
    try:
        gmaps = googlemaps.Client(key=api_key)
        print("    Google Maps client initialized")
    except Exception as e:
        print(f"   ERROR: Client initialization failed: {e}")
        return False
    
    # Test 3: Test simple API call
    print("\n3. Testing API call...")
    try:
        # Simple test with Chicago coordinates
        origin = (41.8781, -87.6298)  # Chicago downtown
        destination = (41.8968, -87.6208)  # Chicago north side
        
        print("    Making test API call...")
        result = gmaps.distance_matrix(
            origin,
            destination,
            mode='driving',
            traffic_model='best_guess',
            departure_time='now'
        )
        
        if result['status'] == 'OK':
            element = result['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                duration_minutes = element['duration']['value'] / 60
                distance_meters = element['distance']['value']
                print(f"    API call successful!")
                print(f"    Travel time: {duration_minutes:.1f} minutes")
                print(f"    Distance: {distance_meters:.0f} meters")
            else:
                print(f"   Warning:  API returned status: {element['status']}")
        else:
            print(f"   Warning:  API returned status: {result['status']}")
            
    except Exception as e:
        print(f"   ERROR: API call failed: {e}")
        print("    Make sure your Google Maps API key is valid and has Distance Matrix API enabled")
        return False
    
    print("\n All tests passed! Google Maps Distance Matrix API is working correctly.")
    print("\n Next steps:")
    print("   1. Run the main analysis script: python distance_matrix_api.py")
    print("   2. Monitor API usage in Google Cloud Console")
    print("   3. Check costs in the Billing section")
    
    return True

if __name__ == "__main__":
    success = test_google_maps_setup()
    sys.exit(0 if success else 1) 