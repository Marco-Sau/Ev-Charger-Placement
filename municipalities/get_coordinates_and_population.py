import csv
import time
import requests
import json
from urllib.parse import quote

def get_coordinates(municipality_name):
    """
    Get the latitude and longitude of a municipality using Nominatim API.
    Adds 'Sardinia, Italy' to the search query for better accuracy.
    """
    # Format the search query
    search_query = f"{municipality_name}, Sardinia, Italy"
    encoded_query = quote(search_query)
    
    # Nominatim API URL
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
    
    # Add user-agent to comply with Nominatim usage policy
    headers = {
        "User-Agent": "SardiniaCoordinatesRetriever/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data and len(data) > 0:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            return float(lat), float(lon)
        else:
            print(f"No results found for {municipality_name}")
            return None, None
    except Exception as e:
        print(f"Error retrieving coordinates for {municipality_name}: {e}")
        return None, None

def fetch_population_data():
    """
    Fetch population data for all Italian municipalities from ISTAT open data.
    Returns a dictionary with municipality names as keys and population as values.
    """
    # This URL contains the latest ISTAT demographic data for Italian municipalities
    url = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_municipalities.json"
    
    try:
        print("Fetching ISTAT population data...")
        response = requests.get(url)
        data = response.json()
        
        # Create a dictionary of municipality name to population
        population_dict = {}
        
        for feature in data['features']:
            properties = feature['properties']
            name = properties.get('name')
            population = properties.get('population')
            region_name = properties.get('region_name')
            
            # Only include municipalities in Sardinia
            if region_name and region_name.lower() == 'sardegna' and name and population:
                population_dict[name] = int(population)
                
                # Also add variations with slight differences in naming conventions
                if "'" in name:
                    alternate_name = name.replace("'", " ")
                    population_dict[alternate_name] = int(population)
                
        print(f"Found population data for {len(population_dict)} Sardinian municipalities")
        return population_dict
    
    except Exception as e:
        print(f"Error fetching population data: {e}")
        return {}

def get_local_population_data():
    """
    Alternative method: Use a manually created dataset of populations for Sardinian municipalities.
    This is a fallback if the API data is unavailable.
    """
    # Data from ISTAT 2021 census, this is just a subset for demonstration
    sardinia_population = {
        "Cagliari": 149477,
        "Sassari": 126218,
        "Olbia": 59368,
        "Alghero": 44048,
        "Quartu Sant'Elena": 69295,
        "Nuoro": 35721,
        "Oristano": 31549,
        "Carbonia": 28755,
        "Selargius": 29082,
        "Iglesias": 26634,
        "Assemini": 26620,
        "Porto Torres": 22545,
        "Arzachena": 13462,
        "Siniscola": 11482,
        "Ozieri": 10713,
        "Tempio Pausania": 13916,
        "Villacidro": 14025,
        "Guspini": 11902,
        "Monserrato": 19856,
        "Macomer": 9906,
        "Tortolì": 11133,
        "La Maddalena": 11170,
        "Sanluri": 8515,
        "Elmas": 9331,
        "Sorso": 13682
    }
    return sardinia_population

def main():
    # Read the municipalities file
    municipalities = []
    with open("sardinia_municipalities.csv", "r", encoding="utf-8") as f:
        for line in f:
            municipality = line.strip()
            if municipality:  # Skip empty lines
                municipalities.append(municipality)
    
    # Get population data from ISTAT
    population_data = fetch_population_data()
    
    # Fallback to local data if needed
    if not population_data:
        print("Using local population data as fallback...")
        population_data = get_local_population_data()
    
    # Create a new CSV file with coordinates and population
    with open("sardinia_municipalities_complete.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Municipality", "Latitude", "Longitude", "Population"])
        
        for municipality in municipalities:
            print(f"Processing {municipality}...")
            
            # Get coordinates
            lat, lon = get_coordinates(municipality)
            
            # Get population from our dataset
            population = population_data.get(municipality)
            if population is None:
                # Try with different capitalization
                population = population_data.get(municipality.title())
            
            writer.writerow([municipality, lat, lon, population])
            
            # Sleep to avoid hitting rate limits for the geocoding API
            time.sleep(1.2)
            
    print("Done! Data saved to sardinia_municipalities_complete.csv")

if __name__ == "__main__":
    main() 