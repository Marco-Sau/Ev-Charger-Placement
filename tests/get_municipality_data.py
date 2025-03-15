import csv
import time
import requests
import json

# Nominatim API for geocoding
def get_coordinates(municipality_name):
    """
    Get the latitude and longitude of a municipality using Nominatim API.
    Adds 'Sardinia, Italy' to the search query for better accuracy.
    """
    search_query = f"{municipality_name}, Sardinia, Italy"
    url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=1"
    
    headers = {
        "User-Agent": "SardiniaDataCollector/1.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data and len(data) > 0:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            return float(lat), float(lon)
        else:
            print(f"No coordinates found for {municipality_name}")
            return None, None
    except Exception as e:
        print(f"Error retrieving coordinates for {municipality_name}: {e}")
        return None, None

# ISTAT Population data
def get_population_data():
    """
    Fetch ISTAT population data for Italian municipalities.
    Returns a dictionary with municipality names as keys and population values.
    """
    # Using the ISTAT endpoint for demographic data
    url = "https://demo.istat.it/api/v1.0/comuni"
    
    try:
        print("Fetching ISTAT population data...")
        population_dict = {}
        
        # Sardinia region code is 20
        # Filter for municipalities in Sardinia
        response = requests.get(f"{url}?region=20")
        data = response.json()
        
        if data:
            for item in data:
                name = item.get("name")
                population = item.get("population")
                if name and population:
                    population_dict[name] = int(population)
        
        return population_dict
    except Exception as e:
        print(f"Error fetching ISTAT data: {e}")
        return {}

# Alternative data source - DBpedia
def get_dbpedia_population(municipality_name):
    """
    Try to get population data from DBpedia if ISTAT data is not available.
    """
    # SPARQL endpoint
    endpoint = "https://dbpedia.org/sparql"
    
    # Query to find population data for a municipality
    query = f"""
    SELECT ?population WHERE {{
      ?city rdfs:label "{municipality_name}"@it .
      ?city dbo:country dbr:Italy .
      ?city dbo:populationTotal ?population .
    }}
    """
    
    try:
        response = requests.get(
            endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/json"}
        )
        
        data = response.json()
        results = data.get("results", {}).get("bindings", [])
        
        if results and len(results) > 0:
            population = int(results[0]["population"]["value"])
            return population
        else:
            return None
    except Exception as e:
        print(f"Error getting DBpedia data for {municipality_name}: {e}")
        return None

# Hardcoded population data for key municipalities as fallback
def get_fallback_population():
    """
    Hardcoded recent population data for major Sardinian municipalities.
    Used as a fallback if online sources fail.
    """
    return {
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

def main():
    # Load the municipality names
    municipalities = []
    with open("municipalities_names_clean.csv", "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:  # Skip empty lines
                municipalities.append(name)
    
    print(f"Loaded {len(municipalities)} municipality names")
    
    # Try to get population data from ISTAT
    population_data = get_population_data()
    
    # Use fallback data if ISTAT data retrieval failed
    if not population_data:
        print("Using fallback population data...")
        population_data = get_fallback_population()
    
    # Create output file with coordinates and population
    with open("sardinia_municipalities_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Municipality", "Latitude", "Longitude", "Population"])
        
        for municipality in municipalities:
            print(f"Processing {municipality}...")
            
            # Get coordinates
            lat, lon = get_coordinates(municipality)
            
            # Get population
            population = population_data.get(municipality)
            
            # If not found in ISTAT data, try DBpedia
            if population is None:
                print(f"Trying DBpedia for {municipality}...")
                population = get_dbpedia_population(municipality)
            
            # Write data to CSV
            writer.writerow([municipality, lat, lon, population])
            
            # Sleep to respect Nominatim usage policy
            time.sleep(1.2)
    
    print("Done! Data saved to sardinia_municipalities_data.csv")

if __name__ == "__main__":
    main() 