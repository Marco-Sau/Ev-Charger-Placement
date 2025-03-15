import csv
import time
import requests
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
    
def main():
    # Read the municipalities file
    municipalities = []
    with open("sardinia_municipalities.csv", "r", encoding="utf-8") as f:
        for line in f:
            municipality = line.strip()
            if municipality:  # Skip empty lines
                municipalities.append(municipality)
    
    # Create a new CSV file with coordinates
    with open("sardinia_municipalities_with_coordinates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Municipality", "Latitude", "Longitude"])
        
        for municipality in municipalities:
            print(f"Processing {municipality}...")
            lat, lon = get_coordinates(municipality)
            writer.writerow([municipality, lat, lon])
            
            # Sleep to avoid hitting rate limits (Nominatim requires max 1 request per second)
            time.sleep(1.2)
            
    print("Done! Coordinates saved to sardinia_municipalities_with_coordinates.csv")

if __name__ == "__main__":
    main() 