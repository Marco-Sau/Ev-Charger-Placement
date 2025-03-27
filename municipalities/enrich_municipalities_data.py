import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import quote

def get_wikipedia_data():
    """
    Scrape the Italian Wikipedia page for Sardinian municipalities to extract
    population and surface area data.
    """
    url = "https://it.wikipedia.org/wiki/Comuni_della_Sardegna"
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with municipalities data
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            print("Could not find the municipalities table on Wikipedia")
            return {}
        
        # Extract data from the table
        municipalities_data = {}
        
        # Process table rows (skip header row)
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 6:  # Make sure we have enough columns
                # Extract municipality name (col 0)
                municipality_name_cell = cols[0].find('a')
                if municipality_name_cell:
                    municipality_name = municipality_name_cell.get_text().strip()
                else:
                    municipality_name = cols[0].get_text().strip()
                
                # Extract population (col 2)
                population_text = cols[2].get_text().strip().replace(".", "")  # Remove thousands separator
                population = int(population_text) if population_text.isdigit() else None
                
                # Extract surface area (col 5)
                surface_text = cols[5].get_text().strip().replace(",", ".")  # Replace comma with dot for decimal
                try:
                    surface = float(surface_text)
                except ValueError:
                    surface = None
                
                # Store the data
                municipalities_data[municipality_name] = {
                    'population': population,
                    'surface_area': surface
                }
        
        print(f"Found data for {len(municipalities_data)} municipalities on Wikipedia")
        return municipalities_data
    
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return {}

def get_coordinates(municipality_name):
    """
    Get the latitude and longitude of a municipality using Nominatim API.
    """
    search_query = f"{municipality_name}, Sardinia, Italy"
    url = f"https://nominatim.openstreetmap.org/search?q={quote(search_query)}&format=json&limit=1"
    
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

def main():
    # Get population and surface area data from Wikipedia
    print("Extracting population and surface area data from Wikipedia...")
    wiki_data = get_wikipedia_data()
    
    if not wiki_data:
        print("Failed to retrieve data from Wikipedia.")
        return
    
    # Create a list to store all municipality data
    all_data = []
    
    # Process each municipality
    for municipality_name, data in wiki_data.items():
        print(f"Processing {municipality_name}...")
        
        # Get coordinates
        lat, lon = get_coordinates(municipality_name)
        
        # Add to our data structure
        all_data.append({
            'Municipality': municipality_name,
            'Latitude': lat,
            'Longitude': lon,
            'Population': data['population'],
            'Surface_Area_km2': data['surface_area']
        })
        
        # Sleep to respect Nominatim usage policy
        time.sleep(1.2)
    
    # Create DataFrame from all data
    df = pd.DataFrame(all_data)
    
    # Try to fill in missing coordinates from existing data file if available
    try:
        existing_df = pd.read_csv("sardinia_municipalities_data.csv")
        
        # Create a dictionary mapping municipality names to coordinates
        coord_map = {row['Municipality']: (row['Latitude'], row['Longitude']) 
                    for _, row in existing_df.iterrows() 
                    if pd.notna(row['Latitude']) and pd.notna(row['Longitude'])}
        
        # Fill in missing coordinates
        for i, row in df.iterrows():
            if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                municipality = row['Municipality']
                if municipality in coord_map:
                    df.at[i, 'Latitude'] = coord_map[municipality][0]
                    df.at[i, 'Longitude'] = coord_map[municipality][1]
    except Exception as e:
        print(f"Note: Could not use existing coordinates data: {e}")
    
    # Save the complete data to CSV
    df.to_csv("sardinia_municipalities_enriched.csv", index=False)
    print("Done! Saved enriched data to sardinia_municipalities_enriched.csv")

if __name__ == "__main__":
    main() 