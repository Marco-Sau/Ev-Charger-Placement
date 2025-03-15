import pandas as pd

def main():
    # Load the CSV files
    print("Loading CSV files...")
    try:
        # Load municipalities data with population and surface area
        municipalities_df = pd.read_csv("sardinia_municipalities.csv")
        print(f"Loaded municipalities data: {len(municipalities_df)} entries")
        
        # Load coordinates data
        coordinates_df = pd.read_csv("sardinia_municipalities_with_coordinates.csv")
        print(f"Loaded coordinates data: {len(coordinates_df)} entries")
        
        # Merge the dataframes on the 'Municipality' column
        print("Merging data...")
        merged_df = pd.merge(
            municipalities_df, 
            coordinates_df, 
            on='Municipality', 
            how='outer'  # Use outer join to keep all municipalities from both files
        )
        
        # Count how many entries have complete data
        complete_entries = merged_df.dropna().shape[0]
        total_entries = merged_df.shape[0]
        print(f"Successfully merged data: {complete_entries} complete entries out of {total_entries} total")
        
        # Check for municipalities without coordinates
        missing_coords = merged_df[merged_df['Latitude'].isna() | merged_df['Longitude'].isna()]
        if not missing_coords.empty:
            print(f"\nMunicipalities missing coordinates ({len(missing_coords)}):")
            for idx, row in missing_coords.iterrows():
                print(f"- {row['Municipality']}")
        
        # Check for municipalities without population/surface data
        missing_data = merged_df[merged_df['Population'].isna() | merged_df['Surface_km2'].isna()]
        if not missing_data.empty:
            print(f"\nMunicipalities missing population or surface area data ({len(missing_data)}):")
            for idx, row in missing_data.iterrows():
                print(f"- {row['Municipality']}")
        
        # Save the merged data to a new CSV file
        output_file = "sardinia_municipalities_complete.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved merged data to {output_file}")
        
        # Display the first few rows of the merged data
        print("\nSample of merged data:")
        print(merged_df.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 