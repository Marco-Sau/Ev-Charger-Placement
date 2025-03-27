from ev_charger import create_urban_scenario, generate_candidate_locations, calculate_installation_costs

def test_main():
    print("Testing main EV charger functionality...")
    
    # Generate urban scenario
    print("Creating urban scenario...")
    scenario = create_urban_scenario(growth_factor=1.0)
    
    # Extract vehicles and districts
    vehicles = scenario['vehicles']
    districts = scenario['districts']
    
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    
    # Show a few district examples
    print("\nSample districts:")
    for i, (name, props) in enumerate(districts.items()):
        if i < 5:  # Just show the first 5
            print(f"  - {name}: center={props['center']}, size={props['size']}, density={props['density']}")
        else:
            break
    
    # Generate candidate locations
    print("\nGenerating candidate locations...")
    locations = generate_candidate_locations(vehicles, districts)
    print(f"Generated {len(locations)} candidate locations")
    
    # Calculate installation costs
    print("\nCalculating installation costs...")
    costs = calculate_installation_costs(locations, districts)
    print(f"Cost range: ${min(costs):.2f} to ${max(costs):.2f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_main() 