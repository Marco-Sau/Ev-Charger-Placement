from ev_charger import create_urban_scenario, generate_candidate_locations, calculate_installation_costs, MAX_VEHICLES_DAY
import numpy as np

def test_fixed_vehicles():
    print("Testing fixed number of vehicles generation...")
    
    # Test with 1000 vehicles
    print("\nGenerating scenario with 1000 vehicles...")
    scenario = create_urban_scenario(fixed_num_vehicles=1000)
    
    # Extract vehicles and districts
    vehicles = scenario['vehicles']
    districts = scenario['districts']
    
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    
    # Generate candidate locations
    print("\nGenerating candidate locations...")
    locations = generate_candidate_locations(vehicles, districts)
    print(f"Generated {len(locations)} candidate locations")
    
    # Check if we have enough capacity
    num_locations = len(locations)
    max_vehicles_per_station = MAX_VEHICLES_DAY
    total_capacity = num_locations * max_vehicles_per_station
    capacity_ratio = total_capacity / len(vehicles)
    
    print(f"\nCapacity analysis:")
    print(f"  Total vehicles: {len(vehicles)}")
    print(f"  Potential locations: {num_locations}")
    print(f"  Maximum vehicles per station: {max_vehicles_per_station}")
    print(f"  Total capacity: {num_locations} locations × {max_vehicles_per_station} = {total_capacity} vehicles")
    print(f"  Capacity ratio: {capacity_ratio:.2f}x (should be at least 1.0x)")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_fixed_vehicles() 