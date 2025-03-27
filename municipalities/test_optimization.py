from ev_charger import create_urban_scenario, optimize_ev_placement, BATTERY_RANGE
import numpy as np
import time

def run_small_test():
    """Run a small test of the optimization with a limited number of vehicles"""
    print("RUNNING SMALL OPTIMIZATION TEST")
    print("===============================")
    
    # Use a higher people-per-vehicle ratio to limit the number of vehicles
    people_per_vehicle = 1000  # 1 vehicle per 1000 people
    
    # Start timing
    start_time = time.time()
    
    # Generate scenario with fewer vehicles
    print("\nGenerating test scenario...")
    scenario = create_urban_scenario(people_per_vehicle=people_per_vehicle)
    
    # Extract vehicles and districts
    vehicles = scenario['vehicles']
    districts = scenario['districts']
    
    vehicle_count = len(vehicles)
    district_count = len(districts)
    print(f"\nTest scenario contains {vehicle_count} vehicles across {district_count} municipalities")
    print(f"Using 1 vehicle per {people_per_vehicle} people")
    
    # For an even smaller test, limit to just 50 vehicles if we have more
    if vehicle_count > 50:
        print(f"\nLimiting to first 50 vehicles for faster testing")
        vehicles = vehicles[:50]
        vehicle_count = 50
    
    # Note about unconstrained stations
    print("\nRunning with unconstrained number of stations")
    
    # Run the optimization
    print("Running optimization...")
    result = optimize_ev_placement(vehicles, districts, BATTERY_RANGE)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if result:
        print("\nOptimization successful!")
        print(f"Total stations placed: {len(result['station_indices'])}")
        print(f"Total cost: ${result['total_cost']:.2f}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        
        # Check vehicle assignments
        assigned = len(result['vehicle_assignments'])
        print(f"Vehicles assigned: {assigned}/{vehicle_count} ({assigned/vehicle_count*100:.1f}%)")
        
        # Print station loads
        print("\nStation loads:")
        for station_idx, load in result['station_loads'].items():
            print(f"  Station {station_idx}: {load} vehicles")
            
        return True
    else:
        print("\nOptimization failed.")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        return False

if __name__ == "__main__":
    run_small_test() 