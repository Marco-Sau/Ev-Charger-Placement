from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import time

# Import visualization functions from the dedicated module
from ev_charger_visualization import plot_urban_solution

# Enable interactive mode for matplotlib - this makes plt.show() non-blocking
plt.ion()

# ============== GLOBAL PARAMETERS ==============
# Key parameters that control the optimization model
BATTERY_RANGE = 50        # Battery range in km (how far a vehicle can travel to a charging station)
MAX_VEHICLES_DAY = 50      # Maximum vehicles per station during day (7am-10pm)
MAX_VEHICLES_NIGHT = 10    # Maximum vehicles per station during night (10pm-7am)
CHARGING_TIME_HOURS = 3   # Hours required to fully charge a vehicle
PEOPLE_PER_VEHICLE = 200  # Number of people per vehicle (lower to generate more vehicles for small sample)
NUM_MUNICIPALITIES = 5    # Number of municipalities to include in the lightweight version

def create_urban_scenario(battery_range=BATTERY_RANGE, people_per_vehicle=PEOPLE_PER_VEHICLE):
    """
    Create a lightweight scenario based on the first few Sardinian municipalities.
    
    Args:
        battery_range: Maximum battery range in km
        people_per_vehicle: Number of people per vehicle
        
    Returns:
        Dictionary with districts information and vehicle locations
    """
    # Load Sardinian municipalities data
    print("Loading Sardinian municipalities data...")
    municipalities_df = pd.read_csv("sardinia_municipalities_complete.csv")
    print(f"Loaded data for {len(municipalities_df)} municipalities")
    
    # Filter out municipalities with missing data
    municipalities_df = municipalities_df.dropna(subset=['Latitude', 'Longitude', 'Surface_km2', 'Population'])
    print(f"Using complete data municipalities")
    
    # Select only the first NUM_MUNICIPALITIES municipalities
    municipalities_df = municipalities_df.head(NUM_MUNICIPALITIES)
    print(f"Limiting to first {NUM_MUNICIPALITIES} municipalities for lightweight version")
    
    # Normalize coordinates to a reasonable range for the simulation
    # First, calculate the center of the region
    center_lat = municipalities_df['Latitude'].mean()
    center_lon = municipalities_df['Longitude'].mean()
    
    # Scale factor to convert from degrees to km (approximate at Sardinia's latitude)
    lat_km_factor = 111.0  # 1 degree latitude ≈ 111 km
    lon_km_factor = 111.0 * math.cos(math.radians(center_lat))  # Adjust for longitude shrinkage at higher latitudes
    
    # Create a dictionary for districts (municipalities)
    districts = {}
    
    # Process each municipality into a district
    for idx, row in municipalities_df.iterrows():
        # Calculate x, y coordinates in km relative to center
        x = (row['Longitude'] - center_lon) * lon_km_factor
        y = (row['Latitude'] - center_lat) * lat_km_factor
        
        # Calculate district size based on surface area
        # Surface area is in km², so diameter = 2 * sqrt(area/π)
        diameter = 2 * math.sqrt(row['Surface_km2'] / math.pi)
        size = (diameter/2, diameter/2)  # Use radius for size
        
        # Calculate density based on population
        # Normalize population to get a reasonable density value
        max_pop = municipalities_df['Population'].max()
        density = row['Population'] / max_pop
        
        # Use a minimum density to ensure all municipalities have some vehicles
        density = max(0.05, density)
        
        # Add to districts dictionary
        districts[row['Municipality']] = {
            'center': (x, y),
            'size': size,
            'density': density,
            'population': row['Population'],
            'surface': row['Surface_km2']
        }
    
    # Generate vehicles based on population data
    vehicles = []
    print(f"Generating vehicles using population data (1 vehicle per {people_per_vehicle} people)")
    total_expected_vehicles = 0
    
    for district in districts.values():
        # Calculate number of vehicles based on population and people_per_vehicle ratio
        population = district['population']
        num_vehicles = max(1, int(population / people_per_vehicle))
        total_expected_vehicles += num_vehicles
        
        # Generate vehicle locations within the district
        center_x, center_y = district['center']
        size_x, size_y = district['size']
        
        for _ in range(num_vehicles):
            # Random position within the district (approximating as rectangle)
            x = center_x + random.uniform(-size_x, size_x)
            y = center_y + random.uniform(-size_y, size_y)
            vehicles.append((x, y))
    
    print(f"Expected vehicles based on population: {total_expected_vehicles}")
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    
    return {'districts': districts, 'vehicles': vehicles}

def optimize_ev_placement(vehicles, districts=None, battery_range=BATTERY_RANGE):
    """
    Directly optimize the placement of EV charging stations.
    
    Args:
        vehicles: List of vehicle coordinates
        districts: Dictionary of district information
        battery_range: Maximum battery range in km (retained for visualization purposes)
        
    Returns:
        Dictionary with optimization results
    """
    # Start timing
    start_time = time.time()
    
    # Convert vehicles to numpy array if needed
    vehicles_array = np.array(vehicles)
    num_vehicles = len(vehicles_array)
    
    print(f"Running direct optimization for {num_vehicles} vehicles")
    
    # Create a mathematical model
    mdl = Model("EV Charger Direct Placement")
    
    # DECISION VARIABLES
    # Binary variables indicating if a station is placed at vehicle location i
    x = mdl.binary_var_list(num_vehicles, name="station_placement")
    
    # Binary variables indicating if vehicle i is assigned to station j
    y = mdl.binary_var_matrix(num_vehicles, num_vehicles, name="vehicle_assignment")
    
    # OBJECTIVE FUNCTION
    # Calculate installation costs if a station is placed at vehicle location i
    # We use the vehicle's location to determine the district and thus the installation cost
    installation_costs = np.ones(num_vehicles) * 120  # Base cost
    
    if districts:
        for i, (vx, vy) in enumerate(vehicles_array):
            for district_name, district in districts.items():
                center_x, center_y = district['center']
                size_x, size_y = district['size']
                
                # Check if vehicle is within this district
                in_x_range = abs(vx - center_x) <= size_x
                in_y_range = abs(vy - center_y) <= size_y
                
                if in_x_range and in_y_range:
                    # Calculate area size
                    area_size = size_x * size_y * 4  # Approximate area
                    
                    # Apply cost multiplier based on area size
                    if area_size < 20:  # Small area
                        installation_costs[i] = 120 * 0.8
                    elif area_size < 50:  # Medium area
                        installation_costs[i] = 120 * 1.0
                    else:  # Large area
                        installation_costs[i] = 120 * 1.5
                    break
    
    # Minimize the total installation cost
    mdl.minimize(mdl.sum(installation_costs[i] * x[i] for i in range(num_vehicles)))
    
    # CONSTRAINTS
    # 1. Each vehicle must be assigned to exactly one charging station
    for i in range(num_vehicles):
        mdl.add_constraint(
            mdl.sum(y[i, j] for j in range(num_vehicles)) == 1,
            ctname=f"vehicle_{i}_assignment_constraint"
        )
    
    # 2. Vehicles can only be assigned to locations with charging stations
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            mdl.add_constraint(
                y[i, j] <= x[j],
                ctname=f"vehicle_{i}_station_{j}_placement_constraint"
            )
    
    # 3. Capacity constraint: each station can serve at most MAX_VEHICLES_DAY vehicles
    for j in range(num_vehicles):
        mdl.add_constraint(
            mdl.sum(y[i, j] for i in range(num_vehicles)) <= MAX_VEHICLES_DAY * x[j],
            ctname=f"station_{j}_capacity_constraint"
        )
    
    # Solve the model with a reasonable time limit (30 seconds)
    print("Solving optimization model...")
    mdl.parameters.timelimit = 30  # Set time limit to 30 seconds
    solution = mdl.solve(log_output=True)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if solution:
        print(f"Optimal solution found in {elapsed_time:.2f} seconds")
        
        # Extract results
        selected_stations = []
        station_loads = {}
        vehicle_assignments = {}
        
        for j in range(num_vehicles):
            if x[j].solution_value > 0.5:
                selected_stations.append(j)
                station_coords = (vehicles_array[j, 0], vehicles_array[j, 1])
                cost = installation_costs[j]
                
                # Count vehicles assigned to this station
                vehicles_assigned = []
                for i in range(num_vehicles):
                    if y[i, j].solution_value > 0.5:
                        vehicles_assigned.append(i)
                        vehicle_assignments[i] = j
                
                station_loads[j] = len(vehicles_assigned)
                print(f"Station at {station_coords} - Cost: ${cost:,.2f} - Vehicles: {len(vehicles_assigned)}")
        
        print(f"\nTotal cost: ${solution.objective_value:,.2f}")
        print(f"Total stations: {len(selected_stations)}")
        
        # Check for unassigned vehicles
        unassigned_vehicles = []
        for i in range(num_vehicles):
            if i not in vehicle_assignments:
                unassigned_vehicles.append(i)
        
        if unassigned_vehicles:
            print(f"WARNING: {len(unassigned_vehicles)} vehicles are not assigned to any charging station!")
        else:
            print("All vehicles have been successfully assigned to charging stations")
        
        return {
            'selected_stations': [vehicles_array[j] for j in selected_stations],
            'station_indices': selected_stations,
            'total_cost': solution.objective_value,
            'station_loads': station_loads,
            'vehicle_assignments': vehicle_assignments
        }
    else:
        print(f"No optimal solution found in {elapsed_time:.2f} seconds")
        
        try:
            infeasible_constraints = mdl.find_conflicts()
            if infeasible_constraints:
                print("Infeasible constraints found:")
                for ct in infeasible_constraints:
                    print(f"  - {ct.name}")
        except:
            print("Could not identify specific infeasible constraints")
        
        print("\nTry adjusting parameters to make the problem feasible")
        return None

# Main execution code
if __name__ == "__main__":
    # Generate urban scenario with vehicles and districts
    scenario = create_urban_scenario(people_per_vehicle=PEOPLE_PER_VEHICLE)
    
    # Extract vehicles and districts from the scenario
    vehicles = scenario['vehicles']
    districts = scenario['districts']
    
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    print(f"Using 1 vehicle per {PEOPLE_PER_VEHICLE} people")
    
    # Run the direct optimization
    result = optimize_ev_placement(vehicles, districts, BATTERY_RANGE)
    
    if result:
        # Plot solution
        plot_urban_solution(
            districts=districts,
            vehicles=np.array(vehicles),
            selected_locations=np.array(result['selected_stations']),
            all_locations=np.array(result['selected_stations']),  # For direct optimization, all locations = selected locations
            battery_range=BATTERY_RANGE,
            title=f"EV Optimization (Lightweight - {NUM_MUNICIPALITIES} municipalities)"
        )
    
    # Keep the program running until the user decides to exit
    print("\nOptimization complete. The plot window will remain open.")
    print("Press Enter to exit the program...")
    input() 