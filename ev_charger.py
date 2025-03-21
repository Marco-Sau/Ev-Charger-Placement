from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math

# Import visualization functions from the dedicated module
from ev_charger_visualization import plot_urban_solution

# Enable interactive mode for matplotlib - this makes plt.show() non-blocking
plt.ion()

# ============== GLOBAL PARAMETERS ==============
# Key parameters that control the optimization model
BATTERY_RANGE = 50        # Battery range in km (how far a vehicle can travel to a charging station)
MAX_VEHICLES_PER_STATION = 10  # Maximum vehicles that can be served by each station
CHARGING_TIME_HOURS = 3   # Hours required to fully charge a vehicle
PEOPLE_PER_VEHICLE = 1000  # Number of people per vehicle (1 vehicle per 100 people)
TIME_LIMIT_SECONDS = 1800  # Maximum time (in seconds) to run the optimization (default: 1 hour)

# Installation cost parameters
BASE_INSTALLATION_COST = 10000  # Base cost for installing a charging station
SMALL_AREA_THRESHOLD = 20       # Area size threshold for small areas (km²)
MEDIUM_AREA_THRESHOLD = 50      # Area size threshold for medium areas (km²)
SMALL_AREA_MULTIPLIER = 1.5     # Cost multiplier for small areas
MEDIUM_AREA_MULTIPLIER = 1.0    # Cost multiplier for medium areas
LARGE_AREA_MULTIPLIER = 0.8     # Cost multiplier for large areas

def create_urban_scenario(battery_range=BATTERY_RANGE, people_per_vehicle=PEOPLE_PER_VEHICLE):
    """
    Create a scenario based on real Sardinian municipalities data.
    
    Args:
        battery_range: Maximum battery range in km
        people_per_vehicle: Number of people per vehicle (e.g., 100 means 1 vehicle per 100 people)
        
    Returns:
        Dictionary with districts information and vehicle locations
    """
    # Load Sardinian municipalities data
    print("Loading Sardinian municipalities data...")
    municipalities_df = pd.read_csv("sardinia_municipalities_complete.csv")
    print(f"Loaded data for {len(municipalities_df)} municipalities")
    
    # Filter out municipalities with missing data
    municipalities_df = municipalities_df.dropna(subset=['Latitude', 'Longitude', 'Surface_km2', 'Population'])
    print(f"Using {len(municipalities_df)} municipalities with complete data")
    
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

def optimize_ev_placement(vehicles, districts, battery_range=BATTERY_RANGE, time_limit=TIME_LIMIT_SECONDS):
    """
    Optimize the placement of EV charging stations to minimize the total number of stations
    while ensuring each vehicle has a charging station within battery range.
    
    Args:
        vehicles: List of vehicle coordinates
        districts: Dictionary of district information
        battery_range: Maximum battery range in km
        time_limit: Maximum time in seconds to run the optimization
        
    Returns:
        Dictionary with optimization results
    """
    # Convert vehicles to numpy array if needed
    vehicles_array = np.array(vehicles)
    num_vehicles = len(vehicles_array)
    
    print(f"Running optimization for {num_vehicles} vehicles with focus on minimizing the number of stations")
    print(f"Time limit set to {time_limit} seconds ({time_limit/3600:.1f} hours)")
    print(f"Battery range: {battery_range} km, Max vehicles per station: {MAX_VEHICLES_PER_STATION}")
    
    # Create a mathematical model
    mdl = Model("EV Charger Placement - Station Minimization")
    
    # Set the time limit
    mdl.parameters.timelimit = time_limit
    
    # DECISION VARIABLES
    # Binary variables indicating if a station is placed at vehicle location i
    x = mdl.binary_var_list(num_vehicles, name="station_placement")
    
    # Binary variables indicating if vehicle i is assigned to station j
    y = mdl.binary_var_matrix(num_vehicles, num_vehicles, name="vehicle_assignment")
    
    # Calculate distances between all pairs of vehicle locations
    distances = np.zeros((num_vehicles, num_vehicles))
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            # Euclidean distance between vehicle i and potential station j
            dx = vehicles_array[i, 0] - vehicles_array[j, 0]
            dy = vehicles_array[i, 1] - vehicles_array[j, 1]
            distances[i, j] = math.sqrt(dx*dx + dy*dy)
    
    # OBJECTIVE FUNCTION: Minimize the total number of charging stations
    mdl.minimize(mdl.sum(x[j] for j in range(num_vehicles)))
    
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
    
    # 3. Battery range constraint: vehicles can only be assigned to stations within battery range
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            if distances[i, j] > battery_range:
                mdl.add_constraint(
                    y[i, j] == 0,
                    ctname=f"vehicle_{i}_range_to_station_{j}_constraint"
                )
    
    # 4. Capacity constraint: each station can serve at most MAX_VEHICLES_PER_STATION vehicles
    for j in range(num_vehicles):
        mdl.add_constraint(
            mdl.sum(y[i, j] for i in range(num_vehicles)) <= MAX_VEHICLES_PER_STATION * x[j],
            ctname=f"station_{j}_capacity_constraint"
        )
    
    # Set solver parameters
    mdl.parameters.mip.tolerances.mipgap = 0.1  # Accept solutions within 10% of optimal
    mdl.parameters.emphasis.mip = 2  # Emphasize finding optimal solutions
    mdl.parameters.mip.strategy.file = 2  # Use disk for branch-and-bound tree
    # Enable symmetry detection (levels 0-5, higher = more aggressive)
    mdl.parameters.preprocessing.symmetry = 3  # Medium-aggressive symmetry detection
    
    # Solve the model
    print("Solving optimization model...")
    solution = mdl.solve(log_output=True)
    
    if solution:
        # Extract results
        selected_stations = []
        station_loads = {}
        vehicle_assignments = {}
        total_distance = 0
        
        for j in range(num_vehicles):
            if x[j].solution_value > 0.5:
                selected_stations.append(j)
                station_coords = (vehicles_array[j, 0], vehicles_array[j, 1])
                
                # Count vehicles assigned to this station and calculate their distances
                vehicles_assigned = []
                station_total_distance = 0
                for i in range(num_vehicles):
                    if y[i, j].solution_value > 0.5:
                        vehicles_assigned.append(i)
                        vehicle_assignments[i] = j
                        station_total_distance += distances[i, j]
                        total_distance += distances[i, j]
                
                avg_distance = station_total_distance/len(vehicles_assigned) if vehicles_assigned else 0
                station_loads[j] = len(vehicles_assigned)
                print(f"Station at {station_coords} - Vehicles: {len(vehicles_assigned)} - Avg Distance: {avg_distance:.2f} km")
        
        print(f"\nTotal stations: {len(selected_stations)}")
        print(f"Total distance (secondary metric): {total_distance:.2f} km")
        print(f"Average distance per vehicle: {total_distance/num_vehicles:.2f} km")
        
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
            'total_stations': len(selected_stations),
            'total_distance': total_distance,
            'station_loads': station_loads,
            'vehicle_assignments': vehicle_assignments
        }
    else:
        print("No feasible solution found. Trying to diagnose the issue...")
        
        try:
            infeasible_constraints = mdl.find_conflicts()
            if infeasible_constraints:
                print("Infeasible constraints found:")
                for ct in infeasible_constraints:
                    print(f"  - {ct.name}")
        except:
            print("Could not identify specific infeasible constraints")
        
        print("\nTry adjusting the battery range to make the problem feasible")
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
            title=f"EV Optimization - Station Minimization (Battery Range: {BATTERY_RANGE} km)"
        )
    
    # Keep the program running until the user decides to exit
    print("\nOptimization complete. The plot window will remain open.")
    print("Press Enter to exit the program...")
    input()
