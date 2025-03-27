from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
from ev_charger_visualization import plot_urban_solution

# Enable interactive mode for matplotlib - this makes plt.show() non-blocking
plt.ion()

# ============== GLOBAL PARAMETERS ==============
# Key parameters that control the optimization model
"""
These parameters define the constraints and characteristics of our EV charging station optimization problem:
- BATTERY_RANGE: Maximum distance (km) an EV can travel to reach a charging station
- MAX_VEHICLES_PER_STATION: Maximum number of vehicles that can be served by a single station (capacity constraint)
- CHARGING_TIME_HOURS: Average time required to fully charge a vehicle (used for capacity calculations)
- PEOPLE_PER_VEHICLE: Population-to-vehicle ratio (used to estimate vehicle count from population data)
- TIME_LIMIT_SECONDS: Maximum runtime for the optimization solver (prevents excessive computation time)
"""
BATTERY_RANGE = 50              # Battery range in km (how far a vehicle can travel to a charging station)
MAX_VEHICLES_PER_STATION = 10   # Maximum vehicles that can be served by each station
# CHARGING_TIME_HOURS = 3         # Hours required to fully charge a vehicle
PEOPLE_PER_VEHICLE = 1000       # Number of people per vehicle (1 vehicle per 100 people)
TIME_LIMIT_SECONDS = 1800       # Maximum time (in seconds) to run the optimization (default: 1 hour)

# Installation cost parameters
"""
These parameters define how installation costs vary based on the size of the area:
- Smaller areas have higher per-unit costs (SMALL_AREA_MULTIPLIER > 1)
- Medium areas have standard costs (MEDIUM_AREA_MULTIPLIER = 1)
- Larger areas benefit from economies of scale (LARGE_AREA_MULTIPLIER < 1)
"""
BASE_INSTALLATION_COST = 10000  # Base cost for installing a charging station
SMALL_AREA_THRESHOLD = 20       # Area size threshold for small areas (km²)
MEDIUM_AREA_THRESHOLD = 50      # Area size threshold for medium areas (km²)
SMALL_AREA_MULTIPLIER = 1.5     # Cost multiplier for small areas
MEDIUM_AREA_MULTIPLIER = 1.0    # Cost multiplier for medium areas
LARGE_AREA_MULTIPLIER = 0.8     # Cost multiplier for large areas

def calculate_installation_cost(surface_area):
    """
    Calculate the installation cost for a charging station based on the surface area of the municipality.
    Larger areas typically have lower per-unit costs due to economies of scale.
    
    Args:
        surface_area: Surface area of the municipality in km²
        
    Returns:
        Installation cost in euros
    """
    # Apply different cost multipliers based on area size thresholds
    # This reflects real-world economics where installation in smaller areas is more expensive per unit
    # due to fixed costs being spread over fewer charging points
    if surface_area < SMALL_AREA_THRESHOLD:
        return BASE_INSTALLATION_COST * SMALL_AREA_MULTIPLIER
    elif surface_area < MEDIUM_AREA_THRESHOLD:
        return BASE_INSTALLATION_COST * MEDIUM_AREA_MULTIPLIER
    else:
        return BASE_INSTALLATION_COST * LARGE_AREA_MULTIPLIER

def create_urban_scenario(battery_range=BATTERY_RANGE, people_per_vehicle=PEOPLE_PER_VEHICLE):
    """
    Create a scenario based on real Sardinian municipalities data.
    
    Args:
        battery_range: Maximum battery range in km
        people_per_vehicle: Number of people per vehicle (e.g., 100 means 1 vehicle per 100 people)
        
    Returns:
        Dictionary with districts information and vehicle locations
    """
    # Step 1: Load real-world data from Sardinia municipalities
    # ---------------------------------------------------------
    print("Loading Sardinian municipalities data...")
    municipalities_df = pd.read_csv("sardinia_municipalities_complete.csv")
    print(f"Loaded data for {len(municipalities_df)} municipalities")
    
    # Step 2: Clean the data by removing entries with missing values
    # -------------------------------------------------------------
    municipalities_df = municipalities_df.dropna(subset=['Latitude', 'Longitude', 'Surface_km2', 'Population'])
    print(f"Using {len(municipalities_df)} municipalities with complete data")
    
    # Step 3: Normalize geographic coordinates to a relative coordinate system
    # ----------------------------------------------------------------------
    # Calculate the center of the region for reference
    center_lat = municipalities_df['Latitude'].mean()
    center_lon = municipalities_df['Longitude'].mean()
    
    # Convert degrees to kilometers for distance calculations
    # (This approximation is valid for relatively small regions like Sardinia)
    lat_km_factor = 111.0  # 1 degree latitude ≈ 111 km
    lon_km_factor = 111.0 * math.cos(math.radians(center_lat))  # Adjust for longitude shrinkage at higher latitudes
    
    # Step 4: Create a dictionary for districts (municipalities)
    # ---------------------------------------------------------
    districts = {}
    
    # Step 5: Process each municipality and calculate its attributes
    # -------------------------------------------------------------
    for idx, row in municipalities_df.iterrows():
        # Convert geographic coordinates to x,y coordinates in km relative to center
        x = (row['Longitude'] - center_lon) * lon_km_factor
        y = (row['Latitude'] - center_lat) * lat_km_factor
        
        # Calculate district size based on surface area
        # Assume circular shape for approximation: diameter = 2 * sqrt(area/π)
        diameter = 2 * math.sqrt(row['Surface_km2'] / math.pi)
        size = (diameter/2, diameter/2)  # Use radius for size
        
        # Calculate density based on population
        # Normalize population to get a reasonable density value for visualization
        max_pop = municipalities_df['Population'].max()
        density = row['Population'] / max_pop
        
        # Use a minimum density to ensure all municipalities have some vehicles
        density = max(0.05, density)
        
        # Add municipality to districts dictionary with all its attributes
        districts[row['Municipality']] = {
            'center': (x, y),
            'size': size,
            'density': density,
            'population': row['Population'],
            'surface': row['Surface_km2']
        }
    
    # Step 6: Generate vehicles based on population data
    # -------------------------------------------------
    vehicles = []
    print(f"Generating vehicles using population data (1 vehicle per {people_per_vehicle} people)")
    total_expected_vehicles = 0
    
    for district in districts.values():
        # Calculate expected number of vehicles based on population and people_per_vehicle ratio
        population = district['population']
        num_vehicles = max(1, int(population / people_per_vehicle))
        total_expected_vehicles += num_vehicles
        
        # Generate vehicle locations within the district
        # Distributed randomly within the district boundaries (approximated as a circle)
        center_x, center_y = district['center']
        size_x, size_y = district['size']
        
        for _ in range(num_vehicles):
            # Random position within the district (approximating as rectangle)
            x = center_x + random.uniform(-size_x, size_x)
            y = center_y + random.uniform(-size_y, size_y)
            vehicles.append((x, y))
    
    # Report statistics on the generated scenario
    print(f"Expected vehicles based on population: {total_expected_vehicles}")
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    
    return {'districts': districts, 'vehicles': vehicles}

def optimize_ev_placement(districts, vehicles, candidate_locations=None, battery_range=30):
    """
    Optimize EV charging station placement using a set covering approach.
    
    Arguments:
        districts: Dictionary of district information
        vehicles: List of vehicle coordinates
        candidate_locations: List of candidate charging station locations (or None to use district centers)
        battery_range: Maximum range of EV on a single charge
        
    Returns:
        selected_locations: Array of selected charging station coordinates
        total_distance: Sum of distances from each vehicle to its nearest station
        station_vehicles: List of lists, each containing vehicles assigned to a station
    """
    # Step 1: Prepare input data
    # --------------------------
    # Convert vehicles to numpy array for efficient calculations
    vehicles_array = np.array(vehicles)
    num_vehicles = len(vehicles_array)
    
    print(f"Running optimization for {num_vehicles} vehicles with focus on minimizing the number of stations")
    print(f"Time limit set to {TIME_LIMIT_SECONDS} seconds ({TIME_LIMIT_SECONDS/3600:.1f} hours)")
    print(f"Battery range: {battery_range} km, Max vehicles per station: {MAX_VEHICLES_PER_STATION}")
    
    # Step 2: Create a mathematical optimization model
    # -----------------------------------------------
    # This uses the DOcplex library to create a Mixed Integer Linear Programming (MILP) model
    mdl = Model("EV Charger Placement - Station Minimization")
    
    # Set the time limit for the solver
    mdl.parameters.timelimit = TIME_LIMIT_SECONDS
    
    # Step 3: Define decision variables
    # ---------------------------------
    # Binary variables indicating if a station is placed at vehicle location i
    x = mdl.binary_var_list(num_vehicles, name="station_placement")
    
    # Binary variables indicating if vehicle i is assigned to station j
    y = mdl.binary_var_matrix(num_vehicles, num_vehicles, name="vehicle_assignment")
    
    # Step 4: Calculate distances between all vehicle locations
    # --------------------------------------------------------
    # Precompute all pairwise distances for use in constraints and objectives
    distances = np.zeros((num_vehicles, num_vehicles))
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            # Euclidean distance between vehicle i and potential station j
            dx = vehicles_array[i, 0] - vehicles_array[j, 0]
            dy = vehicles_array[i, 1] - vehicles_array[j, 1]
            distances[i, j] = math.sqrt(dx*dx + dy*dy)
    
    # Step 5: Define the objective function
    # ------------------------------------
    # OBJECTIVE: Minimize the total number of charging stations
    # This is formulated as the sum of all station placement variables
    mdl.minimize(mdl.sum(x[j] for j in range(num_vehicles)))
    
    # Step 6: Define constraints
    # -------------------------
    # Constraint 1: Each vehicle must be assigned to exactly one charging station
    # This ensures all vehicles are served
    for i in range(num_vehicles):
        mdl.add_constraint(
            mdl.sum(y[i, j] for j in range(num_vehicles)) == 1,
            ctname=f"vehicle_{i}_assignment_constraint"
        )
    
    # Constraint 2: Vehicles can only be assigned to locations with charging stations
    # This links the assignment variables to the station placement variables
    for i in range(num_vehicles):
        for j in range(num_vehicles):
                mdl.add_constraint(
                y[i, j] <= x[j],
                ctname=f"vehicle_{i}_station_{j}_placement_constraint"
                )
    
    # Constraint 3: Battery range constraint - vehicles can only be assigned to stations within battery range
    # This enforces the maximum distance a vehicle can travel to a charging station
    for i in range(num_vehicles):
        for j in range(num_vehicles):
            if distances[i, j] > battery_range:
                mdl.add_constraint(
                    y[i, j] == 0,
                    ctname=f"vehicle_{i}_range_to_station_{j}_constraint"
                )
    
    # Constraint 4: Capacity constraint - each station can serve at most MAX_VEHICLES_PER_STATION vehicles
    # This prevents overloading individual stations
    for j in range(num_vehicles):
            mdl.add_constraint(
            mdl.sum(y[i, j] for i in range(num_vehicles)) <= MAX_VEHICLES_PER_STATION * x[j],
            ctname=f"station_{j}_capacity_constraint"
        )
    
    # Step 7: Set solver parameters to improve performance
    # --------------------------------------------------
    mdl.parameters.mip.tolerances.mipgap = 0.1  # Accept solutions within 10% of optimal (speeds up solver)
    mdl.parameters.emphasis.mip = 2             # Emphasize finding optimal solutions
    mdl.parameters.mip.strategy.file = 2        # Use disk for branch-and-bound tree (helps with large problems)
    mdl.parameters.preprocessing.symmetry = 3   # Medium-aggressive symmetry detection
    
    # Step 8: Solve the optimization model
    # ----------------------------------
    print("Solving optimization model...")
    solution = mdl.solve(log_output=True)
    
    # Step 9: Process the solution if feasible
    # --------------------------------------
    if solution:
        # Extract results from the solution
        selected_locations = []
        station_loads = {}
        vehicle_assignments = {}
        total_distance = 0
        total_cost = 0
        station_costs = []  # List to store costs for each station
        
        # Process each selected station location
        for j in range(num_vehicles):
            if x[j].solution_value > 0.5:
                selected_locations.append(vehicles_array[j])
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
                
                # Find the municipality this station belongs to
                station_cost = BASE_INSTALLATION_COST  # Default cost if municipality not found
                for district_name, district_info in districts.items():
                    district_center = district_info['center']
                    district_size = district_info['size']
                    if (abs(station_coords[0] - district_center[0]) <= district_size[0] and
                        abs(station_coords[1] - district_center[1]) <= district_size[1]):
                        station_cost = calculate_installation_cost(district_info['surface'])
                        break  # Break only if we've found the matching district
                
                total_cost += station_cost
                station_costs.append(station_cost)  # Add the cost to our list
                avg_distance = station_total_distance/len(vehicles_assigned) if vehicles_assigned else 0
                station_loads[j] = len(vehicles_assigned)
                print(f"Station at {station_coords} - Vehicles: {len(vehicles_assigned)} - Avg Distance: {avg_distance:.2f} km - Cost: €{station_cost:,.0f}")
        
        # Step 10: Display summary results
        # ------------------------------
        print(f"\nTotal stations: {len(selected_locations)}")
        print(f"Total distance (secondary metric): {total_distance:.2f} km")
        print(f"Average distance per vehicle: {total_distance/num_vehicles:.2f} km")
        print(f"Total installation cost: €{total_cost:,.0f}")
        print(f"Average cost per station: €{total_cost/len(selected_locations):,.0f}")
        
        # Check if all vehicles are assigned to charging stations
        unassigned_vehicles = []
        for i in range(num_vehicles):
            if i not in vehicle_assignments:
                unassigned_vehicles.append(i)
        
        if unassigned_vehicles:
            print(f"WARNING: {len(unassigned_vehicles)} vehicles are not assigned to any charging station!")
        else:
            print("All vehicles have been successfully assigned to charging stations")
        
        # Return the results
        return selected_locations, total_distance, station_loads, total_cost, station_costs
    else:
        # Step 11: Handle infeasible solutions
        # ----------------------------------
        print("No feasible solution found. Trying to diagnose the issue...")
        
        # Try to identify which constraints are causing infeasibility
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
    # Step 1: Generate the urban scenario with vehicles and districts
    # --------------------------------------------------------------
    # This creates a realistic scenario based on Sardinia municipalities data
    scenario = create_urban_scenario(people_per_vehicle=PEOPLE_PER_VEHICLE)
    
    # Extract the scenario components
    vehicles = scenario['vehicles']
    districts = scenario['districts']
    
    print(f"Generated {len(vehicles)} vehicles across {len(districts)} municipalities")
    print(f"Using 1 vehicle per {PEOPLE_PER_VEHICLE} people")
    
    # Step 2: Run the optimization model
    # ---------------------------------
    # This finds the optimal placement of charging stations to minimize the total number
    # while ensuring every vehicle is served within battery range
    selected_locations, total_distance, station_vehicles, total_cost, station_costs = optimize_ev_placement(
        districts, vehicles, candidate_locations=None, battery_range=BATTERY_RANGE
    )
    
    # Step 3: Visualize the results
    # ---------------------------
    # Plot the solution using the dedicated visualization module
    plot_urban_solution(
        districts, vehicles, selected_locations, 
        all_locations=None,
        battery_range=BATTERY_RANGE,
        costs=station_costs,
        title="Sardinia EV Charging Infrastructure Optimization"
    )
    
    # Notify user about saved results
    print(f"\nResults saved to the 'results' folder.")
