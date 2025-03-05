from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

def create_urban_scenario(growth_factor=2.0):
    # Define city districts (residential, commercial, industrial)
    districts = {
        'residential_1': {'center': (30, 70), 'size': (40, 40), 'density': 0.4 * growth_factor},
        'residential_2': {'center': (150, 160), 'size': (50, 30), 'density': 0.3 * growth_factor},
        'commercial': {'center': (100, 100), 'size': (60, 60), 'density': 0.2 * growth_factor},
        'industrial': {'center': (160, 40), 'size': (50, 40), 'density': 0.1 * growth_factor},
    }
    
    # Generate potential charging station locations (near main intersections and high-traffic areas)
    potential_locations = [
        (40, 40),   # Residential 1 center
        (20, 80),   # Residential 1 edge
        (140, 150), # Residential 2 center
        (160, 170), # Residential 2 edge
        (100, 100), # Commercial district center
        (80, 120),  # Commercial district edge
        (150, 30),  # Industrial area
        (90, 60),   # Between residential and commercial
        (120, 140), # High traffic intersection
        (60, 110)   # Shopping area
    ]
    
    # Generate vehicle locations based on district density
    vehicles = []
    np.random.seed(42)
    
    for district, props in districts.items():
        center_x, center_y = props['center']
        width, height = props['size']
        num_vehicles = int(40 * props['density'])  # Scale factor of 30 for total vehicles
        
        # Generate vehicles with normal distribution around district center
        for _ in range(num_vehicles):
            x = np.random.normal(center_x, width/4)
            y = np.random.normal(center_y, height/4)
            vehicles.append((x, y))
    
    return np.array(potential_locations), np.array(vehicles), districts

def calculate_installation_costs(locations):
    # Base installation cost
    base_cost = 10000
    
    # Additional cost factors based on location
    costs = []
    for x, y in locations:
        # Higher cost in commercial areas (100,100) due to real estate prices
        commercial_factor = 1.5 - np.exp(-((x-100)**2 + (y-100)**2)/(2*50**2))
        # Higher cost in dense areas
        density_factor = 1.2 if (50 <= x <= 150 and 50 <= y <= 150) else 1.0
        
        cost = base_cost * commercial_factor * density_factor
        costs.append(int(cost))
    
    return np.array(costs)

def plot_urban_solution(location_coords, vehicle_coords, selected_locations, battery_range, districts):
    plt.figure(figsize=(15, 10))
    
    # Plot districts
    for district, props in districts.items():
        center_x, center_y = props['center']
        width, height = props['size']
        rect = Rectangle((center_x - width/2, center_y - height/2), 
                        width, height, 
                        alpha=0.2,
                        color='gray' if 'residential' in district else 
                              'orange' if district == 'commercial' else 'brown')
        plt.gca().add_patch(rect)
        plt.text(center_x, center_y, district.replace('_', ' ').title(),
                horizontalalignment='center', verticalalignment='center')
    
    # Plot all potential locations
    plt.scatter(location_coords[:, 0], location_coords[:, 1], 
               c='gray', s=100, alpha=0.7, label='Potential Locations')
    
    # Plot selected locations
    selected_coords = location_coords[selected_locations]
    plt.scatter(selected_coords[:, 0], selected_coords[:, 1], 
               c='red', s=200, marker='*', label='Selected Stations')
    
    # Plot vehicles
    plt.scatter(vehicle_coords[:, 0], vehicle_coords[:, 1], 
               c='blue', s=50, alpha=0.6, label='EVs')
    
    # Plot coverage circles for selected locations
    for loc in selected_coords:
        circle = plt.Circle((loc[0], loc[1]), battery_range, 
                          fill=False, linestyle='--', alpha=0.3, color='green')
        plt.gca().add_patch(circle)
    
    # Customize plot
    plt.xlabel('Distance (km)')
    plt.ylabel('Distance (km)')
    plt.title('Urban EV Charging Station Placement Solution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and adjust limits
    plt.axis('equal')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    
    # Add scale information
    plt.text(5, 195, 'Scale: 1 unit = 1 km', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('urban_ev_charging_solution.png', bbox_inches='tight', dpi=300)
    plt.show()

def optimize_charging_stations(location_coords, vehicle_coords, battery_range, installation_costs, weights=None, max_vehicles_per_station=5):
    """Run the optimization model for a single scenario with capacity constraints"""
    num_locations = len(location_coords)
    num_vehicles = len(vehicle_coords)
    
    if weights is None:
        weights = np.ones(num_vehicles)
    
    print(f"Optimizing for {num_vehicles} vehicles with capacity constraint of {max_vehicles_per_station} vehicles per station...")
    
    # Calculate distance matrix
    distance_matrix = np.zeros((num_vehicles, num_locations))
    for i in range(num_vehicles):
        for j in range(num_locations):
            distance_matrix[i, j] = np.sqrt(
                (vehicle_coords[i, 0] - location_coords[j, 0])**2 + 
                (vehicle_coords[i, 1] - location_coords[j, 1])**2
            )
    
    # Create optimization model
    mdl = Model(f"EV Charger Placement for {num_vehicles} vehicles")
    
    # Decision Variables
    x = mdl.binary_var_list(num_locations, name="x")
    y = mdl.binary_var_matrix(num_vehicles, num_locations, name="y")
    
    # Objective: Minimize installation cost
    mdl.minimize(mdl.sum(installation_costs[j] * x[j] for j in range(num_locations)))
    
    # Constraints
    # 1. Each vehicle must be assigned to at least one charging station
    for i in range(num_vehicles):
        mdl.add_constraint(mdl.sum(y[i, j] for j in range(num_locations)) >= 1)
    
    # 2. Vehicles can only use charging stations within their battery range
    for i in range(num_vehicles):
        for j in range(num_locations):
            mdl.add_constraint(y[i, j] <= (distance_matrix[i, j] <= battery_range))
    
    # 3. Vehicles can only use locations where chargers are installed
    for i in range(num_vehicles):
        for j in range(num_locations):
            mdl.add_constraint(y[i, j] <= x[j])
    
    # 4. NEW CONSTRAINT: Each charging station can serve at most 5 vehicles
    for j in range(num_locations):
        mdl.add_constraint(mdl.sum(y[i, j] for i in range(num_vehicles)) <= max_vehicles_per_station * x[j])
    
    # Solve the model
    solution = mdl.solve(log_output=True)
    
    if solution:
        selected_locations = [j for j in range(num_locations) if x[j].solution_value > 0.5]
        
        # Calculate vehicle assignments for reporting
        station_loads = {}
        for j in selected_locations:
            vehicle_count = sum(1 for i in range(num_vehicles) if y[i, j].solution_value > 0.5)
            station_loads[j] = vehicle_count
            
        return selected_locations, solution.objective_value, num_vehicles, station_loads
    else:
        print("No feasible solution found.")
        return [], 0, num_vehicles, {}

def run_multi_period_optimization():
    """Run optimization for current and future scenarios"""
    print("MULTI-PERIOD EV CHARGING STATION OPTIMIZATION")
    print("=============================================")
    
    # Parameters
    battery_range = 30  # 30km range
    max_vehicles_per_station = 5  # NEW: Capacity constraint
    
    # Generate current scenario
    print("\nGenerating current scenario...")
    location_coords, current_vehicles, districts = create_urban_scenario()
    installation_costs = calculate_installation_costs(location_coords)
    
    # Run optimization for current scenario
    print("\nOptimizing for current scenario...")
    current_selected, current_cost, current_num_vehicles, current_loads = optimize_charging_stations(
        location_coords, current_vehicles, battery_range, installation_costs, 
        max_vehicles_per_station=max_vehicles_per_station
    )
    
    # Generate future scenario (double the number of vehicles)
    print("\nGenerating future scenario with doubled number of vehicles...")
    _, future_vehicles, _ = create_urban_scenario(growth_factor=2.0)
    
    # See if we need clustering for future scenario
    if len(future_vehicles) > 900:  # Approaching CPLEX limit
        print("Using clustering for future scenario due to problem size...")
        # Implement clustering logic as needed
    
    # Run optimization for future scenario
    print("\nOptimizing for future scenario...")
    future_selected, future_cost, future_num_vehicles, future_loads = optimize_charging_stations(
        location_coords, future_vehicles, battery_range, installation_costs,
        max_vehicles_per_station=max_vehicles_per_station
    )
    
    # Display results
    print("\n\nRESULTS SUMMARY")
    print("===============")
    print(f"\nCurrent Scenario ({current_num_vehicles} vehicles):")
    print(f"Total Installation Cost: ${current_cost:,.2f}")
    print(f"Number of charging stations: {len(current_selected)}")
    for j in current_selected:
        location = location_coords[j]
        cost = installation_costs[j]
        vehicles_served = current_loads[j]
        utilization = (vehicles_served / max_vehicles_per_station) * 100
        print(f"- Station at ({location[0]:.1f}, {location[1]:.1f}) - Cost: ${cost:,.2f} - Vehicles: {vehicles_served}/{max_vehicles_per_station} ({utilization:.0f}% utilization)")
    
    print(f"\nFuture Scenario ({future_num_vehicles} vehicles):")
    print(f"Total Installation Cost: ${future_cost:,.2f}")
    print(f"Number of charging stations: {len(future_selected)}")
    for j in future_selected:
        location = location_coords[j]
        cost = installation_costs[j]
        vehicles_served = future_loads[j]
        utilization = (vehicles_served / max_vehicles_per_station) * 100
        print(f"- Station at ({location[0]:.1f}, {location[1]:.1f}) - Cost: ${cost:,.2f} - Vehicles: {vehicles_served}/{max_vehicles_per_station} ({utilization:.0f}% utilization)")
    
    # New stations needed for future
    new_stations = [j for j in future_selected if j not in current_selected]
    if new_stations:
        print(f"\nAdditional stations needed for future: {len(new_stations)}")
        additional_cost = sum(installation_costs[j] for j in new_stations)
        print(f"Additional cost for future expansion: ${additional_cost:,.2f}")
    else:
        print("\nNo additional stations needed for future scenario!")
    
    # Create visualization
    plot_multi_period_solution_with_capacity(
        location_coords, current_vehicles, future_vehicles, 
        current_selected, future_selected, 
        current_loads, future_loads,
        battery_range, districts, max_vehicles_per_station
    )
    
    return current_selected, future_selected, current_cost, future_cost

def plot_multi_period_solution_with_capacity(location_coords, current_vehicles, future_vehicles, 
                                           selected_locations, future_locations, 
                                           current_loads, future_loads,
                                           battery_range, districts, max_capacity):
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Current scenario subplot
    ax1 = plt.subplot(gs[0])
    ax1.set_title("Current Scenario", fontsize=16)
    
    # Plot districts
    for district, props in districts.items():
        center_x, center_y = props['center']
        width, height = props['size']
        rect = Rectangle((center_x - width/2, center_y - height/2), 
                        width, height, 
                        alpha=0.2,
                        color='gray' if 'residential' in district else 
                              'orange' if district == 'commercial' else 'brown')
        ax1.add_patch(rect)
        ax1.text(center_x, center_y, district.replace('_', ' ').title(),
                horizontalalignment='center', verticalalignment='center')
    
    # Plot all potential locations
    ax1.scatter(location_coords[:, 0], location_coords[:, 1], 
               c='gray', s=100, alpha=0.7, label='Potential Locations')
    
    # Plot vehicles
    ax1.scatter(current_vehicles[:, 0], current_vehicles[:, 1], 
               c='blue', s=50, alpha=0.6, label='EVs')
    
    # Plot selected locations with size based on utilization
    for j in selected_locations:
        loc = location_coords[j]
        utilization = current_loads[j] / max_capacity
        size = 100 + (utilization * 150)  # Size increases with utilization
        ax1.scatter(loc[0], loc[1], c='red', s=size, marker='*')
        
        # Add capacity annotation
        ax1.annotate(f"{current_loads[j]}/{max_capacity}", 
                    (loc[0], loc[1]), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8)
        
        # Coverage circle
        circle = plt.Circle((loc[0], loc[1]), battery_range, 
                          fill=False, linestyle='--', alpha=0.3, color='green')
        ax1.add_patch(circle)
    
    # Customize subplot
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Distance (km)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    
    # Similar code for the future scenario subplot...
    # (Code for second subplot would be similar but using future data)
    
    # Add a custom legend for station utilization
    handles = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='0-20% Utilization'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='40-60% Utilization'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=20, label='80-100% Utilization')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.savefig('capacity_constrained_ev_charging_solution.png', bbox_inches='tight', dpi=300)
    plt.show()

# Generate urban scenario
location_coords, vehicle_coords, districts = create_urban_scenario()
num_locations = len(location_coords)
num_vehicles = len(vehicle_coords)
battery_range = 30  # 30km range

# Calculate installation costs
installation_costs = calculate_installation_costs(location_coords)

# Calculate distance matrix
distance_matrix = np.zeros((num_vehicles, num_locations))
for i in range(num_vehicles):
    for j in range(num_locations):
        distance_matrix[i, j] = np.sqrt(
            (vehicle_coords[i, 0] - location_coords[j, 0])**2 + 
            (vehicle_coords[i, 1] - location_coords[j, 1])**2
        )

# Create optimization model
mdl = Model("Urban EV Charger Placement")

# Decision Variables
x = mdl.binary_var_list(num_locations, name="x")
y = mdl.binary_var_matrix(num_vehicles, num_locations, name="y")

# Objective: Minimize installation cost
mdl.minimize(mdl.sum(installation_costs[j] * x[j] for j in range(num_locations)))

# Constraints
for i in range(num_vehicles):
    mdl.add_constraint(mdl.sum(y[i, j] for j in range(num_locations)) >= 1)
    
for i in range(num_vehicles):
    for j in range(num_locations):
        mdl.add_constraint(y[i, j] <= (distance_matrix[i, j] <= battery_range))

for i in range(num_vehicles):
    for j in range(num_locations):
        mdl.add_constraint(y[i, j] <= x[j])

# Solve the model
solution = mdl.solve(log_output=True)

# Display results and create visualization
if solution:
    print("\nUrban EV Charging Station Placement Solution:")
    selected_locations = []
    for j in range(num_locations):
        if x[j].solution_value > 0.5:
            location = location_coords[j]
            cost = installation_costs[j]
            print(f"Install charger at location ({location[0]:.1f}, {location[1]:.1f}) - Cost: ${cost:,.2f}")
            selected_locations.append(j)
    
    print(f"\nTotal Installation Cost: ${solution.objective_value:,.2f}")
    
    # Create visualization
    plot_urban_solution(location_coords, vehicle_coords, selected_locations, battery_range, districts)
else:
    print("No feasible solution found.")
