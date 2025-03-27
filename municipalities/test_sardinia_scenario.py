import matplotlib.pyplot as plt
import numpy as np
from ev_charger import create_urban_scenario, generate_candidate_locations, calculate_installation_costs
from ev_charger_visualization import plot_urban_solution

def test_sardinia_scenario():
    # Create urban scenario using Sardinian municipalities data
    print("Creating urban scenario with Sardinian municipalities data...")
    scenario = create_urban_scenario(growth_factor=1.0)
    
    # Get districts and vehicles from the scenario
    districts = scenario['districts']
    vehicles = scenario['vehicles']
    
    print(f"Created scenario with {len(districts)} municipalities and {len(vehicles)} vehicles")
    
    # Generate candidate locations for charging stations
    print("Generating candidate locations for charging stations...")
    locations = generate_candidate_locations(vehicles, districts)
    print(f"Generated {len(locations)} candidate locations")
    
    # Calculate installation costs
    print("Calculating installation costs...")
    costs = calculate_installation_costs(locations, districts)
    
    # Simulate station selection (just a simple selection of top 20 locations)
    # In a real scenario, this would be determined by the optimization model
    print("Selecting sample charging stations...")
    
    # For demonstration, select 20 locations with lowest costs
    indices = np.argsort(costs)[:20]
    selected_locations = locations[indices]
    
    # Visualize the scenario
    print("Visualizing scenario...")
    plot_urban_solution(
        districts=districts,
        vehicles=vehicles,
        selected_locations=selected_locations,
        all_locations=locations,
        battery_range=30,
        costs=costs,
        selected_indices=indices,
        title="Sardinian Municipalities EV Charging Network",
        save_path="sardinia_ev_scenario.png"
    )
    
    print("Visualization complete! Check sardinia_ev_scenario.png")

if __name__ == "__main__":
    test_sardinia_scenario() 