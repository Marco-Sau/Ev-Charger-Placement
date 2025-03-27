"""
Visualization functions for EV Charger Placement Optimization.

This module contains functions for visualizing the results of EV charging station
placement optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
import matplotlib.gridspec as gridspec
import math
import os

# Default parameters (these can be overridden when calling the functions)
"""
Default visualization parameters that control how the plots are generated:
- DEFAULT_BATTERY_RANGE: Standard battery range for visualization (km)
- DEFAULT_MAX_VEHICLES_DAY/NIGHT: Maximum vehicles per station during different time periods
- DEFAULT_CHARGING_TIME_HOURS: Time required to charge, affects station capacity calculations
"""
DEFAULT_BATTERY_RANGE = 30        # Battery range in km
DEFAULT_MAX_VEHICLES_DAY = 5      # Maximum vehicles per station during day (7am-10pm)
DEFAULT_MAX_VEHICLES_NIGHT = 1    # Maximum vehicles per station during night (10pm-7am)
DEFAULT_CHARGING_TIME_HOURS = 3   # Hours required to fully charge a vehicle

def ensure_results_dir():
    """
    Create the results directory if it doesn't exist
    
    Returns:
        Path to the results directory
    """
    # Create a dedicated directory for saving visualization results
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def plot_urban_solution(districts, vehicles, selected_locations, all_locations=None, 
                        battery_range=30, costs=None, selected_indices=None, 
                        title="Urban EV Charging Station Optimization", save_path=None):
    """
    Plot the urban scenario with vehicles, districts, and charging stations.
    
    Args:
        districts: Dictionary of district information
        vehicles: List of vehicle coordinates
        selected_locations: Array of selected charging station coordinates
        all_locations: Array of all candidate charging station coordinates (optional)
        battery_range: Maximum battery range in km
        costs: List of installation costs (optional)
        selected_indices: Indices of selected locations (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Step 1: Ensure the results directory exists for saving plots
    # -----------------------------------------------------------
    results_dir = ensure_results_dir()
    
    # Step 2: Generate and save three separate visualization plots
    # -----------------------------------------------------------
    # Main map showing the geographical distribution of districts, vehicles, and stations
    plot_main_map(districts, vehicles, selected_locations, all_locations, battery_range, 
                 costs, selected_indices, title, os.path.join(results_dir, "map_view.png"))
    
    # Histogram showing the distribution of distances from vehicles to nearest stations
    plot_distance_histogram(vehicles, selected_locations, battery_range, 
                           title, os.path.join(results_dir, "distance_histogram.png"))
    
    # Visualization of how many vehicles are served by each station (load distribution)
    plot_station_loads(selected_locations, vehicles, battery_range,
                      title, os.path.join(results_dir, "station_loads.png"))
    
    print(f"Plots saved to '{results_dir}' folder.")

def plot_main_map(districts, vehicles, selected_locations, all_locations=None, 
                 battery_range=30, costs=None, selected_indices=None, 
                 title="Urban EV Charging Station Optimization", save_path=None):
    """
    Plot the main map view with districts, vehicles, and charging stations
    
    Args:
        districts: Dictionary of district information
        vehicles: List of vehicle coordinates
        selected_locations: Array of selected charging station coordinates
        all_locations: Array of all candidate charging station coordinates (optional)
        battery_range: Maximum battery range in km
        costs: List of installation costs (optional)
        selected_indices: Indices of selected locations (optional)
        title: Plot title
        save_path: Path to save the figure
    """
    # Step 1: Create a new figure for the map visualization
    # ---------------------------------------------------
    plt.figure(figsize=(16, 12))
    
    # Convert vehicles and selected_locations to numpy arrays if they're not already
    vehicles = np.array(vehicles)
    selected_locations = np.array(selected_locations)
    
    # Also convert all_locations if it's provided
    if all_locations is not None:
        all_locations = np.array(all_locations)
    
    # Step 2: Define colors for different district types
    # -------------------------------------------------
    district_colors = {
        'residential': '#AAFFAA',  # Light green for residential areas
        'commercial': '#AAAAFF',   # Light blue for commercial areas
        'industrial': '#FFAAAA',   # Light red for industrial areas
    }
    
    # Step 3: Determine if we're using real data or synthetic scenario
    # --------------------------------------------------------------
    # Check for real data by looking for population attribute in districts
    is_real_data = any('population' in district for district in districts.values())
    
    # Create a list to store legend elements
    legend_elements = []
    
    # Step 4: Visualize districts based on data type (real vs synthetic)
    # ----------------------------------------------------------------
    if is_real_data:
        # For real municipalities data, use circles scaled by population
        for name, district in districts.items():
            center_x, center_y = district['center']
            radius = district['size'][0]  # Use radius from size
            
            # Scale circle color by population density
            normalized_density = min(1.0, district['density'] * 5)
            color = (0.8 - 0.3 * normalized_density, 
                     0.9 - 0.2 * normalized_density, 
                     1.0 - 0.3 * normalized_density)
            
            # Draw circle for each municipality
            circle = Circle((center_x, center_y), radius, 
                            alpha=0.5, edgecolor='black', 
                            linewidth=0.5, facecolor=color)
            plt.gca().add_patch(circle)
            
            # For major municipalities (high population), add labels
            if district.get('population', 0) > 20000:
                plt.text(center_x, center_y, name, 
                         fontsize=8, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Add municipality circle to legend
        legend_elements.append(Patch(facecolor='lightblue', edgecolor='black', alpha=0.5,
                               label='Municipalities'))
    else:
        # For synthetic data, use rectangles with predefined colors
        for name, district in districts.items():
            center_x, center_y = district['center']
            width, height = district['size'][0] * 2, district['size'][1] * 2  # Double size as we use radius
            
            # Determine district type from its name
            district_type = None
            for key in district_colors.keys():
                if key in name.lower():
                    district_type = key
                    break
            
            color = district_colors.get(district_type, '#CCCCCC')
            
            # Draw rectangle for district
            rect = Rectangle((center_x - width/2, center_y - height/2), width, height, 
                             alpha=0.5, edgecolor='black', facecolor=color)
            plt.gca().add_patch(rect)
            plt.text(center_x, center_y, name, fontsize=12, ha='center')
    
    # Step 5: Plot vehicles as small blue dots
    # ---------------------------------------
    plt.scatter(vehicles[:, 0], vehicles[:, 1], color='blue', alpha=0.3, s=8, label='Vehicles')
    legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='Vehicles'))
    
    # Step 6: Plot candidate locations (if provided)
    # --------------------------------------------
    if all_locations is not None:
        plt.scatter(all_locations[:, 0], all_locations[:, 1], marker='s', color='gray', alpha=0.5, s=50)
        legend_elements.append(Patch(facecolor='gray', alpha=0.5, label='Candidate Locations'))
    
    # Step 7: Plot selected charging station locations
    # ----------------------------------------------
    # Use color gradient if costs are provided, otherwise use red
    station_colors = costs_to_colors(costs, selected_indices) if costs is not None else 'red'
    plt.scatter(selected_locations[:, 0], selected_locations[:, 1], marker='s', color=station_colors, s=100)
    
    # Add cost gradient legend if we have multiple stations with costs
    if isinstance(station_colors, list) and len(station_colors) > 0:
        # Add low cost and high cost stations to legend
        legend_elements.append(Patch(facecolor='green', label='Lower Cost Stations'))
        legend_elements.append(Patch(facecolor='red', label='Higher Cost Stations'))
    else:
        # Just add a generic stations entry
        legend_elements.append(Patch(facecolor='red', label='Charging Stations'))
    
    # Step 8: Add visual indicators for station coverage areas
    # ------------------------------------------------------
    for i, loc in enumerate(selected_locations):
        # Draw a circle with radius = battery_range
        if is_real_data:
            # For real data, use smaller circles to avoid overwhelming the plot
            circle_range = battery_range / 4  # Show a smaller range for visibility
            circle = Circle((loc[0], loc[1]), circle_range, alpha=0.1, edgecolor='red', facecolor='red')
        else:
            circle = Circle((loc[0], loc[1]), battery_range, alpha=0.1, edgecolor='red', facecolor='red')
        plt.gca().add_patch(circle)
    
    # Add battery range to legend
    legend_elements.append(Patch(facecolor='red', alpha=0.1, 
                          label=f'Battery Range ({battery_range} km)'))
    
    # Step 9: Set plot limits with appropriate padding
    # ----------------------------------------------
    if is_real_data:
        # For real data, calculate bounds from all elements
        all_points = np.vstack([vehicles, selected_locations])
        if all_locations is not None:
            all_points = np.vstack([all_points, all_locations])
        
        min_x, min_y = np.min(all_points, axis=0) - battery_range/2
        max_x, max_y = np.max(all_points, axis=0) + battery_range/2
    else:
        # For synthetic data, use fixed bounds
        min_x, min_y = -20, -20
        max_x, max_y = 220, 220
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # Step 10: Add labels, legend, and formatting
    # -----------------------------------------
    plt.xlabel('X Coordinate (km)', fontsize=16)
    plt.ylabel('Y Coordinate (km)', fontsize=16)
    plt.title(title, fontsize=20)
    
    # Add comprehensive legend with all elements
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14, 
               framealpha=0.9, title="Map Elements")
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Ensure equal aspect ratio for proper distance representation
    plt.gca().set_aspect('equal')
    
    # Pad the limits to make sure the aspect ratio is properly enforced
    x_range = max_x - min_x
    y_range = max_y - min_y
    max_range = max(x_range, y_range)
    
    # Calculate center points
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    
    # Set limits to be centered and equal in both directions
    plt.gca().set_xlim(x_center - max_range/2, x_center + max_range/2)
    plt.gca().set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    # Increase tick label size
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    
    # Add summary statistics as text on the plot
    summary_text = f"Total Stations: {len(selected_locations)}\n"
    summary_text += f"Battery Range: {battery_range} km\n"
    summary_text += f"Total Municipalities: {len(districts)}"
    
    # Add text box with summary statistics
    plt.text(0.02, 0.02, summary_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Step 11: Save the figure to file
    # -------------------------------
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map view saved to: {save_path}")
    
    plt.close()

def plot_distance_histogram(vehicles, selected_locations, battery_range=30, 
                           title="Urban EV Charging Station Optimization", save_path=None):
    """
    Plot histogram of distances from vehicles to nearest charging station
    
    Args:
        vehicles: List of vehicle coordinates
        selected_locations: Array of selected charging station coordinates
        battery_range: Maximum battery range in km
        title: Plot title
        save_path: Path to save the figure
    """
    # Step 1: Create a new figure for the histogram
    # -------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Convert inputs to numpy arrays if they're not already
    vehicles = np.array(vehicles)
    selected_locations = np.array(selected_locations)
    
    # Step 2: Calculate distances from each vehicle to its nearest station
    # -----------------------------------------------------------------
    distances = []
    for vehicle in vehicles:
        distances.append(min(np.sqrt(np.sum((selected_locations - vehicle)**2, axis=1))))
    
    # Convert distances to numpy array
    distances = np.array(distances)
    
    # Step 3: Calculate key statistics about distances
    # ---------------------------------------------
    num_vehicles = len(vehicles)
    num_stations = len(selected_locations)
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    
    # Count vehicles within battery range
    vehicles_in_range = np.sum(distances <= battery_range)
    coverage_percentage = (vehicles_in_range / num_vehicles) * 100
    
    # Calculate average vehicles per station
    avg_vehicles_per_station = num_vehicles / num_stations if num_stations > 0 else 0
    
    # Step 4: Create a histogram of distances
    # -------------------------------------
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=battery_range, color='red', linestyle='--', label=f'Battery Range ({battery_range} km)')
    plt.xlabel('Distance to Nearest Charging Station (km)', fontsize=16)
    plt.ylabel('Number of Vehicles', fontsize=16)
    plt.title('Distance Distribution', fontsize=20)
    plt.legend(fontsize=14)
    
    # Increase tick label size
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    
    # Step 5: Add text box with statistics
    # ----------------------------------
    stats_text = f"""
    Distance Metrics:
    ----------------
    Mean Distance: {mean_distance:.2f} km
    Median Distance: {median_distance:.2f} km
    Maximum Distance: {max_distance:.2f} km
    
    Coverage:
    --------
    Vehicles within Range: {vehicles_in_range} of {num_vehicles}
    Coverage Percentage: {coverage_percentage:.2f}%
    """
    
    # Add a text box with the statistics
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Step 6: Save the figure to file
    # -----------------------------
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance histogram saved to: {save_path}")
    
    plt.close()

def plot_station_loads(selected_locations, vehicles, battery_range=30,
                      title="Urban EV Charging Station Optimization", save_path=None):
    """
    Plot station load distribution - how many vehicles are assigned to each station
    
    Args:
        selected_locations: Array of selected charging station coordinates
        vehicles: List of vehicle coordinates
        battery_range: Maximum battery range in km
        title: Plot title
        save_path: Path to save the figure
    """
    # Step 1: Create a new figure for the station load visualization
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Convert inputs to numpy arrays if they're not already
    vehicles = np.array(vehicles)
    selected_locations = np.array(selected_locations)
    
    # Step 2: Calculate which vehicles are assigned to which stations
    # -------------------------------------------------------------
    # Assign each vehicle to its nearest station within battery range
    station_loads = [0] * len(selected_locations)
    unassigned = 0
    
    for vehicle in vehicles:
        # Calculate distances to all stations
        distances = np.sqrt(np.sum((selected_locations - vehicle)**2, axis=1))
        
        # Find nearest station within battery range
        if np.min(distances) <= battery_range:
            nearest_station = np.argmin(distances)
            station_loads[nearest_station] += 1
        else:
            unassigned += 1
    
    # Step 3: Create histogram of station loads
    # ---------------------------------------
    plt.hist(station_loads, bins=range(0, max(station_loads)+2), align='left', 
             color='green', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Number of Vehicles Assigned', fontsize=16)
    plt.ylabel('Number of Stations', fontsize=16)
    plt.title('Station Load Distribution', fontsize=20)
    
    # Increase tick label size
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    
    # Step 4: Add text box with summary statistics
    # -----------------------------------------
    stats_text = f"""
    Station Load Statistics:
    ----------------------
    Total Stations: {len(selected_locations)}
    Total Vehicles: {len(vehicles)}
    Average Load: {np.mean(station_loads):.2f} vehicles/station
    Maximum Load: {max(station_loads)} vehicles
    Unassigned Vehicles: {unassigned}
    """
    
    # Add a text box with the statistics
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Step 5: Save the figure to file
    # -----------------------------
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Station load distribution saved to: {save_path}")
    
    plt.close()

def costs_to_colors(costs, selected_indices=None):
    """
    Convert installation costs to a color gradient for visualization
    
    Args:
        costs: List of installation costs or a single cost value
        selected_indices: Indices of selected locations (optional)
    
    Returns:
        List of RGB color tuples or 'red' if costs is None
    """
    # Return default color if no costs provided
    if costs is None:
        return 'red'
    
    # Handle case where costs is a single value
    if not isinstance(costs, (list, np.ndarray)) or (isinstance(costs, np.ndarray) and costs.size == 1):
        return 'red'  # Just return a default color for a single value
    
    # If selected indices are provided, use only those costs
    if selected_indices is not None:
        if len(selected_indices) != len(costs):
            costs = [costs[i] for i in selected_indices]
    
    # Step 1: Normalize costs between 0 and 1
    # -------------------------------------
    min_cost = min(costs)
    max_cost = max(costs)
    
    if min_cost == max_cost:  # All costs are the same
        return ['red'] * len(costs)
    
    normalized_costs = [(cost - min_cost) / (max_cost - min_cost) for cost in costs]
    
    # Step 2: Create a color gradient from green (low cost) to red (high cost)
    # ---------------------------------------------------------------------
    colors = []
    for norm_cost in normalized_costs:
        r = min(1.0, norm_cost * 2)         # Red increases with cost
        g = min(1.0, 2 - norm_cost * 2)     # Green decreases with cost
        b = 0.0                             # No blue
        colors.append((r, g, b))
    
    return colors