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

# Enable interactive mode for matplotlib - this makes plt.show() non-blocking
plt.ion()

# Default parameters (these can be overridden when calling the functions)
DEFAULT_BATTERY_RANGE = 30        # Battery range in km
DEFAULT_MAX_VEHICLES_DAY = 5      # Maximum vehicles per station during day (7am-10pm)
DEFAULT_MAX_VEHICLES_NIGHT = 1    # Maximum vehicles per station during night (10pm-7am)
DEFAULT_CHARGING_TIME_HOURS = 3   # Hours required to fully charge a vehicle

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
    # Create a figure with two subplots - using a much larger figure size
    plt.figure(figsize=(28, 20))
    
    # Convert vehicles to numpy array if it's not already
    vehicles = np.array(vehicles)
    
    # Set up the grid for subplots (2 columns layout)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    
    # ============== LEFT PLOT: MAIN MAP ==============
    ax1 = plt.subplot(gs[0])
    
    # Plot districts
    district_colors = {
        'residential': '#AAFFAA',
        'commercial': '#AAAAFF',
        'industrial': '#FFAAAA',
    }
    
    # Determine if we're using real data by checking district attributes
    is_real_data = any('population' in district for district in districts.values())
    
    # Different visualization for real data vs synthetic scenario
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
            ax1.add_patch(circle)
            
            # For major municipalities (high population), add labels
            if district.get('population', 0) > 20000:
                ax1.text(center_x, center_y, name, 
                         fontsize=8, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    else:
        # For synthetic data, use rectangles with predefined colors
        for name, district in districts.items():
            center_x, center_y = district['center']
            width, height = district['size'][0] * 2, district['size'][1] * 2  # Double size as we use radius
            
            # Determine district type
            district_type = None
            for key in district_colors.keys():
                if key in name.lower():
                    district_type = key
                    break
            
            color = district_colors.get(district_type, '#CCCCCC')
            
            # Draw rectangle for district
            rect = Rectangle((center_x - width/2, center_y - height/2), width, height, 
                             alpha=0.5, edgecolor='black', facecolor=color)
            ax1.add_patch(rect)
            ax1.text(center_x, center_y, name, fontsize=12, ha='center')
    
    # Plot vehicles
    ax1.scatter(vehicles[:, 0], vehicles[:, 1], color='blue', alpha=0.3, s=8, label='Vehicles')
    
    # Plot all candidate locations if provided
    if all_locations is not None:
        ax1.scatter(all_locations[:, 0], all_locations[:, 1], marker='s', color='gray', alpha=0.5, s=50, label='Candidate Locations')
    
    # Plot selected charging stations
    station_colors = costs_to_colors(costs, selected_indices) if costs is not None else 'red'
    ax1.scatter(selected_locations[:, 0], selected_locations[:, 1], marker='s', color=station_colors, s=100, label='Selected Stations')
    
    # Add some visual elements to show station coverage
    for i, loc in enumerate(selected_locations):
        # Draw a circle with radius = battery_range
        if is_real_data:
            # For real data, use smaller circles to avoid overwhelming the plot
            circle_range = battery_range / 4  # Show a smaller range for visibility
            circle = Circle((loc[0], loc[1]), circle_range, alpha=0.1, edgecolor='red', facecolor='red')
        else:
            circle = Circle((loc[0], loc[1]), battery_range, alpha=0.1, edgecolor='red', facecolor='red')
        ax1.add_patch(circle)
    
    # Set plot limits with some padding
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
    
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    
    # Add labels and legend
    ax1.set_xlabel('X Coordinate (km)', fontsize=16)
    ax1.set_ylabel('Y Coordinate (km)', fontsize=16)
    ax1.set_title('Urban EV Charging Infrastructure', fontsize=20)
    ax1.legend(loc='upper right', fontsize=14)
    
    # Add grid
    ax1.grid(alpha=0.3)
    
    # Ensure equal aspect ratio for proper distance representation
    ax1.set_aspect('equal')
    
    # Pad the limits to make sure the aspect ratio is properly enforced
    x_range = max_x - min_x
    y_range = max_y - min_y
    max_range = max(x_range, y_range)
    
    # Calculate center points
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    
    # Set limits to be centered and equal in both directions
    ax1.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax1.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    # Increase tick label size
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # ============== RIGHT PLOT: STATISTICS & METRICS ==============
    ax2 = plt.subplot(gs[1])
    
    # Prepare data for the stats display
    num_vehicles = len(vehicles)
    num_stations = len(selected_locations)
    
    # Calculate distances from vehicles to nearest charging station
    distances = []
    for vehicle in vehicles:
        distances.append(min(np.sqrt(np.sum((selected_locations - vehicle)**2, axis=1))))
    
    # Convert distances to numpy array
    distances = np.array(distances)
    
    # Calculate statistics
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    max_distance = np.max(distances)
    
    # Count vehicles within battery range
    vehicles_in_range = np.sum(distances <= battery_range)
    coverage_percentage = (vehicles_in_range / num_vehicles) * 100
    
    # Calculate average vehicles per station
    avg_vehicles_per_station = num_vehicles / num_stations if num_stations > 0 else 0
    
    # Create a distance histogram
    ax2.hist(distances, bins=20, color='skyblue', edgecolor='black')
    ax2.axvline(x=battery_range, color='red', linestyle='--', label=f'Battery Range ({battery_range} km)')
    ax2.set_xlabel('Distance to Nearest Charging Station (km)', fontsize=16)
    ax2.set_ylabel('Number of Vehicles', fontsize=16)
    ax2.set_title('Distance Distribution', fontsize=20)
    ax2.legend(fontsize=14)
    
    # Increase tick label size
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Add stats as text below the histogram
    stats_text = f"""
    Optimization Results:
    ---------------------
    Number of Vehicles: {num_vehicles}
    Number of Charging Stations: {num_stations}
    Vehicles per Station: {avg_vehicles_per_station:.2f}
    
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
    ax2.text(0.5, -0.3, stats_text, transform=ax2.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center', bbox=props)
    
    # Adjust layout
    plt.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to leave space for suptitle
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def costs_to_colors(costs, selected_indices=None):
    """Convert installation costs to a color gradient"""
    if costs is None:
        return 'red'
    
    # If only selected indices are provided, filter costs
    if selected_indices is not None:
        costs = np.array(costs)[selected_indices]
    
    # Normalize costs to range [0, 1]
    min_cost = np.min(costs)
    max_cost = np.max(costs)
    if max_cost > min_cost:
        normalized_costs = (costs - min_cost) / (max_cost - min_cost)
    else:
        normalized_costs = np.zeros_like(costs)
    
    # Create colors based on normalized costs (green to red)
    colors = []
    for cost_norm in normalized_costs:
        r = min(1.0, 0.3 + 0.7 * cost_norm)
        g = min(1.0, 0.8 - 0.6 * cost_norm)
        b = 0.3
        colors.append((r, g, b))
    
    return colors 