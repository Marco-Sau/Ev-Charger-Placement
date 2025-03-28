# EV Charging Station Placement Optimization

This project implements an optimization approach for the strategic placement of EV (Electric Vehicle) charging stations across Sardinian municipalities, leveraging operations research techniques to find the optimal configuration that minimizes costs while ensuring coverage for all vehicles.

## Project Overview

The goal of this project is to determine the optimal placement of charging stations across municipalities in Sardinia, based on real geographic and demographic data. The optimization aims to:

- Minimize the total number of charging stations needed
- Ensure all vehicles are within battery range of at least one station
- Respect capacity constraints of each charging station
- Consider installation costs that vary by municipality size

## Problem Formulation

The EV charging station placement problem is formulated as an Integer Linear Program (ILP):

- **Decision Variables**:
  - `x_j`: Binary variable indicating if a charging station is placed at location j
  - `y_ij`: Binary variable indicating if vehicle i is assigned to station j

- **Objective Function**:
  - Minimize the total number of stations: `Min ∑ x_j`

- **Constraints**:
  - Each vehicle must be assigned to exactly one station: `∑ y_ij = 1, for all i`
  - Vehicles can only be assigned to locations with stations: `y_ij ≤ x_j, for all i,j`
  - Battery range constraint: `y_ij = 0 if distance(i,j) > BATTERY_RANGE`
  - Capacity constraint: `∑ y_ij ≤ MAX_VEHICLES_PER_STATION * x_j, for all j`

## Data Description

The project uses real data from Sardinian municipalities, including:
- Geographic coordinates (latitude/longitude)
- Population data
- Surface area data
- Municipality boundaries

This data is processed to generate:
- Municipality centers and sizes
- Vehicle distribution based on population density
- Distance calculations for optimization

## Installation Requirements

To run this project, you'll need:

```
Python 3.7+
docplex (IBM CPLEX Python API)
numpy
matplotlib
pandas
```

You can install the required packages via pip:

```bash
pip install docplex numpy matplotlib pandas
```

Note: The `docplex` package requires access to IBM CPLEX, which is available for academic use through the IBM Academic Initiative.

## How to Run the Project

1. Ensure all dependencies are installed
2. Place the `sardinia_municipalities_complete.csv` file in the project directory
3. Run the main optimization script:

```bash
python ev_charger.py
```

4. To visualize results or modify parameters, you can import the module in your own script:

```python
from ev_charger import create_urban_scenario, optimize_ev_placement
from ev_charger_visualization import plot_urban_solution

# Create scenario
scenario = create_urban_scenario(people_per_vehicle=1000)
vehicles = scenario['vehicles']
districts = scenario['districts']

# Run optimization
selected_locations, total_distance, station_loads, total_cost, station_costs = optimize_ev_placement(
    districts, vehicles, battery_range=50
)

# Visualize results
plot_urban_solution(
    districts, vehicles, selected_locations, 
    battery_range=50,
    costs=station_costs
)
```

## Key Parameters

You can adjust several parameters to test different scenarios:

- `BATTERY_RANGE`: Maximum distance (km) an EV can travel to reach a charging station
- `MAX_VEHICLES_PER_STATION`: Maximum capacity of each charging station
- `PEOPLE_PER_VEHICLE`: Population-to-vehicle ratio (how many people per vehicle)
- `TIME_LIMIT_SECONDS`: Maximum runtime for the optimization solver

## Understanding the Output

The optimization produces:

1. **Console Output**:
   - Number of municipalities and vehicles
   - Station placement details with assigned vehicles
   - Total cost and distance metrics
   - Optimization statistics

2. **Visualization**:
   - Map of Sardinia with municipalities
   - Vehicle locations (blue dots)
   - Selected charging stations (color-coded by cost)
   - Battery range circles showing coverage areas
   - Legend explaining all visual elements
   - Summary statistics

## Optimization Approaches

The project implements two advanced optimization approaches:

### Branch-and-Bound Algorithm

Branch-and-bound is a systematic enumeration method that:
1. Solves LP relaxations of the integer problem at each node
2. Branches on fractional variables to create new subproblems
3. Prunes branches that cannot lead to better solutions
4. Continues until finding the optimal integer solution

See `branch_and_bound_ev_schema.tex` for a detailed explanation and visualization.

### Branch-and-Cut Algorithm

Branch-and-cut extends branch-and-bound by incorporating cutting planes:
1. Solves LP relaxations like branch-and-bound
2. Adds cutting planes to tighten the relaxations:
   - Covering cuts: Ensure enough stations to serve vehicle clusters
   - Neighborhood cuts: Ensure stations within range of vehicles
   - Clique inequalities: Handle mutually incompatible vehicle groups
3. Branches when cuts alone can't find integer solutions

See `branch_and_cut_ev_schema.tex` for a detailed explanation and visualization.

Branch-and-cut typically finds optimal solutions faster and with fewer branches than pure branch-and-bound.

## Project Structure

```
.
├── ev_charger.py                    # Main optimization module
├── ev_charger_visualization.py      # Visualization functions
├── sardinia_municipalities_complete.csv  # Municipality data
├── branch_and_bound_ev_schema.tex   # LaTeX explanation of branch-and-bound
├── branch_and_cut_ev_schema.tex     # LaTeX explanation of branch-and-cut
└── README.md                        # This file
```

## Visualization Details

The visualization includes:
- **Municipalities**: Shown as circles with size representing area
- **Vehicles**: Displayed as small blue dots
- **Charging Stations**: Colored circles with gradient based on installation cost
- **Battery Range**: Semi-transparent circles showing the coverage area for selected stations
- **Legend**: Comprehensive legend explaining all visual elements
- **Statistics**: Summary statistics showing total stations, costs, and coverage metrics

## Acknowledgments

This project uses real geographical and demographic data from the municipalities of Sardinia, Italy. 