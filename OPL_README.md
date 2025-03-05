# EV Charging Station Placement - OPL Model

This directory contains the OPL (Optimization Programming Language) implementation of the electric vehicle charging station placement optimization model. This model is designed to work with IBM ILOG CPLEX Optimization Studio.

## Files

- `ev_charger.mod` - The OPL model file containing the mathematical model definition
- `ev_charger.dat` - The data file containing urban scenario parameters
- `run.js` - JavaScript file to pre-process data and execute the model

## How to Use with IBM CPLEX Optimization Studio

### Setup

1. Open IBM ILOG CPLEX Optimization Studio
2. Create a new project: `File > New > OPL Project`
3. Name your project (e.g., "EV_Charger_Placement")
4. Copy the three files (`ev_charger.mod`, `ev_charger.dat`, and `run.js`) into your project folder
5. In the Project Explorer view, right-click on your project and select `Refresh`

### Running the Model

1. Right-click on `run.js` in the Project Explorer
2. Select `Run As > OPL Run Configuration`
3. Make sure that:
   - The model file is set to `ev_charger.mod`
   - The data file is set to `ev_charger.dat`
   - The run file is set to `run.js`
4. Click `Run`

### Modifying the Model

#### To adjust the urban scenario:

Edit `ev_charger.dat` to modify:
- The number of locations and vehicles
- Coordinates of potential charging stations
- Vehicle positions
- Installation costs
- Maximum vehicle capacity per station
- Battery range

#### To adjust the optimization model:

Edit `ev_charger.mod` to change:
- The objective function
- Constraints
- Output formatting

## Multi-Period Planning

To implement multi-period planning (comparing current and future scenarios), you would:

1. Create a second data file `ev_charger_future.dat` with increased number of vehicles
2. Modify `run.js` to:
   - Solve the model with the current data
   - Save the solution
   - Solve again with the future data
   - Compare the solutions

## Benefits of Using OPL

- Full access to CPLEX without size limitations
- No need for Python license configuration
- Native integration with IBM CPLEX Optimizer
- Better performance for large-scale problems
- Rich modeling language designed specifically for optimization

## Visualization

Unlike the Python version, visualization isn't built directly into the OPL model. To visualize the results:

1. Export the solution data using the `execute` block in the model
2. Use external tools (Excel, Python, etc.) to create visualizations
3. Alternatively, use the IBM CPLEX Optimization Studio visualization tools 