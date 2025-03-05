/*********************************************
 * OPL Model for EV Charger Placement
 * This model determines optimal locations for electric vehicle charging stations
 * in an urban environment, minimizing installation costs while ensuring coverage.
 *********************************************/

// Parameters that will be read from data file
int numLocations = ...;
int numVehicles = ...;
float batteryRange = ...;

// Coordinates
tuple LocationPoint {
  float x;
  float y;
}

tuple VehiclePoint {
  float x;
  float y;
}

// Sets
{LocationPoint} locationCoords = ...;
{VehiclePoint} vehicleCoords = ...;
range Locations = 1..numLocations;
range Vehicles = 1..numVehicles;

// Weights for vehicles (optional)
float vehicleWeights[Vehicles] = ...;

// Installation costs for each location
float installationCosts[Locations] = ...;

// Maximum vehicles per station
int maxVehiclesPerStation = ...;

// Precomputed distance matrix
float distanceMatrix[Vehicles][Locations] = ...;

// Decision Variables
dvar boolean x[Locations];             // Whether to install a charger at location j
dvar boolean y[Vehicles][Locations];   // Whether vehicle i is assigned to location j

// Objective: Minimize installation costs
minimize sum(j in Locations) installationCosts[j] * x[j];

// Constraints
subject to {
  // 1. Each vehicle must be assigned to at least one charging station
  forall(i in Vehicles)
    sum(j in Locations) y[i,j] >= 1;
  
  // 2. Vehicles can only use charging stations within battery range
  forall(i in Vehicles, j in Locations)
    y[i,j] <= (distanceMatrix[i,j] <= batteryRange);
  
  // 3. Vehicles can only use locations where chargers are installed
  forall(i in Vehicles, j in Locations)
    y[i,j] <= x[j];
  
  // 4. Capacity constraint: Each charging station can serve at most maxVehiclesPerStation vehicles
  forall(j in Locations)
    sum(i in Vehicles) y[i,j] <= maxVehiclesPerStation * x[j];
}

// Solution output information
execute {
  writeln("SOLUTION SUMMARY");
  writeln("===============");
  
  var selectedLocations = new Array();
  var totalCost = 0;
  
  for (var j = 1; j <= numLocations; j++) {
    if (x[j] > 0.5) {
      selectedLocations.push(j);
      totalCost += installationCosts[j];
      
      // Calculate station load
      var vehicleCount = 0;
      for (var i = 1; i <= numVehicles; i++) {
        if (y[i][j] > 0.5) {
          vehicleCount++;
        }
      }
      
      var location = Opl.item(locationCoords, j-1);
      var utilization = (vehicleCount / maxVehiclesPerStation) * 100;
      
      writeln("- Station at (", location.x, ", ", location.y, 
              ") - Cost: $", installationCosts[j], 
              " - Vehicles: ", vehicleCount, "/", maxVehiclesPerStation,
              " (", utilization.toFixed(0), "% utilization)");
    }
  }
  
  writeln("\nTotal Installation Cost: $", totalCost);
  writeln("Number of charging stations: ", selectedLocations.length);
} 