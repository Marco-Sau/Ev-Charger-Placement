/*********************************************
 * OPL Run Script for EV Charger Placement
 * This script pre-processes data and runs the model
 *********************************************/

// Main function to execute
function main() {
  // Load the model and data
  var source = new IloOplModelSource("ev_charger.mod");
  var cplex = new IloCplex();
  var def = new IloOplModelDefinition(source);
  var model = new IloOplModel(def, cplex);
  var data = new IloOplDataSource("ev_charger.dat");
  model.addDataSource(data);
  model.generate();
  
  // Calculate distance matrix before solving
  calculateDistanceMatrix(model);
  
  // Display problem size
  writeln("Problem size:");
  writeln("- Variables: ", cplex.getNcols());
  writeln("- Constraints: ", cplex.getNrows());
  
  // Solve model
  if (cplex.solve()) {
    // Solution found
    writeln("\nOptimal solution found!");
    writeln("Objective value: ", cplex.getObjValue());
    
    // Process and display solution (handled by execute block in model)
    model.postProcess();
  } else {
    writeln("No solution found");
  }
}

// Function to calculate the distance matrix
function calculateDistanceMatrix(model) {
  writeln("Calculating distance matrix...");
  
  var numVehicles = model.numVehicles;
  var numLocations = model.numLocations;
  var locationCoords = model.locationCoords;
  var vehicleCoords = model.vehicleCoords;
  
  // Create the distance matrix
  for (var i = 1; i <= numVehicles; i++) {
    var vehicle = Opl.item(vehicleCoords, i-1);
    for (var j = 1; j <= numLocations; j++) {
      var location = Opl.item(locationCoords, j-1);
      
      // Calculate Euclidean distance
      var dx = vehicle.x - location.x;
      var dy = vehicle.y - location.y;
      var distance = Math.sqrt(dx*dx + dy*dy);
      
      // Store in the model's distance matrix
      model.distanceMatrix[i][j] = distance;
    }
  }
  
  writeln("Distance matrix calculated.");
}

// Run the main function
main(); 