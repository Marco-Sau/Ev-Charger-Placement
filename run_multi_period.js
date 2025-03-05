/*********************************************
 * OPL Run Script for Multi-Period EV Charger Placement
 * This script runs both current and future scenarios
 *********************************************/

// Main function to execute
function main() {
  writeln("MULTI-PERIOD EV CHARGING STATION OPTIMIZATION");
  writeln("=============================================");
  
  // First run the current scenario
  writeln("\nCURRENT SCENARIO");
  writeln("----------------");
  var currentSelectedLocations = runScenario("ev_charger.dat");
  
  // Then run the future scenario
  writeln("\nFUTURE SCENARIO");
  writeln("--------------");
  var futureSelectedLocations = runScenario("ev_charger_future.dat");
  
  // Compare the two scenarios
  compareScenarios(currentSelectedLocations, futureSelectedLocations);
}

// Function to run a single scenario
function runScenario(dataFile) {
  // Load the model and data
  var source = new IloOplModelSource("ev_charger.mod");
  var cplex = new IloCplex();
  var def = new IloOplModelDefinition(source);
  var model = new IloOplModel(def, cplex);
  var data = new IloOplDataSource(dataFile);
  model.addDataSource(data);
  model.generate();
  
  // Calculate distance matrix
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
    
    // Process solution
    model.postProcess();
    
    // Collect selected locations
    var selectedLocations = [];
    for (var j = 1; j <= model.numLocations; j++) {
      if (model.x[j] > 0.5) {
        selectedLocations.push(j);
      }
    }
    
    return selectedLocations;
  } else {
    writeln("No solution found");
    return [];
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
}

// Function to compare current and future scenarios
function compareScenarios(currentLocations, futureLocations) {
  writeln("\nCOMPARISON OF SCENARIOS");
  writeln("======================");
  
  writeln("\nCurrent scenario selected locations: ", currentLocations);
  writeln("Future scenario selected locations: ", futureLocations);
  
  // Find new stations needed for future
  var newStations = [];
  for (var i = 0; i < futureLocations.length; i++) {
    var location = futureLocations[i];
    if (currentLocations.indexOf(location) == -1) {
      newStations.push(location);
    }
  }
  
  writeln("\nAdditional stations needed for future: ", newStations.length);
  if (newStations.length > 0) {
    writeln("Locations of new stations: ", newStations);
  } else {
    writeln("No additional stations needed for future scenario!");
  }
}

// Run the main function
main(); 