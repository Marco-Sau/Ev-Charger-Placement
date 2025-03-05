import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

# Generate random data
np.random.seed(42)
n_customers = 10  # Number of customers
K = 3  # Number of vehicles
Q = 15  # Vehicle capacity

# Generate random coordinates for customers and depot
x_coord = np.random.rand(n_customers + 1) * 100
y_coord = np.random.rand(n_customers + 1) * 100

# Generate random demand values for customers
demand = np.random.randint(1, 5, size=n_customers)

# Generate random time windows for each customer
time_windows = [(np.random.randint(0, 50), np.random.randint(50, 100)) for _ in range(n_customers)]

# Generate random service time for each customer
service_time = np.random.randint(1, 5, size=n_customers)

# Define depot location
depot = (x_coord[0], y_coord[0])

# Define customer indices
N = [i for i in range(1, n_customers + 1)]  # Customer indices
V = [0] + N  # All vertices including depot

# Function to calculate Euclidean distance between two points
def distance(i, j):
    return np.sqrt((x_coord[i] - x_coord[j])**2 + (y_coord[i] - y_coord[j])**2)

# Compute distance matrix
c = {(i, j): distance(i, j) for i in V for j in V if i != j}

# Initialize optimization model
mdl = Model("VRPTW")

# Define binary decision variables for vehicle routes
x = mdl.binary_var_dict(c, name='x')

# Define continuous decision variables for vehicle load after serving customers
u = mdl.continuous_var_dict(N, lb=0, ub=Q, name='u')

# Define continuous decision variables for arrival times at customers
t = mdl.continuous_var_dict(V, lb=0, name='t')

# Objective function: Minimize total travel cost
mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in c))

# Constraints
# Each customer must be visited exactly once
for i in N:
    mdl.add_constraint(mdl.sum(x[j, i] for j in V if j != i) == 1)
    mdl.add_constraint(mdl.sum(x[i, j] for j in V if j != i) == 1)

# Time window constraints for customers
for i in N:
    mdl.add_constraint(t[i] >= time_windows[i-1][0])
    mdl.add_constraint(t[i] <= time_windows[i-1][1])

# Vehicle capacity constraints
for i in N:
    mdl.add_constraint(u[i] >= demand[i-1])
    mdl.add_constraint(u[i] <= Q)

# Flow conservation at depot
mdl.add_constraint(mdl.sum(x[0, j] for j in N) <= K)
mdl.add_constraint(mdl.sum(x[i, 0] for i in N) <= K)

# Subtour elimination and capacity constraints
M = 1000  # Big M constant
for i in N:
    for j in N:
        if i != j:
            # Capacity constraints
            mdl.add_constraint(u[i] - u[j] + demand[j-1] <= Q * (1 - x[i, j]))
            
            # Time window constraints with service time
            mdl.add_constraint(
                t[i] + service_time[i-1] + c[i, j] <= t[j] + M * (1 - x[i, j])
            )

# Add depot time window constraint (assuming depot is available from time 0 to end of day)
max_time = max(tw[1] for tw in time_windows)
mdl.add_constraint(t[0] == 0)  # Depot starts at time 0

# Solve the model
if not mdl.solve():
    print("No feasible solution found. Check constraints.")
else:
    print("Optimal solution found.")

# Extract the solution, listing the selected routes
solution = [(i, j) for i, j in c if x[i, j].solution_value is not None and x[i, j].solution_value > 0.9]
if not solution:
    print("No solution found. Ensure constraints are correct.")
else:
    print("Optimal Routes:", solution)

# Visualization
plt.scatter(x_coord[1:], y_coord[1:], color='blue')  # Plot customer locations
plt.scatter(*depot, color='red', marker='*', label='Depot')  # Plot depot location

# Annotate demand values at each customer location
for i, txt in enumerate(demand, start=1):
    plt.annotate(f"{txt}", (x_coord[i]+2, y_coord[i]+2))

# Plot selected vehicle routes
for i, j in solution:
    plt.plot([x_coord[i], x_coord[j]], [y_coord[i], y_coord[j]], 'k-')

plt.legend()
plt.show()
