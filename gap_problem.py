"""
Call Center Workforce Scheduling using the Generalized Assignment Problem (GAP)
with IBM CPLEX via the DOcplex library, including visualization of the results.

Scenario:
    - A 24/7 call center requires coverage for 21 shifts per week (3 shifts/day for 7 days).
    - There are 10 employees, each with a maximum number of shifts they can work.
    - Each shift is assigned a type (Morning, Afternoon, Night) with corresponding base costs.
    - Some employees (e.g., E3) are restricted from working disallowed shifts (e.g., night shifts).

The objective is to minimize the total cost of shift assignments while meeting all coverage and capacity constraints.
The script then visualizes the weekly schedule as a table and the distribution of shifts per employee.
"""

from docplex.mp.model import Model
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define Data
# ----------------

# List of employees and their maximum shifts (capacity)
employees = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
capacities = {
    'E1': 5, 'E2': 5, 'E3': 4, 'E4': 5, 'E5': 4,
    'E6': 5, 'E7': 5, 'E8': 4, 'E9': 5, 'E10': 5
}

# Define the days of the week and shift types
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
shift_types = ['Morning', 'Afternoon', 'Night']

# Build the list of shifts: each shift is a tuple (day, shift_type)
shifts = []
for day in days:
    for stype in shift_types:
        shifts.append((day, stype))
# Total number of shifts is 21
num_shifts = len(shifts)

# Define disallowed assignments.
# For instance, employee 'E3' is not allowed to work 'Night' shifts.
not_allowed = {
    'E3': ['Night']
}

# Create a cost matrix for each employee and shift.
# Base costs: Morning=10, Afternoon=12, Night=15.
# For disallowed assignments, we assign a high cost and add an explicit constraint.
cost = {}
for e in employees:
    cost[e] = {}
    for j, (day, stype) in enumerate(shifts):
        if e in not_allowed and stype in not_allowed[e]:
            cost[e][j] = 999  # High cost for disallowed shifts.
        else:
            if stype == 'Morning':
                cost[e][j] = 10
            elif stype == 'Afternoon':
                cost[e][j] = 12
            elif stype == 'Night':
                cost[e][j] = 15

# 2. Build the Optimization Model
# ---------------------------------

mdl = Model(name="CallCenterScheduling")

# Decision variables: x[e, j] = 1 if employee e is assigned to shift j, else 0.
x = {}
for e in employees:
    for j in range(num_shifts):
        x[e, j] = mdl.binary_var(name="x_{}_{}".format(e, j))

# Objective Function: Minimize the total assignment cost
mdl.minimize(mdl.sum(cost[e][j] * x[e, j] for e in employees for j in range(num_shifts)))

# 3. Add Constraints
# -------------------

# 3.1. Each shift must be covered by exactly one employee.
for j in range(num_shifts):
    mdl.add_constraint(mdl.sum(x[e, j] for e in employees) == 1,
                       ctname="shift_assign_{}".format(j))

# 3.2. Each employee cannot be assigned more shifts than their capacity.
for e in employees:
    mdl.add_constraint(mdl.sum(x[e, j] for j in range(num_shifts)) <= capacities[e],
                       ctname="cap_{}".format(e))

# 3.3. Disallow assignments for employees not permitted on certain shifts.
for e in employees:
    for j, (day, stype) in enumerate(shifts):
        if e in not_allowed and stype in not_allowed[e]:
            mdl.add_constraint(x[e, j] == 0,
                               ctname="not_allowed_{}_{}".format(e, j))

# 4. Solve the Model
# -------------------
solution = mdl.solve(log_output=True)

if solution:
    print("Solution found:\n")
    
    # Extract the schedule: for each shift, determine the assigned employee.
    schedule = []
    for j, (day, stype) in enumerate(shifts):
        for e in employees:
            if solution.get_value(x[e, j]) > 0.5:  # Binary variable is 1
                schedule.append({"Day": day, "Shift": stype, "Employee": e})
                break

    # Create a pandas DataFrame from the schedule list.
    schedule_df = pd.DataFrame(schedule)
    
    # Display the schedule as a table.
    print("Weekly Shift Schedule:")
    print(schedule_df)
    
    # 4.1. Plot the schedule as a table using matplotlib.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=schedule_df.values, 
                     colLabels=schedule_df.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title("Call Center Weekly Shift Schedule")
    plt.tight_layout()
    plt.show()

    # 4.2. Plot a bar chart showing the number of shifts assigned to each employee.
    shift_counts = schedule_df['Employee'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    shift_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Employee')
    plt.ylabel('Number of Shifts')
    plt.title('Number of Shifts Assigned per Employee')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

else:
    print("No solution found.")

# Optionally, print additional model information.
mdl.print_information()