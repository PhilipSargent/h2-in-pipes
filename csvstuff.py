import csv

# Create a list of data
data = [
    ['H2', 33.2, 13.0, -0.22, 2.015],
    ['CH4', 190.56, 45.99, 0.011, None],
    ['C2H6', 305.32, 48.72, 0.099, None],
    ['C3H8', 369.15, 42.48, 0.152, None],
    ['nC4', 425, 38, 0.2, None],
    ['iC4', 407.7, 36.5, 0.2, None],
    ['iC5', 407.7, 36.5, 0.2, None],
    ['nC5', 407.7, 36.5, 0.2, None],
    ['CO2', 304.2, 73.8, 0.225, None],
    ['H2O', 647.1, 220.6, 0.345, None],
    ['N2', 126.21, 33.9, 0.0401, None],
]

# Open a CSV file for writing
with open('PR_constants.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['Component', 'Tc', 'Pc', 'omega', 'Mw'])

    # Write the data rows
    writer.writerows(data)

print("PR_constants.csv file created successfully!")