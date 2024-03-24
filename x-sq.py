import matplotlib.pyplot as plt
import math

P = 201
# Define the function
def func(x):
  return math.sqrt(P - x)

# Define the x-axis values
x = range(0, 200)  # Adjust the range if needed

# Calculate the corresponding y-axis values
y = [func(i) for i in x]

# Create the plot
plt.plot(x, y, label=f"y = sqrt(201-x)")

# Add labels and title
plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.savefig("x2.png")  
