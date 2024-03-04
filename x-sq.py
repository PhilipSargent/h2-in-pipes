import matplotlib.pyplot as plt

P = 1e5
# Define the function
def func(x):
  return P - x**2

# Define the x-axis values
x = range(0, 200)  # Adjust the range if needed

# Calculate the corresponding y-axis values
y = [func(i) for i in x]

# Create the plot
plt.plot(x, y, label=f"y = {P} - x^2")

# Add labels and title
plt.xlabel("x")
plt.ylabel("y")

plt.legend()

plt.savefig("x2.png")  
