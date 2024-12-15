import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

"""This script will plot the original data and the fitted parabola. The numpy.polyfit function returns the coefficients of the fitted polynomial, highest power first. In this case, it returns the coefficients a, b, and c of the parabola equation y=ax2+bx+c
"""
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# data: digitised from fitted line in Fig.2 of Mehrafarin and Nima Pourtolami (2008) by me
data = pd.read_csv('nikuradse.csv')

# Extract the x and y values
f = data.iloc[:, 0].values
r = data.iloc[:, 1].values
Re = data.iloc[:, 1].values

# Plot the original data
plt.scatter(Re, f, label='Digitised data')

for n in [ 2]:
    # Fit a second degree polynomial to the data
    coefficients = np.polyfit(x, y, n)

    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate y-values for the polynomial on a set of x-values
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = polynomial(x_fit)

 
    # Plot the fitted polynomial
    #plt.plot(x_fit, y_fit, 'r', label=f'polynomial order-{n}')



# Take the logarithm of the data
log_x = np.log(x)
log_y = np.log(y)

# Fit a line to the logarithmic data
coefficients = np.polyfit(log_x, log_y, 1)

# Create a polynomial function from the coefficients
polynomial = np.poly1d(coefficients)

# Generate y-values for the polynomial on a set of x-values
x2_fit = np.linspace(min(x), max(x), 500)
y2_fit = np.exp(polynomial(np.log(x_fit)))



# Plot the fitted power law
plt.plot(x2_fit, y2_fit, 'r', label='Fitted power law')


# Fit a cubic spline to the data
# cs = CubicSpline(x, y)

# Generate y-values for the spline on a set of x-values
# x_fit = np.linspace(min(x), max(x), 500)
# y_fit = cs(x_fit)

# Plot the fitted cubic spline
# plt.plot(x_fit, y_fit, 'r', label='Fitted cubic spline')

plt.xlabel('(ε/D)Re^6/(8+3η)')

plt.ylabel('100f Re^(2+3η)/(8+3η)')


plt.legend()
plt.grid(True)
plt.savefig("goldenf_1.png")
plt.close()
