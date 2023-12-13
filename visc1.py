"""
Sure, here’s the Python program that calculates the viscosity of natural gas mixtures using the method of Lee, Gonzalez, and Eakin in SI units and then plots the results. This method is a simple relation which gives quite accurate results for typical natural gas mixtures with low non-hydrocarbon content.

Python
AI-generated code. Review and use carefully. More info on FAQ.


This program calculates the viscosity of natural gas for a range of temperatures and densities, assuming a constant molecular weight for the gas. It then creates a contour plot of the viscosity as a function of temperature and density.

Please note that you’ll need the numpy and matplotlib libraries to run this program. You can install them using pip:

pip install numpy matplotlib

Also, remember to replace the MWg value with the actual molecular weight of your natural gas mixture. The temperature T should be in Kelvin (K) and the density rho_g should be in kg/m^3. The resulting viscosity is expressed in Pascal-seconds (Pa.s).
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_viscosity(MWg, T, rho_g):
    # Constants for the Lee, Gonzalez, and Eakin method
    kv = (9.4 + 0.02 * MWg) * T**1.5 / (209 + 19 * MWg + T)
    xv = (3.5 + 986 / T + 0.01 * MWg) * np.exp(-0.025 * MWg)
    yv = 2.4 - 0.2 * xv

    # Calculate the viscosity
    mu_g = 1e-7 * kv * np.exp(xv * (rho_g / 1000)**yv)

    return mu_g

# Define the range of temperatures and densities
T_values = np.linspace(111, 222, 100)  # Temperature in Kelvin
rho_g_values = np.linspace(1, 10, 100)  # Density in kg/m^3

# Assume a constant molecular weight for the gas
MWg = 20  # Molecular weight

# Calculate the viscosity for each temperature and density
viscosity_values = np.array([[calculate_viscosity(MWg, T, rho_g) for T in T_values] for rho_g in rho_g_values])

# Plot the results
plt.figure(figsize=(10, 6))
plt.contourf(T_values, rho_g_values, viscosity_values, cmap='viridis')
plt.colorbar(label='Viscosity (Pa.s)')
plt.xlabel('Temperature (K)')
plt.ylabel('Density (kg/m^3)')
plt.title('Natural Gas Viscosity')
plt.savefig("visc1.png")

