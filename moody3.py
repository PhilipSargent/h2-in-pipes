import numpy as np
import matplotlib.pyplot as plt

# Constants for Moody diagram
K = 0.005  # Roughness coefficient
D = 0.035  # Pipe diameter in meters

# Laminar flow (Re < 2000)
Re_laminar = np.linspace(1, 2000, 100)
f_laminar = 64 / Re_laminar

# Transitional flow (2000 < Re < 4000)
Re_transitional = np.linspace(2000, 4000, 100)
f_transitional = 0.3164 * Re_transitional ** (-0.25)

# Turbulent flow (Re > 4000)
Re_turbulent = np.logspace(np.log10(4000), 8, 100)
f_turbulent = (1 / (-1.8 * np.log10(6.9 / Re_turbulent + (K / D) / 3.7))) ** 2

# Combine all flow regimes
Re = np.concatenate((Re_laminar, Re_transitional, Re_turbulent))
f = np.concatenate((f_laminar, f_transitional, f_turbulent))

# Plot the Moody diagram
plt.figure(figsize=(10, 6))
plt.loglog(Re, f, label='Moody Diagram')
plt.loglog(Re_laminar, f_laminar, 'r', label='Laminar Flow')
plt.title('Moody Diagram with Laminar Flow Line')
plt.xlabel('Reynolds number, Re')
plt.ylabel('Darcy-Weisbach friction factor, f')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('moody3_diagram.png')
#plt.show()
