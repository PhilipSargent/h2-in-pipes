"""Sure, here’s a Python program that uses the matplotlib and numpy libraries to plot the given data on a log(viscosity) against temperature plot and fit a straight line to each gas, thus calculating the power law exponent of viscosity on temperature for each gas.

Python
AI-generated code. Review and use carefully. More info on FAQ.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from https://myengineeringtools.com/Data_Diagrams/Viscosity_Gas.html
# omitting 600C data and -150 and -100C data
temperature = np.array([0, 100, 200, 300, 400, 500])
viscosity = { # in mPs.s
    'O2': np.array([0.0192, 0.0244, 0.029, 0.0332, 0.037, 0.0404]),
    'Air': np.array([ 0.0176, 0.022, 0.026, 0.0298, 0.033, 0.036]),
    'N2': np.array([ 0.0168, 0.021, 0.0248, 0.028, 0.0308, 0.0332]),
    'CO2': np.array([ 0.0136, 0.0183, 0.0224, 0.0265, 0.0302, 0.0338]),
    'H2O': np.array([ 0.009, 0.0128, 0.0164, 0.0202, 0.024, 0.0278]),
    'H2': np.array([ 0.0083, 0.0104, 0.01202, 0.014, 0.0156, 0.017])
}

# Convert temperature to Kelvin
temperature = temperature + 273.15

# Plotting
plt.figure(figsize=(10, 6))
for gas, vis in viscosity.items():
    vis = vis / 1000 #Convert mPa.s to Ps.a
    #mask = np.isfinite(vis)
    mask = [value is not None for value in vis]
    slope, intercept = np.polyfit(np.log10(temperature[mask]), np.log10(vis[mask]), 1)
    print(f"{gas:6}    {slope:.3f}  {intercept:.2f}")
    plt.plot(np.log10(temperature), np.log10(vis), 'o', label=f'{gas}, slope: {slope:.2f}')
    plt.plot(np.log10(temperature), slope * np.log10(temperature) + intercept, '-')
plt.xlabel('log10(Temperature) (K)')
plt.ylabel('log10(Viscosity) (Pa.s)')
plt.legend()
plt.grid(True)
plt.title('Log-Viscosity vs Log-Temperature for Different Gases')
plt.savefig("visc-temp.png")
plt.close()