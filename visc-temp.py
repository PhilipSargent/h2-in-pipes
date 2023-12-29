"""Sure, here’s a Python program that uses the matplotlib and numpy libraries to plot the given data on a log(viscosity) against temperature plot and fit a straight line to each gas, thus calculating the power law exponent of viscosity on temperature for each gas.
18 Dec. 2023
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from https://myengineeringtools.com/Data_Diagrams/Viscosity_Gas.html
# omitting 600C data and -150 and -100C data
temperature = np.array([0, 100, 200, 300, 400, 500]) # degreees C
# Convert temperature to Kelvin
temperature = temperature + 273.15

viscosity = { # in mPa.s
    'O2': np.array([0.0192, 0.0244, 0.029, 0.0332, 0.037, 0.0404]),
    'Air': np.array([ 0.0176, 0.022, 0.026, 0.0298, 0.033, 0.036]),
    'N2': np.array([ 0.0168, 0.021, 0.0248, 0.028, 0.0308, 0.0332]),
    'CO2': np.array([ 0.0136, 0.0183, 0.0224, 0.0265, 0.0302, 0.0338]),
    'H2O': np.array([ 0.009, 0.0128, 0.0164, 0.0202, 0.024, 0.0278]),
    'H2': np.array([ 0.0083, 0.0104, 0.01202, 0.014, 0.0156, 0.017]),
}
for gas, vis in viscosity.items():
    vis = vis / 1000 # Convert mPa.s to Pa.s
    viscosity[gas] = vis

temperature2 = np.array([300, 400, 500, 600]) #degrees K
viscosity2 = { # in microPa.s
    'CH4': np.array([ 9.7, 13, 16.4, 19.8]),
    'C2H6': np.array([ 9.4, 12.2, 14.8, 17.1]),
    'C3H8': np.array([ 8.2, 10.8, 13.3, 15.6]),
    'nC4': np.array([ 7.5, 9.9, 12.2, 14.5]),
    'iC4': np.array([ 7.5, 9.9, 12.2, 14.4]),
    'nC5': np.array([ 6.7, 9.2, 11.4, 13.4]),
    'iC5': np.array([ 7.5, 9.9, 12.2, 14.4]),
    
    
    'Ar': np.array([ 22.7, 28.6, 33.9, 38.8]),
    'He': np.array([ 19.9, 24.3, 28.3, 32.2]),
    'C6': np.array([ 6.29, 8.6, 10.8, 12.82]), # 300 K for gaseous not liquid C6, at 0.218 bar. Close enough.
}
for gas, vis in viscosity2.items():
    vis = vis / 1e6 # Convert microPa.s to Pa.s
    viscosity2[gas] = vis

temperature3 = np.array([300, 400, 500]) #degrees K
viscosity3 = { # in microPa.s
   'neoC5': np.array([6.9326, 9.1783, 11.18]), # not same source. 
 }

for gas, vis in viscosity3.items():
    vis = vis / 1e6 # Convert microPa.s to Pa.s
    viscosity3[gas] = vis

# Plotting
plt.figure(figsize=(10, 6))

for gas, vis in viscosity.items():
    mask = [value is not None for value in vis]
    slope, intercept = np.polyfit(np.log10(temperature[mask]), np.log10(vis[mask]), 1)
    print(f"{gas:6}    {slope:.3f}  {intercept:.2f}  ({1e6*pow(10, intercept):.3f} μK)")
    plt.plot(np.log10(temperature), np.log10(vis), 'o', label=f'{gas}, slope: {slope:.2f}')
    plt.plot(np.log10(temperature), slope * np.log10(temperature) + intercept, '-')

for gas, vis in viscosity2.items():
    mask = [value is not None for value in vis] 
    slope, intercept = np.polyfit(np.log10(temperature2[mask]), np.log10(vis[mask]), 1)
    print(f"{gas:6}    {slope:.3f}  {intercept:.2f}  ({1e6*pow(10, intercept):.3f} μK)")
    plt.plot(np.log10(temperature2), np.log10(vis), 'o', label=f'{gas}, slope: {slope:.2f}')
    plt.plot(np.log10(temperature2), slope * np.log10(temperature2) + intercept, '-')
    
for gas, vis in viscosity3.items():
    mask = [value is not None for value in vis] 
    slope, intercept = np.polyfit(np.log10(temperature3[mask]), np.log10(vis[mask]), 1)
    print(f"{gas:6}    {slope:.3f}  {intercept:.2f}  ({1e6*pow(10, intercept):.3f} μK)")
    plt.plot(np.log10(temperature3), np.log10(vis), 'o', label=f'{gas}, slope: {slope:.2f}')
    plt.plot(np.log10(temperature3), slope * np.log10(temperature3) + intercept, '-')
    
plt.xlabel('log10(Temperature) (K)')
plt.ylabel('log10(Viscosity) (Pa.s)')
plt.legend()
plt.grid(True)
plt.title('Log-Viscosity vs Log-Temperature for Different Gases')
plt.savefig("visc-temp.png")
plt.close()