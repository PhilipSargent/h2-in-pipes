import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mks
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

# data: digitised  by https://www.researchgate.net/publication/301791243_United_Formula_for_the_Friction_Factor_in_the_Turbulent_Region_of_Pipe_Flow
# reformatted by me by hand
data = pd.read_csv('nikuradse.csv')

# Extract the x and y values
f = data.iloc[:, 0].values
r = data.iloc[:, 1].values
Re = data.iloc[:, 2].values

# Plot the original data


print(len(data))

# This colour scheme and marker symbol exactly match those used by Jianjun Tao in
# Critical Instability and Friction Scaling of Fluid Flows through Pipes with Rough Inner Surfaces
mk = [mks.MarkerStyle('<'), 
     mks.MarkerStyle("^", fillstyle='none'), 
     mks.MarkerStyle("<", fillstyle='none'), 
    '+', 
      mks.MarkerStyle('s', fillstyle='none'), 
    'x',
    '+', 
    '+']

colours =['red', 'orange', 'lightblue', 'm', 'lightgreen', 'blue']

# Restructure this into one list per roughness
fdata = {}
Redata = {}
sym = {}
for i in range(len(Re)):
    fdata[r[i]] =[]
    Redata[r[i]] =[]
    
for i in range(len(Re)):
    fdata[r[i]].append(f[i])
    Redata[r[i]].append(Re[i])



ax = plt.gca()
ax.set_xscale('log')

j = 0
for r in fdata:
    sym = mk[j]
    col = colours[j]
    j += 1
    plt.scatter(Redata[r], fdata[r], label=f"D/ε = {r:.0f}", marker=sym, color=col)
    
#plt.scatter(Re, f, label='Digitised data')
symbols = ['+', 'x', ]
plt.ylabel('Darcy-Weisbach friction factor  f')
plt.xlabel("Reynolds' number")
plt.title(f"Nikuradse (1933) original results: {len(data)} data points")

plt.legend()
plt.grid(True)
plt.savefig("goldenf_2.png")
plt.close()

# - - - - - - - - - - - -- - - - - -- - - - - -- - - - - -- - - - - -
x = {}
y = {}
for r in fdata:
    x[r] = []
    y[r] = []
    for k in range(len(Redata[r])):
        xx = np.power(Redata[r][k], 3/4) / r
        yy = np.power(Redata[r][k], 1/4) * fdata[r][k]
        
        x[r].append(xx)
        y[r].append(yy)
        
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

j = 0
for r in fdata:
    sym = mk[j]
    col = colours[j]
    j += 1
    plt.scatter(x[r], y[r], label=f"D/ε = {r:.0f}", marker=sym, color=col)
    
plt.ylabel("$Re^{1/4} \cdot f$")
plt.xlabel("$Re^{3/4} \cdot (ε/D)$")
plt.legend()
plt.grid(True)
plt.savefig("goldenf_fre.png")
plt.close()
exit()

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
