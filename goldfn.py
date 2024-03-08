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

print(len(Re),f" Nikuradse points")
# Data from Orgeon and Princeton machines is published in
# Friction factors for smooth pipe flow, J. Fluid Mech. (2004), vol. 511, pp. 41–44
# DOI: 10.1017/S0022112004009796
# https://www.researchgate.net/publication/231901224_Friction_factors_for_smooth_pipe_flow

# roughness 3.8e-4 for Bauer & Galavics in steam pipes.

# roughness 1/26000 for Langelands and same pipe Shultz et al "Flow in a Commercial Steel Pipe"
# Shultz : 16th Australasian Fluid Mechanics Conference
# Crown Plaza, Gold Coast, Australia
# 2-7 December 2007

# Oregon data CJ Swanson, B Julian, GG Ihas, RJ Donnelly
# Journal of Fluid Mechanics, 2002
# https://scholar.google.com/scholar_lookup?&title=&journal=J.%20Fluid%20Mech.&doi=10.1017%2FS0022112002008595&volume=461&publication_year=2002&author=Swanson%2CC.J&author=Julian%2CB&author=Ihas%2CG.G&author=Donnelly%2CR.J

#  diameter to be 0.4672 ± 0.0010 cm i.e. 4.672 mm
# e < 0.1 microns

# ff_oregon_lam.csv is the data before the turbulent region
# Load the data from a CSV file
for csv in ['ff_princeton.csv', 'ff_oregon.csv']:
    data = pd.read_csv(csv)

    # Extract the x and y values
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    if csv == 'ff_princeton.csv':
        print(len(x),f" Princeton points")
        rr = 26000
    else:
        print(len(x),f" Oregon points")
        rr = 4.672 /(0.1 * 1e-3)
    z = np.ones(len(x))*rr

    f = np.concatenate((y, f))
    r = np.concatenate((z, r))
    Re = np.concatenate((x, Re))
print(len(Re),f" Nikuradse + Princeton points")


# Plot the original data


# This colour scheme and marker symbol exactly match those used by Jianjun Tao in
# Critical Instability and Friction Scaling of Fluid Flows through Pipes with Rough Inner Surfaces
mk = [mks.MarkerStyle("o", fillstyle='none'), 
    '+', 
    mks.MarkerStyle('<'), 
    mks.MarkerStyle("^", fillstyle='none'), 
    mks.MarkerStyle("<", fillstyle='none'), 
    '+', 
    mks.MarkerStyle('s', fillstyle='none'), 
    'x',
    '+']

colours =['olive', 'black','red', 'orange', 'lightblue', 'm', 'lightgreen', 'blue', ]

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
title = f"Nikuradse, Princeton, Oregon: {len(Re)} data points"
xlabel = "$Re^{6/(8+3\\eta)} \\cdot (2ε/D)$"
symbols = ['+', 'x', ]
plt.ylabel('Darcy-Weisbach friction factor  f')
plt.xlabel("Reynolds' number")
plt.title(title)

plt.legend()
plt.grid(True)
plt.savefig("goldenf_2.png")
plt.close()

# - - - - - - - - - - - -- - - - - -- - - - - -- - - - - -- - - - - -
eta = 0.02
denom = 8 + 3*eta
nom = 2+3*eta
expon =  nom/denom
expon2 = 6 / denom

x = {}
y = {}
for r in fdata:
    x[r] = []
    y[r] = []
    for k in range(len(Redata[r])):
        xx = np.power(Redata[r][k], expon2) * 2 / (r)
        yy = np.power(Redata[r][k], expon) * fdata[r][k]
        
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
    plt.scatter(x[r], y[r], label=f"ε/D = 1/{r:.0f}", marker=sym, color=col)

plt.title(title)
plt.ylabel("$f \\cdot Re^{((2+3\\eta)/(8+3\\eta))}$")
plt.xlabel(xlabel)
plt.legend()
plt.grid(True)
plt.savefig("goldenf_fre.png")
plt.close()

ax = plt.gca()
ax.set_xscale('linear')
ax.set_yscale('linear')

j = 0
for r in fdata:
    sym = mk[j]
    col = colours[j]
    j += 1
    plt.scatter(x[r], y[r], label=f"ε/D = 1/{r:.0f}", marker=sym, color=col)

plt.title(title)
plt.ylabel("$f \\cdot Re^{((2+3\\eta)/(8+3\\eta))}$")
plt.xlabel(xlabel)
plt.legend()
plt.grid(True)
plt.savefig("goldenf_fre_lin.png")
plt.close()
exit()


