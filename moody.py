"""
Bing wrote this code with only one little bug:

Below is a Python program that uses the `matplotlib` and `numpy` libraries to plot a Moody diagram, which is a graph that shows the relationship between the Reynolds number, relative roughness, and the Darcy-Weisbach friction factor for fluid flow in pipes. The program will save the plot as an image file named `moody_diagram.png`."""

import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# We memoize some functions so that they do not get repeadtedly called with
# the same arguments. Yet still be retain a more obvius way of writing the program.

def memoize(func):
    """Standard memoize function to use in a decorator, see
    https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
    """
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func


lowest_f = 0.000001

# Define laminar flow (Re < 2000)
def laminar(reynolds):
    f_laminar = 64 / reynolds
    return f_laminar
    
def blasius(reynolds):
    f = (0.316 / (reynolds**0.25))
    if f > 0.009 and reynolds < 1e10 : # to get diagram the right shape
        return f
    
    return None
     
    
def smooth(reynolds):
    return colebrook(reynolds, 0.0)
    
def haarland(reynolds, relative_roughness):
    """The accuracy of the Darcy friction factor solved from this equation
    is claimed to be within about ±2 % of the Colebrook value, 
    if the Reynolds number is greater than 3,000 
    Kiij¨arvi (2011)
    But I have made a mistake and it produces numbers all about 0.2 not about 0.02,
    only valid re > 3,000"""
    if reynolds < 4000 or reynolds > 1e8 or relative_roughness < 1e-6 or relative_roughness > 5e-2:
        return 0.02
    inverse_f = -1.8 * np.log((relative_roughness/3.7)**1.11 + 6.9/reynolds) # typo in one  paper: 37 not 3.7
    return 1/np.sqrt(inverse_f)

@memoize 
def colebrook(reynolds, relative_roughness):
    """Define the Colebrook equation as an implicit function and solve it
    """
    # Initial guess for the friction factor
    f_initial = 0.02
    #f_initial = haarland(reynolds, relative_roughness) #fails 
    
    # Define the implicit Colebrook equation
    def f(f, re, rr):
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (re * np.sqrt(f)))

        
    # Solve for the friction factor using the Newton-Raphson method
    f_solution = f_initial
    f_new = f_initial
    epsilon = 0.02
    for _ in range(20): # not Newton method, just iterate 20x
        if f_solution < 0:
            f_new = lowest_f
        else:
            f_new -= f(f_solution, reynolds, relative_roughness) / (-0.5 * f_solution**(-1.5) - 2.51 / (reynolds * np.sqrt(f_solution)) * (-0.5 * f_solution**(-1.5)))
        if abs((f_solution - f_new)/f_solution) < epsilon:
            return f_new
        f_solution = f_new
    return f_solution

rr_piggot = np.logspace(-1.3, -5, 100) # roughness values between 0.01 and 0.00001

def piggot_point(rr):
    """The Piggot line is where the Colebrook curve flattens out as a function of Re
    Source is comment by RJS Piggot on page 680 of Rouse (1944).
    """
    eps = 0.01 # to within 1%
    fix = 3500 # Piggot's number
    n = 0
    for re in reynolds:
        n += 1
        if abs(re * rr - fix)/(re * rr) < eps:
           f = colebrook(re, rr)
           return f, re

    return None

def smooth_piggot(points):
    def find_next_valid(n=0):
        # defaults to starting at beginning of list
        first_valid = None
        for i in range(n, len(points)):
            if points[i]:
                return i, points[i]
        return None, None

    i, f = find_next_valid()  
    if not i:
        print("All None")
        return points # give up: a list entirely of None values
    
    print(f"First OK at {i}")
    j = i
    while j < len(points) - 1:
        j = j+1
        if points[j]:
            i = j
            continue
        k, f = find_next_valid(j)  
        if not k:
           print(f"All None from {j} to end ")
           return points # No more valid values left
        print(f"at {j} invalid   found next valid {k} {f:.4f}")
        v = j-1
        # valid points at j-1 and at k
        
        gap = k - j 
        q = 0
        for m in range(j, k):
            q += 1
            incr = -(points[v] - points[k])/gap
            points[m] = points[v] + q*incr
            #print(f"  Averaging: {j} to {k}  = {q}/{gap} {points[v]:.5f} {points[k]:.5f}     {points[m]:.5f}")
        j = k
    
    print(f"Done all")
    return points

def piggot():
    """Calculate the Piggot points then re-order for plotting"""
    points = {}
    for rr in rr_piggot:
        if pp := piggot_point(rr):
            f, re = pp
            points[re] = f

    piggot_list = []
    #last_valid = None
    for re in reynolds:
        if re in points:
            #last_valid = points[re]
            piggot_list.append(points[re])
        else:
            piggot_list.append(None)
    
    p = smooth_piggot(piggot_list)
    # for i in range(len(p)-1):
        # if p[i] and p[i+1]:
            # print(f"{i}:{p[i]:.5f} {p[i]-p[i+1]:.5f}")
    return p
    
def plot_diagram(title, filename, plot="loglog"):
    # Calculate the friction factor for each relative roughness
    friction_factors = {}
    for rr in relative_roughness_values:
        friction_factors[rr] = [colebrook(re, rr) for re in reynolds]
        # [print(re, rr, haarland(re, rr)) for re in reynolds]


    friction_laminars = [laminar(re) for re in reynolds_laminar]
    friction_smooth = [smooth(re) for re in reynolds]
    friction_blasius = [blasius(re) for re in reynolds]
    
    plt.figure(figsize=(10, 6))
    # Plot the Moody diagram
    # plt.loglog(reynolds, friction_smooth, label=f'Smooth: ε/D = 0')
    if plot == "loglog":
        plt.loglog(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
        plt.loglog(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
        if fp:
            plt.loglog(reynolds, fp, label='Piggot', linestyle='dashdot')
        for rr, ff in friction_factors.items():
            plt.loglog(reynolds, ff, label=f'ε/D = {rr}')
    if plot == "linear":
        plt.plot(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
        plt.plot(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
        if fp:
            plt.plot(reynolds, fp, label='Piggot')
        for rr, ff in friction_factors.items():
            plt.plot(reynolds, ff, label=f'ε/D = {rr}')


    plt.xlabel('Reynolds number, Re')
    plt.ylabel('Darcy-Weisbach friction factor, f')
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(filename)
    
# Define the Reynolds number range and relative roughness values
# only need 10 points for the straight line
reynolds_laminar = np.logspace(2.9, 3.9, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.4, 9.0, 500) # 10^7.7 = 5e7
relative_roughness_values = [0.01, 0.001, 0.0001, 0.00001]

fp = piggot()

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

plot_diagram('Moody Diagram', 'moody_diagram.png', plot="loglog")

# Plot enlarged diagram
reynolds_laminar = np.logspace(2.9, 3.4, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.4, 5.0, 500) 
relative_roughness_values = [0.01, 0.003, 0.001]

# fp = piggot() # not in view on the enlarged plot
fp = None
plot_diagram('Moody Diagram Transition region', 'moody_enlarge.png',plot="loglog")

reynolds_laminar = np.logspace(2.9, 3.4, 50) # 10^2.7 = 501, 10^3.4 = 2512

plot_diagram('Moody Diagram Transition region', 'moody_enlarge_lin.png',plot="linear")


# plt.show() # does not work as this is a non-interactive run of the program
'''
Before running this program, ensure you have the required libraries installed. You can install them using pip:
pip install numpy matplotlib scipy

This program uses the Colebrook equation to calculate the friction factor for a range of Reynolds numbers and relative roughness values. It then plots these values on a log-log scale to create the Moody diagram. The resulting plot is saved as an image file named `moody_diagram.png`. If you need any further assistance or modifications, feel free to ask! 😊'''