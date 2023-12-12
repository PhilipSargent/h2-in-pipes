"""
Bing wrote this code with only one little bug:

Below is a Python program that uses the `matplotlib` and `numpy` libraries to plot a Moody diagram, which is a graph that shows the relationship between the Reynolds number, relative roughness, and the Darcy-Weisbach friction factor for fluid flow in pipes. The program will save the plot as an image file named `moody_diagram.png`."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define laminar flow (Re < 2000)
def laminar(reynolds):
    f_laminar = 64 / reynolds
    return f_laminar
    
def blasius(reynolds):
    if reynolds < 1e5:
        return (0.316 / (reynolds**0.25))
    else:
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

# Define the Colebrook equation as a function
def colebrook(reynolds, relative_roughness):
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
    for _ in range(20):
        f_new -= f(f_solution, reynolds, relative_roughness) / (-0.5 * f_solution**(-1.5) - 2.51 / (reynolds * np.sqrt(f_solution)) * (-0.5 * f_solution**(-1.5)))
        if abs((f_solution - f_new)/f_solution) < epsilon:
            return f_new
        f_solution = f_new
    return f_solution

def plot_diagram(filename='moody2_diagram.png'):
    # Calculate the friction factor for each relative roughness
    friction_factors = {}
    for rr in relative_roughness_values:
        friction_factors[rr] = [colebrook(re, rr) for re in reynolds]
        # [print(re, rr, haarland(re, rr)) for re in reynolds]


    friction_laminars = [laminar(re) for re in reynolds_laminar]
    friction_smooth = [smooth(re) for re in reynolds]
    friction_blasius = [blasius(re) for re in reynolds]
    
    # Plot the Moody diagram
    plt.figure(figsize=(10, 6))
    plt.loglog(reynolds_laminar, friction_laminars, label=f'Laminar')
    plt.loglog(reynolds, friction_smooth, label=f'Smooth: ε/D = 0')
    plt.loglog(reynolds, friction_blasius, label=f'Blasius')

    for rr, ff in friction_factors.items():
        plt.loglog(reynolds, ff, label=f'ε/D = {rr}')

    plt.xlabel('Reynolds number, Re')
    plt.ylabel('Darcy-Weisbach friction factor, f')
    plt.title('Moody Diagram')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(filename)
    #plt.savefig('moody_diagram.eps')
    
# Define the Reynolds number range and relative roughness values
reynolds_laminar = np.logspace(2.7, 4.4, 1000) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.4, 7.7, 1000) # 10^7.7 = 5e7
relative_roughness_values = [0.00003,  0.0001, 0.0003, 0.001, 0.003,  0.01, 0.03]

plot_diagram()

reynolds_laminar = np.logspace(2.7, 4.4, 1000) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.4, 4.7, 1000) 
relative_roughness_values = [0.0003, 0.001, 0.003,  0.01, 0.03]

plot_diagram('moody_enlarge.png')



# plt.show() # does not work as this is a non-interactive run of the program
'''
Before running this program, ensure you have the required libraries installed. You can install them using pip:
pip install numpy matplotlib scipy

This program uses the Colebrook equation to calculate the friction factor for a range of Reynolds numbers and relative roughness values. It then plots these values on a log-log scale to create the Moody diagram. The resulting plot is saved as an image file named `moody_diagram.png`. If you need any further assistance or modifications, feel free to ask! 😊'''