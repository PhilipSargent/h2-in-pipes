"""
Below is a Python program that uses the `matplotlib` and `numpy` libraries to plot a Moody diagram, which is a graph that shows the relationship between the Reynolds number, relative roughness, and the Darcy-Weisbach friction factor for fluid flow in pipes. The program will save the plot as an image file named `moody_diagram.png`."""

import functools
import numpy as np
import matplotlib.pyplot as plt
import virt_nik as vn
import pyfrac_yj as pf
import warnings

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

def logistic_transition(x, v1, v2, midpoint, steepness):
    """
    Computes the transition between two functions using a logistic curve.

    Args:
        x: Input values.
        func1: The first function.
        func2: The second function.
        midpoint: The midpoint of the transition.
        steepness: The steepness of the transition (higher value = steeper).

    Returns:
        The transitioned values at the input points.
    """

    y = (1 / (1 + np.exp(-steepness * (x - midpoint)))) * v2 + (1 - (1 / (1 + np.exp(-steepness * (x - midpoint))))) * v1
    return y


def d_func(func, reynolds, rr):
    """Differential of a function wrt to Re"""
    δ = reynolds/1e5
    e1 = func(reynolds-δ, rr)
    e2 = func(reynolds+δ, rr)
    
    return (e1-e2)/2*δ


# Define laminar flow (Re < 2000)
def laminar(reynolds):
    f_laminar = 64 / reynolds
    return f_laminar
    
def blasius(reynolds):
    f = (0.316 / (reynolds**0.25))
    if f > 0.005 and reynolds < 1e10 : # to get diagram the right shape
        return f
    
    return None

@memoize 
def gioia_chakraborty_friction_factor(Re, epsilon):
    """
    Calculates the friction factor using the Gioia and Chakraborty method.

    Args:
        Re: Reynolds number
        k: Roughness height
        D: Pipe diameter

    Returns:
        Friction factor
    """

    A = 30 + 8.8 * epsilon + 7.1 * epsilon**2 + 2.4 * epsilon**3
    B = 550 + 33 * epsilon**2

    f = 0.8 / (Re / A)**(1/3) * (1 + 12 / (Re / B)**2)**(-1/2)

    return f/4
    
@memoize 
def virtual_nikuradse(reynolds, relative_roughness):
    # Now using pyfrac code, which uses Fanning ff so is 4* too small
    # but while this fits Blasius end, the high-Re end is STILL 4x too small..
    
    # warnings.filterwarnings("error")
    
    sigma = 1/relative_roughness
 
    f =  4 * pf.FF_YangJoseph(reynolds, sigma)
    return f
    #This vn.vm DOES NOT WORK - my imperfect conversion from fortran not fixed yet:
    # return vn.vm(reynolds, sigma)

def smooth(reynolds):
    return colebrook(reynolds, 0.0)
    
def swarmee(Re, r):
    """Swarmee as quoted by Brackbill"""
    t1 = (64/Re)**8
    
    p5 = (2500/Re)**6
    
    j1 = r/3.7 + 5.74/Re**0.9
    
    j2 = np.log(j1)
    j3 = j2 - p5
    
    k1 = 9.5 * j3**-16
    
    f = ( t1 + k1)**(1/8)
    return f
    
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

        
    # Solve for the friction factor using the Newton-Raphson method ?
    f_solution = f_initial
    f_new = f_initial
    epsilon = 0.02
    for _ in range(20): # not Newton method, just iterate 20x
        if f_solution < 0:
            f_new = lowest_f
        else:
            # print(f"### {reynolds}, {relative_roughness} Fail to find, need to iterate ###")
            term = 2.51 / (reynolds * np.sqrt(f_solution))
            f_new -= f(f_solution, reynolds, relative_roughness) / (-0.5 * f_solution**(-1.5) -  term * (-0.5 * f_solution**(-1.5)))
        if abs((f_solution - f_new)/f_solution) < epsilon:
            return f_new
        f_solution = f_new
    return f_solution

@memoize 
def afzal_b(reynolds, relative_roughness):
    """Afzal but with Blasius upper bound
    
    Note that this is 'before' we introduce the Nikuradse transition to laminar, so
    we can have the beginning of the transition at a lower Re than one might expect"""
    Re_upper = 4500
    Re_lower = 1000

    a = afzal(reynolds, relative_roughness)
    if reynolds > Re_upper:
        return a
    b = blasius(reynolds)
    if reynolds < Re_lower:
        return b
    
    return np.sqrt(a*b)
    
@memoize 
def afzal_mod(reynolds, relative_roughness):
    """Afzal but with transiton to laminar"""
    Re_upper = 5000
    Re_lower = 1000
    
   
    a = afzal_b(reynolds, relative_roughness)
    

    if reynolds > Re_upper:
        return a

    L = laminar(reynolds)
    if reynolds < Re_lower:
        return L
    
    midpoint = 0.6 * (Re_upper - Re_lower)
    steepness = 0.01
    return logistic_transition(reynolds, L, a, midpoint, steepness)
    
@memoize 
def h2_ratio(reynolds, relative_roughness):
    re_ratio = 0.4103
    
    a = afzal_mod(reynolds, relative_roughness)
    h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    increase = h2/a # 100 * (h2-a)/a
    return increase

@memoize 
def p2_h2_ratio(reynolds, relative_roughness):
    """ rho_ratio * v_ratio**2  = 1.03512 """
    re_ratio = 0.4103
    
    v_ratio  = 3.076
    rho_ratio = 0.1094
    
    a = afzal_mod(reynolds, relative_roughness)
    h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    increase =  (h2/a) * rho_ratio * v_ratio**2 # 100 * ((h2-a)/a) * rho_ratio * v_ratio**2
    return increase

@memoize 
def w2_h2_ratio(reynolds, relative_roughness):
    """ rho_ratio * v_ratio**2  = 1.03512 
    Work = pressure_drop * v_ratio
    """
     
    v_ratio  = 3.076
        
    increase = p2_h2_ratio(reynolds, relative_roughness) * v_ratio
    return increase
    
@memoize 
def afzal_shift(reynolds, relative_roughness):
    re_ratio = 0.4103
    
    h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    return h2

@memoize 
def afzal(reynolds, relative_roughness):
    """Define the afzal variant fff equation as an implicit function and solve it
    10.1115/1.2375129
    https://www.researchgate.net/publication/238183949_Alternate_Scales_for_Turbulent_Flow_in_Transitional_Rough_Pipes_Universal_Log_Laws
    """
    # Initial guess for the friction factor
    f_initial = 0.02
    #f_initial = haarland(reynolds, relative_roughness) #fails 
    
    # Define the implicit afzal equation
    def f(f, re, rr):
        j = 11
        t = np.exp(-j*5.66 /(rr*re*np.sqrt(f)))
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(+2.51 / (re * np.sqrt(f)) + rr *t / 3.7 )

        
    # Solve for the friction factor using the Newton-Raphson method (?) 
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

# roughness range for piggot line
rr_piggot = np.logspace(-1.3, -6.1, 100) # roughness values between 0.01 and 0.00001

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
        # print("All None")
        return points # give up: a list entirely of None values
    
    # print(f"First OK at {i}")
    j = i
    while j < len(points) - 1:
        j = j+1
        if points[j]:
            i = j
            continue
        k, f = find_next_valid(j)  
        if not k:
           # print(f"All None from {j} to end ")
           return points # No more valid values left
        # print(f"at {j} invalid   found next valid {k} {f:.4f}")
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
    
    # print(f"Done all")
    return points

def piggot():
    """Calculate the Piggot points then re-order for plotting"""
    points = {}
    for rr in rr_piggot:
        if pp := piggot_point(rr):
            f, re = pp
            if f < 0.043:
                # do not ahve the Piggot lie too far above the roughest pipe
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

    
def plot_diagram(title, filename, plot="loglog", fff=colebrook, gradient=False, h2=False, w2=False):
    """Calculate the friction factor for each relative roughness,
    OK this does stuff several times and should be disentangled really
    
    fff : the friction factor function
    """
    if not type(fff) is list:
        fff = [fff]

    plt.figure(figsize=(10, 6))
    if moody_ylim:
        plt.ylim(0.004, 0.11)
    friction_laminars = [laminar(re) for re in reynolds_laminar]
    friction_smooth = [smooth(re) for re in reynolds]
    friction_blasius = [blasius(re) for re in reynolds]
    if not gradient and not h2:
        
        if plot == "loglog":
            plt.loglog(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
            plt.loglog(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
            if fp:
                plt.loglog(reynolds, fp, label='Piggot line', linestyle='dashdot')
        if plot == "linear":
            plt.plot(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
            plt.plot(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
            if fp:
                plt.plot(reynolds, fp, label='Piggot')

    for f in fff:
        friction_factors = {}
        for rr in relative_roughness_values:
            if gradient:
                friction_factors[rr] = [ d_func(f, re, rr) for re in reynolds]
                 
            else:
                friction_factors[rr] = [f(re, rr) for re in reynolds]
 
        # Plot the Moody diagram
        if plot == "loglog":
            for rr, ff in friction_factors.items():
                plt.loglog(reynolds, ff, label=f'ε/D = {rr}')
        if plot == "linear":
            for rr, ff in friction_factors.items():
                plt.plot(reynolds, ff, label=f'ε/D = {rr}')
        if plot == "linlog":
            for rr, ff in friction_factors.items():
                plt.semilogx(reynolds, ff, label=f'ε/D = {rr}')


    plt.xlabel('Reynolds number, Re')
    if not gradient:
        plt.ylabel('Darcy-Weisbach friction factor, f')
    else:
        plt.ylabel('d(f)/d(Re) Darcy-Weisbach friction factor gradient')
    if h2:
        #plt.ylim(-20,150)
        plt.xlabel('Reynolds number Re for natural gas')
        if fff==p2_h2_ratio:
            plt.ylabel('Pressure drop ratio ')
        else:
            plt.ylabel('Darcy-Weisbach friction factor ratio ')
    if w2:
        #plt.ylim(-80,500)
        plt.ylabel('Compressor power ratio ')
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(filename)

def export_f_table():
    """Produce a text file with f= f(Re)"""
    reynolds = np.logspace(2.4, 9.0, 200)

    rr = 1e-5
    with open('f_table.txt', 'w') as ff:
       ff.write(f"{'f':8},   {'Re':8} for rr = {rr:9.2f}\n") 
       for re in reynolds:
           f = afzal_mod(re, rr)
           ff.write(f"{f:8.3f}, {re:10.4f}\n") 
            
# - - -- - - - -- - - - -- - - - -- - - - -- - - - -- - - - -- - - - -- - - - -- - 
# Define the Reynolds number range and relative roughness values
# only need 10 points for the straight line

export_f_table()

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

moody_ylim = True

reynolds_laminar = np.logspace(2.9, 3.9, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(2.4, 9.0, 1000) # 10^7.7 = 5e7
relative_roughness_values = [0.01, 0.001, 0.0001, 0.00001,  0.000001] #
#relative_roughness_values = list(reversed(relative_roughness_values))
fp = piggot()

# plot_diagram('Moody Diagram (Colebrook)', 'moody_colebrook.png', plot="loglog")

moody_ylim = False
#plot_diagram('Moody Diagram (Afzal)', 'moody_afzal.png', plot="loglog", fff=afzal_mod)
plot_diagram('', 'moody_afzal.png', plot="loglog", fff=afzal_mod)

plot_diagram('factor increase in f between H2 and NG', 'h2_ratio.png', plot="linlog", fff=h2_ratio, h2=True)

#plot_diagram('factor increase in Pressure drop between H2 and NG', 'p2_h2_ratio.png', plot="linlog", fff=p2_h2_ratio, h2=True)
plot_diagram('', 'p2_h2_ratio.png', plot="linlog", fff=p2_h2_ratio, h2=True)

plot_diagram('factor increase in compressor work between H2 and NG', 'w2_h2_ratio.png', plot="linlog", fff=w2_h2_ratio, w2=True)

plot_diagram('Moody Diagram (Swarmee)', 'moody_swarmee.png', plot="loglog", fff=swarmee)
# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm.png', plot="loglog", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])

# Plot enlarged diagram

reynolds_laminar = np.logspace(2.9, 3.4, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(2.8, 5.0, 500) 
relative_roughness_values = [0.01, 0.003, 0.001, 1e-5]

# fp = piggot() # not in view on the enlarged plot
fp = None
plot_diagram('factor increase in f between H2 and NG', 'h2_ratio_enlarge.png', plot="linlog", fff=h2_ratio, h2=True)

# plot_diagram('Moody (Colebrook) Transition region', 'moody_colebrook_enlarge.png',plot="loglog")
plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge.png',plot="loglog", fff=[afzal_mod]) #, afzal_shift

plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_ll.png',plot="linlog", fff=[afzal_mod, afzal_shift])
plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_d_ll.png',plot="linlog", fff=[afzal_mod, afzal_shift], gradient=True)


# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm_enlarge.png', plot="loglog", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])

reynolds_laminar = np.logspace(2.9, 3.4, 500) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.0, 4.0, 500) 

plot_diagram('factor increase in f between H2 and NG', 'h2_ratio_enlarge_lin.png', plot="linear", fff=h2_ratio, h2=True)

# plot_diagram('Moody (Colebrook) Transition region', 'moody_colebrook_enlarge_lin.png',plot="linear")
plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_lin.png',plot="linear", fff=[afzal_mod]) #, afzal_shift
# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm_enlarge_lin.png', plot="linear", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])

exit()

re = 1e9
print(f"For high Re = {re:6.0e}")
for rr in [0.01, 0.001, 0.0001, 0.00001,  0.000001]:
    for fff in [colebrook, afzal, swarmee, virtual_nikuradse]:
        print(f"{fff.__name__:17} {rr:6} {fff(re, rr):.5f}")
    print("")

'''
Before running this program, ensure you have the required libraries installed. You can install them using pip:
pip install numpy matplotlib scipy

This program uses the Colebrook equation to calculate the friction factor for a range of Reynolds numbers and relative roughness values. It then plots these values on a log-log scale to create the Moody diagram. The resulting plot is saved as an image file named `moody_diagram.png`. If you need any further assistance or modifications, feel free to ask! 😊'''