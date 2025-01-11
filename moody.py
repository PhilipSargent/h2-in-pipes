"""
Below is a Python program that uses the `matplotlib` and `numpy` libraries to plot a Moody diagram, which is a graph that shows the relationship between the Reynolds number, relative roughness, and the Darcy-Weisbach friction factor for fluid flow in pipes. The program will save the plot as an image file named `moody_diagram.png`.

TO DO:
 - Refactor the plotting routines
 - Factor out the boiler bits into a speratae .py file.
 - add Prantdl Karman PvK line to Moody plots
 - re-do the iterative solution for the Afzal equation properly.
 - Do work / new paper for function that matches Goldenberg equation: properly replace Colebrook & everyone.
 - Write a paper on oxygen enrched and pressurised condensers
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import pyfrac_yj as pf
import virt_nik as vn  # unused
from peng import (
    T273,
    Atm,
    R,
    colour,
    do_mm_rules,
    dzdp,
    get_density,
    get_Hc,
    get_v_ratio,
    get_viscosity,
    get_Vm,
    get_z,
    get_Δp_ratio_br,
    get_μ_ratio,
    get_ϱ_ratio,
    set_mix_rule,
    style,
)
from peng_utils import memoize

# We memoize some functions so that they do not get repeadtedly called with
# the same arguments. Yet still retain a more obvious way of writing the program.

global P, T
lowest_f = 0.000001
η = 0.02
b_exponent = (2+3*η)/(8+3*η)
ib_factor = 0.316 *   3e3**(b_exponent- 0.25)

T8C = T273 + 3 # 3 degrees C
T = T8C # default temp
P = Atm + 40/1000 # 40 mbar default pressure

# roughness range for piggot line
rr_piggot = np.logspace(-1.3, -9.1, 100) # roughness values between 0.01 and 0.00001

# Set the default mixture rule for viscosities in a mixed gas
visc_f = set_mix_rule()

T50C = T273 + 50 # K
T25C = T273 + 25 # K
T15C = T273 + 15 # K
T8C = T273 + 8 # K
T3C = T273 + 3 # K
T250 = T273 -20 #  -20 C
T230 = T273 -40 #  -40 C
    
# print(f"Intermittent-Blasius factor: {ib_factor:.3f}")

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
@memoize 
def laminar(reynolds):
    f_laminar = 64 / reynolds
    return f_laminar
    
@memoize 
def blasius(reynolds):
    f = (0.316 / (reynolds**0.25))
    if f > 0.005 and reynolds < 1e10 : # to get diagram the right shape
        return f
    
    return None
    
@memoize 
def iblasius(reynolds):
    # with intermittency correction
    f = (ib_factor / (reynolds**b_exponent))
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

    a = afzal_basic(reynolds, relative_roughness)
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
def get_re_ratio(g, p, T):
    #Re = density . v . D / viscosity
    # All the ratio funcitons are for the value for g divided by the value for NG
    v_ratio = get_v_ratio(g, P, T)
    μ_ratio = get_μ_ratio(g, P, T, visc_f)
    ϱ_ratio = get_ϱ_ratio(g, P, T)
    re_ratio = ϱ_ratio * v_ratio / μ_ratio
    print(f"{T=:3.0f}K ({T-T273:5.1f}°C) {P=:8.4f} {v_ratio=:.4f} {μ_ratio=:.4f} {ϱ_ratio=:.4f}  {re_ratio=:.4f}  {visc_f.__name__}")
    return re_ratio
    
#@memoize 
def h2_ratio(reynolds, relative_roughness):
    # Just the friction factor, but still needs re_ratio
    global P, T
    re_ratio = get_re_ratio('H2', P, T)
    # re_ratio = 0.4103
    
    a = afzal_mod(reynolds, relative_roughness)
    h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    increase = h2/a # 100 * (h2-a)/a
    return increase

#@memoize # not memoize when using P and T globally
def p2_h2_ratio(reynolds, relative_roughness):
    """ rho_ratio * v_ratio**2  = 1.03512 at 1atm
    For other temperatures and pressures we must call the Peng-Robinson EOS
    functions.
    """
    global P, T
    g='H2'
    Δp_ratio = get_Δp_ratio_br(g, P, T) # just for Blasius, do NOT use
    re_ratio = get_re_ratio(g, P, T)
    
    
    v_ratio  = get_v_ratio(g, P, T) #3.076 # this includes the boielr efficiency number
    rho_ratio = get_ϱ_ratio(g, P, T) # 0.1094
    
    f = afzal_mod(reynolds, relative_roughness)
    f_h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    increase =  (f_h2/f) * rho_ratio * v_ratio**2 
    return increase

#@memoize 
def w2_h2_ratio(reynolds, relative_roughness):
    """ rho_ratio * v_ratio**2  = 1.03512 
    Work = pressure_drop * v_ratio
    """
    global P, T
     
    v_ratio  = get_v_ratio('H2', P, T) #3.076 # this includes the boielr efficiency number
        
    increase = p2_h2_ratio(reynolds, relative_roughness) * v_ratio
    return increase
    
@memoize 
def afzal_shift(reynolds, relative_roughness):
    global P, T
    re_ratio = get_re_ratio('H2',P,T)
    
    h2 = afzal_mod(reynolds*re_ratio, relative_roughness)
    
    return h2

@memoize 
def afzal_basic(reynolds, relative_roughness):
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
                # do not have the Piggot lie too far above the roughest pipe
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

def plot_pt_diagram(title, filename, plot="loglog", fff=colebrook, gradient=False,  w2=False):
    # Derived from plot_diagram(), both need refactoring
    global P, T
    rr = 1e-7
    title = title + f" (ε/D = {rr})"
    if type(fff) is not list:
        fff = [fff]
    plt.figure(figsize=(10, 6))
    
    for f in fff: # several different friction factor functions
        friction_factors = {}
        
        for pt in [(1,50), (30,8), (74,8), (150,-40)]:
            P = pt[0]
            t = pt[1]
            T = T273+t
            label =  f" ({T-T273:.0f}°C, {P:.0f} bar)"
            # Calculate the curves 
            if gradient:
                friction_factors[rr] = [ d_func(f, re, rr) for re in reynolds]
            else:
                friction_factors[rr] = [f(re, rr) for re in reynolds]
            # Plot the calculated curves on the Moody diagram
            if plot == "loglog":
                for rr, ff in friction_factors.items():
                    plt.loglog(reynolds, ff, label=label)
            if plot == "linear":
                for rr, ff in friction_factors.items():
                    plt.plot(reynolds, ff, label=label)
            if plot == "linlog":
                for rr, ff in friction_factors.items():
                    plt.semilogx(reynolds, ff, label=label)

        plt.xlabel('Reynolds number, Re')
        if not gradient:
            plt.ylabel('Darcy-Weisbach friction factor $f$')
        else:
            plt.ylabel('$d(f)/d(Re)$ Darcy-Weisbach friction factor gradient')
        if not w2:
            plt.xlabel('Reynolds number Re for natural gas')
            if f == p2_h2_ratio:
                plt.ylabel('Pressure gradient ratio ')
            else:
                plt.ylabel('Darcy-Weisbach friction factor ratio ')
        else:
            #plt.ylim(-80,500)
            plt.ylabel('Friction loss power ratio ')
        plt.title(title)
        plt.grid(True, which='both', ls='--')
        plt.legend()
        plt.savefig(filename)
        plt.close()

@memoize
def kg_from_GW(g, Q_GW):
    """Input Qh is GW  (GJ/s) of combustion energy in HHV per second."""
    T25C = T273 + 25
    _, _, hc = get_Hc(g, T25C) # MJ/mol HHV always at 25 C
    Q_moles = Q_GW *1e3 / hc #  (MJ/s) / (MJ/mol) => mol/s
    
    m = do_mm_rules(g) # (g/mol)
    Q_kg = Q_moles * m * 1e-3 # (mol/s) * (g/mol) * 1e-3 => kg/s
    
    # check
    Q_ck = GW_from_kg(g, Q_kg)
    ck = Q_ck - Q_GW
    if abs(ck) > 1e-12:
         print(f"***{g:7} {Q_GW=:9.2f} [{ck=}]")
    #print(f"-- {g:7}{Q_GW=:10.4f} GW Q_moles={Q_moles*1e-3:8.4f} kmol/s  {Q_kg=:9.5f} kg/s  ")
    
    return Q_kg

@memoize
def GW_from_kg(g, Qg):
    # Qg is in kg/s
    T25C = T273 + 25
    
    # calc mol/s from kg/s
    m = do_mm_rules(g) # in g/m3, convert to kg/m3:
    Qm = Qg / (m * 1e-3) # mol/s

    _, _, hc = get_Hc(g, T25C) # MJ/mol HHV always at 25 C
    
    Q = Qm * hc #  (mol/s) *  (MJ/mol) => MJ/s i.e. MW
    Qh = Q * 1e-3 # MW * 1e-3 => GW
    return Qh


@memoize
def get_v_from_Q(g, Tt, Pp, Qh, D):
    # Qh is in GW - what whatever gas g is.
    Qg = kg_from_GW(g, Qh)
    ϱ = get_density(g, Pp, Tt) 
    Qv = Qg / ϱ # in m³/s
    A = get_A(D)
    v = Qv / A # m/s
    #print(f"-- {g:7}  {P=:9.1f} bar {v=:9.5f} m/s  {ϱ=:9.5f} kg/m³ {Qg=:9.5f} kg/s {Qv=:9.5f} m³/s")
    return v

@memoize
def get_Q_from_v(g, Tt, Pp, v, D):
    # v (m/s)
    # Q (GW)
    #print(f"***{g:7}  get_Q_from_v {v=:9.2f} m/s [{Tt=}] {D=}")  
    if Pp < 0 :
        print(f"***{g:7}  get_Q_from_v {v=:9.2f} m/s [{Tt=}] {D=}")    
    A = get_A(D)
    ϱ = get_density(g, Pp, Tt) 
    Qv = v * A
    Qg = Qv *  ϱ
    Qh = GW_from_kg(g, Qg)
    
    v_ck = get_v_from_Q(g, Tt, Pp, Qh, D)
    ck = v_ck - v
    if abs(ck) > 1e-12:
         print(f"***{g:7} {v=:9.2f} m/s [{ck=}]")    
    return Qh # GW
    
@memoize
def get_A(D):
    return np.pi * D**2 / 4

@memoize
def funct_B_sub(g, Qg, D):
    """When doing calculations with B we need to use the gas constant in terms of P
    and cubic metres, not in terms of bar and litres, because we are mixing up kg and km.m/s² to make the units work.
    
    Qg (kg/s)
    D  (m)
    
    Must return value in Pa² / K.m
    
    P* = ( Q / A ) * sqrt ( R T / m) in units of Pascals
    """
    Rg = 8.31446261815324 # J/K.mol # R = 0.083144626  # l.bar/(mol.K) 
    A = get_A(D) # m²
    m = do_mm_rules(g) # g/mol
    m_kg = m * 1e-3 # kg/mol
    Bs = np.power(Qg/A,2) * Rg  / m_kg
    #print(f".. {g:7} B/T: {Bs:9.4f} {m=:8.4f} g/mol {A=:9.5} m²  {D=} m  {Qg=:9.5f} kg/s")
    return Bs
    
@memoize
def funct_B(g, Qg, T, D):
    # B = Q² R T / A² m # Pa² / m
    B = T * funct_B_sub(g, Qg, D) # Pascals-squared
    #sqrtB = np.sqrt(B)/1e5        # bar
    #print(f"-- {g:7} {T=:7.2f} K  ({T-T273:5.1f}°C) B: {B:10.4e} sqrt(B):{sqrtB:9.4f} bar²")
    return B
    
@memoize
def p_star(g, Qg, T, D):
    # B = Q² R T / A² m # Pa²
    # B = P*²
    B = funct_B(g, Qg, T, D) # Pa²
    p_star = np.sqrt(B)/1e5        # Pa -> bar
    return p_star
    
@memoize
def p_star_v(g, v, T):
    """Calculate P* but in terms of gas velocity, not in terms of mass flow rate per cross-section area.
    We do not lose any generality by simply setting D = 1 metre and P = 1 atm
    As P* is related to the speed of sound, it only depends on T and not on P or anything else.
    """
    D = 1
    A = get_A(D)
    P = Atm
    # Qg = mass flow rate = v . density . A  = m/s * kg/m^3 * m²
    ϱ = get_density(g, P, T)
    Qg = v * ϱ * A
    return p_star(g, Qg, T, D)

@memoize
def running_dp38(L, g, T, P0, f_function, rr, D, Qh):
    """Does the integral from x: 0 to L of all the dP/dx to get P = P(L)
    This understands x i.e. L
    This steps in 1km steps from zero to L, and adds up the pressure gradients.
    L (m)
    T (K)
    P0 (bar)
    D (m)
    Qh (GW)
    
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    scipy.integrate.quad(func, a, b, args=())
    Integrate func from x=[a to b]   QUADPACK.
    func(x,..)
    """
    def dpdx(x,p):
        # Have to do this because I can't find the args=() documentation for scipy
        if type(p) is not float:
            p = p[0] # scipy likes to set this to a numpty.ndarray of a numpy.float64 value instead of a simple float
        if p < 0:
            #print(f"negative P. Abort. def dpdx(x,p) {T} {P0} {x} {p}")
            #exit(-1) 
            p = 1
        gradient = dp38(x, p, g, T, f_function, rr, D, Qh)
        return gradient
        
    p = P0 # 84 bar, for Yamal pipe
    
    # Initial condition p = [P0]
    x_span = (0, L)
    # Solve the ODE
    solution = solve_ivp(dpdx, x_span, [P0]) # has to be a list for last variable

    #print(solution.t)
    #print(solution.y)
    p = solution.y[0][-1]
    grad = dpdx(L,[p])
    #print(f"{int(L/1000):5} km  ({p:5.1f} bar) P0-p:{p-P0:8.3f} bar  gradient:{grad*1000:9.5f} bar/km")
    
    if p < 0:
        print(f"**** Negative pressure - pressure drop too large. {L=} {g=} {T=} {P0=} {f_function=} {rr=} {D=} {Qh=}")
        exit(-1)
    return p
   
@memoize
def get_Re(g, T, P, Qh, D):
    if P < 0:
        print(f"negative P. Abort.  get_Re(g, T, P, Qh, D): {T=} {P=}")
        exit(-1)    
    ϱ = get_density(g, P, T) # kg/m³
    v = get_v_from_Q(g, T, P, Qh, D)
    μ = get_viscosity(g, P, T, visc_f) #in μPa.s 
    μ = μ * 1e-6 # in Pa.s
    Re = ϱ * v * D / μ # dimensionless as Pa ≡ N/m² and kg.m/s² ≡ N
    return Re

def eq36(g, T, P, B, ff, Z, D):
    r""" Equation 36 {Sargent2024b}
    \begin{equation}
    \label{eqn:force9a}
    \frac{\delta P}{\delta x} =     \frac{Z f}{2 D \left[  \frac{Z}{P} - \frac{P}{B} -  \left( \frac{\partial Z}{\partial P} \right)_T   \right]}
    \end{equation}
    
    Yamal data
    
    In all this we need to be careful with units: all in SI m³ and N and Pa,
    not litres or bars or g/mol.
    
    TO BE DEBUGGED !!
    
    T (K)
    P (Pa)
    B (Pa^-2)
    ff (-)
    Z (-)
    D (m)
    Qh (GW)
    """
    # print (f"-- {g:7}  {ff=:.3e} {Re=:0.3e}") #  {ϱ=:8.4f}  {v=:0.3f} m/s {μ=:8.2e} Pa.s")
    nom = Z * ff / (2 * D) # units (1/m)
   
    # g  =  -B * ff * Z  / (2 * D * P) #  Pa /m
    # return g  #  Pa /m    
    dZdP = dzdp(T, P*1e-5, g)     # P must be in bar, otherzie Z is completely wrong.
    dZdP = dZdP *1e-5 # convert to Pa^-1
    denom = (Z/P) - (P/B) - dZdP 
    denom = (Z/P) - (P/B) 
    gr = nom /denom # Pa/m
    #print (f"-- {g:7} Z={Z:8.4f} {P=:.0f} Pa {denom=:0.5f}  (Z/P):{(Z/P)*1e5:0.5f}  -(P/B):{-(P/B)*1e5:0.5f} -dZdP:{-dZdP*1e5:0.5f} (Pa^-1) ")    
    # g  =  -B * ff * Z  / (2 * D * P) #  Pa /m
    return gr  #  Pa /m


def eq38(B, ff, Z, D, P):
    g  =  -B * ff * Z  / (2 * D * P) #  Pa /m
    return g  #  Pa /m

@memoize
def dp38(x, P, g, T, f_function, rr, D, Qh):
    """This is the 'near-ideal gas' version of the equation (38)
    where the inertial term and 
    the dZ/dP terms are omitted.
    
    This calculates the dP/dx for given P, T, Qh. It does not understand 'x'.
    But it seems to have to have it so that the quad() integration works

    input variable Qh is in GW of Yamal gas, 
    so we need to convert this to Q (kg/s) of whatever gas we are plotting
    and calculate the gas velocity from that too.
    
    In all this we need to be careful with units: all in SI m³ and N and Pa,
    not litres or bars or g/mol.
    
    T (K)
    P (bar)
    D (m)
    Qh (GW)
    
    dP/dx = - B f Z / (2 D P) """
        
    if P < 0:
        print(f"negative P. Abort. dp38(x, P, g, T, f_function, rr, D, Qh) {T=} {P=}")
        exit(-1)    
    
    Qg = kg_from_GW(g, Qh) # kg/s
    B = funct_B(g, Qg, T, D) # Pa² 

    # dimensionless
    Z = get_z(g, P, T)
    Re =  get_Re(g, T, P, Qh, D)
    ff = f_function(Re, rr) # afzal(reynolds, relative_roughness), dimensionless
    # print (f"--{g:7}  {ff=:.3e} {Re=:0.3e}  {ϱ=:8.4f}  {Z=:0.5f}  {v=:0.3f} m/s {μ=:8.2e} Pa.s")
    
    P = P *1e5 # convert bar to Pa
    # equation 38 or 36:
    #print (f"][ {g:7} ({T-T273:5.1f}°C)  P:{P/1e5:8.4f} bar  dZ/dp={dzdp(T, P, g):8.4f} Pa/m")

    #gradient  = eq38(B, ff, Z, D, P) #  Pa /m
    gradient  = eq36(g, T, P, B, ff, Z, D) #  Pa /m
    gradient = gradient * 1e-5 # bar/m
    #print (f"--{g:7} P={P/1e5:0.5f} bar  {B=:0.5e} Pa² gradient={gradient*1000:0.5e} bar/km")
    return gradient # bar/m

def plot_p_star(): 
    """Plot P* for NG and H2 for a range of temperatures
    """
         
    temperatures = np.linspace(233.15, 323.15, 100)  
    plt.figure(figsize=(10, 5))
    
    for g in ['NG', 'H2']:
        for v in [20, 10, 1, 0.1]:
            label=f"{v:4.1f} m/s"
            txt = f"{g} {label}"
            pstar_v = [p_star_v(g, v, t) for t in temperatures]
            
            # print(f"P* {g:7} at  {v:4.1f} m/s")
            plt.semilogy(temperatures - T273, pstar_v,  label=txt, **plot_kwargs(g))

    plt.title(f'P* as a function of temperature for several gas velocities')
    plt.xlabel('Gas temperature (°C)')
    plt.ylabel('P* = ( Q / A ) * sqrt ( R T / m) (bar)')
    plt.legend()
    plt.grid(True)

    plt.savefig("pipe_pstar.png")
    plt.close()
#@memoize
def LPM(g, T, P_ent, L_seg, D, Qh, rr=1e-5, f_function=afzal_mod):
    """LinePackMetric for a gas in a pipe of length L (m), diameter D (m), carrying Qh (GW) of gas
    But perhaps GW is the wrong thing to keep constant, maybe all for same gas velocity ? 
    At each pressure, maintaining the same MJ/s for Ng at 5 m/s ?
    """
    def energy_in_pipe(Pp):
        v = get_v_from_Q(g, T, Pp, Qh, D)
        ϱ = get_density(g, Pp, T) 
        ϱv = ϱ * v
        
        Vm = get_Vm(g, Pp, T) # m³/mol - molar volume at P
        hhvv = hc_g / Vm # MJ/m³ # HHV per cubic metre
        E_seg = V_seg * hhvv *1e6 # J = Ws = amount of gas when linepacked 100%
        #print(f"++ {g:5} ({T-T273:5.1f}°C) {P=:5.1f} {ϱ=:8.4f} kg/m³  {v=:8.4f} m/s  {ϱv=:8.3f} kg/s.m² {hhvv=:8.3f} MJ/m³ Vm{Vm*1000:8.3f} m³/kmol")
        return E_seg
        
    A = get_A(D)
    V_seg = L_seg * A # volume
    _, _, hc_g = get_Hc(g, T) # MJ/mol 

    #P_exit = running_dp38(L_seg, g, T, P_ent, f_function, rr, D, Qh)
    P_exit = get_final_pressure(g, T, P_ent, L_seg, D, Qh)
    if P_exit < 0:
        print(f"## Negative exit pressure. {g} {T} {P} {L_seg}")
    v_ent = get_v_from_Q(g, T, P_ent, Qh, D)
    #print(f"++ {g:5} ({T-T273:5.1f}°C)   L={L_seg/1e3:5.1f}km {v_ent=:5.1f} {P_ent=:5.1f} {P_exit=:5.1f} {P_ent-P_exit=:7.3f} bar")

    E_ent = energy_in_pipe(P_ent)*1e-9 # GJ =GWs
    E_exit = energy_in_pipe(P_exit)*1e-9 # GJ = GWs
    
    # Now subtract the amount of gas which is in the pipe under normal flow
    # But remember, the P/x curve is almost linear, so the line[ack triangle
    # is half the difference.
    
    lpm = 0.5* (E_ent-E_exit)/(Qh) # GWs / GW
    #lpm = (E_ent)/(Qh) # GWs / GW

    lpm_km = lpm/(L_seg * 1e-3)
    #print(f"+++{g:5} ({T-T273:5.1f}°C) {P_ent=:7.1f} bar  E_ent={E_ent:8.0f} E_exit={E_exit:8.0f} lpm={lpm:8.1e} s  ={lpm/3600:7.3f} hours  lpm/km={lpm_km:8.1f} s/km")
    # print(f"+++{g:5} ({T-T273:5.1f}°C) {P=:7.1f} bar V_seg={V_seg/1e3:9.4f} 1000.m³ lpm={lpm:8.1e} s  ={lpm/3600:7.3f} hours  lpm/km={lpm_km:8.1f} s/km")

    return lpm # in seconds

def plot_lpm(L_seg = 12): # L_seg (km)
    """Plot LPM  for pure hydrogen and natural gases
    Standardise on 4 m/s for NG, and equivalent energy velocity for H2
    """
    def get_Q_NG(p, T):
        # Normalises to same Q (GW) as for NG v=4m/s at this T,P
        v_ng = 4 # m/s
        Q_NG = get_Q_from_v("NG", T, p, v_ng, D)
        
        return Q_NG
         
    pressures = np.linspace(8, 80, 50)  # bar
    plt.figure(figsize=(10, 5))
    
    for g in ['NG', 'H2']:
          # km
        Lm = L_seg * 1e3 # m
        D = 0.2 # 200 mm
        v = 4 # m/s
        #for T in [ T8C]:
        for T in [T230, T8C, T25C, T50C]:
            # for p in pressures:
                # v_gas = get_v_from_Q(g, T, p, get_Q(g,p), D)
                # print(f"|| {g:7} {v_gas=:9.4f} m/s")
            label=f"{T-T273:4.0f}°C"
            txt = f"{g} {label}"
            p_lpm = [LPM(g, T, p, Lm, D, get_Q_NG(p, T)) for p in pressures]
            
            print(f"Linepack Metric {g:7} at  {T-T273:4.1f}°C")
            plt.plot(pressures, p_lpm,  label=txt, **plot_kwargs(g))

    plt.title(f'Linepack Metric - for NG v ={v:2.0f} m/s ({L_seg} km, {D*1000:.0f} mm dia.)')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Linepack Metric (seconds)')
    plt.legend()
    plt.grid(True)

    plt.savefig("pipe_lpm.png")
    plt.close()
    
    # This is eqn(38) direct calculated differential curve
    # COPY FOR DEBUGGING LPM
    output="pipeLPM"
    f = afzal_mod
    
    plt.figure(figsize=(10, 6))
    fn = f"{f.__name__}"
    #title = title_in + f" (ε/D = {rr0} [{fn}])"
    filename = output + "_p_" + fn + ".png"
    x_range = np.linspace(1, L_seg*1000-1, 50) # 500 km

    
    p_final ={}
    for g in ['Yamal', 'H2']:
        p_final[g] = {}
        for T in [T230, T8C, T25C, T50C]:
            lab =  f" ({T-T273:.1f}°C)"

            p_x = {}
            label = f"{g:7}" + lab
            # print(label)
            p_x[T] = [running_dp38(x, g, T, 30, f, 1e-5, D, get_Q_NG(30, T)) for x in x_range] # bar
            plotit(g, p_x, "linear", f, label, x_range)
            #p_final[g][T] =   p_x[T][-1] # printed at end of program
    plt.ylabel('Pressure (bar)')
    saveit(f'Pressure - for NG v ={v:2.0f} m/s ({L_seg} km, {D*1000:.0f} mm dia.)',filename)
    # END OF COPY FOR DEBUGGING LPM


    print(f"Now plot the ratio")
    # Plot LPM  RATIO hydrogen : natural gases
    if True:
        pressures = np.linspace(6, 80, 50)  # bar
        plt.figure(figsize=(10, 5))
        for g in ['NG']:
            Lm = L_seg * 1e3 # m
            D = 0.2 # 200 mm
            #for T in [ T8C]:
            for T in [ T25C, T8C, T273]:
                label=f"{T-T273:4.0f}°C"
                txt = f"{g} {label}"
                p_lpm = [LPM('H2', T, p, Lm, D, get_Q_NG(p,T))/LPM(g, T, p, Lm, D, get_Q_NG(p,T))  for p in pressures]
                plt.plot(pressures, p_lpm,  label=label, **plot_kwargs(g))

        plt.title(f'Linepack Metric Ratio H2/NG - for NG v ={v:2.0f} m/s ({L_seg} km, {D*1000:.0f} mm dia.)')
        plt.xlabel('Pressure (bar)')
        plt.ylabel(f'Linepack Metric ratio H2/NG')
        plt.legend()
        plt.grid(True)

        plt.savefig("pipe_lpm_ratio.png")
        plt.close()
        
def get_final_pressure(g, T, P_zero, L_pipe, D, Qh, rr=1e-5, f=afzal_mod):
    """     Pfinal = SQRT (P0² - B f Z L/D )
    L_pipe (m)
    P_zero (bar)
    """
    def estimate(g, T, P_zero, P_mean):
        
        Qg = kg_from_GW(g, Qh)
        B = funct_B(g, Qg, T, D)
        Z = get_z(g, P_mean, T)
        Re =  get_Re(g, T, P_mean, Qh, D)
        ff = f(Re, rr)
        t1 = (P_zero*1e5)**2
        t2 = B * ff * Z * L_pipe / D
        if t1 > t2 + 1:
            fp = np.sqrt(t1 - t2 ) # Pa
        else:
            fp = 1
            print(f"## Floor P=1 bar  {g:7} {T-T273:4.0f}°C {P_zero=:8.3f} {P_mean=:8.3f} {t1:.3e} {t2:.3e}")
        return fp * 1e-5 # bar
        
    fp1 = estimate(g, T, P_zero, P_zero)
    mean_P = (P_zero + fp1)/2
    rms_P = np.sqrt((P_zero**2 + fp1**2)/2)
    geo_P = np.sqrt(P_zero * fp1)
    fp2 = estimate(g, T, P_zero, mean_P)
    #print(g, P_zero, fp1, fp2, rms_P, mean_P, geo_P)
    return fp2
        
def plotit(g, p_x, plot, ff, label, x_range):
    # Plot the calculated curves 
    if plot == "loglog":
        for rr, p in p_x.items():
            plt.loglog(x_range/1000, ff, label=label, **plot_kwargs(g))
    if plot == "linear":
        for rr, ff in p_x.items():
            plt.plot(x_range/1000, ff, label=label, **plot_kwargs(g))
    if plot == "linlog":
        for rr, ff in p_x.items():
            plt.semilogx(x_range/1000, ff, label=label, **plot_kwargs(g))

def saveit(title,filename):
    plt.title(title)
    plt.xlabel('Distance (km)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    # print(f"*** {filename} ***")
        
def plot_pipeline(title_in, output, plot="linear", fff=afzal_mod):
    # Derived from plot_pt_diagram(), all need refactoring
    
    # Be Careful. Many variables rely on Python scoping rules between functions and
    # included functions. These rules are implicit.
    global P, T


        
    def print_finals():
        for g in ['Yamal', 'H2']:
            for t in t_range:
                T = T273+t
                pf_i = p_final[g][T]
                pf = get_final_pressure(g, T, P0, L_pipe, D, Qh, rr0, f)
                error = 100*(pf_i - pf)/pf_i
                print(f"   {g:7} ({T-T273:5.1f}°C) P_final:{pf_i:5.1f} bar eqn39 estimate:{pf:5.1f} bar {error=:5.2f} %")
        for t in t_range:
            T = T273+t
            ratio = (84 - p_final['H2'][T])/(84 - p_final['Yamal'][T])
            print(f"   {'':7} ({T-T273:5.1f}°C) pressure drop ratio:{ratio:8.4f} H2/Yamal {100*(ratio-1):6.2f} % greater")
        if False:
            for t in [-40, -30, -10, 0, 8, 20, 30, 42.5, 50]:
                T = T273+t
                for Lkm in [10, 20, 100, 200, 800 ]: # km
                    Length = Lkm*1e3 # m
                    p_H2 = get_final_pressure('H2', T, P0, Length, D, Qh, rr0, f)
                    p_NG = get_final_pressure('Yamal', T, P0, Length, D, Qh,rr0, f)
                    ratio = (84 - p_H2)/(84 - p_NG)
                    print(f"   {Lkm:5.0f}km  ({T-T273:5.1f}°C) r:{ratio:8.4f} H2/Yamal +{100*(ratio-1):6.2f} % {p_H2:8.4f} {p_NG:8.4f}")

               

    t_range = [42.5, 8, -40]
    D = 1.3836 # m
    A = np.pi*D**2/4
    rr0 = 2.2e-5 # Yamal
    P0 = 84 # bar Yamal
    # 33 billion m³(STP) / year says Gazprom, but maybe not for this bit.
    Qstp = 2019950 / 3600 # m³(STP) /hour => 561 m³(STP)/s 
    print(f"\nYAMAL PIPELINE Qv={Qstp:9.4f} m³(STP) /s {fff}")
    ϱstp = get_density('Yamal', Atm, T273+15) # STP about 0.7 kg/m³
    ϱ = get_density('Yamal', P0, T273+42.5) # kg/m³, about 58 kg/m³
    Qv = Qstp * ϱstp / ϱ  # m³/s  actually at 84 bar
    Vmstp = get_Vm('Yamal',Atm, T273+15) # about 23 m³
    Qmoles = Qstp / Vmstp
    print(f"YAMAL PIPELINE Vm={Vmstp:9.4f} m³(STP) /mol   Qm={Qmoles*1e-3:9.4f} kmoles/s")
  
    Qg = Qv * ϱ # m³/s * kg/m³  => kg/s
    # Qg = 390.63 # kg /s Yamal gas
    Qh = GW_from_kg('Yamal', Qg)
    # Qh = 21.1851 # GW = 10^3 MJ/s - use this as baseline and convert to Q for each gas
    v = get_v_from_Q('Yamal', T273+42.5, P0, Qh, D)
    
    print(f"=> {Qv=:9.4f} m³/s  {ϱ=:9.4f} kg/m³ at {P0} bar and 42.5 C (ϱ(STP):{ϱstp:9.4f} kg/m³)")
    print(f"=> {Qg=:9.4f} kg/s {Qh=:9.4f} GW {v=:9.4f} m/s of Yamal gas.")
    L_pipe = 800e3
    x_range = np.linspace(1, L_pipe-1000, 50) # 500 km


    if type(fff) is not list:
        fff = [fff]
    
    for f in fff: # several different friction factor functions
        plt.figure(figsize=(10, 5))
        
        # This is eqn(38) direct calculated differential curve
        plt.figure(figsize=(10, 5))
        fn = f"{f.__name__}"
        title = title_in + f" (ε/D = {rr0} [{fn}])"
        filename = output + "_p_" + fn + ".png"
        
        p_final ={}
        for g in ['Yamal', 'H2']:
            p_final[g] = {}
            for t in t_range:
                T = T273+t
                lab =  f" ({T-T273:.1f}°C)"

                p_x = {}
                label = f"{g:7}" + lab
                # print(label)
                p_x[T] = [running_dp38(x, g, T, P0, f, rr0, D, Qh) for x in x_range] # bar
                plotit(g, p_x, plot, f, label, x_range)
                p_final[g][T] =   p_x[T][-1] # printed at end of program
        plt.ylabel('Pressure (bar)')
        saveit( title,filename)

        # Now take the results from the last run and re-calculate the gradient at each point
        plt.figure(figsize=(10, 5))
        fn = f"{f.__name__}"
        title = title_in + f" (ε/D = {rr0} [{fn}])"
        filename = output + "_g_" + fn + ".png"
        
        for t in t_range:
            T = T273+t
            lab =  f" ({T-T273:.1f}°C)"

            g_x = {}
            for g in ['Yamal', 'H2']:
                label = f"{g:7}" + lab
                # print(label)
                g_x[T] = [dp38(x, running_dp38(x, g, T, P0, f, rr0, D, Qh), g, T, f, rr0, D, Qh)*1000  for x in x_range] # bar
                plotit(g, g_x, plot, f, label, x_range)

        plt.ylabel('Pressure gradient (bar/km)')
        saveit(title,filename)
        
        # Now take the results from the last run and calculate the velocity
        plt.figure(figsize=(10, 5))
        fn = f"{f.__name__}"
        title = f"Velocity of gas along pipeline  (ε/D = {rr0} [{fn}])"
        filename = output + "_v_" + fn + ".png"
        
        for t in t_range:
        #for t in [8]:
            T = T273+t
            lab =  f" ({T-T273:.1f}°C)"

            v_x = {}
            for g in ['Yamal', 'H2']:
                label = f"{g:7}" + lab
                lpm = LPM(g, T, P0, 139*1e3, D, Qh) # Qh (GW) L (m)
                v_x[T] = [get_v_from_Q(g, T, running_dp38(x, g, T, P0, f, rr0, D, Qh), Qh, D)  for x in x_range] # bar
                plotit(g, v_x, plot, f, label, x_range)

        plt.ylabel('Gas velocity (m/s)')
        saveit(title,filename)

        
        print_finals()
 
        
def plot_diagram(title, filename, plot="loglog", fff=colebrook, gradient=False, h2=False, w2=False):
    """Calculate the friction factor for each relative roughness,
    OK this does stuff several times and should be disentangled and refactored
    
    fff : the friction factor function
    """
    global P, T
    if title:
        title = title + f" ({T-T273:.0f}°C, {P:.0f} bar)"
    if type(fff) is not list:
        fff = [fff]

    plt.figure(figsize=(10, 6))
    if moody_ylim:
        plt.ylim(0.004, 0.11)
    friction_laminars = [laminar(re) for re in reynolds_laminar]
    friction_smooth = [smooth(re) for re in reynolds]
    friction_blasius = [blasius(re) for re in reynolds]
    friction_iblasius = [iblasius(re) for re in reynolds]
    if not gradient and not h2:
        # These are the theory lines for restricted theories, shoudl add PvK here
        
        if plot == "loglog":
            plt.loglog(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
            plt.loglog(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
            plt.loglog(reynolds, friction_iblasius, label=f'i-Blasius', linestyle='dashed')
            if fp:
                plt.loglog(reynolds, fp, label='Piggot line', linestyle='dashdot')
        if plot == "linear":
            plt.plot(reynolds_laminar, friction_laminars, label=f'Laminar', linestyle='dotted')
            plt.plot(reynolds, friction_blasius, label=f'Blasius', linestyle='dashed')
            plt.plot(reynolds, friction_iblasius, label=f'i-Blasius', linestyle='dashed')
            if fp:
                plt.plot(reynolds, fp, label='Piggot')

    for f in fff: # several different friction factor functions
        friction_factors = {}
        for rr in relative_roughness_values:
            if gradient:
                friction_factors[rr] = [ d_func(f, re, rr) for re in reynolds]
                 
            else:
                friction_factors[rr] = [f(re, rr) for re in reynolds]
 
        # Plot the calculated curves on the Moody diagram
        if plot == "loglog":
            for rr, ff in friction_factors.items():
                plt.loglog(reynolds, ff, label=f'ε/D = {rr}')
        if plot == "linear":
            for rr, ff in friction_factors.items():
                plt.plot(reynolds, ff, label=f'ε/D = {rr}')
        if plot == "linlog":
            for rr, ff in friction_factors.items():
                plt.semilogx(reynolds, ff, label=f'ε/D = {rr}')

    # Plot all the diagrams with ratios of H2 to NG
    plt.xlabel('Reynolds number, Re')
    if not gradient:
        plt.ylabel('Darcy-Weisbach friction factor $f$')
    else:
        plt.ylabel('$d(f)/d(Re)$ Darcy-Weisbach friction factor gradient')
    if h2:
        #plt.ylim(-20,150)
        plt.xlabel('Reynolds number Re for natural gas')
        if f == p2_h2_ratio:
            plt.ylabel('Pressure gradient ratio ')
        else:
            plt.ylabel('Darcy-Weisbach friction factor ratio ')
    if w2:
        #plt.ylim(-80,500)
        plt.ylabel('Friction loss (power loss) ratio ')
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def export_f_table():
    """Produce a text file with f= f(Re)"""
    reynolds = np.logspace(2.4, 9.0, 200)

    rr = 1e-5
    with open('f_table.txt', 'w') as ff:
       ff.write(f"{'f':8},   {'Re':8} for rr = {rr:9.2f}\n") 
       for re in reynolds:
           f = afzal_mod(re, rr)
           ff.write(f"{f:8.3f}, {re:10.4f}\n") 
            
def plot_kwargs(g):
    linestyle=style(g)
    # color=colour(g)            Want H2 lines different colours
    #return  {'color': color, 'linestyle': linestyle}
    return  {'linestyle': linestyle}
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

T8C = T273 +8
T = T8C
P = 30
plot_p_star()
plot_lpm() # using defaults for rr and f_function

moody_ylim = True


reynolds_laminar = np.logspace(2.9, 3.9, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(2.4, 14.0, 1000) # 10^7.7 = 5e7
relative_roughness_values = [0.01, 0.001, 0.0001, 0.000003,  1e-7, 1e-9] #
#relative_roughness_values = list(reversed(relative_roughness_values))
fp = piggot()

moody_ylim = False
plot_diagram('', 'moody_afzal.png', plot="loglog", fff=afzal_mod)

plot_diagram('$f$  ratio between H2 and NG', 'h2_ratio.png', plot="linlog", fff=h2_ratio, h2=True)

plot_diagram('Pressure gradient ratio between H2 and NG', 'p2_h2_ratio.png', plot="linlog", fff=p2_h2_ratio, h2=True)

plot_diagram('Ratio of friction loss between H2 and NG', 'w2_h2_ratio.png', plot="linlog", fff=w2_h2_ratio, w2=True)

plot_diagram('Moody Diagram (Swarmee)', 'moody_swarmee.png', plot="loglog", fff=swarmee)
plot_diagram('Moody Diagram (Colebrook)', 'moody_colebrook.png', plot="loglog", fff=swarmee)
plot_diagram('Moody Diagram (Haarland)', 'moody_haarland.png', plot="loglog", fff=haarland) # BUGGY
# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm.png', plot="loglog", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])

# now do comparative plot, but for just one roughness valuesT250 = T273 -40
reynolds = np.logspace(2.4, 11.0, 1000) # 10^7.7 = 5e7

relative_roughness_values = [1e-5] # not used

# This re-sets global variables P, T
plot_pt_diagram('Pressure gradient ratio between H2 and NG', 'p2_h2_ratio_pt.png', plot="linlog", fff=p2_h2_ratio)
plot_pt_diagram('Ratio of friction loss (power) between H2 and NG', 'w2_h2_ratio_pt.png', plot="linlog", fff=w2_h2_ratio, w2=True)
# so reset them afterwards
T = T250
P = 30

# --- PLOT PRESSURE _ DISTANCE along PIPELINE
plot_pipeline("Pressure drop along pipeline", "pipe", plot="linear", fff=[afzal_mod])

# Plot enlarged diagrams

reynolds_laminar = np.logspace(2.9, 3.4, 5) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(2.8, 5.0, 500) 
relative_roughness_values = [0.01, 0.003, 0.001, 1e-5]

# fp = piggot() # not in view on the enlarged plot
fp = None
plot_diagram('$f$ ratio between H2 and NG', 'h2_ratio_enlarge.png', plot="linlog", fff=h2_ratio, h2=True)

# plot_diagram('Moody (Colebrook) Transition region', 'moody_colebrook_enlarge.png',plot="loglog")
plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge.png',plot="loglog", fff=[afzal_mod]) #, afzal_shift

plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_ll.png',plot="linlog", fff=[afzal_mod, afzal_shift])
plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_d_ll.png',plot="linlog", fff=[afzal_mod, afzal_shift], gradient=True)


# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm_enlarge.png', plot="loglog", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])

reynolds_laminar = np.logspace(2.9, 3.4, 500) # 10^2.7 = 501, 10^3.4 = 2512
reynolds = np.logspace(3.0, 4.0, 500) 

plot_diagram('$f$ ratio between H2 and NG', 'h2_ratio_enlarge_lin.png', plot="linear", fff=h2_ratio, h2=True)

plot_diagram('Moody (Afzal) Transition region', 'moody_afzal_enlarge_lin.png',plot="linear", fff=[afzal_mod]) #, afzal_shift
# plot_diagram('Moody Diagram (Virtual Nikuradze)', 'moody_vm_enlarge_lin.png', plot="linear", fff=[virtual_nikuradse,gioia_chakraborty_friction_factor])




exit()

'''
Before running this program, ensure you have the required libraries installed. You can install them using pip:
pip install numpy matplotlib scipy

This program uses the Colebrook equation to calculate the friction factor for a range of Reynolds numbers and relative roughness values. It then plots these values on a log-log scale to create the Moody diagram. The resulting plot is saved as an image file named `moody_diagram.png`. If you need any further assistance or modifications, feel free to ask! 😊'''