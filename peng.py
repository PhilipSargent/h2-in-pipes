import functools 
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys

import sonntag as st

from cycler import cycler
from gas_data import gas_data, gas_mixtures,gas_mixture_properties

"""This code written Philip Sargent, starting in December 2023, by  to support
a paper on the replacement of natural gas in the UK distribution grid with hydrogen.

For general information, commercial natural gas typically contains 85 to 98 percent methane, with the remainder mainly nitrogen and ethane, and has a calorific value of approximately 38 megajoules (MJ) per cubic metre

Wobbe Number (WN) (i) ≤51.41 MJ/m³, and  (ii)  ≥47.20 MJ/m³
https://www.nationalgas.com/data-and-operations/quality
≥46.5MJ/m3 after 6 April 2025 https://www.hse.gov.uk/gas/gas-safety-management-regulation-changes.htm

UNITS: bar, K, litres NOTE: bar is not SI ! Need to convert to Pascal.
"""
# This algorithm does NOT deal with the temperature dependence of alpha properly. 
# The code should be rearranged to calculate alpha for each point on the plot for each gas.

# Peng-Robinson Equation of State constants for Hydrogen and Methane
# Omega is the acentric factor is a measure of the non-sphericity of molecules; 
# a higher acentric factor indicates greater deviation from spherical shape
# PR constants data from ..aargh lost it.

R = 0.083144626  # l.bar/(mol.K)  SI after 2019 redefinition of Avagadro and Boltzmann constants
# 1 bar is today defined as 100,000 Pa not 1atm

Atm = 1.01325 # bar 
#Atm = 0.9256 # bar 
#Atm = 1.0536  # bar 
# Lowest recorded pressure in UK 925.6mb at Ochertyre, near Crieff, Perthshire on the 26th January 1884
# highest recorded pressure in UK 1053.6 mbar, Aberdeen 31.1.1902
T273 = 273.15 # K

# gas_data now imported from a separate file.

# 20% H2, remainder N.Sea gas. BUT may need adjusting to maintain Wobbe value, by adding N2 probably.
fifth = {}
fifth['H2'] = 0.2
ng = gas_mixtures['NG']
for g in ng:
    fifth[g] = ng[g]*0.8
gas_mixtures['NG+20%H2'] = fifth

air = {}
air['H2O'] = 0.0084 # 50% RH at 15
ag = gas_mixtures['dryAir']
for g in ag:
    air[g] = ag[g]*(1 - air['H2O'])
gas_mixtures['Air'] = air

enrich = [0.3, 0.5, 0.8]
air_list = ['Air']
for o in enrich:
    original_o = gas_mixtures['Air']['O2']
    increase = o - original_o  # i.e. 9.1 % for 30%O2
    factor  = (1 - o) / ( 1 - original_o)
    name_airN = f"Air{o:.0%}O2"
    airN = {}
    ag = gas_mixtures['Air']
    for g in ag:
        airN[g] = ag[g] * factor
    airN['O2'] = o
    gas_mixtures[name_airN] = airN
    air_list.append(name_airN)
air_list.append('O2') # pure oxygen

# Binary interaction parameters for hydrocarbons for Peng-Robinson
# based on the Chueh-Prausnitz correlation
# from https://wiki.whitson.com/eos/cubic_eos/
# also from Privat & Jaubert, 2023 (quoting a 1987 paper).
# Note that the default value (in the code) is -0.019 as this represents ideal gas behaviour.

# There is a full table of BIP using teh GCM method on https://wiki.whitson.com/eos/bips/index.html#coutinho-et-al-correlation
# using these might be better than the Coutinho equation - future work.

# These are used in function estimate_k_?(g1, g2) which estimates these parameters from gas data.
# NOT NOW USED, instead weuse the Coutinho estimation procedure in estimate_k()
# The difference is undetectable in our use at ambient conditions.
k_ij = {
    'CH4': {'C2H6': 0.0021, 'C3H8': 0.007, 'iC4': 0.013, 'nC4': 0.012, 'iC5': 0.018, 'nC5': 0.018, 'C6': 0.021, 'CO2': 0},
    'C2H6': {'C3H8': 0.001, 'iC4': 0.005, 'nC4': 0.004, 'iC5': 0.008, 'nC5': 0.008, 'C6': 0.010},
    'C3H8': {'iC4': 0.001, 'nC4': 0.001, 'iC5': 0.003, 'nC5': 0.003, 'C6': 0.004},
    'iC4': {'nC4': 0.0, 'iC5': 0.0, 'nC5': 0.0, 'C6': 0.001}, # ?
    'nC4': {'iC5': 0.001, 'nC5': 0.001, 'C6': 0.001}, # ?
    'iC5': {'C6': -0.019}, # placeholder
    'nC5': {'C6': -0.019}, # placeholder    
    #'C6': {'C6': -0.019}, # placeholder
    'CO2': {'C6': -0.019}, # placeholder
    'H2O': {'C6': -0.019}, # placeholder
    'N2': {'C6': -0.019}, # placeholder
    'He': {'C6': -0.019}, # placeholder
    'H2': {'C6': -0.019}, # placeholder
    'O2': {'C6': -0.019}, # placeholder
    'Ar': {'C6': -0.019}, # placeholder
}

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

@memoize   
def estimate_k_fixed(gas1, gas2, T=298):
    """Using the data table for k_ij"""
    k = -0.019 # value for ideal solution Privat & Jaubert, 2023
    if gas2 in k_ij[gas1]:
        k = k_ij[gas1][gas2]
    if gas1 in k_ij[gas2]:
        k = k_ij[gas2][gas1]
    return k

@memoize
def estimate_k_gao(gas1, gas2, T=298):
    """An estimate of the temperature-independent binary interaction parameters, eq.(29) in 
    Privat & Jaubert 2023, which is due to Gao reworking the method of Chueh & Prausnitz (1960s)
    
    BUT temperature-independent BIPs are known to be inaccurate.
"""
    Tc1 = gas_data[gas1]["Tc"]
    Tc2 = gas_data[gas2]["Tc"]
    
    Pc1 = gas_data[gas1]["Pc"]
    Pc2 = gas_data[gas2]["Pc"]
    
    Zc1 = peng_robinson(Tc1, Pc1, gas1)
    Zc2 = peng_robinson(Tc2, Pc2, gas2)
    
    power = 0.5*(Zc1 + Zc2)
    term = 2 * np.sqrt(Tc1*Tc2) / (Tc1 + Tc2)
    k = 1 - pow(term, power)
    return k  
    
@memoize    
def estimate_k(gas1, gas2, T=298):
    """Courtinho method implemented here.
    
    BUT We should REALLY be using the temperature-dependent group-contribution method as described in
    author = {Romain Privat and Jean-Nol Jaubert},
    doi = {10.5772/35025},
    book = {Crude Oil Emulsions- Composition Stability and Characterization},
    pages = {71-106},
    publisher = {InTech},
    title = {Thermodynamic Models for the Prediction of Petroleum-Fluid Phase Behaviour},
    year = {2012},
    which has data for all the components we are dealing with (check this..)"""

    a1, b1 = a_and_b(gas1, T)
    a2, b2 = a_and_b(gas2, T)
    term = 2 * np.sqrt(b1*b2) / (b1 + b2)
    k = 1 - 0.885 * pow(term, -0.036)
    return k
    
def check_composition(mix, composition, n=0):
    """Checks that the mole fractions add up to 100%
    This gives warnings and prints out revised compositions for manual fixing,
    but after doing that, it normalises everything perfectly using float division
    so that all calculations on the data are using perfectly normalised compositions, even if they
    don't quite match what the data declaration says."""
    eps = 0.000001
    warn = 0.02 # 2 %
    
    x = 0
    norm = 1
    for gas, xi in composition.items():
       x += xi
    norm = x

    if abs(x - 1.0) > eps:
        if abs(x - 1.0) < warn:
            print(f"--------- Warning gas mixture '{mix}', {100*(1-warn)}% > {100*x:.5f} > {100*(1+warn)}%. Normalizing.")
        else:
            print(f"######### BAD gas mixture '{mix}', molar fractions add up to {x} !!!")
            
        # Normalising is not done exactly, but with rounded numbers to 6 places of decimals.
        print(f"Stated:\n   '{mix}': {gas_mixtures[mix]},") 
        for g in gas_mixtures[mix]:
            gas_mixtures[mix][g] = float(f"{gas_mixtures[mix][g]/norm:.6f}")
        print(f"Normed:\n   '{mix}': {gas_mixtures[mix]},") 
        
        # Recursive call to re-do normalization, still with 6 places of decimals.
        n += 1
        if n < 5:
            newcomp = gas_mixtures[mix]
            check_composition(mix, newcomp, n)
        else:
            print(f"Cannot normalise using rounded 6 places of decimals, doing it exactly:") 
            gas_mixtures[mix][g] = gas_mixtures[mix][g]/norm
            print(f"Normed:\n   '{mix}': {gas_mixtures[mix]},") 
        
        
    # Normalise all the mixtures perfectly, however close they already are to 100%
    for gas, xi in composition.items(): 
        x = xi/norm
        gas_mixtures[mix][gas] = x
           
def density_actual(gas, T, P):
    """Calculate density for a pure gas at temperature T and pressure = P
    """
    ϱ = P * gas_data[gas]['Mw'] / (peng_robinson(T, P, gas) * R * T)
    return ϱ

@memoize   
def viscosity_actual(gas, T, P):
    """Calculate viscosity for a pure gas at temperature T and pressure = P
    """
    if len(gas_data[gas]['Vs']) == 3:
        vs0, t, power  = gas_data[gas]['Vs'] # at T=t  
    else:
        vs0, t  = gas_data[gas]['Vs'] # at T=t 
        power = 0.5

    vs = pow(T/t, power) * vs0 # at 1 atm

    return vs

def viscosity_values(mix, T, P):

    values = {}
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # this is where we call the function to calculate the viscosity
        vs = viscosity_actual(gas, T, P) 
        values[gas] = vs
    return values

@memoize       
def do_mm_rules(mix):
    """Calculate the mean molecular mass of the gas mixture"""
    
    if mix in gas_data:
        # if a pure gas
        return gas_data[mix]['Mw']
        
    mm_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        mm_mix += x * gas_data[gas]['Mw']
    
    return mm_mix

@memoize       
def do_flue_rules(mix, X_):
    """Calculate the mean molecular Carbon number and hydrogen number"""
    
    if mix in gas_data:
        # if a pure gas
        if X_ in gas_data[mix]:
            return gas_data[mix][X_]
        
    X_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for number of C and H atoms
        if X_ in gas_data[gas]:
           X_mix += x * gas_data[gas][X_]
    # for mixtures this must be normalised using only the combustible gases
    ff = get_fuel_fraction(mix)
    if ff > 0 :
        return X_mix/ff
   
    return 0
    
@memoize
def linear_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
    """
    value_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        value_mix += x * values[gas]
    
    return value_mix
    
@memoize
def explog_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
                 
    This exp(log()) mixing rue was used by Xiong 2023 for the Peng-Robinson FT case. eqn.(6).
    """
    ln_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # exp(ln()) mixing rule for volume factor
        ln_mix += x * np.log(values[gas]) # natural log
    
    return np.exp(ln_mix)

@memoize
def hernzip_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    using the Herning & Zipper mixing rule
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
    """
    composition = gas_mixtures[mix]
    # sum_of_sqrt(Mw)
    x = 0
    sqrt_Mw = 0
    for gas, x in composition.items():
        sqrt_Mw += x * np.sqrt(gas_data[gas]['Mw'])
 
    value_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        value_mix += x * values[gas] * np.sqrt(gas_data[gas]['Mw']) / sqrt_Mw
    
    return value_mix

        
def do_notwilke_rules(mix):
    """Calculate the mean viscosity of the gas mixture"""
    vs_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        vs, _ = gas_data[gas]['Vs'] # ignore T, so value for hexane will be bad
        vs_mix += x * vs
    
    return vs_mix

@memoize
def z_mixture_rules(mix, T):
    """
    Calculate the Peng-Robinson constants for a mixture of hydrocarbon gases.
    
    This uses the (modified) Van der Waals mixing rules and assumes that the
    binary interaction parameters are non-zero between all pairs of components
    that we have data for.
    
    Zc is the compressibility factor at the critical point    
    """
    
    # Initialize variables for mixture properties
    a_mix = 0
    b_mix = 0
    Zc_mix = 0
    
    composition = gas_mixtures[mix]
    # Calculate the critical volume and critical compressibility for the mixture
    # Vc_mix = 0
    # for gas, xi in composition.items(): 
        # Tc = gas_data[gas]['Tc']
        # Pc = gas_data[gas]['Pc']
        # Vc_mix += xi * (0.07780 * Tc / Pc)
   
    # Calculate the mixture critical temperature and pressure using mixing rules
    for gas1, x1 in composition.items():
        Tc1 = gas_data[gas1]['Tc']
        Pc1 = gas_data[gas1]['Pc']
         
        a1, b1 = a_and_b(gas1, T) 
        
        # Linear mixing rule for volume factor
        b_mix += x1 * b1
           
        # Van der Waals mixing rules for 'a' factor
        for gas2, x2 in composition.items(): # pairwise, but also with itself.
            Tc2 = gas_data[gas2]['Tc']
            Pc2 = gas_data[gas2]['Pc']
            #omega2 = gas_data[gas2]['omega']
            a2, b2 = a_and_b(gas2, T) 
            
            # Use mixing rules for critical properties
            k = estimate_k(gas1, gas2, T)
             
            a_mix += x1 * x2 * (1 - k) * (a1 * a2)**0.5  
            
       # Return the mixture's parameters for the P-R law
    return { mix: 
        {
            'a_mix': a_mix,
            'b_mix': b_mix,
         }
    }

"""This function uses simple mixing rules to calculate the mixture’s critical properties. The kij parameter, which accounts for the interaction between different gases, is assumed to be 0 for simplicity. In practice, kij may need to be adjusted based on experimental data or literature values for more accurate results.
 """


def get_LMN(omega):
    """Twu (1991) suggested a replacement for the alpha function, which instead of depending
        only on T & omega, depends on T, L, M, N (new material constants)
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8851615/pdf/ao1c06519.pdf
        
        We now have the PDF for the H2 + n-butane Jaubert et al. 2013 paper which has 
        all these parameters
    """
    # These equations are from Privat & Jaubert (2023)
    # https://www.sciencedirect.com/science/article/pii/S0378381222003168
    L = 0.0544 + 0.7536 * omega + 0.0297 * omega**2
    M = 0.8678 - 0.1785 * omega + 0.1401 * omega**2
    N = 2
    
    return L, M, N

@memoize
def a_and_b(gas, T):
    """Calculate the a and b intermediate parameters in the Peng-Robinson forumula 
    a : attraction parameter
    b : repulsion parameter
    
    Assume  temperature of 25 C if temp not given
    """
    # Reduced temperature and pressure
    Tc = gas_data[gas]['Tc']
    Pc = gas_data[gas]['Pc']

    Tr = T / Tc
    omega = gas_data[gas]['omega']
    
    
    # We do not use the L,M,N formulation as we have omega for
    # all our gases, and H2 just doesn't work with L,M,N at the pressures we use.
    if False:
        if 'L' in gas_data[gas]:
            L = gas_data[gas]['L']
            M = gas_data[gas]['M']
            N = gas_data[gas]['N']
        else:
            L, M, N = get_LMN(gas_data[gas]['omega'])            
        
        alpha1 = Tr ** (N*(M-1)) * np.exp(L*(1 - Tr**(M*N)))
    
    # updated wrt many compoudns, Pina-Martinez 2019:
    kappa = 0.3919 + 1.4996 * omega - 0.2721 * omega**2 + 0.1063 * omega**3
    
    
    # https://www.sciencedirect.com/science/article/abs/pii/S0378381218305041
    # 1978 Robinson and Peng
    if omega < 0.491: # omega for nC10, https://www.sciencedirect.com/science/article/abs/pii/S0378381205003493
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    else:
        kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.16666 * omega**3
        
    # Alpha function
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

    # Coefficients for the cubic equation of state
    a = 0.45724 * (R * Tc)**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc

    return a, b

@memoize
def solve_for_Z(T, p, a, b):
   
    # Solve cubic equation for Z the compressibility
    A = a * p / (R * T)**2 # should have alpha in here? No..
    B = b * p / (R * T)
    
    c3 = 1
    c2 = -1 +B
    c1 = A - 3 * B**2 - 2 * B
    c0 = -A * B + B**2 + B**3

    # Solve the cubic equation for Z
    roots = np.roots([c3, c2, c1, c0])
 
    # Filter out complex roots and select the appropriate real root
    real_roots = roots[np.isreal(roots)].real
    Z = np.max(real_roots)  # Assuming vapor phase 
    
    return Z

@memoize
def peng_robinson(T, P, gas): # Peng-Robinson Equation of State
    if gas not in gas_mixtures:    
        a, b = a_and_b(gas, T)
    else:
        constants = z_mixture_rules(gas, T)
        a = constants[gas]['a_mix']
        b = constants[gas]['b_mix'] 
        
    Z = solve_for_Z(T, P, a, b)
    return Z


def viscosity_LGE(Mw, T_k, ϱ):
    """The  Lee, Gonzalez, and Eakin method, originally expressed in 'oilfield units'
    of degrees Rankine and density in g/cc, with a result in centiPoise
    doi.org/10.2118/1340-PA 1966
    Updated to SI: PetroWiki. (2023). 
    https://petrowiki.spe.org/Natural_gas_properties. 
    """
    raise # don't use this 
    
    T = T_k * 9/5 # convert Kelvins to Rankine
   
    # Constants for the Lee, Gonzalez, and Eakin #1
    k = (7.77 + 0.0063 * Mw) * T**1.5 / (122.4 + 12.9 * Mw + T)
    x = 2.57 + 1914.5 / T + 0.0095 * Mw # * np.exp(-0.025 * MWg) hallucination!
    y = 1.11 - 0.04 * x

    # Constants for the Lee, Gonzalez, and Eakin #2
    k = (9.4 + 0.02 * Mw) * T**1.5 / (209 + 19 * Mw + T)
    x = 3.5 + 986 / T + 0.01 * Mw
    y = 2.4 - 0.2 * x

    mu = 0.1 * k * np.exp(x * (ϱ / 1000)**y) #microPa.s

    return mu 

def print_bip():
    """Print out the binary interaction parameters
    """
    for g1 in gas_data:
        if g1 in k_ij:
            print("")
            for g2 in gas_data:
               if g2 in k_ij[g1]:
                pass
                print(f"{g1}:{g2} {k_ij[g1][g2]} - {estimate_k(g1,g2):.3f} {k_ij[g1][g2]/estimate_k(g1,g2):.3f}", end="\n")
            print("")

    for g1 in gas_data:
        for g2 in gas_data:
           pass
           print(f"{g1}:{g2}  {estimate_k(g1,g2):.3f}  ", end="")
        print("")

@memoize
def get_z(g, p, T):
    if g in gas_data:
        return peng_robinson(T, p, g)
        
    constants = z_mixture_rules(g, T)
    a = constants[g]['a_mix']
    b = constants[g]['b_mix']
    Z_mix = solve_for_Z(T, p, a, b)
    return Z_mix
    
@memoize
def get_density(mix, p, T):
    if mix in gas_data:
        g = mix
        ϱ_pg = p * gas_data[g]['Mw'] / (peng_robinson(T, p, g) * R * T)
        return ϱ_pg
        
    constants = z_mixture_rules(mix, T)
    a = constants[mix]['a_mix']
    b = constants[mix]['b_mix']
    Z_mix = solve_for_Z(T, p, a, b)
    mm = do_mm_rules(mix) # mean molar mass
    # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
    ϱ = p * mm / (Z_mix * R * T)
    return ϱ

@memoize
def get_blasius_factor(g, p, T):
    ϱ = get_density(g, p, T)
    μ =  get_viscosity(g, p, T)

    b_factor = pow(μ, 0.25) * pow(ϱ, 0.75)
    return b_factor
    
def get_Hc(g, T):
    """If the data is there, return the standard heat of combustion, 
    but in MJ/m³ not MJ/mol
    Uses molar volume at (15 degrees C, 1 atm) even though reference T for Hc is 25 C"""
    if g in gas_mixture_properties and 'Hc' in gas_mixture_properties[g]:
        hc = gas_mixture_properties[g]['Hc']
    elif g in gas_data and 'Hc' in gas_data[g]:
        hc = gas_data[g]['Hc']
    else:
        hc = 0
        composition = gas_mixtures[g]
        for pure_gas, x in composition.items():
            # Linear mixing rule for volume factor
            hc += x * gas_data[pure_gas]['Hc']
    # hc is in MJ/mol, so we need to divide by the molar volume at  (25 degrees C, 1 atm) in m³
    Mw = do_mm_rules(g)/1000 # Mw now in kg/mol not g/mol
    ϱ_0 = get_density(g, Atm, T) # in kg/m³
    molar_volume = Mw / ϱ_0  # in m³/mol
    
    if hc:
        return molar_volume, hc/molar_volume, hc
    else:
        return molar_volume, None, None

def get_Cp(g):
    """If the data is there, return the specific heat in J/mol.K, 
    """
    if g in gas_mixture_properties and 'Cp' in gas_mixture_properties[g]:
        cp = gas_mixture_properties[g]['Cp']
    elif g in gas_data and 'Cp' in gas_data[g]:
        cp = gas_data[g]['Cp']
    else:
        cp = 0
        composition = gas_mixtures[g]
        for pure_gas, x in composition.items():
            # Linear mixing rule for volume factor
            if 'Cp' in gas_data[pure_gas]:
                pure_cp = gas_data[pure_gas]['Cp']
                contrib = x * pure_cp
                cp += contrib
                #print(f"{g:5} {pure_gas:5} Cp:{pure_cp:8.4f} Cp contrib:{contrib:8.4f}")
    # Cp is in J/mol.K
    if cp:
        return cp
    else:
        return None

def get_fuel_fraction(g):
    if g in gas_data:
        if 'Hc' in gas_data[g]:
            if gas_data[g]['Hc'] > 0:
                return 1.0
            else:
                return 0
        else:
            return 0
            
    mff = 0
    composition = gas_mixtures[g]
    for pure_gas, x in composition.items():
        # Linear mixing rule for volume factor
        ff = get_fuel_fraction(pure_gas)
        mff += x * ff
    return mff
    
def print_fuelgas(g, oxidiser='Air'):
    """Oxidiser had been DryAir, now it is Air, i.e. 15%RH at 15C,
    but we are going to doing enriched air."""
    mm = do_mm_rules(g) # mean molar mass

    if g in gas_data:
        c_ = 0
        h_ = 0
        if 'C_' in gas_data[g]:
            c_ = gas_data[g]['C_']
        if 'H_' in gas_data[g]:
            h_ = gas_data[g]['H_']
    else:
        c_ = do_flue_rules(g, 'C_')
        h_ = do_flue_rules(g, 'H_')

    # mole fraction fuel

    mff = get_fuel_fraction(g)

    mv, hcmv, hc = get_Hc(g, 298)
    if hc:
        # for one mole of fuel gas
        # moles O2 = h_/2 + c_
        # but * 1.15 for excess air
        # moles N2 = moles O2 * (79.05/20.95)
        o2 = (h_/2 + c_) * 1.15
        #n2 = gas_mixtures['dryAir']['O2']
        n2 = gas_mixtures[oxidiser]['N2'] # now 'Air' = 50% RH at 15C, and N2 not O2 ! - unused
        dew_C = dew_point(g, oxidiser)
        print(f"{g:15} {mm:6.3f} {dew_C:5.3f} {c_:5.3f}   {h_:5.3f} {hc*1000:9.3f} {mff*100:8.4f} %")
    
@memoize
def get_viscosity(g, p, T):
    if g in gas_data:
        μ = viscosity_actual(g, T, p)
    else:
        values = viscosity_values(g, T, p)
        μ = linear_mix_rule(g, values)
    return  μ

def print_density(g, p, T):
    ϱ = get_density(g, p, T)
    mm = do_mm_rules(g) # mean molar mass
    μ =  get_viscosity(g, p, T)
    z =  get_z(g, p, T)
    print(f"{g:15} {mm:6.3f}  {ϱ:.5f}   {μ:8.5f} {z:9.6f}")
 
def print_wobbe(g, T15C):
    """HHV and Wobbe much be in MJ/m³, but at 15 C and 1 atm, not p and T as given
    UK NTS WObbe limits from     https://www.nationalgas.com/data-and-operations/quality
    
    "gas that is permitted in gas networks in Great Britain must have a relative density of ≤0.700"
    https://www.hse.gov.uk/gas/gas-safety-management-regulation-changes.htm
    also, relative density must be >0.7 and CO2 less than 2.5 mol.%
    """
    too_light = ""
    best = (47.20 + 51.41) / 2 # wobbe limits
    lowest = 47.20
    highest = 51.41
    width = 51.41 - 47.20
    
    # Wobbe is at STP: 15 C and 1 atm
    ϱ_0 = get_density(g, Atm, T15C)
    ϱ_air = get_density('Air', Atm, T15C)
    
    relative_ϱ = (ϱ_0/ϱ_air)
 
    wobbe_factor_ϱ = 1/np.sqrt(ϱ_0/ϱ_air)
    
    mv, hcmv, hc = get_Hc(g, T15C) 

    if relative_ϱ > 0.7:
        too_light = f"Rϱ > 0.7 ({relative_ϱ:.3f} = {ϱ_0:.3f} kg/m³)"
        
    # yeah, yeah: 'f' strings are great
    if hc:
        w = wobbe_factor_ϱ * hcmv
        niceness = 100*(w - best)/width  # 100*w/best - 100
        flag = f"{'nice':^8} {niceness:+.1f} %"
        if w < lowest:
            flag = f"{'LOW':^8}"
        if w  > highest:
            flag = f"{'HIGH':^8}"

        w = f"{w:>.5f}"
        hc = f"{hc:^12.4f}"
        hcmv = f"{hcmv:^11.4f}"
    else:
        w = f"{'-':^10}"
        hc = f"{'-':^12}"
        hcmv = f"{'-':^11}"
        flag = f"{'  -            '}"
    
    print(f"{g:15} {hc} {mv:.7f} {hcmv}{wobbe_factor_ϱ:>11.5f}   {w} {flag} {too_light}")
    
# @memoize  
# def condensing_fraction(g, t):
    # """The fraction of water in the flue gas that condenses to a liquid,
    # for this fuel gas g
    # for this condensing temperature t
    # """
    
    # #print(f"{g:4} {t-273.15:8.2f} {water_fraction=:8.4f}")
    # return water_fraction

@memoize  
def get_vapour_moles(g, t, oxidiser):
    """Calculate the amount of H2O(g) compared to H2O(l) at
    a particular temperature - in the flue gas
    """
    vp = st.sonntag_vapor_pressure(t) # in Pascals
    # Convert to fraction of an atmosphere
    vp_a = vp * 1e-5 / Atm # Atm = 1.0325 bar 
    
    
    # Work in whole moles, not  mole fractions
    n = get_moles_flue_for_1mol_fuel_gas(g,oxidiser)
    
    flue = get_flue_composition(g, oxidiser)
    h2o_f = gas_mixtures[flue]['H2O'] # fractional value
    h2o_mol = n * h2o_f        # moles of h2o
    not_mol = n * (1 - h2o_f)  # moles of non-condensibles
    
    if vp_a >= 1: # Sonntag equation is not precisely correct, so we need to fix this.
        vp_a = 1
        vap_mol = h2o_mol
    else:
        vap_mol = (vp_a/(1-vp_a)) * not_mol # rearranged equation as per s'sheet
    
    if vap_mol >= h2o_mol:
        vap_mol = h2o_mol
    
    return vap_mol

@memoize  
def get_water_moles(g, t, oxidiser):
    """get number of moles of liquid water condensed for fuel g
    and at temperature t 
    for one mole of fuel gas"""
    n = get_moles_flue_for_1mol_fuel_gas(g, oxidiser)

    flue = get_flue_composition(g, oxidiser)
    h2o_f = gas_mixtures[flue]['H2O'] # fractional value
    h2o_mol = n * h2o_f        # moles of h2o
    
    vap_mol = get_vapour_moles(g, t, oxidiser)
    liq_mol = h2o_mol - vap_mol

    return liq_mol
    
# @memoize  
# def get_vapour_moles__(g, t):
    # """get number of moles of  water vapour, NOT condensed for fuel g
    # and at temperature t 
    # for one mole of fuel gas"""
    # flue_moles = get_moles_flue_for_1mol_fuel_gas(g)
    # h2o_moles = flue_moles * gas_mixtures['flue']['H2O']
    
    # vapour_fraction =  1 - condensing_fraction(g, t)
    # vapour = vapour_fraction * h2o_moles
    # v2 = vapour_moles2(g, t)
    # print(f"{g:4} {t-273.15:8.2f} v1:{vapour:8.4f} v2:{v2:8.4f} ")
    # return vapour
   
@memoize
def latent_lost(g, t, oxidiser):
    # we need the actual number of moles of water, not just the proportion
    # This is the extra heat LOST because it is emitted a vapour, not an extra
    # amount we gain because it condenses.
    # STP at 298K assumes it is all condensed to water
    lost = get_vapour_moles(g, t, oxidiser)     
    LH = gas_data['H2O']['LH'] # kJ/mol    
    
    latent_heat =   lost * LH * 1000 # convert to J from kJ
    latent = latent_heat/1000
    #print(f"{g:4} {t-273.15:8.2f} Latent loss:{latent:8.4f} Lost (moles):{lost:8.4f} vp:{st.sonntag_vapor_pressure(t)/Atm/1e5:7.4f} ")
    return latent_heat
 
@memoize  
def get_moles_flue_for_1mol_fuel_gas(g, oxidiser):
    """ Calculates the components of the flue gas
    as a side-effect, puts the composition into 'flue_gas' and then gas_mixture['flue_gas_{oxidiser}']
    """
    # First we calculate the number of moles of each component in the input air
    # and adjust for the O2 which has got burned.
    o2_in, o2_burned = get_moles_O2_for_1mol_fuel_gas(g)
    air_mol = get_moles_air_for_1mol_fuel_gas(g, oxidiser)
    
    flue_gas = {} # moles (not fractions)
    for c in gas_mixtures[oxidiser]:
        flue_gas[c] = air_mol *  gas_mixtures[oxidiser][c]
    # check
    # print(f"O2 in {flue_gas['O2']} {o2_in}")
    
    flue_gas['O2'] = flue_gas['O2'] - o2_burned
    
    # Now we add in the inerts from the fuel gas, which we have 1 mole of
    # unless it is a pure gas in which case nothing is added
    if g not in gas_data:
         for c in gas_mixtures[g]:
            if gas_data[c]['C_'] == 0 and gas_data[c]['H_'] == 0 : # skip the combustibles
                if c in flue_gas:
                    flue_gas[c] += 1 *gas_mixtures[g][c] # 1 mole
                else:
                    flue_gas[c] = 1 * gas_mixtures[g][c] # 1 mole

    # Calculate the output (product) gases for 1 mole of fuel gas
    co2_out =  do_flue_rules(g,'C_') 
    h2o_out =  do_flue_rules(g,'H_')/2
    
    # we know the dry air has CO2 in it, so no need to check that
    # and we know it has no moisture
    flue_gas['CO2'] += co2_out
    flue_gas['H2O'] = h2o_out
    
    # add up the number of moles
    n = 0
    for c in flue_gas:
        n += flue_gas[c]
    print(f"Number of moles in flue gas for 1 mole fuel: {n:8.4f} for {g:6} and {oxidiser}")
      
      # Normalise
    for c in flue_gas:
        flue_gas[c] = flue_gas[c]/n
    gas_mixtures[g+"_flue_"+oxidiser] = flue_gas
    return n

@memoize  
def get_moles_O2_for_1mol_fuel_gas(g):
    ff = get_fuel_fraction(g)
    if ff < 0.001:
        print(f"{g:5} Not a fuel {ff}")
        return
    o2_fuel = do_flue_rules(g,'C_') + do_flue_rules(g,'H_')/4 # directly need to burn
    o2_burned = o2_fuel * ff # actual needed, as not all that 1 mol on fuel is combustible
    o2_in = o2_burned * 1.15 # 15% excess air
    return o2_in, o2_burned

@memoize  
def get_moles_air_for_1mol_fuel_gas(g, oxidiser):
    o2_in, _ = get_moles_O2_for_1mol_fuel_gas(g)
    
    if oxidiser == 'O2':
        x = 1.0
    else:
        x = gas_mixtures[oxidiser]['O2'] 
    air_mol_0 = o2_in * 1/x # where x is fraction of O2 in air
    
    # Alternatiove synthetic air calc to match spreadsheet
    o2s, n2s = 0.2095, 0.7905
    n2_synth = o2_in * n2s/o2s
    air_mol = o2_in + n2_synth
    #print(f"{g:4} Moles of air per mol of fuel : {air_mol_0:8.4f} {air_mol:8.4f}")
    return  air_mol_0

@memoize
def get_flue_composition(g, oxidiser):
    """For one mole of fuel gas, return the composition of the flue gas
    ALWAYS recalculate this as we run with several fuels
    re-think how this is stored..?"""
    
    n = get_moles_flue_for_1mol_fuel_gas(g, oxidiser)
    return g+'_flue_'+oxidiser
        
@memoize
def sensible_fuel(g, t):
    """For 1 mole of pseudo-gas g, how much is the sensible heat required to heat it
    from t to 298 K ?
    This includes inerts as well as combustible gases
    """
    fuel_mol = 1.0 # start with 1 mol of fuel gas
    fuel_cp = get_Cp(g)
    sensible_heat = fuel_mol * fuel_cp * (298 - t)
    #print(f"{g:5} {t} fuel {sensible_heat=:8.3f}")
    return sensible_heat
    
@memoize    
def sensible_air(g, t, oxidiser):
    """For 1 mole of pseudo-gas g, how much is the sensible heat required to heat it
    from t to 298 K ?
    """
    air_mol = get_moles_air_for_1mol_fuel_gas(g, oxidiser)
    air_cp = get_Cp(oxidiser)
    
    sensible_heat = air_mol * air_cp * (298 - t)
    #print(f"Air   {t} Air {sensible_heat=:8.3f}")
    return sensible_heat
    
@memoize    
def sensible_flue(g, t, oxidiser):
    """Sensible heat needed to'heat' flue gas from 298 to exit temp
    g : the fuel gas
    """
    flue_moles = get_moles_flue_for_1mol_fuel_gas(g, oxidiser)
    flue_cp = get_Cp(g+'_flue_'+oxidiser)
    sensible_heat = flue_moles * flue_cp * (t- 298) # flue ext temp is greater than 298 (nearly always)
    # print(f"{g:5}{flue_cp:8.3f} {t:5.1f} Flue {sensible_heat=:8.3f}")
    return sensible_heat

@memoize    
def sensible_water(g, t, oxidiser):
    """Sensible heat needed to'heat' liquid water from 298 to exit temp"""
    water_moles = get_water_moles(g, t, oxidiser) # for one mole fuel
    water_cp = gas_data['H2O']['CpL']
    sensible_heat = water_moles *water_cp * (t- 298) 
    return sensible_heat

@memoize
def sensible_in(g, T, t_fuel, t_air, oxidiser):
    """Sensible heat needed to heat inlet fuel and air up to 298"""
    fuel_in = sensible_fuel(g, t_fuel) # 1 mole of fuel
    air_in = sensible_air(g, t_air, oxidiser) # yes, this is for 1 mole of fuel
    return fuel_in + air_in
    
@memoize
def sensible_out(g, T, oxidiser):
    """Sensible heat needed to'heat' flue gas from 298 to exit temp"""
    flue_out = sensible_flue(g, T, oxidiser) # yes, for 1 mole fuel
    water_out = sensible_water(g, T, oxidiser) # for one mole fuel
    return flue_out + water_out
    
def d_condense(T, pressure, g, oxidiser):
    """Differential of the efficiency/ condensations temperature plot"""
    δ = 0.1
    e1 = condense(T-δ, pressure, g, oxidiser)
    e2 = condense(T+δ, pressure, g, oxidiser)
    
    return (e1-e2)/2*δ

@memoize
def condense(T, pressure, g, oxidiser):
    """Return the efficiency (%) of the boiler assuming flue gas is all condensed
    at temperature T (K)
    """
    t_fuel = 273.15 + 8
    t_air  = 273.15 + 5
    
    ff = get_fuel_fraction(g)
    if ff < 0.001:
        print(f"Not a fuel {ff}")
        return None  
    _, _, hc_MJ = get_Hc(g, 298) 
    hc = hc_MJ * 1000 * 1000

    heat_out = hc - latent_lost(g, T, oxidiser) - sensible_in(g, T, t_fuel, t_air, oxidiser) - sensible_out(g, T, oxidiser)
    η = 100 * heat_out/hc
    return η

def find_intersection(g1, g2, oxidiser):
    """For a condensing boiler at 1 atm, at what temperature are
    the efficiences equal between these two fuels?"""
    p = Atm
    def objective(T):
        obj = condense(T, p, g1, oxidiser) - condense(T, p, g2, oxidiser)
        #print(f"{T-T273:5.3f} {condense(T, p, g1):8.4f} {condense(T, p, g2):8.4f}")
        return obj
        
    eps = 1e-5
    T = T273 + 60
    n = 0
    while True:
        n += 1
        if T > T273 + 100:
            break
        obj = objective(T)
        #print(f"{T-T273:5.3f} Obj:{obj} ")
        if abs(obj) < eps:
            break
        delta_T = 3 * obj
        T += delta_T
        
        
    eff = condense(T, p, g1, oxidiser)
    if obj > 2* eps:
        print("ABORT")
    print(f"At {T-T273:5.2f} degrees C the fuels '{g1}' and '{g2}' have the same efficiency of {eff:8.5f} %  ({n}) with {oxidiser}")
 
def export_η_table(oxidiser='dryAir'):
    """Produce a text file with boiler efficiences"""
    p = Atm
    t_condense = np.linspace(T273+0,T273+100, 101)  
    c_H2 = [condense(T, p, 'H2', oxidiser) for T in t_condense]
    c_NG = [condense(T, p, 'NG', oxidiser) for T in t_condense]

    with open('η_table.txt', 'w') as η:
       pass
       pass
       η.write(f"{'Temp.(C)':8}   {'eta (H2)':8} {'eta (NG)':8}\n") 
       for T in t_condense:
            η.write(f"{T-T273:8.1f} {condense(T, p, 'H2', oxidiser):8.4f} {condense(T, p, 'NG', oxidiser):8.4f}\n") 
   
def get_h2o_pp(g, oxidiser):
    # REFACTOR ALL THIS now that it is being done properly elsewhwere
    
    # Calculate the reagent (input) gases for 1 mol of fuel gas g
    o2_in, o2_burned = get_moles_O2_for_1mol_fuel_gas(g)

    # now calculate inerts, start off with those in the fuel
    n2_gas = 0
    co2_gas = 0
    if g not in gas_data:
        if 'N2' in  gas_mixtures[g]:
            n2_gas = gas_mixtures[g]['N2']
        if 'CO2' in  gas_mixtures[g]:
            n2_gas = gas_mixtures[g]['CO2']
        
    #n2_gas = 1 - ff # everything that isn't O2 in the fuel

    #print(f"\n{g:5}\t C_={do_flue_rules(g,'C_'):6.4f} H_={do_flue_rules(g,'H_'):6.4f} O2:fuel {o2_fuel:6.4f} O2:gas ratio {o2_gas:6.4f}")
    
    
    o2 = gas_mixtures[oxidiser]['O2'] # partial pressue, i.e. molar fraction
    # h2o = gas_mixtures['dryAir']['H2O']
    n2 = 1- o2
    n2_in = o2_in * n2/o2
    # synthetic air, not used now
    o2s, n2s = 0.2095, 0.7905
    n2_synth = o2_in * n2s/o2s
    #print(f"O2 in: {o2_in:6.4f}, N2 in {n2_in:6.4f}  {n2_synth:6.4f}")
    
    # Calculate the output (product) gases
    o2_out = o2_in - o2_burned
    n2_out = n2_in + n2_gas
    n2_out_synth = n2_synth + n2_gas
    
    co2_out =  do_flue_rules(g,'C_') + co2_gas 
    h2o_out =  do_flue_rules(g,'H_')/2   # no H2O in the fuel gas, but there might be in air
    #print(f"O2 out: {o2_out:6.4f}, N2 out {n2_out:6.4f} {n2_out_synth:6.4f}  CO2 {co2_out:6.4f}  H2O {h2o_out:6.4f} ")
    tot_out = (o2_out + n2_out_synth + co2_out + h2o_out)
    t = tot_out/100
    #print(f"O2 out: {o2_out/t:6.3f}, N2 out {n2_out/t:6.3f} {n2_out_synth/t:6.3f}  CO2 {co2_out/t:6.3f}  H2O {h2o_out/t:6.3f} ")
   # OK, h2o_out/t is the partial pressure of H2O which determines the dew point. At last.
    h2o_pp = Atm * h2o_out/tot_out # in mol %, so convert to pressure in bar
    h2o_pp = h2o_pp * 1e5 # now in Pascals
    return h2o_pp

def dew_point(g, oxidiser):
    h2o_pp = get_h2o_pp(g, oxidiser)
    dew_C = st.get_dew_point(h2o_pp)-T273
    #print(f"{h2o_out/tot_out:.4f} {dew_C:.4f} C")
    return dew_C

def print_gas(g, oxidiser):
    dew_C = dew_point(g, oxidiser)
    if dew_C:
        print(f"{g} Dew point: {dew_C:.4f} C")
     
def print_fuel(g, s):
    print(f"\n{s}")
    #print(f"{g:3}")
    if g not in gas_mixtures:
        f = g
        mv, hcmv, hc = get_Hc(f, 298)
        if not hc:
            hc = 0
        print(f"{f:5}\t{100:8.4f} %{gas_data[f]['C_']:3}{gas_data[f]['H_']:3} {hc*1000:6.1f} kJ/mol ")
    else:
        nts = gas_mixtures[g]
        for f in nts:
            mv, hcmv, hc = get_Hc(f, 298)
            if not hc:
                hc = 0
            print(f"{f:5}\t{nts[f]*100:8.5f} %{gas_data[f]['C_']:3}{gas_data[f]['H_']:3} {hc*1000:6.1f} kJ/mol ")
    print_gas(g, oxidiser='dryAir') 

def style(mix):
    if mix in gas_data:
        return 'dashed'
    else:
        return 'solid'

# see https://matplotlib.org/stable/users/explain/colors/colors.html (bottom of page)
colours =  {'H2': 'xkcd:red',
   'CH4': 'xkcd:azure',
   'C2H6': 'xkcd:orchid',
   'NG+20%H2': 'xkcd:gold',
   'NG': 'xkcd:violet'}

# see https://matplotlib.org/cycler/
#colour_cycle = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def colour(g):
    if g in colours:
        return colours[g]
    return None # default behavior 
    splat = colour_cycle() # called as a generator, but always returns first item
    for c in splat:
       break
    print(g, c)
    return c['color']
    
def plot_kwargs(g):
    linestyle=style(g)
    color=colour(g)

    return  {'color': color, 'linestyle': linestyle}
# ---------- ----------main program starts here---------- ------------- #
def main():
    program = sys.argv[0]
    stem = str(pl.Path(program).with_suffix(""))
    fn={}
    for s in ["z", "ϱ", "μ", "bf", "bf_NG", "η", "ηη","dη","Tη"]:
        f = stem  + "_" + s
        fn[s] = pl.Path(f).with_suffix(".png") 

    for mix in gas_mixtures:
        composition = gas_mixtures[mix]
        check_composition(mix, composition)

    print_fuel('H2', "Hydrogen")
    print_fuel('NG', "NatGas at Fordoun NTS 20th Jan.2021")
    
    dp = 40
    tp = 15 # C
    t8 = 8 # C
    pressure =  Atm + dp/1000 # 1atm + 47.5 mbar, halfway between 20 mbar and 75 mbar
    T15C = T273 + tp # K
    T8C = T273 + t8 # K

    display_gases = ["NG"]    

    # Print the densities at 8 C and 15 C  - - - - - - - - - - -

    print(f"\nDensity of gas at (kg/m³)at T={tp:.1f}°C and P={dp:.1f} mbar above 1 atm, i.e. P={pressure:.5f} bar")

    plot_gases = []
    for g in display_gases:
        plot_gases.append(g)
    for g in ["H2", "CH4"]:
        plot_gases.append(g)


    print(f"{'gas':13}{'Mw(g/mol)':6}  {'ϱ(kg/m³)':5}  {'μ(Pa.s)':5} {'z (-)':5} T={tp:.1f}°C ")
    for g in plot_gases:
        print_density(g, pressure, T15C)

    print(f"\n{'gas':13}{'Mw(g/mol)':6}  {'ϱ(kg/m³)':5}  {'μ(Pa.s)':5}  {'z (-)':5} T={t8:.1f}°C ")
    for g in plot_gases:
        print_density(g, pressure, T8C)

    dp = 55
    pressure =  Atm + dp/1000
    print(f"\nDensity of gas at (kg/m³)at T={8:.1f}°C and P={dp:.1f} mbar above 1 atm, i.e. P={pressure:.5f} bar")
    print(f"\n{'gas':13}{'Mw(g/mol)':6}  {'ϱ(kg/m³)':5}  {'μ(Pa.s)':5}  {'z (-)':5} T={8:.1f}°C ")
    for g in plot_gases:
        print_density(g, pressure, T8C)

    print(f"\nHc etc. all at 15°C and 1 atm = {Atm} bar. Wobbe limit is  47.20 to 51.41 MJ/m³")
    print(f"W_factor_ϱ =  1/(sqrt(ϱ/ϱ(air))) ")
    print(f"{'gas':13} {'Hc(MJ/mol)':12} {'MV₀(m³/mol)':11} {'Hc(MJ/m³)':11}{'W_factor_ϱ':11} Wobbe(MJ/m³) ")
    for g in plot_gases:
        print_wobbe(g, T15C)
     
    print("'nice' values range from -50% to +50% from the centre of the valid Wobbe range.")

    print(f"\n[H2O][CO2] of fuel gas")
    print(f"{'gas':13}{'Mw(g/mol)':6} {'Dew Pt':6}  {'C_':5}   {'H_':5}{'Hc(kJ/mol)':5}  fuel")
    for g in ['H2', 'CH4', 'C2H6']:
        print_fuelgas(g)
    for g in gas_mixtures:
        print_fuelgas(g)

    export_η_table()
    
    #- - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -- - - - - - - - - - -
    # Plot defaults
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 6),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

   # Plot the condensing curve  - - - - - - - - - - -
    p = Atm
    t_condense = np.linspace(273.15+20, 273.15+100, 1000)  
    plt.figure(figsize=(10, 6))
    c_H2 = [condense(T, p, 'H2', 'Air') for T in t_condense]
    c_NG = [condense(T, p, 'NG', 'Air') for T in t_condense]
    
    plt.plot(t_condense-273.15, c_H2, label='Pure hydrogen', **plot_kwargs('H2'))
    plt.plot(t_condense-273.15, c_NG, label='Natural Gas', **plot_kwargs('NG'))
   
    #plt.title(f'Maximum boiler efficiency vs Condensing Temperature at {p} bar')
    plt.xlabel('Flue gas temperature (°C)')
    plt.ylabel('Maximum boiler efficiency (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["η"])
    plt.close()
    
    # Plot the condensing curves at different % oxygen  - - - - - - - - - - -
    get_Cp
    p = Atm
    t_condense = np.linspace(273.15+20, 273.15+100, 1000)  
    plt.figure(figsize=(10, 6))
    for o in air_list:
        # print(f"Cp {o:8} {get_Cp(o):.3f}")
        c_NG = [condense(T, p, 'NG', o) for T in t_condense]
        
        if o == 'Air':
            k = 'NG'
        else:
            k = o
        plt.plot(t_condense-273.15, c_NG, label='Natural Gas + '+ o, **plot_kwargs(k))
       
    plt.xlabel('Flue gas temperature (°C)')
    plt.ylabel('Maximum boiler efficiency (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["ηη"])
    plt.close()
    
    # Plot the Differential of the condensing curve  - - - - - - - - - - -
    p = Atm
    t_condense = np.linspace(273.15+20, 273.15+100, 1000)  
    plt.figure(figsize=(10, 6))
    c_H2 = [d_condense(T, p, 'H2', 'Air') for T in t_condense]
    c_NG = [d_condense(T, p, 'NG', 'Air') for T in t_condense]
     
    plt.plot(t_condense-273.15, c_H2, label='Pure hydrogen', **plot_kwargs('H2'))
    plt.plot(t_condense-273.15, c_NG, label='Natural Gas', **plot_kwargs('NG'))
   
    plt.title(f'd(η)/d(T) vs Condensing Temperature at {p} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Boiler efficiency gradient d(η)/d(T) (%/K)')
    plt.legend()
 
    plt.savefig(fn["dη"])
    plt.close()
    
    find_intersection('H2', 'NG', 'Air') # where do the efficiences cross?
    find_intersection('H2', 'NG+20%H2', 'Air') # where do the efficiences cross?

   # Plot the compressibility  - - - - - - - - - - -

    # Calculate Z0 for each gas
    Z0 = {}
    for gas in gas_data:
        Z0[gas] = peng_robinson(T273+25, pressure, gas)

    # Plot Z compressibility factor for pure hydrogen and natural gases
    temperatures = np.linspace(233.15, 323.15, 100)  
      # bar

    plt.figure(figsize=(10, 6))


    # Plot for pure hydrogen
    Z_H2 = [peng_robinson(T, pressure, 'H2') for T in temperatures]


    plt.plot(temperatures - T273, Z_H2, label='Pure hydrogen', **plot_kwargs('H2'))

        
    # Plot for pure methane
    # Z_CH4 = [peng_robinson(T, pressure, 'CH4') for T in temperatures]
    # plt.plot(temperatures - T273, Z_CH4, label='Pure methane', **plot_kwargs('CH4'))


    # Plot for natural gas compositions. Now using correct temperature dependence of 'a'
    ϱ_ng = {}
    μ_ng = {}

    for mix in plot_gases:
        if mix in gas_data:
            continue
        mm = do_mm_rules(mix) # mean molar mass
        ϱ_ng[mix] = []
        μ_ng[mix] = []

        Z_ng = []
        for T in temperatures:
            # for Z, the averaging across the mixture (a, b) is done before the calc. of Z
            constants = z_mixture_rules(mix, T)
            a = constants[mix]['a_mix']
            b = constants[mix]['b_mix']
            Z_mix = solve_for_Z(T, pressure, a, b)
            Z_ng.append(Z_mix)
            
            # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
            ϱ_mix = pressure * mm / (Z_mix * R * T)
            ϱ_ng[mix].append(ϱ_mix)

            # μ_mix = viscosity_LGE(mm, T, ϱ_mix)
            μ_mix = get_viscosity(mix, pressure, T)
            μ_ng[mix].append(μ_mix)

        plt.plot(temperatures - T273, Z_ng, label=mix, **plot_kwargs(mix))

    #plt.title(f'Z  Compressibility Factor vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Z Compressibility Factor')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["z"])
    plt.close()

    # Viscosity plot  EXPTL values at 298K - - - - - - - - - - -

    P = pressure
    μ_g = {}
    for mix in plot_gases:
        μ_g[mix] = []
        for T in temperatures:
            μ = get_viscosity(mix,P,T)
            # μ = hernzip_mix_rule(mix, values) # Makes no visible difference wrt to linear!
            # μ = explog_mix_rule(mix, values) # very slight change by eye
            μ_g[mix].append(μ)
        plt.plot(temperatures - T273, μ_g[mix], label= mix, **plot_kwargs(mix))

    plt.title(f'Dynamic Viscosity [data] vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Dynamic Viscosity (μPa.s) - linear Mw mixing rule')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["μ"])
    plt.close()

    # Blasius Parameter plot  - - - - - - - - - - -
    bf_gases = []
    for g in display_gases:
        bf_gases.append(g)
    for g in ["H2", "CH4", "C2H6"]:
        bf_gases.append(g)
        
    P = pressure
    bf_g = {}
    for mix in bf_gases:
        bf_g[mix] = []
        for T in temperatures:
            bf = get_blasius_factor(mix,P,T)
            bf_g[mix].append(bf)
        plt.plot(temperatures - T273, bf_g[mix], label= mix, **plot_kwargs(mix))


    #plt.title(f'Blasius Parameter  ϱ^3/4.μ^1/4 vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Blasius Parameter  ϱ^3/4.μ^1/4 ')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["bf"])
    plt.close()


    # Blasius Parameter plot NORMALIZED wrt NG  - - - - - - - - - - -

    P = pressure
    bf_g = {}
    for mix in bf_gases:
        bf_g[mix] = []
        for T in temperatures:
            bf = get_blasius_factor(mix,P,T)
            bf_g[mix].append(bf)
     
    for mix in bf_gases:
        if mix == "NG":
             continue
        for i in range(len(temperatures)):
            T = temperatures[i]
            #print(mix, T, bf_g[mix][i]/ bf_g['NG'][i], bf_g[mix][i], bf_g['NG'][i])
            bf_g[mix][i] = bf_g[mix][i]/ bf_g['NG'][i]
        plt.plot(temperatures - T273, bf_g[mix], label= mix, **plot_kwargs(mix))

    plt.title(f'Blasius Parameter ϱ^3/4.μ^1/4  normalised to NG value at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Blasius Parameter ratio ')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["bf_NG"])
    plt.close()
    
    bf_g = {}
    t = 0
    for P in [pressure, 4+Atm, 7+Atm]:
        t += 1
        print(f"\nBlasius Parameter ϱ^3/4.μ^1/4 (normalised by NG) between {temperatures[0]-T273:4.1f}C and {temperatures[-1]-T273:4.1f}C at {P} bar")
        for mix in bf_gases:
            bf_g[mix] = []
            for T in temperatures:
                bf = get_blasius_factor(mix,P,T)
                bf_g[mix].append(bf)
            if mix == "NG":
                 continue
            for i in range(len(temperatures)):
                T = temperatures[i]
                bf_g[mix][i] = bf_g[mix][i]/ bf_g['NG'][i]
            plt.plot(temperatures - T273, bf_g[mix], label= mix, **plot_kwargs(mix))
            bf_g[mix].sort()
            mn = bf_g[mix][0]
            mx = bf_g[mix][-1]
            mean = (mx + mn)/2
            rng = (mx - mn)/2
            pct = 100*rng/mean
            print(f"{mix:5} {mean:9.4f} ±{rng:7.4f}  {pct:5.2f}%")
            plt.title(f'Blasius Parameter ϱ^3/4.μ^1/4  normalised to NG value at {P} bar')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Blasius Parameter ratio ')
            plt.legend()
        plt.savefig(f"peng_bf_NG_{t}.png")
        plt.close()


    # ϱ/Viscosity plot Kinematic EXPTL values at 298K - - - - - - - - - - -

    P = pressure
    re_g = {}

    for mix in plot_gases + ['He']:
        ϱ_ng[mix] =  [get_density(mix, P, T) for T in temperatures]
        μ_ng[mix] = [get_viscosity(mix, P, T) for T in temperatures]
            
        re_g[mix] = []
        for i in range(len(μ_ng[mix])):
            re_g[mix].append(  μ_ng[mix][i]/ϱ_ng[mix][i])
        plt.plot(temperatures - T273, re_g[mix], label= mix, **plot_kwargs(mix))
        
    #plt.title(f'Kinematic Viscosity vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Dynamic Viscosity/Density (μPa.s)/(kg/m³) ')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_re.png")
    plt.close()

    # Density plot  - - - - - - - - - - -

    # pure gases
    for g in ["H2", "CH4"]: 
        ϱ_pg = [pressure * gas_data[g]['Mw'] / (peng_robinson(T, pressure, g) * R * T)  for T in temperatures]
        plt.plot(temperatures - T273, ϱ_pg, label = "pure " + g, **plot_kwargs(g))

    # Density plots for gas mixtures
    for mix in display_gases:
        plt.plot(temperatures - T273, ϱ_ng[mix], label=mix, **plot_kwargs(mix))

    plt.title(f'Density vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density (kg/m³)')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["ϱ"])
    plt.close()

    # Plot the compressibility  as a function of Pressure - - - - - - - - - - -
    T = T273+8
    P= None

    # Plot Z compressibility factor for pure hydrogen and natural gases
    pressures = np.linspace(0, 80, 100)  # bar

    plt.figure(figsize=(10, 6))

    for g, txt in [('H2','Pure hydrogen'), ('CH4','Pure methane'),('He','Pure helium')]:
        Z = [peng_robinson(T, p, g) for p in pressures]
        plt.plot(pressures, Z, label=txt, **plot_kwargs(g))


    # Plot for natural gas compositions. Now using correct temperature dependence of 'a'
    ϱ_ng = {}
    μ_ng = {}

    for mix in gas_mixtures:
        mm = do_mm_rules(mix) # mean molar mass
        ϱ_ng[mix] = []
        μ_ng[mix] = []

        Z_ng = []
        for p in pressures:
            # for Z, the averaging across the mixture (a, b) is done before the calc. of Z
            constants = z_mixture_rules(mix, T)
            a = constants[mix]['a_mix']
            b = constants[mix]['b_mix']
            Z_mix = solve_for_Z(T, p, a, b)
            Z_ng.append(Z_mix)
            
            # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
            ϱ_mix = p * mm / (Z_mix * R * T)
            ϱ_ng[mix].append(ϱ_mix)

        plt.plot(pressures , Z_ng, label=mix, **plot_kwargs(mix))

    plt.title(f'Z  Compressibility Factor vs Pressure at {T} K')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Z Compressibility Factor')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_z_p.png")
    plt.close()

    # Plot Blasius Parameter for pure hydrogen and natural gases
    pressures = np.linspace(1, 8.1, 100)  # bar
    T = T273+8

    plt.figure(figsize=(10, 6))

    bf_g = {}

    for g in plot_gases:
        bf_g[g] = []

        for p in pressures:
            bf = get_blasius_factor(g,p,T)
            bf_g[g].append(bf)

        plt.plot(pressures , bf_g[g], label=g, **plot_kwargs(g))

    plt.title(f'Blasius Parameter vs Pressure at {T} K')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Blasius Parameter ϱ^3/4.μ^1/4')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_bf_p.png")
    plt.close()

    # Plot viscosity as a function of pressure - looking for bugs
    pressures = np.linspace(0, 80, 100)  # bar
    T = T273+25

    plt.figure(figsize=(10, 6))


    bf_g = {}

    for g in bf_gases:
        bf_g[g] = []

        for p in pressures:
            bf = get_viscosity(g,P,T)
            bf_g[g].append(bf)

        plt.plot(pressures , bf_g[g], label=g, **plot_kwargs(g))

    plt.title(f'Viscosity vs Pressure at {T} K')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Dynamic Viscosity (μPa.s)')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_μ_p.png")
    plt.close()

    # Plot density as a function of pressure - looking for bugs
    pressures = np.linspace(0, 4.5, 100)  # bar
    T = T273+25

    plt.figure(figsize=(10, 6))


    bf_g = {}

    for g in plot_gases:
        bf_g[g] = []

        for p in pressures:
            bf = get_density(g,p,T)
            bf_g[g].append(bf)

        plt.plot(pressures , bf_g[g], label=g, **plot_kwargs(g))

    plt.title(f'Density vs Pressure at {T} K')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_ϱ_p.png")
    plt.close()
    
if __name__ == '__main__':
    sys.exit(main())  
