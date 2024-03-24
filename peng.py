import functools 
import numpy as np
import matplotlib.pyplot as plt

import pathlib as pl
import sys
import warnings

import sonntag as st

# from scipy.optimize import newton_raphson

from cycler import cycler
from scipy.optimize import fsolve

from gas_data import gas_data, gas_mixtures, gas_mixture_properties, ng_gases, enrich, air_list, k_ij
from peng_utils import memoize

"""This code written Philip Sargent, starting in December 2023, by  to support
a paper on the replacement of natural gas in the UK distribution grid with hydrogen.

THIS IS NOT TO BE USED FOR ENGINEERING DESIGN

The equation of state, mixing rules and viscosity models for the gases are "cartoon"
versions of the real equations that should be used where real plant is being 
designed and real human safety is at risk. 
For engineering design use the GERG-2008 equation of state which is under preparation 
for adoption as an international standard (ISO 20765-2 und ISO 20765-3).
See https://pubs.acs.org/doi/10.1021/je300655b

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

# gas_data now imported from a separate file. It is initialised on import.

# We memoize some functions so that they do not get repeatedly called with
# the same arguments. Yet still be retain a more obvius way of writing the program.
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
        for g in gas_mixtures[mix]:
            print(f"{g:9}: {gas_mixtures[mix][g]*100:7.3f} %")
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
def μ_ϱ_gradient():
    """Should only get run once, depsite being deeply nested"""
    T25C = T273 + 25
    μ_1 = viscosity_actual('H2', T25C, Atm, force=True) # STP
    μ_100 = 9.1533 # microPa.s

    ϱ_1 = get_density('H2',Atm, T25C)
    ϱ_100 = get_density('H2',100, T25C)
    gradient = (μ_100 - μ_1) / (ϱ_100 - ϱ_1) # microPa.s / kgm^-3
    # print(f"μ_ϱ_gradient {μ_1=:0.4f} {ϱ_1=:0.4f} {gradient}")
    return μ_1, ϱ_1, gradient

@memoize   
def viscosity_H2(T, P):
    """Higher accuracy viscosity just for hydrogen because of its importance to this 
    project.
    
    
    So use "present correlation" Table 4:
    valid: T:100 to 990 K, for 0.1 to 220 Mpa (1 to 2,200 bar)
    
    C. Li, W. Jia, and X. Wu, “Temperature prediction for high pressure high temperature condensate gas flow through chokes,” Energies, vol. 5, no. 3, pp. 670–682, 2012, doi: 10.3390/en5030670.
    """
    @memoize  
    def AB(ϱ):
        warnings.filterwarnings("error")
        # our function gives density in kg/m^3
        # assume Li et al want g/cc
        ϱ = ϱ /1000

        try:
            A = np.exp(5.73 + np.log(ϱ) + 65.0 * np.power(ϱ, 3/2) - 6e-6 * np.exp(135*ϱ))
        except RuntimeWarning:
            print('Runtime Warning')
            print(f"H2   {P=:3.1f} {T=:3.0f} {ϱ=:0.4e}  {vs=:10.5f}  ")
            A = 0
        # np.log() is natural log. np.log10() is log base 10.
        # We don't know what units Li et al. are using for density.
         
        B = 1 *(10 + 8*(np.power(ϱ/0.07,6) - np.power(ϱ/0.07,3/2)) - 18 * np.exp(-59*np.power(ϱ/0.07,3)))      
        warnings.resetwarnings()
        return A, B
        
    @memoize  
    def ϱ_H2_g_cc(P,T):
        # P is in bar for the density calc.
        # we know the result is correct because we have checked it against data,
        # density ~ 100x more than 1 bar at 100 bar etc.
        ϱ = P * gas_data['H2']['Mw'] / (peng_robinson(T, P, 'H2') * R * T)
        return ϱ
        

    @memoize  
    def Δμ_linear(ϱ):
        """Linear fit to REFPROP8 at 100 bar (10MPa) and 25C at 9.1533 microPa s
        """
        μ_1, ϱ_1, g = μ_ϱ_gradient()
        
        μ =  μ_1 + g * (ϱ - ϱ_1)
        Δμ = μ - μ_1
        return Δμ
        
    @memoize  
    def Δvs_H2(ϱ):
        A, B = AB(ϱ)
        Δvs = A * np.exp(B/T) # this is far too large, by a factor 
        # Assume undocumented units issue in published paper:
        Δvs = Δvs /10
        return Δvs
        
    # Equation as used for other gases to get vx0 for H2
    vs0, t0, power  = gas_data['H2']['Vs'] # at T=t 
    vs_1 = pow(T/t0, power) * vs0 # at 1 atm
    
    ϱ = ϱ_H2_g_cc(P,T)
    ϱ_1 = ϱ_H2_g_cc(Atm,T) # at 1 bar
    
    Δvs = Δvs_H2(ϱ)
    Δvs_1 = Δvs_H2(ϱ_1)
    vs = vs_1 + Δvs - Δvs_1
   
    Δμ =  Δμ_linear(ϱ) 
    μ = Δμ + vs_1
    # print(f"H2   {P=:3.1f} {T-T273:6.1f} {ϱ=:0.4f} {μ=:10.5f} {vs_1=:10.5f} {Δvs=:10.5f} {Δvs_1=:10.5f} ")
    return μ
    
@memoize   
def viscosity_ng(μ, T, P):
    """Using gas industry approx. for natural gas mixtures"""
    vs0 = μ
    a2 = -0.00002207 # /K.bar
    a3 = 0.00434531 # /bar
    vs2 = a2 * 13 * P * (T-T273)
    vs3 = a3 * 13 * P
    vs = vs0 +   vs2 + vs3
    #print(f"{T-T273:.1f} {P=:.0f} {vs2=:.5f} {vs3=:.5f} {vs0=:.5f}")
    return vs
    
@memoize   
def viscosity_actual(gas, T, P, force=False):
    """Calculate viscosity for a pure gas at temperature T and pressure = P
    """
    if not force and gas == 'H2':
        return viscosity_H2(T, P)

    if len(gas_data[gas]['Vs']) == 3:
        vs0, t, power  = gas_data[gas]['Vs'] # at T=t  
    else:
        vs0, t  = gas_data[gas]['Vs'] # at T=t 
        power = 0.5

    vs = pow(T/t, power) * vs0 # at 1 atm

    return vs
    
@memoize   
def viscosity_values(mix, T, P):
    if not P:
        print(mix,  T, P)
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
def get_linear(mix, property):
    """Calculate the linear sum of the property for the gas mixture
    weighted by molecular fraction"""
    
    if mix in gas_data:
        # if a pure gas
        return gas_data[mix][property]
        
    o_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        o_mix += x * gas_data[gas][property]
    
    return o_mix
    
@memoize       
def get_omega(mix):
    """Calculate the linear sum of the omegas for the gas mixture
    weighted by molecular fraction"""
           
    o_mix = get_linear(mix, 'omega')
    return o_mix

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

def set_mix_rule():
    global visc_f
    visc_f = wilke_mix_rule
    return visc_f

@memoize
def linear_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    https://en.wikipedia.org/wiki/Viscosity_models_for_mixtures
    
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
    https://en.wikipedia.org/wiki/Viscosity_models_for_mixtures
    This is the Arrhenius law for viscosity of mixtures of liquids
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
                 
    This exp(log()) mixing rule was used by Xiong 2023 for the Peng-Robinson FT case. eqn.(6).
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
    For viscosity, thsi is the Car model:
    the momentum-weighted sum of partial viscosities
    https://en.wikipedia.org/wiki/Viscosity_models_for_mixtures
    
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

@memoize
def M_terms(i, j):
    """Dimensionless constant formed from pairwise interactions of the 
    component gases. Independent of T and P
    
    This is not symmetric in (i, j): it divides i  by j
    """
    mi = gas_data[i]['Mw']
    mj = gas_data[j]['Mw']

    denom = np.sqrt(8 * (1 + mi/mj))
    M_bit = np.power(mj/mi, 1/4)
    return denom, M_bit

@memoize
def Φ(i, j, μi, μj):
    """Dimensionless constant formed from pairwise interactions of the 
    component gases
    Depends on viscosity so deopends on T and P
    
    Equation (4) in Arrhenius & Buker (2022)
    """
    if i == j:
        return None
    denom, M_bit = M_terms(i, j)
    μ_term = np.sqrt(μi/μj)
    ϕ_ij = np.power((1 + μ_term * M_bit), 2) / denom
    #print (i, j, f"{μi:.4f} {μj:.4f} {ϕ_ij:.4f}")
    #print (i, j,f"  {ϕ_ij=:.4f}")
    return ϕ_ij
        
@memoize
def wilke_mix_rule(mix, values):
    """Calculate the mean value of a property for a mixture
    using the Wilke mixing rule. The Carr  (hernzip) rule is a simplified version of this
    
    The input viscosity values depend on P and T so vary a lot, but @memoize is valid
    
    Many typos in the literature.
    https://idaes-pse.readthedocs.io/en/stable/explanations/components/property_package/general/transport_properties/viscosity_wilke.html is wrong
    
    but Davidson is correct
    and the Arrhenius refactors Davidson'seqn.
    
    I have checked this, the Henning commnetd-out line dioes indeed reproduce the 
    separate calcualtion of the Henning average.
    
    values: dict {gas1: v1, gas2: v2, gas3: v3 etc}
                 where gas1 is one of the component gases in the mix, and v1 is value for that gas
    """
    composition = gas_mixtures[mix]
    μ = 0
    for i, xi in composition.items():
        μi = values[i]
        dn = 0
        for j, xj in composition.items():
            if i == j:
                continue
            μj = values[j]
            ϕ_ij = Φ(i, j, μi, μj)
            #ϕ_ij =  np.sqrt(gas_data[j]['Mw']/gas_data[i]['Mw']) # Herning simplification
            t = xj * ϕ_ij # Davidson incorrectc too, copied into in Arrhenius & Buker
            dn += t
            #print (f"    {xj:.2f} * ϕ_ij  {t:.4f} ")
        
        num_i = xi * μi
        denom_i = xi + dn
        μ_inc = num_i /denom_i
        μ += μ_inc 
        #print (f"{num_i=:.4f} {denom_i=:.4f} {μ_inc=:.4f}")
    return  μ
    
@memoize
def z_mixture_rules(mix, T):
    """
    Calculate the Peng-Robinson constants for a mixture of hydrocarbon gases.
    
    This uses the (modified) Van der Waals mixing rules and assumes that the
    binary interaction parameters are non-zero between all pairs of components
    that we have data for.
    
    Zc is the compressibility factor at the critical point    
    """
    if mix in gas_data:
        return a_and_b(mix, T)
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
    return a_mix,  b_mix


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
def get_kappa(omega):
    # updated wrt many compoudns, Pina-Martinez 2019:
    kappa = 0.3919 + 1.4996 * omega - 0.2721 * omega**2 + 0.1063 * omega**3
    
    # https://www.sciencedirect.com/science/article/abs/pii/S0378381218305041
    # 1978 Robinson and Peng
    if omega < 0.491: # omega for nC10, https://www.sciencedirect.com/science/article/abs/pii/S0378381205003493
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    else:
        kappa = 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.16666 * omega**3
        
    return kappa

@memoize
def estimate_a_and_b(gas, Tc, Pc, T):
    """Give a guess at Tc and Pc, return what this would mean that the a and be are
    at Tr = 0.7 Tc"""
 
    omega = get_omega(gas) 
    kappa = get_kappa(omega)

    if Tc < 0:
        #print('#########',gas, Tc, Pc, T)
        Tc = 1e-6
        
    Tr = T/Tc
        
    # Alpha function
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

    # Coefficients for the cubic equation of state
    a = 0.45724 * (R * Tc)**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc

    return a, b
    
@memoize
def a_and_b(gas, T):
    """Calculate the a and b intermediate parameters in the Peng-Robinson forumula 
    a : attraction parameter
    b : repulsion parameter
    
    Assume  temperature of 25 C if temp not given
    """
    """This function uses simple mixing rules to calculate the mixture’s critical properties. The kij parameter, which accounts for the interaction between different gases, is assumed to be 0 for simplicity. In practice, kij may need to be adjusted based on experimental data or literature values for more accurate results.
 """
    # Reduced temperature and pressure
    Tc = gas_data[gas]['Tc'] # Kelvn
    Pc = gas_data[gas]['Pc'] # bar 

    Tr = T / Tc
    omega = gas_data[gas]['omega']
    kappa = get_kappa(omega)
     
    # Alpha function
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2

    # Coefficients for the cubic equation of state
    # R is in l.bar/(mol.K) - see top of file
    a = 0.45724 * (R * Tc)**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc

    return a, b
    
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

@memoize
def solve_for_Z(T, p, a, b):
    # R is in l.bar/(mol.K) - see top of file
   
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
        a, b = z_mixture_rules(gas, T)
        
    Z = solve_for_Z(T, P, a, b)
    return Z

@memoize
def pz(T, P, gas): # return the Pressure divided by Z        
    Z = peng_robinson(T, P, gas)
    return P/Z

@memoize
def dzdp(T, P, gas): # return the Pressure divided by Z        
    Z1 = peng_robinson(T, P*0.999, gas)
    Z2 = peng_robinson(T, P*1.001, gas)
    return (Z2 - Z1)/ (P*1.001 - P*0.999)

@memoize
def peng_robinson_invert(a,b): # Peng-Robinson Equation of State
    # This does not seem to work at all, Units problem?
    # UNUSED
    Tc = (8*a / 27*b*R)
    Pc = (a / 27*b**2)
    return Tc, Pc

@memoize
def  guess_critical(gas):
    """Starting point for interating to find the Tc and Pc for a gas mixture"""
    Tc = get_linear(gas, 'Tc')
    Pc = get_linear(gas, 'Pc')

    return Tc, Pc

def critical_properties_PR(gas, a, b, T):
    """
    Calculates critical temperature (Tc) and critical pressure (Pc) of a gas MIXTURE
    from the Peng-Robinson equation of state (EOS) parameters a and b
    
    HOWEVER, a and b for a pure gas have to be calulated using omega, and 
    *at a specific temperature* . The coefficient a depends on temperature.
    This is fine when calcualting a and b  from Tc, Pc, omega (pure gas) and the 
    temperature T at which you want to use the P-R - usually to calculate Z.
    
    HOWEVER inverting P-R to back-calculate Tc and Pc is a bit different.
    1. The averaing rule for omega is also needed, and 
    2. you need to know what temperature was used when the a and b were generated.
    3. But actually it is both Tc and T which are used when calculating a (via alpha)
       so there is another implicit loop there.

    Args:
      a_mix (float): Peng-Robinson coefficient a for the mixture.
      b_mix (float): Peng-Robinson coefficient b for the mixture.

    Returns:
      tuple[float, float]: A tuple containing critical temperature (Tc) and
                           critical pressure (Pc) in Kelvin and Pascal, respectively.
    """
    if gas in gas_data:
        gTc = gas_data[gas]['Tc']
        gPc = gas_data[gas]['Pc']

    # R = 8.314472  # J/mol*K but we use l.bar/mol.K
    omega = get_omega(gas) # estimate for this gas mixture
    kappa = get_kappa(omega)
    alpha = 1 + kappa * 0.027 # at Tr = 0.7, but this is a guess - and wrong.
    
    # Once we calculate Tc, Pc we should  recalculate kappa and alpha at that temp
    # and iterate again
    
    # Initial guesses 
    Tc, Pc = guess_critical(gas)

   # Solve the system of non-linear equations using fsolve
    # try:
        # Tc, Pc = fsolve(critical_property_equation, (Tc_guess, Pc_guess))
    # except Exception as e:
        # raise ValueError(f"Critical property calculation failed: {e}")
    # return Tc, Pc
    
    # This is what estimate_a_and_b does:
    # a = 0.45724 * (R * Tc)**2 / Pc * alpha
    # b = 0.07780 * R * Tc / Pc
    # alpha = 1 + kappa * 0.027

    tolerance = 1e-6
    max_iterations = 100
    Tc_ = Tc
    Pc_ = Pc
    
    a_, b_ = estimate_a_and_b(gas, Tc_, Pc_, T)
 
    for _ in range(max_iterations):
        # We iterate Tc and b in an inner loop. This converges rapidly.
        # The outer loop with a and Pc is more difficult.
        for _ in range(5):
            a_, b_ = estimate_a_and_b(gas, Tc_, Pc_, T)
 
            Tc_ *= 1.0* ( 1 + (b - b_)/b)
            if Tc_ <0:
                # print('### Tc',gas, a, b)
                Tc = 0

            if abs(b - b_) < tolerance:
                #print("Tc converged")
                break
            
        Pc1 = 1.001*Pc
        a1, _ = estimate_a_and_b(gas, Tc_, Pc1, T)
        da_by_dPc = (a1 -a_)/(Pc1-Pc_)
        k = da_by_dPc 
        factor = (a - a_) / da_by_dPc
        if factor > 1e-3:
            mult = 0.5
        else:
            mult = 0.1
        Pc_ = Pc_ - mult*factor
        if Pc_ <0:
            print('### Pc',gas, a, b)
            Pc = 0
        
        if abs(a - a_) < tolerance:
            #print("--- Pc converged")
            break
        #print(f"{a:9.5f} {a-a_:8.1e}  {k:9.2e} {factor:9.2e}  {b:9.5f} {b-b_:8.1e}  {b_:9.5f} {Tc:.1f}  {Tc_:.1f} {Pc:5.1f} {Pc_:9.4f}")
    if Tc_ < 0:
        Tc_ = float('NaN')
    return Tc_, Pc_

def viscosity_LGE(Mw, T_k, ϱ):
    """The  Lee, Gonzalez, and Eakin method, originally expressed in 'oilfield units'
    of degrees Rankine and density in g/cc, with a result in centiPoise
    doi.org/10.2118/1340-PA 1966
    Updated to SI: PetroWiki. (2023). 
    https://petrowiki.spe.org/Natural_gas_properties. 
    
    I could not get this to work: produced results at variance from tabulated viscosities
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
        
    a, b = z_mixture_rules(g, T)
    Z_mix = solve_for_Z(T, p, a, b)
    return Z_mix
    
@memoize
def get_density(mix, p, T):
    if mix in gas_data:
        g = mix
        ϱ_pg = p * gas_data[g]['Mw'] / (peng_robinson(T, p, g) * R * T)
        return ϱ_pg
        
    a, b = z_mixture_rules(mix, T)
    Z_mix = solve_for_Z(T, p, a, b)
    mm = do_mm_rules(mix) # mean molar mass
    # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
    ϱ = p * mm / (Z_mix * R * T)
    return ϱ

@memoize
def get_μ_ratio(g, p, T, visc_f, g2='NG'):
    '''Used by moody.py but not in this file'''
    μ_ratio = get_viscosity(g, p, T, visc_f)/get_viscosity(g2, p, T, visc_f)
    return μ_ratio
    
@memoize
def get_ϱ_ratio(g, p, T, g2='NG'):
    '''Used by moody.py but not in this file'''
    ϱ_ratio = get_density(g, p, T)/get_density(g2, p, T)
    return ϱ_ratio
    

def get_v_ratio(g, p, T, g2='NG'):
    T25C = 273.15 + 25
    _, _, hc_g = get_Hc(g, T25C) # molar_volume, hc/molar_volume, hc = get_Hc(g, T)
    _, _, hc_ng = get_Hc(g2, T25C)
    hhvr = hc_ng / hc_g
    v_ratio = hhvr * pz(T, p,'NG')/pz(T, p, g)
    #print(f"{T=:.0f} {p=:8.4f} {v_ratio=:.4f} ")
    return v_ratio

@memoize
def get_Δp_ratio_br(g, p, T, g2='NG'):
    # Only used to calc the ratios of H2:NG in the Blasius regime
    b_ratio = get_blasius_factor(g, p, T) / get_blasius_factor(g2, p, T)
    v_ratio = get_v_ratio(g, p, T)
    Δp_ratio = b_ratio * pow(v_ratio, 7/4)
    
    return Δp_ratio

@memoize
def get_blasius_factor(g, p, T):
    ϱ = get_density(g, p, T)
    μ =  get_viscosity(g, p, T, visc_f)

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
    
def print_fuelgas(g, oxidiser):
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
def get_viscosity(g, P, T, visc_f):
    # if 'visc_f' not in locals():
        # #print("UNDEFINED viscosity mean value algorithm, using Wilke")
        # visc_f = set_mix_rule()
    
    if g in gas_data:
        μ = viscosity_actual(g, T, P)
    else:
        values = viscosity_values(g, T, P)
        μ = visc_f(g, values)
        
    if g in ng_gases:
        μ = viscosity_ng(μ, T, P)
    return  μ

def print_density(g, p, T, visc_f):
    ϱ = get_density(g, p, T)
    mm = do_mm_rules(g) # mean molar mass
    μ =  get_viscosity(g, p, T, visc_f)
    z =  get_z(g, p, T)
    
    a, b = z_mixture_rules(g, T)
    Tc, Pc = critical_properties_PR(g, a, b, T) 
    
    if g in gas_data:
        s = f"({gas_data[g]['Pc']:0.1f} bar {gas_data[g]['Tc']:6.1f} K)" 
    else:
        s = "(approx.)"
        
    print(f"{g:15} {mm:6.3f}  {ϱ:9.5f}   {μ:8.5f} {z:9.6f} {ϱ/μ:9.5f} {Pc:9.1f} bar {Tc:9.1f} K {s}")

def print_viscosity(g, p, T, visc_f):
 
    mm = do_mm_rules(g) # mean molar mass
    μ =  get_viscosity(g, p, T, visc_f)
    print(f"{g:15} {mm:6.5f}   {μ:8.5f} {visc_f.__name__}")


def print_wobbe(plot_gases, g, T15C):
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
    
    print(f"\nHc etc. all at 15°C and 1 atm = {Atm} bar. Wobbe limit is  47.20 to 51.41 MJ/m³")
    print(f"W_factor_ϱ =  1/(sqrt(ϱ/ϱ(air))) ")
    print(f"{'gas':13} {'Hc(MJ/mol)':12} {'MV₀(m³/mol)':11} {'Hc(MJ/m³)':11}{'W_factor_ϱ':11} Wobbe(MJ/m³) ")


    for g in plot_gases:
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
    print("'nice' values range from -50% to +50% from the centre of the valid Wobbe range.")

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
    
    # we know the dry air has CO2 in it, so no need to check that.. except for H2 burning in pure O2..
    # and we know it has no moisture
    if 'CO2' in flue_gas:
        flue_gas['CO2'] += co2_out
    else:
        flue_gas['CO2'] = co2_out
        
    flue_gas['H2O'] = h2o_out
    
    # add up the number of moles
    n = 0
    for c in flue_gas:
        n += flue_gas[c]
    # print(f"Number of moles in flue gas for 1 mole fuel: {n:8.4f} for {g:6} and {oxidiser}")
      
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
 
def export_η_table(oxidiser):
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
    #print(f"{h2o_out/tot_out:.4f} {dew_C:.4f}°C")
    return dew_C

def print_gas(g, oxidiser):
    dew_C = dew_point(g, oxidiser)
    if dew_C:
        print(f"{g} Dew point: {dew_C:.4f}°C")
     
def print_fuel(g, s, oxidiser):
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
    print_gas(g, oxidiser) 

def print_some_gas_data(plot_gases, P, dp=None):
    # ignore P if dp (mbar) supplied
    global T50C, T25C, T15C, T8C, T3C, T250, T230, T273

    if dp:
        P = Atm + dp/1000
        pstr = f"P={dp:.1f} mbar above 1 atm, i.e. P={P:.5f} bar"
    else:
        pstr = f"P={P:.0f} bar" 
        
    print(f"\nDensity of gas (kg/m³) at {pstr}")
    for T in [T8C, T15C]:
        print(f"{'gas':13}{'Mw(g/mol)':6}  {'ϱ(kg/m³)':5}  {'μ(Pa.s)':5}    {'Z (-)':5}      {'ϱ/μ(Mkg/sm)':5}  T={T-T273:.1f}°C ")
        for g in plot_gases:
            print_density(g, P, T, visc_f)
        
def style(mix):
    if mix in gas_data:
        return 'dashed'
    else:
        return 'solid'

# see https://matplotlib.org/stable/users/explain/colors/colors.html (bottom of page)
colours =  {'H2': 'xkcd:red',
   'O2': 'xkcd:turquoise',
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
    global visc_f
    global T50C, T25C, T15C, T8C, T3C, T250, T230, T273
    
    program = sys.argv[0]
    stem = str(pl.Path(program).with_suffix(""))
    fn={}
    for s in ["z", "ϱ", "μ", "bf", "bf_NG", "η", "η20", "ηη", "ηηη","dη","Tη"]:
        f = stem  + "_" + s
        fn[s] = pl.Path(f).with_suffix(".png") 

    for mix in gas_mixtures:
        composition = gas_mixtures[mix]
        check_composition(mix, composition)

    # print_fuel('H2', "Hydrogen", 'dryAir')
    # print_fuel('NG', "NatGas at Fordoun NTS 20th Jan.2021", 'dryAir')
    
    dp = 40
 
    pressure =  Atm + dp/1000 # 1atm + 47.5 mbar, halfway between 20 mbar and 75 mbar
    T50C = T273 + 50 # K
    T25C = T273 + 25 # K
    T15C = T273 + 15 # K
    T8C = T273 + 8 # K
    T3C = T273 + 3 # K
    T250 = T273 -20 #  -20 C
    T230 = T273 -40 #  -40 C

    display_gases = ["NG"]   

    # Natural gases - print in order of density
    T=T8C
    print(f"{'gas':13}{'Mw(g/mol)':6}   {'ϱ(kg/m³)':5}   {'μ(Pa.s)':5}   {'Z (-)':5}      {'ϱ/μ(Mkg/sm)':5}  Pc (bar)      Tc (K)  T={T-T273:.1f}°C P=1 atm")
    ϱ={}
    for g in ng_gases:
    #for g in ['NG','CH4']:
        ϱ[g] =  get_density(g, Atm, T)
     # Sort by value
    ϱ = dict(sorted(ϱ.items(), key=lambda item: item[1]))
    for g in ϱ:
        print_density(g, Atm, T8C, wilke_mix_rule)
        
    # Viscosity averaging function global - - - - - - - - - - -

    if False:
        for g in ['NG', 'Groening', 'NoGas', 'North Sea', 'UW', 'HeOx','ArH2' ]:
            print(" ")
            for PP in [1, 220]:
                print(f"Viscosity of gas (kg/m³) at {T8C=} T={T8C-T273:.1f}°C and P={PP:.5f} bar")
                print(f"{'':14} {'Mw(g/mol)':8} {'μ(Pa.s)':5}  T={T8C-T273:.1f}°C P={PP:.0f}")
                for visc_f in [linear_mix_rule, explog_mix_rule, hernzip_mix_rule, wilke_mix_rule]:
                    print_viscosity(g, PP, T3C,visc_f)

    visc_f = wilke_mix_rule
     
    print(f"--- using '{visc_f.__name__}'")
    # Print the densities at 8 C and 15 C  - - - - - - - - - - -
    
    plot_gases = []
    for g in display_gases:
        plot_gases.append(g)
    for g in ["H2", "CH4"]:
        plot_gases.append(g)
    
    # This next line was for when I was testing the reverse calculation for Tc and Pc from a,b
    # print_some_gas_data( ["H2",  "Ar", "O2", "CH4", "C2H6", "CO2", "He","N2"], 20) # 20 bar
    if False:
        print_some_gas_data(plot_gases, 0, 50) # 50 mbar
        print_some_gas_data(plot_gases, 20) # 20 bar
        print_some_gas_data(plot_gases, 220)

        print_wobbe(plot_gases,g, T15C)

        print(f"\n[H2O][CO2] of fuel gas")
        print(f"{'gas':13}{'Mw(g/mol)':6} {'Dew Pt':6}  {'C_':5}   {'H_':5}{'Hc(kJ/mol)':5}  fuel")
        for g in ['H2', 'CH4', 'C2H6']:
            print_fuelgas(g, 'Air')
        for g in gas_mixtures:
            print_fuelgas(g, 'Air')
        
    export_η_table('Air')
    
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
    c_20H = [condense(T, p, 'NG+20%H2', 'Air') for T in t_condense]
    
    plt.plot(t_condense-273.15, c_H2, label='Pure hydrogen', **plot_kwargs('H2'))
    plt.plot(t_condense-273.15, c_NG, label='Natural Gas', **plot_kwargs('NG'))
   
    #plt.title(f'Maximum boiler efficiency vs Condensing Temperature at {p} bar')
    plt.xlabel('Flue gas temperature (°C)')
    #plt.ylim([80, 100])
    plt.ylabel('Maximum boiler efficiency (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["η"])
    plt.plot(t_condense-273.15, c_20H, label='NG+20%H2', **plot_kwargs('NG+20%H2'))
    plt.legend()
    plt.savefig(fn["η20"])
  
    
    plt.close()
    
    # Plot the condensing curves at different % oxygen  - - - - - - - - - - -
    p = Atm
    for g in ['NG', 'H2']:
        plt.figure(figsize=(10, 6))
        c_g = {}
        for o in air_list:
            # print(f"Cp {o:8} {get_Cp(o):.3f}")
            c_g[o] = [condense(T, p, g, o) for T in t_condense]
            
            if o == 'Air':
                k = g  # linestyle for plot
            else:
                k = o
            plt.plot(t_condense-273.15, c_g[o], label=f"{g}+{o}", **plot_kwargs(k))
           
        plt.xlabel('Flue gas temperature (°C)')
        plt.ylabel('Maximum boiler efficiency (%)')
        #plt.ylim([80, 100])
        plt.legend()
        plt.grid(True)

        plt.savefig(f"peng_ηη_{g}.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        for o in air_list:
            if o == 'Air':
                continue
            e = list()
            for item1, item2 in zip(c_g[o], c_g['Air']):
                item = item1 - item2
                #item = item1/item2
                e.append(item)
            plt.plot(t_condense-273.15, e, label=f"{g}+{o}", **plot_kwargs(o))
        
        plt.xlabel('Flue gas temperature (°C)')
        plt.ylabel('Increase in maximum boiler efficiency (%)')
        #plt.ylim([80, 100])
        plt.legend()
        plt.grid(True)

        plt.savefig(f"peng_ηηη_{g}.png")
        plt.close()
      
    # Plot the Differential of the condensing curve  - - - - - - - - - - -
    p = Atm
    plt.figure(figsize=(10, 6))
    c_H2 = [d_condense(T, p, 'H2', 'Air') for T in t_condense]
    c_g = [d_condense(T, p, 'NG', 'Air') for T in t_condense]
     
    plt.plot(t_condense-273.15, c_H2, label='Pure hydrogen', **plot_kwargs('H2'))
    plt.plot(t_condense-273.15, c_g, label='Natural Gas', **plot_kwargs('NG'))
   
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
            a, b = z_mixture_rules(mix, T)
            Z_mix = solve_for_Z(T, pressure, a, b)
            Z_ng.append(Z_mix)
            
            # For density, the averaging across the mixture (Mw) is done before the calc. of ϱ
            ϱ_mix = pressure * mm / (Z_mix * R * T)
            ϱ_ng[mix].append(ϱ_mix)

            # μ_mix = viscosity_LGE(mm, T, ϱ_mix)
            μ_mix = get_viscosity(mix, pressure, T, visc_f)
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
            μ = get_viscosity(mix,P,T, visc_f)
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
    for g in ["H2"]:
        bf_gases.append(g)
        
    P = pressure
    bf_g = {}
    for mix in bf_gases:
        bf_g[mix] = []
        for T in temperatures:
            bf = get_blasius_factor(mix,P,T)
            bf_g[mix].append(bf)
        plt.plot(temperatures - T273, bf_g[mix], label= mix, **plot_kwargs(mix))


    plt.title(f'Blasius Parameter  ϱ^3/4.μ^1/4 vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Blasius Parameter  ϱ^3/4.μ^1/4 ')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["bf"])
    plt.close()


    # Blasius Parameter plot NORMALIZED wrt NG  - - - - - - - - - - -

    P = pressure
   
    bf_g = {}
    t = 0
    for P in [pressure, 2+Atm, 7+Atm, 19+Atm, 199+Atm]:
        t += 1
        #print(f"\nBlasius Parameter ϱ^3/4.μ^1/4 (normalised by NG) between {temperatures[0]-T273:4.1f}C and {temperatures[-1]-T273:4.1f}C at {P} bar")
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
            # plt.plot(temperatures - T273, bf_g[mix], label= mix+f" {P:5.0f} bar", **plot_kwargs(mix))
            plt.plot(temperatures - T273, bf_g[mix], label= mix+f" {P:5.0f} bar")
            bf_g[mix].sort()
            mn = bf_g[mix][0]
            mx = bf_g[mix][-1]
            mean = (mx + mn)/2
            rng = (mx - mn)/2
            pct = 100*rng/mean
            #print(f"{mix:5} {mean:9.4f} ±{rng:7.4f}  {pct:5.2f}%")
    plt.title(f'Normalised Blasius Parameter ϱ^3/4.μ^1/4  ratio of H2/NG values')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Normalised Blasius Parameter ratio wrt NG ')
    plt.grid(True)
    plt.legend()
    #plt.savefig(f"peng_bf_NG_{t}.png")
    plt.savefig(f"peng_bf_NG_.png")
    plt.close()

    # Pressure-drop plot NORMALIZED wrt NG  - - - - - - - - - - -
    Δp_g = {}
    t = 0
    for P in [pressure, 2+Atm, 7+Atm, 19+Atm, 199+Atm]:
        t += 1
        #print(f"\nΔp Blasius-fit Pressure drop ratio (normalised by NG) between {temperatures[0]-T273:4.1f}C and {temperatures[-1]-T273:4.1f}C at {P} bar")
        for mix in bf_gases:
            Δp_g[mix] = []
            if mix == "NG":
                 continue            
            for i in range(len(temperatures)):
                T = temperatures[i]
                # print(f"{i=} {T=} {mix=}")
                Δp_g[mix].append(get_Δp_ratio_br(mix,P,T))
            # plt.plot(temperatures - T273, Δp_g[mix], label= mix+f" {P:5.0f} bar", **plot_kwargs(mix))
            plt.plot(temperatures - T273, Δp_g[mix], label= mix+f" {P:5.0f} bar")
            Δp_g[mix].sort()
            mn = Δp_g[mix][0]
            mx = Δp_g[mix][-1]
            mean = (mx + mn)/2
            rng = (mx - mn)/2
            pct = 100*rng/mean
            #print(f"Δp {mix:5} {mean:9.4f} ±{rng:7.4f}  {pct:5.2f}%")
    plt.title(f'Δp Blasius-fit Pressure drop - ratio of H2/NG values')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Δp Pressure drop  ratio wrt NG ')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"peng_Δp_ratio_br.png")
    plt.close()
    
    # ϱ/Viscosity plot Kinematic EXPTL values at 298K - - - - - - - - - - -
    P = pressure
    re_g = {}

    for mix in plot_gases + ['He']:
        ϱ_ng[mix] =  [get_density(mix, P, T) for T in temperatures]
        μ_ng[mix] = [get_viscosity(mix, P, T, visc_f) for T in temperatures]
            
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
    pressure = Atm + 20
    plt.figure(figsize=(10, 5))

    # pure gases
    for g in ["H2"]: 
        ϱ_pg = [pressure * gas_data[g]['Mw'] / (peng_robinson(T, pressure, g) * R * T)  for T in temperatures]
        plt.plot(temperatures - T273, ϱ_pg, label = "pure " + g, **plot_kwargs(g))

    # Density plots for gas mixtures
    for mix in display_gases:
         ϱ_ng[mix] =  [get_density(mix, pressure, T) for T in temperatures]
         plt.plot(temperatures - T273, ϱ_ng[mix], label=mix, **plot_kwargs(mix))

    plt.title(f'Density vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density (kg/m³)')
    plt.legend()
    plt.grid(True)

    plt.savefig(fn["ϱ"])
    plt.close()
 
    # Plot the compressibility  as a function of Pressure - - - - - - - - - - -
    T = T8C
    P= None
    plt.figure(figsize=(10, 6))

    # Plot Z compressibility factor for pure hydrogen and natural gases
    pressures = np.linspace(1, 250, 100)  # bar

    plt.figure(figsize=(10, 5))

    for g, txt in [('H2','Pure hydrogen'), ('NG','Natural gas'), ('CH4','Pure methane')]:
        Z = [peng_robinson(T, p, g) for p in pressures]
        plt.plot(pressures, Z, label=txt, **plot_kwargs(g))

    plt.title(f'Z  Compressibility Factor vs Pressure at {T-T273:4.1f}°C')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Z Compressibility Factor')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_z_p.png")
    plt.close()

    # Plot P/Z  for pure hydrogen and natural gases

    for g, txt in [('H2','Pure hydrogen'), ('NG','Natural gas'), ('CH4','Pure methane')]:
        p_z = [pz(T, p, g) for p in pressures]
        plt.plot(pressures, p_z, label=txt, **plot_kwargs(g))
        
    plt.title(f'P / Z  vs Pressure at {T-T273:4.1f}°C')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Pressure / Z')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_z_pp.png")
    plt.close()

   # Plot dZ/dP v P  for pure hydrogen and natural gases

    for g, txt in [('H2','Pure hydrogen'), ('NG','Natural gas')]:
        dzdp_ = [dzdp(T, p, g) for p in pressures]
        plt.plot(pressures, dzdp_, label=txt, **plot_kwargs(g))
        
    plt.title(f'dZ/dP  vs Pressure at {T-T273:4.1f}°C')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('dZ/dP')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_z_pp.png")
    plt.close()
    
    # Plot velocity ratio for pure hydrogen and natural gas
    T8C = T273 + 8
    v_ratio = get_v_ratio('H2',94,T8C) 
    #print(f"Velocity ratio {v_ratio:.3f} at 94 bar {T8C-T273:4.1f}°C")
   
    for T in [T230, T250, T3C, T25C, T50C]:
    
        v_ratio = [ get_v_ratio('H2',p,T) for p in pressures]        
        vr_max = max(v_ratio)
        #print(f"Velocity ratio min:{v_ratio[0]:.3f} max:{vr_max:.3f} at  {T-T273:4.1f}°C")
        plt.plot(pressures, v_ratio, label=f"{T-T273:4.0f}°C")
    
    plt.title(f'Velocity ratio v(H2)/v(NG)  vs Pressure ')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Velocity ratio v(H2)/v(NG)')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_v_ratio.png")
    plt.close()

    
    # Plot Blasius Parameter for pure hydrogen and natural gases
    pressures = np.linspace(1, 8.1, 100)  # bar
    T = T273+8

    plt.figure(figsize=(10, 5))

    bf_g = {}

    for g in plot_gases:
        bf_g[g] = []

        for p in pressures:
            bf = get_blasius_factor(g,p,T)
            bf_g[g].append(bf)

        plt.plot(pressures , bf_g[g], label=g, **plot_kwargs(g))

    plt.title(f'Blasius Parameter vs Pressure at {T-T273:4.1f}°C')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Blasius Parameter ϱ^3/4.μ^1/4')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_bf_p.png")
    plt.close()    
    # Plot Blasius Parameter for pure hydrogen and natural gases
    pressures = np.linspace(1, 8.1, 100)  # bar
    T = T273+8

    plt.figure(figsize=(10, 5))

    bf_g = {}

    for g in plot_gases:
        bf_g[g] = []

        for p in pressures:
            bf = get_blasius_factor(g,p,T)
            bf_g[g].append(bf)

        plt.plot(pressures , bf_g[g], label=g, **plot_kwargs(g))

    plt.title(f'Blasius Parameter vs Pressure at {T-T273:4.1f}°C')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Blasius Parameter ϱ^3/4.μ^1/4')
    plt.legend()
    plt.grid(True)

    plt.savefig("peng_bf_p.png")
    plt.close()

    # Plot viscosity as a function of pressure - looking for bugs
    pressures = np.linspace(0.001, 220, 100)  # bar
    for g in ['NG','H2']:
        plt.figure(figsize=(10, 5))

        mu_g = {}
  
        for T in [T50C, T25C, T8C, T250,T230]:
            mu_g[g] = []
            for p in pressures:
                mu = get_viscosity(g,p,T, visc_f)
                mu_g[g].append(mu)

            plt.plot(pressures , mu_g[g], label=f"{g} {T-T273:.0f}°C")

        plt.title(f'Viscosity vs Pressure ')
        plt.xlabel('Pressure (bar)')
        plt.ylabel('Dynamic Viscosity (μPa.s)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f"peng_{g}μ_p.png")
        plt.close()

    # Plot viscosity as a function of temp - looking for bugs
    temps = np.linspace(-50, 50, 100)  # bar
    for g in ['NG','H2']:
        plt.figure(figsize=(10, 5))

        mu_g = {}
  
        for P in [220, 150,30,1,0.01]:
            mu_g[g] = []
            for T in temps:
                mu = get_viscosity(g,P,T+T273, visc_f)
                mu_g[g].append(mu)

            plt.plot(temps , mu_g[g], label=f"{g} {P:.0f} bar")

        plt.title(f'Viscosity vs Temperature ')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Dynamic Viscosity (μPa.s)')
        plt.legend()
        plt.grid(True)

        plt.savefig(f"peng_{g}μ_T.png")
        plt.close()

    # Plot density as a function of pressure - looking for bugs
    pressures = np.linspace(0.001, 100, 100)  # bar
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
