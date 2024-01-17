import functools 
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys

from cycler import cycler

"""This code written Philip Sargent, starting in December 2023, by  to support a paper on the replacement of natural
gas in the UK distribution grid with hydrogen.

For general information, commercial natural gas typically contains 85 to 90 percent methane, with the remainder mainly nitrogen and ethane, and has a calorific value of approximately 38 megajoules (MJ) per cubic metre

Wobbe Number (WN) (i) ≤51.41 MJ/m³, and  (ii)  ≥47.20 MJ/m³
https://www.nationalgas.com/data-and-operations/quality
≥46.5MJ/m3 after 6 April 2025 https://www.hse.gov.uk/gas/gas-safety-management-regulation-changes.htm

UNITS: bar, K, litres
"""
# This algorithm does NOT deal with the temperature dependence of alpha properly. 
# The code should be rearranged to calculate alpha for each point ont he plot for each gas.

# Peng-Robinson Equation of State constants for Hydrogen and Methane
# Omega is the acentric factor is a measure of the non-sphericity of molecules; 
# a higher acentric factor indicates greater deviation from spherical shape
# PR constants data from ..aargh lost it.

R = 0.083144626  # l.bar/(mol.K)  SI after 2019 redefinition of Avagadro and Boltzmann constants
# 1 bar is today defined as 100,000 Pa not 1atm

Atm = 1.01325 # bar 
T273 = 273.15 # K


# "In this Schedule, the reference conditions are 15C and 1.01325 bar"
# UK law for legal Wobbe limits and their calculation.
# https://www.legislation.gov.uk/uksi/2023/284/made

# L M N for H2 from https://www.researchgate.net/figure/Coefficients-for-the-Twu-alpha-function_tbl3_306073867
# 'L': 0.7189,'M': 2.5411,'N': 10.2,
# for cryogenic vapour pressure. This FAILS for room temperature work, producing infinities. Use omega instead.
# possibly re-visit using the data from Jaubert et al. on H2 + n-butane.

# Hc is standard enthalpy of combustion at 298K, i.e. HHV,  in MJ/mol (negative sign not used). 
#     Data from Wikipedia & https://www.engineeringtoolbox.com/standard-heat-of-combustion-energy-content-d_1987.html
# Wb is Wobbe index: MJ/m³
# RD is relative density (air is  1)
# Vs is viscosity and temp. f measurement as tuple (microPa.s, K, exponent(pressure))
# All viscosities from marcia l. huber and allan h. harvey,
#https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=907539

# Viscosity temp ratameters calibrated by log-log fit to https://myengineeringtools.com/Data_Diagrams/Viscosity_Gas.html
# using aux. program visc-temp.py in this repo.

gas_data = {
    'H2': {'Tc': 33.2, 'Pc': 13.0, 'omega': -0.22, 'Mw':2.015, 'Vs': (8.9,300, 0.692), 'Hc':0.285826, 'H_':2},
    'CH4': {'Tc': 190.56, 'Pc': 45.99, 'omega': 0.01142, 'Mw': 16.04246, 'Vs': (11.1,300, 1.03), 'Hc':0.890602, 'C_':1, 'H_':4}, # Hc from ISO_6976
    'C2H6': {'Tc': 305.32, 'Pc': 48.72, 'omega': 0.099, 'Mw': 30.07, 'Vs': (9.4,300, 0.87), 'Hc':1.5639, 'C_':2, 'H_':6}, # 
    'C3H8': {'Tc': 369.15, 'Pc': 42.48, 'omega': 0.1521, 'Mw': 44.096, 'Vs': (8.2,300, 0.93), 'Hc':2.21866, 'C_':3, 'H_':8}, # https://www.engineeringtoolbox.com/propane-d_1423.html
    'nC4': {'Tc': 425, 'Pc': 38,  'omega': 0.20081, 'Mw': 58.1222, 'Vs': (7.5,300, 0.950), 'Hc':2.876841, 'C_':4, 'H_':10}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Butane.html https://www.engineeringtoolbox.com/butane-d_1415.html 
    'iC4': {'Tc': 407.7, 'Pc': 36.5, 'omega': 0.1835318, 'Mw': 58.1222, 'Vs': (7.5,300, 0.942), 'Hc':2.87728, 'C_':4, 'H_':10}, # omega  http://www.coolprop.org/fluid_properties/fluids/IsoButane.html https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=1F https://webbook.nist.gov/cgi/cbook.cgi?Name=butane&Units=SI Viscocity assumed same as nC4
    'nC5': {'Tc': 469.8, 'Pc': 33.6, 'omega': 0.251032, 'Mw': 72.1488, 'Vs': (6.7,300, 1.0), 'Hc':3.53609, 'C_':5, 'H_':12}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Pentane.html     
    'iC5': {'Tc': 461.0, 'Pc': 33.8, 'omega': 0.2274, 'Mw': 72.1488, 'Vs': (6.7,300, 0.94), 'Hc':3.52917, 'C_':5, 'H_':12}, # omega http://www.coolprop.org/fluid_properties/fluids/Isopentane.html  Viscocity assumed same as nC5 
    
    'neoC5': {'Tc': 433.8, 'Pc': 31.963, 'omega': 0.1961, 'Mw': 72.1488, 'Vs': (6.9326,300, 0.937), 'Hc':3.51495, 'C_':5, 'H_':12},
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C463821&Units=SI&Mask=4#Thermo-Phase
    # omega from http://www.coolprop.org/fluid_properties/fluids/Neopentane.html
    
    'C6':  {'Tc': 507.6, 'Pc': 30.2, 'omega': 0.1521, 'Mw': 86.1754, 'Vs': (8.6,400, 1.03), 'Hc':4.19475, 'C_':6, 'H_':14}, # omega is 0.2797 isohexane    
    'CO2': {'Tc': 304.2, 'Pc': 73.8, 'omega': 0.228, 'Mw': 44.01, 'Vs': (15.0,300, 0.872), 'Hc':0, 'C_':0, 'H_':0}, # https://en.wikipedia.org/wiki/Acentric_factor
    'H2O': {'Tc': 647.1, 'Pc': 220.6, 'omega': 0.344292, "Mw": 18.015, 'Vs': (9.8,300, 1.081), 'Hc':0, 'C_':0, 'H_':0}, # https://link.springer.com/article/10.1007/s10765-020-02643-6/tables/1
    'N2': {'Tc': 126.21, 'Pc': 33.958, 'omega': 0.0372, 'Mw':28.013, 'Vs': (17.9,300, 0.658), 'Hc':0, 'C_':0, 'H_':0}, #  omega http://www.coolprop.org/fluid_properties/fluids/Nitrogen.html
    'He': {'Tc': 5.2, 'Pc': 2.274, 'omega': -0.3836, 'Mw': 4.0026, 'Vs': (19.9,300, 0.69), 'Hc':0, 'C_':0, 'H_':0},  # omega http://www.coolprop.org/fluid_properties/fluids/Helium.html
    # https://eng.libretexts.org/Bookshelves/Chemical_Engineering/Distillation_Science_(Coleman)/03%3A_Critical_Properties_and_Acentric_Factor
    # N2 https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.2c00363/suppl_file/ie2c00363_si_001.pdf
    # N2 omega is from https://en.wikipedia.org/wiki/Acentric_factor
    'Ar': {'Tc': 150.687, 'Pc': 48.630, 'omega': 0, 'Mw': 39.948, 'Vs': (22.7,300, 0.77), 'Hc':0, 'C_':0, 'H_':0}, #https://en.wikipedia.org/wiki/Acentric_factor
    'O2': {'Tc': 154.581, 'Pc': 50.43, 'omega': 0.022, 'Mw': 31.9988, 'Vs': (20.7,300, 0.72), 'Hc':0, 'C_':0, 'H_':0},# http://www.coolprop.org/fluid_properties/fluids/Oxygen.html
    }

# Natural gas compositions (mole fractions)
gas_mixtures = {
    
    'NG': {'CH4': 0.895514, 'C2H6': 0.051196, 'C3H8': 0.013549, 'nC4': 0.001269, 'iC4': 0.002162, 'nC5': 2e-05, 'iC5': 0.000344, 'neoC5': 0.003472, 'C6': 0.002377, 'CO2': 0.020743, 'N2': 0.009354}, # Normalized.email John Baldwin 30/12/2023 - Fordoun
    # 'Fordoun': { 'CH4':  0.900253, 'C2H6':  0.051467, 'C3H8':  0.013621, 'iC4':  0.001276, 'nC4':  0.002173, 'neoC5':  0.000020,'iC5':  0.000346, 'nC5':  0.003490,  'C6':  0.002390, 'CO2':  0.020853, 'N2':  0.009404, }, # original email John Baldwin 30/12/2023, unnormalized
    'Groening': {'CH4': 0.813, 'C2H6': 0.0285, 'C3H8': 0.0037, 'nC4': 0.0014, 'nC5': 0.0004, 'C6': 0.0006, 'CO2': 0.0089, 'N2': 0.1435, 'O2': 0}, # Groeningen gas https://en.wikipedia.org/wiki/Groningen_gas_field
    
    'Tokyo': {'CH4': 0.896, 'C2H6': 0.056, 'C3H8': 0.0034, 'iC4': 0.0007, 'nC4': 0.0007, 'N2': 0.0432 }, # Tokyo town gas, with assumed missing gas all N2 http://members.igu.org/html/wgc2009/papers/docs/wgcFinal00580.pdf
    
    'Biomethane': {'CH4': 0.92,  'C3H8': 0.04, 'CO2': 0.04 }, # wobbe central, not a real natural gas  https://www.gasgovernance.co.uk/sites/default/files/ggf/Impact%20of%20Natural%20Gas%20Composition%20-%20Paper_0.pdf
 
    '10C2-10N': {'CH4': 0.80,  'C3H8': 0.1, 'N2': 0.1 }, # RH corner of allowable wobbe polygon ?
    '7C2-2N': {'CH4': 0.91,  'C3H8': 0.07, 'N2': 0.02 }, # top corner of allowable wobbe polygon ?
  
    'mix6': {'CH4': 0.8, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02, 'N2': 0.10}, # ==mix6 from      https://backend.orbit.dtu.dk/ws/files/131796794/FPE_D_16_00902R1.pdf - no, somewhere else..

    'NTS79': {'CH4': 0.9363, 'C2H6': 0.0325, 'C3H8': 0.0069, 'nC4': 0.0027, 'CO2': 0.0013, 'N2': 0.0178, 'He': 0.0005, 'nC5': 0.002}, # https://en.wikipedia.org/wiki/National_Transmission_System
    # https://en.wikipedia.org/wiki/National_Transmission_System
    # This NTS composition from Wikipedia actually comes from 1979 !  Cassidy, Richard (1979). Gas: Natural Energy. London: Frederick Muller Limited. p. 14.


    '11D': { 'CH4':  0.88836, 'C2H6':  0.04056, 'C3H8':  0.00997, 'iC4':  0.00202, 'nC4':  0.00202, 'iC5':  0.00050, 'nC5':  0.00050, 'neoC5':  0.00050, 'C6':  0.00049, 'CO2':  0.01512, 'N2':  0.03996, }, # normlized 11D gas from Duchowny22, doi:10.1016/j.egyr.2022.02.289
    
    'Algerian': {'CH4': 0.867977, 'C2H6': 0.085862, 'C3H8': 0.011514, 'iC4': 0.000829, 'nC4': 0.001044, 'iC5': 0.000205, 'nC5': 0.000143, 'C6': 0.000164, 'CO2': 0.018505, 'N2': 0.012927, 'He': 0.000829}, # NORMALIZED # Algerian NG, Romeo 2022, C6+
 
    'North Sea': {'CH4': 0.836, 'C2H6': 0.0748, 'C3H8':0.0392, 'nC4':0.0081, 'iC4':0.0081, 'nC5':0.0015, 'iC5':0.0014, 'CO2':0.0114, 'N2':0.0195}, # North Sea gas [Hassanpou] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7347886/
        
    # 'ethane': {'C2H6': 1.0}, # ethane, but using the mixing rules: software test check
    # 'propane': {'C3H8': 1.0}, # ethane, but using the mixing rules: software test check
    
    # dry air from Picard2008, who is quoting 
    'dryAir':  {'N2': 0.780872, 'O2': 0.209396, 'CO2': 0.00040, 'Ar': 0.009332},
    #'Air':  {'N2': 0.761749, 'O2': 0.204355, 'CO2': 0.00039, 'Ar': 0.009112, 'He': 5.0e-06, 'H2O': 0.024389} # https://www.thoughtco.com/chemical-composition-of-air-604288
    # But ALSO adding 2.5% moisture to the air and normalising
}

"""reduce the lower limit for Wobbe Index from 47.2 MJ/m³  to 46.50 MJ/m³ was approved by HSE. 
This shall enter into force from 6 April 2025
"Gas Ten Year Statement December 2023"
https://www.nationalgas.com/document/144896/download
"""
gas_mixture_properties = {
    'Algerian': {'Wb': 49.992, 'HHV': 39.841, 'RD': 0.6351}, #Algerian NG, Romeo 2022, C6+ BUT not according to my calcs. for Hc and wobbe.
    'Air': {'Hc': 0} 
}


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


# Binary interaction parameters for hydrocarbons for Peng-Robinson
# based on the Chueh-Prausnitz correlation
# from https://wiki.whitson.com/eos/cubic_eos/
# also from Privat & Jaubert, 2023 (quoting a 1987 paper).
# Note that the default value (in the code) is -0.019 as this represents ideal gas behaviour.

# There is a full table of BIP using teh GCM method on https://wiki.whitson.com/eos/bips/index.html#coutinho-et-al-correlation
# using these might be better than teh Coutinho equation - future work.

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
    
def print_fuelgas(g):
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
        n2 = gas_mixtures['dryAir']['O2']
        print(f"{g:15} {mm:6.3f}  {c_:5.3f}   {h_:5.3f} {hc*1000:9.3f} {mff*100:8.4f} %")
    
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

def print_ng():
    g = "NG"
    print(f"NatGas at Fordoun NTS 20th Jan.2021")
    nts = gas_mixtures[g]
    for f in nts:
        mv, hcmv, hc = get_Hc(f, 298)
        if not hc:
            hc = 0
        # print(f"{f:5}\t{nts[f]*100:8.5f} %{gas_data[f]['C_']:3}{gas_data[f]['H_']:3} {hc*1000:6.1f} kJ/mol ")
    o2_fuel = do_flue_rules(g,'C_') + do_flue_rules(g,'H_')/4
    o2_gas = o2_fuel * get_fuel_fraction(g)
    # print("")
    print(f"{'NG':5}\t C_={do_flue_rules(g,'C_'):6.4f} H_={do_flue_rules(g,'H_'):6.4f} O2:fuel {o2_fuel:6.4f} O2:gas ratio {o2_gas:6.4f}")
    
    o2_in = o2_fuel * get_fuel_fraction(g) * 1.15 # 15% excess air
    n2_in = 1/gas_mixtures['dryAir']['O2']
    print(air, o2_in, n2_in)

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
    for s in ["z", "ϱ", "μ", "bf", "bf_NG"]:
        f = stem  + "_" + s
        fn[s] = pl.Path(f).with_suffix(".png") 

    for mix in gas_mixtures:
        composition = gas_mixtures[mix]
        check_composition(mix, composition)

    print_ng()
    
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
    print(f"{'gas':13}{'Mw(g/mol)':6}   {'C_':5}   {'H_':5}{'Hc(kJ/mol)':5}  fuel")
    for g in ['H2', 'CH4', 'C2H6']:
        print_fuelgas(g)
    for g in gas_mixtures:
        print_fuelgas(g)

    # Plot defaults
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 6),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

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
    Z_CH4 = [peng_robinson(T, pressure, 'CH4') for T in temperatures]
    plt.plot(temperatures - T273, Z_CH4, label='Pure methane', **plot_kwargs('CH4'))


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

    plt.title(f'Z  Compressibility Factor vs Temperature at {pressure} bar')
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


    plt.title(f'Blasius Parameter ϱ^3/4.μ^1/4 vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Blasius Parameter ')
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


    # ϱ/Viscosity plot Kinematic EXPTL values at 298K - - - - - - - - - - -

    P = pressure
    re_g = {}

    for mix in plot_gases:
        if mix in gas_data:
            ϱ_ng[mix] =  [get_density(mix, P, T) for T in temperatures]
            μ_ng[mix] = [get_viscosity(mix, P, T) for T in temperatures]
            
        re_g[mix] = []
        for i in range(len(μ_g[mix])):
            re_g[mix].append( ϱ_ng[mix][i] / μ_g[mix][i])
        plt.plot(temperatures - T273, re_g[mix], label= mix, **plot_kwargs(mix))
        
    plt.title(f'Density / Dynamic Viscosity vs Temperature at {pressure} bar')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density/ Dynamic Viscosity (kg/m³)/(μPa.s) ')
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
    pressures = np.linspace(1, 7, 100)  # bar
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
