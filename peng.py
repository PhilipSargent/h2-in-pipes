import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import sys

"""For general information, commercial natural gas typically contains 85 to 90 percent methane, with the remainder mainly nitrogen and ethane, and has a calorific value of approximately 38 megajoules (MJ) per cubic metre

Wobbe Number (WN) (i) ≤51.41 MJ/m3, and  (ii)  ≥47.20 MJ/m3
https://www.nationalgas.com/data-and-operations/quality

UNITS: bar, K, litres
"""
# This algorithm does NOT deal with the temperature dependence of alpha properly. 
# The code should be rearranged to calculate alpha for each point ont he plot for each gas.

# Peng-Robinson Equation of State constants for Hydrogen and Methane
# Omega is the acentric factor is a measure of the non-sphericity of molecules; 
# a higher acentric factor indicates greater deviation from spherical shape
# PR constants data from ..aargh lost it.

R = 0.08314462  # l.bar/(mol.K)

# L M N for H2 from https://www.researchgate.net/figure/Coefficients-for-the-Twu-alpha-function_tbl3_306073867
# 'L': 0.7189,'M': 2.5411,'N': 10.2,
# for cryogenic vapour pressure. This FAILS for room temperature work, producing infinities. Use omega instead.

# Hc is heat of combusiton in kJ/mol
# HHV in mJ/M3
# Wb is Wobbe index: MJ/m3
# RD is relative density (air is  1)
# Vs is viscosity and temp. f measurement as tuple (microPa.s, K)
# All viscosities from marcia l. huber and allan h. harvey,
#https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=907539

gas_data = {
    'H2': {'Tc': 33.2, 'Pc': 13.0, 'omega': -0.22, 'Mw':2.015, 'Vs': (8.9,300)},
    'CH4': {'Tc': 190.56, 'Pc': 45.99, 'omega': 0.01142, 'Mw': 16.0428, 'Vs': (11.1,300)},
    'C2H6': {'Tc': 305.32, 'Pc': 48.72, 'omega': 0.099, 'Mw': 30.07, 'Vs': (9.4,300)}, # 
    'C3H8': {'Tc': 369.15, 'Pc': 42.48, 'omega': 0.1521, 'Mw': 44.096, 'Vs': (8.2,300)}, # https://www.engineeringtoolbox.com/propane-d_1423.html
    'nC4': {'Tc': 425, 'Pc': 38,  'omega': 0.20081, 'Mw': 58.1222, 'Vs': (7.5,300), 'Hc':2-877.5}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Butane.html https://www.engineeringtoolbox.com/butane-d_1415.html 
    'iC4': {'Tc': 407.7, 'Pc': 36.5, 'omega': 0.1835318, 'Mw': 58.1222, 'Vs': (7.5,300)}, # omega  http://www.coolprop.org/fluid_properties/fluids/IsoButane.html https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=1F https://webbook.nist.gov/cgi/cbook.cgi?Name=butane&Units=SI Viscocity assumed same as nC4
    'nC5': {'Tc': 469.8, 'Pc': 33.6, 'omega': 0.251032, 'Mw': 72.1488, 'Vs': (6.7,300)}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Pentane.html     
    'iC5': {'Tc': 461.0, 'Pc': 33.8, 'omega': 0.2274, 'Mw': 72.1488, 'Vs': (6.7,300)}, # omega http://www.coolprop.org/fluid_properties/fluids/Isopentane.html  Viscocity assumed same as nC5    
    'C6':  {'Tc': 507.6, 'Pc': 30.2, 'omega': 0.1521, 'Mw': 86.1754, 'Vs': (8.6,400)}, # omega is 0.2797 isohexane    
    'CO2': {'Tc': 304.2, 'Pc': 73.8, 'omega': 0.228, 'Mw': 44.01, 'Vs': (15.0,300)}, # https://en.wikipedia.org/wiki/Acentric_factor
    'H2O': {'Tc': 647.1, 'Pc': 220.6, 'omega': 0.344292, "Mw": 18.015, 'Vs': (9.8,300)}, # https://link.springer.com/article/10.1007/s10765-020-02643-6/tables/1
    'N2': {'Tc': 126.21, 'Pc': 33.958, 'omega': 0.0372, 'Mw':28.013, 'Vs': (17.9,300)}, #  omega http://www.coolprop.org/fluid_properties/fluids/Nitrogen.html
    'He': {'Tc': 5.2, 'Pc': 2.274, 'omega': -0.3836, 'Mw': 4.0026, 'Vs': (19.9,300)},  # omega http://www.coolprop.org/fluid_properties/fluids/Helium.html
    # https://eng.libretexts.org/Bookshelves/Chemical_Engineering/Distillation_Science_(Coleman)/03%3A_Critical_Properties_and_Acentric_Factor
    # N2 https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.2c00363/suppl_file/ie2c00363_si_001.pdf
    # N2 omega is from https://en.wikipedia.org/wiki/Acentric_factor
    'Ar': {'Tc': 150.687, 'Pc': 48.630, 'omega': 0, 'Mw': 39.948, 'Vs': (22.7,300)}, #https://en.wikipedia.org/wiki/Acentric_factor
    'O2': {'Tc': 154.581, 'Pc': 50.43, 'omega': 0.022, 'Mw': 31.9988, 'Vs': (20.7,300)},# http://www.coolprop.org/fluid_properties/fluids/Oxygen.html
    }

# Natural gas compositions (mole fractions)
gas_mixtures = {
    'GG': {'CH4': 0.827, 'C2H6': 0.03, 'C3H8': 0.003, 'CO2': 0, 'N2': 0.14}, # Groeningen gas

    'Wobbe mix': {'CH4': 0.9,  'C3H8': 0.04,  'N2': 0.06}, # wobbe central, not a real natural gas  https://www.gasgovernance.co.uk/sites/default/files/ggf/Impact%20of%20Natural%20Gas%20Composition%20-%20Paper_0.pdf
    
    'NTS': {'CH4': 0.8, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02, 'N2': 0.10}, # ==mix6 from      https://backend.orbit.dtu.dk/ws/files/131796794/FPE_D_16_00902R1.pdf

    'Algerian': {'CH4': 0.86486, 'C2H6': 0.08788, 'C3H8': 0.01179, 'iC4': 0.00085,  'nC4': 0.00107,
         'iC5': 0.00021, 'nC5': 0.00015,'C6': 0.00017,'CO2': 0.01894, 'N2': 0.01323, 'He': 0.00085}, # Algerian NG, Romeo 2022, C6+
         
    'North Sea': {'CH4': 0.836, 'C2H6': 0.0748, 'C3H8':0.0392, 'nC4':0.0081, 'iC4':0.0081, 
        'nC5':0.0015, 'iC5':0.0014, 'CO2':0.0114, 'N2':0.0195}, # North Sea gas https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7347886/
        
    # 'ethane': {'C2H6': 1.0}, # ethane, but using the mixing rules: software test check
    # 'propane': {'C3H8': 1.0}, # ethane, but using the mixing rules: software test check
    'Air': {'N2': 0.78084, 'O2': 0.209476, 'CO2':0.0004,'Ar': 0.00934, 'He': 0.00000524}, # https://www.thoughtco.com/chemical-composition-of-air-604288
}


gas_mixture_properties = {
    'Algerian': {'Wb': 49.992, 'HHV': 39.841, 'RD': 0.6351} #Algerian NG, Romeo 2022, C6+
}

# 20% H2, remainder N.Sea gas
fifth = {}
fifth['H2'] = 0.2
nsea = gas_mixtures['NTS']
for g in nsea:
    fifth[g] = nsea[g]*0.8
gas_mixtures['NTS+20% H2'] = fifth
    
# Binary interaction parameters for hydrocarbons for Peng-Robinson
# based on the Chueh-Prausnitz correlation
# from https://wiki.whitson.com/eos/cubic_eos/
# also from Privat & Jaubert, 2023 (quoting a 1987 paper)

# BUT we should be calculating these from other thermodynamic data really...?
k_ij = {
    'CH4': {'C2H6': 0.0021, 'C3H8': 0.007, 'iC4': 0.013, 'nC4': 0.012, 'iC5': 0.018, 'nC5': 0.018, 'C6': 0.021, 'CO2': 0},
    'C2H6': {'C3H8': 0.001, 'iC4': 0.005, 'nC4': 0.004, 'iC5': 0.008, 'nC5': 0.008, 'C6': 0.010},
    'C3H8': {'iC4': 0.001, 'nC4': 0.001, 'iC5': 0.003, 'nC5': 0.003, 'C6': 0.004},
    'iC4': {'nC4': 0.0, 'iC5': 0.0, 'nC5': 0.0, 'C6': 0.001},
    'nC4': {'iC5': 0.001, 'nC5': 0.001, 'C6': 0.001},
    'iC5': {'C6': 0.0}, # placeholder
    'nC5': {'C6': 0.0}, # placeholder    
    'C6': {'C6': 0.0}, # placeholder
    'CO2': {'C6': 0.0}, # placeholder
    'N2': {'C6': 0.0}, # placeholder
    'He': {'C6': 0.0}, # placeholder
    'H2': {'C6': 0.0}, # placeholder
    'O2': {'C6': 0.0}, # placeholder
    'Ar': {'C6': 0.0}, # placeholder
}

def check_composition(mix, composition):
    """Checks that the mole fractions add up to 100%"""
    eps = 0.00001
    warn = 0.02 # 2 %
    
    x = 0
    norm = 1
    for gas, xi in composition.items():
       x += xi
    norm = x
    if abs(x - 1.0) > eps:
        if abs(x - 1.0) < warn:
            print(f"--------- Warning gas mixture '{mix}', {100*(1-warn)}% > {100*x:.2f} > {100*(1+warn)}%. Normalizing.")
        else:
            print(f"######### BAD gas mixture '{mix}', molar fractions add up to {x} !!!")
            
    # Normalise all the mixtures, even if they are close to 100%
    for gas, xi in composition.items(): 
        x = xi/norm
        gas_mixtures[mix][gas] = x
        
def do_mm_rules(mix):
    """Calculate the mean molecular mass of the gas mixture"""
    mm_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        mm_mix += x * gas_data[gas]['Mw']
    
    return mm_mix

        
def do_notwilke_rules(mix):
    """Calculate the mean viscosity of the gas mixture"""
    vs_mix = 0
    composition = gas_mixtures[mix]
    for gas, x in composition.items():
        # Linear mixing rule for volume factor
        vs, _ = gas_data[gas]['Vs'] # ignore T, so value for hexane will be bad
        vs_mix += x * vs
    
    return vs_mix

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
            k = 0
            if gas2 in k_ij[gas1]:
                k = k_ij[gas1][gas2]
            if gas1 in k_ij[gas2]:
                k = k_ij[gas2][gas1]
                
            # while b just depends on the mixture, a is temperature dependent. 
            # we fudge it with a fixed temp. a for the moment..
            
            a_mix += x1 * x2 * (1 - k) * (a1 * a2)**0.5  
            
       # Return the mixture's parameters for the P-R law
    return { mix: 
        {
            'a_mix': a_mix,
            'b_mix': b_mix,
         }
    }

"""This function uses simple mixing rules to calculate the mixture’s critical properties. The kij parameter, which accounts for the interaction between different gases, is assumed to be 0 for simplicity. In practice, kij may need to be adjusted based on experimental data or literature values for more accurate results.

Please note that this is a simplified approach and may not be accurate for all gas mixtures. For precise calculations, especially for complex mixtures or those under extreme conditions, it’s recommended to use specialized software or databases that provide more sophisticated mixing rules and interaction parameters. """


def get_LMN(omega):
    """Twu (1991) suggested a replacement for the alpha function, which instead of depending
        only on T & omega, depends on T, L, M, N (new material constants)
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8851615/pdf/ao1c06519.pdf
    """
    # These equations are from Privat & Jaubert (2023)
    # https://www.sciencedirect.com/science/article/pii/S0378381222003168
    L = 0.0544 + 0.7536 * omega + 0.0297 * omega**2
    M = 0.8678 - 0.1785 * omega + 0.1401 * omega**2
    N = 2
    
    return L, M, N

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

def solve_for_Z(T, P, a, b):
   
    # Solve cubic equation for Z the compressibility
    A = a * P / (R * T)**2 # should have alpha in here? No..
    B = b * P / (R * T)
    
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

    
# Peng-Robinson Equation of State
def peng_robinson(T, P, gas):
    if gas not in gas_mixtures:    
        a, b = a_and_b(gas, T)
    else:
        constants = z_mixture_rules(gas, T)
        a = constants[gas]['a_mix']
        b = constants[gas]['b_mix'] 
        
    Z = solve_for_Z(T, P, a, b)
    return Z

def viscosity_LGE(Mw, T_k, rho):
    """The  Lee, Gonzalez, and Eakin method, originally expressed in 'oilfield units'
    of degrees Rankine and density in g/cc, with a result in centiPoise
    doi.org/10.2118/1340-PA 1966
    Updated to SI: PetroWiki. (2023). 
    https://petrowiki.spe.org/Natural_gas_properties. 
    """

    T = T_k * 9/5 # convert Kelvins to Rankine
   
    # Constants for the Lee, Gonzalez, and Eakin #1
    k = (7.77 + 0.0063 * Mw) * T**1.5 / (122.4 + 12.9 * Mw + T)
    x = 2.57 + 1914.5 / T + 0.0095 * Mw # * np.exp(-0.025 * MWg) hallucination!
    y = 1.11 - 0.04 * x

    # Constants for the Lee, Gonzalez, and Eakin #2
    k = (9.4 + 0.02 * Mw) * T**1.5 / (209 + 19 * Mw + T)
    x = 3.5 + 986 / T + 0.01 * Mw
    y = 2.4 - 0.2 * x

    mu = 0.1 * k * np.exp(x * (rho / 1000)**y) #microPa.s

    return mu 
    
# ---------- main program starts here ------------- #

program = sys.argv[0]
stem = str(pl.Path(program).with_suffix(""))
fn={}
for s in ["z", "rho", "mu"]:
    f = stem  + "_" + s
    fn[s] = pl.Path(f).with_suffix(".png") 

for mix in gas_mixtures:
    composition = gas_mixtures[mix]
    check_composition(mix, composition)

# Plot the compressibility  - - - - - - - - - - -
for gas in gas_data:
    a , b = a_and_b(gas, 298.15)
    #print(gas, a, b)

# test the P-R parameter mixtures rules
for mix in gas_mixtures:
    mixture_constants = z_mixture_rules(mix, 298.15)
    #print(mixture_constants, " at T=298.15")
    
# Plot Z compressibility factor for pure hydrogen and natural gases
temperatures = np.linspace(243.15, 323.15, 100)  
pressure = 1.075  # bar

plt.figure(figsize=(10, 6))

# Plot for pure hydrogen
Z_H2 = [peng_robinson(T, pressure, 'H2') for T in temperatures]
plt.plot(temperatures - 273.15, Z_H2, label='Pure hydrogen', linestyle='dashed')

    
# Plot for pure methane
Z_CH4 = [peng_robinson(T, pressure, 'CH4') for T in temperatures]
plt.plot(temperatures - 273.15, Z_CH4, label='Pure methane', linestyle='dashed')


# Plot for natural gas compositions. Now using correct temperature dependence of 'a'
rho_ng = {}
μ_ng = {}

for mix in gas_mixtures:
    mm = do_mm_rules(mix) # mean molar mass
    rho_ng[mix] = []
    μ_ng[mix] = []

    Z_ng = []
    for T in temperatures:
        constants = z_mixture_rules(mix, T)
        a = constants[mix]['a_mix']
        b = constants[mix]['b_mix']
        Z_mix = solve_for_Z(T, pressure, a, b)
        Z_ng.append(Z_mix)
        
        rho_mix = pressure * mm / (Z_mix * R * T)
        rho_ng[mix].append(rho_mix)


        μ_mix = viscosity_LGE(mm, T, rho_mix)
        μ_ng[mix].append(μ_mix)

    if mix == "Air":
        continue 
    plt.plot(temperatures - 273.15, Z_ng, label=mix)

plt.title(f'Z  Compressibility Factor vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Z Compressibility Factor')
plt.legend()
plt.grid(True)

plt.savefig(fn["z"])
plt.close()

# Viscosity plot  LGE - - - - - - - - - - -
# Vicosity for gas mixtures LGE
for mix in gas_mixtures:
   plt.plot(temperatures - 273.15, μ_ng[mix], label=mix)

# Viscosity plots for pure gases
μ_g = {}
for g in ["H2", "CH4", "N2"]:
    μ_g[g] = []
    mm_g = gas_data[g]['Mw']
    for T in temperatures:
        rho_g= pressure * mm / (peng_robinson(T, pressure, g) * R * T)
        μ = viscosity_LGE(mm_g, T, rho_g)
        μ_g[g].append(μ)
    plt.plot(temperatures - 273.15, μ_g[g], label= "pure " + g, linestyle='dashed')


plt.title(f'Dynamic Viscosity [LGE] vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Dynamic Viscosity (μPa.s)')
plt.legend()
plt.grid(True)

plt.savefig("peng_mu_LGE.png")
plt.close()

# Viscosity plot  FIX - - - - - - - - - - -
# Vicosity for gas mixtures - data
μ_g = {}
for mix in gas_mixtures:
    μ_g[g] = []
    vs = do_notwilke_rules(mix)
    for T in temperatures:
        μ = vs
        μ_g[g].append(μ)
    plt.plot(temperatures - 273.15, μ_g[g], label= mix)
  

# Viscosity plots for pure gases
μ_g = {}
for g in ["H2", "CH4", "N2", "O2"]:
    μ_g[g] = []
    vs, t = gas_data[g]['Vs']
    for T in temperatures:
        μ = vs
        μ_g[g].append(μ)
    plt.plot(temperatures - 273.15, μ_g[g], label= "pure " + g, linestyle='dashed')


plt.title(f'Dynamic Viscosity [data] vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Dynamic Viscosity (μPa.s) - molar fraction mixing rule')
plt.legend()
plt.grid(True)

plt.savefig(fn["mu"])
plt.close()

# Density plot  - - - - - - - - - - -

# pure hydrogen
mm_H2 = gas_data['H2']['Mw']
rho_H2 = [pressure * mm_H2 / (peng_robinson(T, pressure, 'H2') * R * T)  for T in temperatures]
plt.plot(temperatures - 273.15, rho_H2, label='Pure hydrogen', linestyle='dashed')

# pure methane
mm_CH4 = gas_data['CH4']['Mw']
rho_CH4 = [pressure * mm_CH4 / (peng_robinson(T, pressure, 'CH4') * R * T)  for T in temperatures]
plt.plot(temperatures - 273.15, rho_CH4, label='Pure methane', linestyle='dashed')

# Density plots for gas mixtures
for mix in gas_mixtures:
    plt.plot(temperatures - 273.15, rho_ng[mix], label=mix)

plt.title(f'Density vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density (kg/m³)')
plt.legend()
plt.grid(True)

plt.savefig(fn["rho"])
plt.close()

