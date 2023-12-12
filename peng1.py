import numpy as np
import matplotlib.pyplot as plt

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

PR_constants = {
    'H2': {'Tc': 33.2, 'Pc': 13.0, 'omega': -0.22},
    'CH4': {'Tc': 190.6, 'Pc': 45.99, 'omega': 0.011},
    'C2H6': {'Tc': 305.32, 'Pc': 48.72, 'omega': 0.099}, # 305.556, 48.299, 0.1064 for SRK eos
    'C3H8': {'Tc': 369.8, 'Pc': 42.48, 'omega': 0.152},
    'nC4': {'Tc': 306.152, 'Pc': 38,  'omega': 0.15}, # omega is WRONG, guessed. Tc Pc from https://www.engineeringtoolbox.com/butane-d_1415.html
    'iC4': {'Tc': 407.7, 'Pc': 36.5, 'omega': 0.15}, # omega is WRONG, guessed. https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=1F
    'CO2': {'Tc': 304.2, 'Pc': 73.8, 'omega': 0.225},
    'H2O': {'Tc': 647.1, 'Pc': 220.6, 'omega': 0.345}, # https://link.springer.com/article/10.1007/s10765-020-02643-6/tables/1
    'N2': {'Tc': 126.21, 'Pc': 33.9, 'omega': 0.0401}, 
    # N2 https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.2c00363/suppl_file/ie2c00363_si_001.pdf
    # N2 omega is from https://en.wikipedia.org/wiki/Acentric_factor
}

# Natural gas compositions (mole fractions)
# Assuming hypothetical compositions for demonstration purposes
natural_gas_compositions = {
    'oH2': {'H2': 1}, # hydrogen, but using the mixing rules: software test check
    'NG1': {'CH4': 0.9, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02},
    'NG2': {'CH4': 0.85, 'C2H6': 0.07, 'C3H8': 0.05, 'CO2': 0.03},
    'NG3': {'CH4': 0.8, 'C2H6': 0.1, 'C3H8': 0.05, 'CO2': 0.05},
    'NSEA': {'CH4': 0.836, 'C2H6': 0.0748, 'C3H8':0.0392, 'nC4':0.0081, 'iC4':0.81, 
        'CO2':0.0114, 'N2':0.0195},#  'nC5':0.15, 'iC5':0.14
        # north sea gas https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7347886/
    'MIX6': {'CH4': 0.8, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02, 'N2': 0.10}, # mix6 from https://backend.orbit.dtu.dk/ws/files/131796794/FPE_D_16_00902R1.pdf
    'GG': {'CH4': 0.827, 'C2H6': 0.03, 'C3H8': 0.003, 'CO2': 0, 'N2': 0.14},
    'ethane': {'C2H6': 1.0}, # hydrogen, but using the mixing rules: software test check
}
# 20% H2, remainder N.Sea gas
fifth = {}
fifth['H2'] = 0.2
nsea = natural_gas_compositions['NSEA']
for g in nsea:
    fifth[g] = nsea[g]*0.8
natural_gas_compositions['NG20H2'] = fifth
    
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
    'iC5': {'nC5': 0.0, 'C6': 0.0}, # placeholder
    'nC5': {'C6': 0.0}, # placeholder
    'CO2': {'C6': 0.0}, # placeholder
    'N2': {'C6': 0.0}, # placeholder
    'H2': {'C6': 0.0}, # placeholder
}

def calculate_PR_constants_for_mixture(name_mix):
    """
    Calculate the Peng-Robinson constants for a mixture of hydrocarbon gases.
    
    This uses the (modified) Van der Waals mixing rules and assumes that the
    binary interaction parameters are non-zero between all pairs of components
    that we have data for.
    
    Zc is the compressibility factor at the critical point    
    """
    eps = 0.00001
    warn = 0.02 # 2 %
    
    # Initialize variables for mixture properties
    a_mix = 0
    b_mix = 0
    Zc_mix = 0
    
    # Calculate the critical volume and critical compressibility for the mixture
    Vc_mix = 0
    x = 0
    norm = 1
    
    composition = natural_gas_compositions[name_mix]
    for gas, xi in composition.items():
           x += xi
    if abs(x - 1.0) > eps:
        if abs(x - 1.0) < warn:
            print(f"--------- Warning gas mixture '{name_mix}', {100*(1-warn)}% > molar fractions > {100*(1+warn)}%. Normalizing.")
            norm = x
        else:
            print(f"######### BAD gas mixture '{name_mix}', molar fractions add up to {x}")

    for gas, xi in composition.items():
        x += xi/norm
 
        Tc = PR_constants[gas]['Tc']
        Pc = PR_constants[gas]['Pc']
        Vc_mix += xi * (0.07780 * Tc / Pc)
    
   
    # Calculate the mixture critical temperature and pressure using mixing rules
    for gas1, x1 in composition.items():
        Tc1 = PR_constants[gas1]['Tc']
        Pc1 = PR_constants[gas1]['Pc']
        #omega1 = PR_constants[gas1]['omega']
        
        a1, b1 = a_and_b(gas1) # assume T=298.15 K
        b_mix += x1 * b1 # Linear mixing rule for volume factor
        
        
        
        for gas2, x2 in composition.items(): # pairwise, but also with itself (?!)
            # if gas2 == gas1:
                # continue
            Tc2 = PR_constants[gas2]['Tc']
            Pc2 = PR_constants[gas2]['Pc']
            #omega2 = PR_constants[gas2]['omega']
            a2, b2 = a_and_b(gas2) # assume T=298.15 K
            
            # Use mixing rules for critical properties
            k = 0
            if gas2 in k_ij[gas1]:
                k = k_ij[gas1][gas2]
            if gas1 in k_ij[gas2]:
                k = k_ij[gas2][gas1]
                
            # The AI got this completely wrong. These should be the a and b parameters, not the Tc and Pc constants!!
            # and while b just depends on the mixture, a is temperature dependent. 
            
            # we fudge it with a fixed temp. a for the moment..
            
            a_mix += x1 * x2 * (1 - k) * (a1 * a2)**0.5  
            
       # Return the mixture's parameters for the P-R law
    return { name_mix: 
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

def a_and_b(gas, T=298.15):
    """Calculate the a and b intermediate parameters in the Peng-Robinson forumula 
    a : attraction parameter
    b : repulsion parameter
    
    Assume  temperature of 25 C if temp not given
    """
    # Reduced temperature and pressure
    Tc = PR_constants[gas]['Tc']
    Pc = PR_constants[gas]['Pc']

    Tr = T / Tc
    
    
    # Constants, valid only if amega < 0.49
    omega = PR_constants[gas]['omega']
    if 'L' in PR_constants[gas]:
        L = PR_constants[gas]['L']
        M = PR_constants[gas]['M']
        N = PR_constants[gas]['N']
    else:
        L, M, N = get_LMN(PR_constants[gas]['omega'])
        if gas == "H2":
            print(gas, L, M, N)
            

    
    alpha1 = Tr ** (N*(M-1)) * np.exp(L*(1 - Tr**(M*N)))
    
    if True:
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

    if T == 298.15 and gas.endswith("H2"):
        # why is thins fine for everyhting except H2 ?
        print(gas, alpha/alpha1)
        pass
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
     
    a, b = a_and_b(gas, T)

    Z = solve_for_Z(T, P, a, b)
    return Z

# test the P-R parameter mixtures rules
for mix in natural_gas_compositions:
    mixture_constants = calculate_PR_constants_for_mixture(mix)
    print(mixture_constants, " at T=298.15")
    
# Plot Z compressibility factor for pure hydrogen and natural gases
temperatures = np.linspace(273.15, 308.15, 100)  # 0°C to 35°C in Kelvin
pressure = 1.075  # bar

plt.figure(figsize=(10, 6))

# Plot for pure hydrogen
Z_H2 = [peng_robinson(T, pressure, 'H2') for T in temperatures]
plt.plot(temperatures - 273.15, Z_H2, label='Pure hydrogen', linestyle='dashed')

# Plot for pure methane
Z_CH4 = [peng_robinson(T, pressure, 'CH4') for T in temperatures]
plt.plot(temperatures - 273.15, Z_CH4, label='Pure methane', linestyle='dashed')

Z_C2H6 = [peng_robinson(T, pressure, 'C2H6') for T in temperatures]
plt.plot(temperatures - 273.15, Z_C2H6, label='Pure ethane', linestyle='dashed')

Z_C3H8 = [peng_robinson(T, pressure, 'C3H8') for T in temperatures]
plt.plot(temperatures - 273.15, Z_C3H8, label='Pure propane', linestyle='dashed')

Z_C3H8 = [peng_robinson(T, pressure, 'nC4') for T in temperatures]
plt.plot(temperatures - 273.15, Z_C3H8, label='Pure nC4', linestyle='dashed')

# Plot for natural gas compositions
for mix in natural_gas_compositions:
    constants = calculate_PR_constants_for_mixture(mix)
    a = constants[mix]['a_mix']
    b = constants[mix]['b_mix']

    Z_ng = []
    for T in temperatures:
       Z_mix = solve_for_Z(T, pressure, a, b)
       Z_ng.append(Z_mix)
    plt.plot(temperatures - 273.15, Z_ng, label=mix)

plt.title(f'Z Compressibility Factor vs Temperature at {pressure} bar')
plt.xlabel('Temperature (°C)')
plt.ylabel('Z Compressibility Factor')
plt.legend()
plt.grid(True)

plt.savefig("peng1.png")
#plt.show()
