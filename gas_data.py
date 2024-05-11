import sys
import numpy as np
from peng_utils import memoize

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

# Some Cp values taken from the Cp_data at 25 C:
gas_data = {
    'H2': {'Tc': 33.2, 'Pc': 13.0, 'omega': -0.22, 'Mw':2.015, 'Vs': (8.9,300, 0.692), 'Hc':0.285826, 'C_':0, 'H_':2, 'Cp': 28.76 },
    'CH4': {'Tc': 190.56, 'Pc': 45.99, 'omega': 0.01142, 'Mw': 16.04246, 'Vs': (11.1,300, 1.03), 'Hc':0.890602, 'C_':1, 'H_':4, 'Cp':35.69}, # Hc from ISO_6976
    'C2H6': {'Tc': 305.32, 'Pc': 48.72, 'omega': 0.099, 'Mw': 30.07, 'Vs': (9.4,300, 0.87), 'Hc':1.5639, 'C_':2, 'H_':6, 'Cp': 52.8891}, # 
    'C3H8': {'Tc': 369.15, 'Pc': 42.48, 'omega': 0.1521, 'Mw': 44.096, 'Vs': (8.2,300, 0.93), 'Hc':2.21866, 'C_':3, 'H_':8, 'Cp': 73.7033}, # https://www.engineeringtoolbox.com/propane-d_1423.html
    'nC4': {'Tc': 425, 'Pc': 38,  'omega': 0.20081, 'Mw': 58.1222, 'Vs': (7.5,300, 0.950), 'Hc':2.876841, 'C_':4, 'H_':10, 'Cp': 99.3473}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Butane.html https://www.engineeringtoolbox.com/butane-d_1415.html 
    'iC4': {'Tc': 407.7, 'Pc': 36.5, 'omega': 0.1835318, 'Mw': 58.1222, 'Vs': (7.5,300, 0.942), 'Hc':2.87728, 'C_':4, 'H_':1, 'Cp': 96.9858}, # omega  http://www.coolprop.org/fluid_properties/fluids/IsoButane.html https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=1F https://webbook.nist.gov/cgi/cbook.cgi?Name=butane&Units=SI Viscocity assumed same as nC4
    'nC5': {'Tc': 469.8, 'Pc': 33.6, 'omega': 0.251032, 'Mw': 72.1488, 'Vs': (6.7,300, 1.0), 'Hc':3.53609, 'C_':5, 'H_':12, 'Cp': 123.3789}, # omega http://www.coolprop.org/fluid_properties/fluids/n-Pentane.html     
    'iC5': {'Tc': 461.0, 'Pc': 33.8, 'omega': 0.2274, 'Mw': 72.1488, 'Vs': (6.7,300, 0.94), 'Hc':3.52917, 'C_':5, 'H_':12, 'Cp': 118.9}, # omega http://www.coolprop.org/fluid_properties/fluids/Isopentane.html  Viscocity assumed same as nC5, but we have a model for this: see iC5.py in this folder.
    
    'neoC5': {'Tc': 433.8, 'Pc': 31.963, 'omega': 0.1961, 'Mw': 72.1488, 'Vs': (6.9326,300, 0.937), 'Hc':3.51495, 'C_':5, 'H_':12, 'Cp': 120.82},
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C463821&Units=SI&Mask=4#Thermo-Phase
    # omega from http://www.coolprop.org/fluid_properties/fluids/Neopentane.html
    
    'C6':  {'Tc': 507.6, 'Pc': 30.2, 'omega': 0.1521, 'Mw': 86.1754, 'Vs': (8.6,400, 1.03), 'Hc':4.19475, 'C_':6, 'H_':14, 'Cp': 147.6375}, # omega is 0.2797 isohexane    
    'CO2': {'Tc': 304.2, 'Pc': 73.8, 'omega': 0.228, 'Mw': 44.01, 'Vs': (15.0,300, 0.872), 'Hc':0, 'C_':0, 'H_':0, 'Cp':37.12}, # https://en.wikipedia.org/wiki/Acentric_factor
    'H2O': {'Tc': 647.1, 'Pc': 220.6, 'omega': 0.344292, "Mw": 18.015, 'Vs': (9.8,300, 1.081), 'Hc':0, 'C_':0, 'H_':0, 'Cp': 32.81, 'LH': 43.99, 'CpL':75.63}, # CpL in liquid phase LH latent heat  https://link.springer.com/article/10.1007/s10765-020-02643-6/tables/1
    'N2': {'Tc': 126.21, 'Pc': 33.958, 'omega': 0.0372, 'Mw':28.013, 'Vs': (17.9,300, 0.658), 'Hc':0, 'C_':0, 'H_':0, 'Cp': 29.12}, #  omega http://www.coolprop.org/fluid_properties/fluids/Nitrogen.html, CpL is Cp for liquid
    'He': {'Tc': 5.2, 'Pc': 2.274, 'omega': -0.3836, 'Mw': 4.0026, 'Vs': (19.9,300, 0.69), 'Hc':0, 'C_':0, 'H_':0, 'Cp': 126.153},  # omega http://www.coolprop.org/fluid_properties/fluids/Helium.html
    # https://eng.libretexts.org/Bookshelves/Chemical_Engineering/Distillation_Science_(Coleman)/03%3A_Critical_Properties_and_Acentric_Factor
    # N2 https://pubs.acs.org/doi/suppl/10.1021/acs.iecr.2c00363/suppl_file/ie2c00363_si_001.pdf
    # N2 omega is from https://en.wikipedia.org/wiki/Acentric_factor
    'Ar': {'Tc': 150.687, 'Pc': 48.630, 'omega': 0, 'Mw': 39.948, 'Vs': (22.7,300, 0.77), 'Hc':0, 'C_':0, 'H_':0, 'Cp':20.786}, #https://en.wikipedia.org/wiki/Acentric_factor
    'O2': {'Tc': 154.581, 'Pc': 50.43, 'omega': 0.022, 'Mw': 31.9988, 'Vs': (20.7,300, 0.72), 'Hc':0, 'C_':0, 'H_':0, 'Cp':29.34},# http://www.coolprop.org/fluid_properties/fluids/Oxygen.html
    }

# SPECIFIC HEAT parameters
# From Table A2. of Cengel, Y.A., Boles, M.A., 2002. Thermodynamics: an Engineering Approach, 4th ed. McGraw Hill
Cp_data = {
    "H2": {"a": 29.11, "b": -0.1916e-2, "c": 0.4003e-5, "d": -0.8704e-9},
    "CH4": {"a": 19.89, "b": 5.024e-2, "c": 1.269e-5, "d": -11.01e-9},
    "C2H6": {"a": 6.900, "b": 17.27e-2, "c": -6.406e-5, "d": 7.285e-9},
    "C3H8": {"a": -4.04, "b": 30.48e-2, "c": -15.72e-5, "d": 31.74e-9},
    "nC4": {"a": 3.96, "b": 37.15e-2, "c": -18.34e-5, "d": 35.00e-9},
    "iC4": {"a": -7.913, "b": 41.60e-2, "c": -23.01e-5, "d": 49.91e-9},
    "nC5": {"a": 6.774, "b": 45.43e-2, "c": -22.46e-5, "d": 42.29e-9},
    "C6":  {"a": 6.938, "b": 55.22e-2, "c": -28.65e-5, "d": 57.69e-9},
    
    "CO2": {"a": 22.26, "b": 5.981e-2, "c": -3.501e-5, "d": 7.469e-9},
    "H2O": {"a": 32.24, "b": 0.1923e-2, "c": 1.055e-5, "d": -3.595e-9},        
    "N2": {"a": 28.90, "b": -0.1571e-2, "c": 0.8081e-5, "d": -2.873e-9},
    #"iC5": 
    #"neoC5": 
    #"He": 
    #"Ar": 
    
    "O2": {"a": 25.48, "b": 1.520e-2, "c": -0.7155e-5, "d": 1.312e-9},
    
    #"Air": {"a": 28.11, "b": 0.1967e-2, "c": 0.4802e-5, "d": -1.966e-9},

    #"Ethylene": {"Formula": "C2H4", "a": 3.95, "b": 15.64e-2, "c": -8.344e-5, "d": 17.67e-9},
    #"Propylene": {"Formula": "C3H6", "a": 3.15, "b": 23.83e-2, "c": -12.18e-5, "d": 24.62e-9},    
    }
        
# Natural gas compositions (mole fractions)
# We do not do LPG yet, but here are the propane/butane ratios: https://www.mylpg.eu/useful/lpg-mixture
gas_mixtures = {
    
    'NG': {'CH4': 0.895514, 'C2H6': 0.051196, 'C3H8': 0.013549, 'nC4': 0.001269, 'iC4': 0.002162, 'nC5': 2e-05, 'iC5': 0.000344, 'neoC5': 0.003472, 'C6': 0.002377, 'CO2': 0.020743, 'N2': 0.009354}, # Normalized.email John Baldwin 30/12/2023 - Fordoun
    # 'Fordoun': { 'CH4':  0.900253, 'C2H6':  0.051467, 'C3H8':  0.013621, 'iC4':  0.001276, 'nC4':  0.002173, 'neoC5':  0.000020,'iC5':  0.000346, 'nC5':  0.003490,  'C6':  0.002390, 'CO2':  0.020853, 'N2':  0.009404, }, # original email John Baldwin 30/12/2023, unnormalized
    'Groening': {'CH4': 0.813, 'C2H6': 0.0285, 'C3H8': 0.0037, 'nC4': 0.0014, 'nC5': 0.0004, 'C6': 0.0006, 'CO2': 0.0089, 'N2': 0.1435, 'O2': 0}, # Groeningen gas https://en.wikipedia.org/wiki/Groningen_gas_field
    
    'UW': {'CH4': 0.96803, 'C2H6': 0.01773, 'C3H8': 0.00395, 'iC4': 0.00063, 'nC4': 0.00057, 'nC5': 0.00007,  'iC5': 0.0001, 'C6': 0.00009, 'CO2': 0.00115, 'N2': 0.00768, 'O2': 0}, # DOI: 10.15199/17.2023.12.1 Uilhoorn & Witek 2023 Polish gas, no provenance recorded, blended with H2 at 10% and 30% in calcs. 
   
    # All these come from Grinten(2020) for gases in the 'OGE' network
    # https://oge.net/en/we/purpose/oge-in-europe
    'NoGas': {'CH4': 0.906, 'C2H6': 0.0578, 'C3H8': 0.0068, 'nC4': 0.0009, 'iC4': 0.001,'nC5': 0.0002, 'iC5': 0.0002, 'C6': 0.0002, 'CO2': 0.018, 'N2': 0.0088, 'O2': 0, 'neoC5': 0.0001}, 
    'RuGas': {'CH4': 0.9624, 'C2H6': 0.0271, 'C3H8': 0.002, 'nC4': 0.0003, 'iC4': 0.0005,'nC5': 0.0, 'iC5': 0.0001, 'C6': 0.0001, 'CO2': 0.0034, 'N2': 0.0041, 'O2': 0, 'neoC5': 0.0}, 
    
    
    'Yamal': {'CH4': 0.983455, 'C2H6': 0.006104, 'C3H8': 0.001572, 'nC4': 0.000253, 'iC4': 0.000299,'nC5': 0.00004, 'iC5': 0.000055, 'C6': 0.0, 'CO2': 0.007918, 'N2': 0.000303, 'O2': 0, 'neoC5': 0.0}, 
    
    
    
    
    'BioGas': {'CH4': 0.9615, 'CO2': 0.029, 'N2': 0.0075, 'O2': 0.002, }, 
    'NL-H': {'CH4': 0.88853, 'C2H6': 0.05078, 'C3H8': 0.01133, 'nC4': 0.00221, 'iC4': 0.00154,'nC5': 0.00037, 'iC5': 0.00046, 'C6': 0.0005, 'CO2': 0.01208, 'N2': 0.0322, 'O2': 0, 'neoC5': 0.0}, 
    'NL-L': {'CH4': 0.84343, 'C2H6': 0.03874, 'C3H8': 0.00512, 'nC4': 0.00089, 'iC4': 0.00085,'nC5': 0.00021, 'iC5': 0.00026, 'C6': 0.00052, 'CO2': 0.01504, 'N2': 0.09494, 'O2': 0, 'neoC5': 0.0}, 
    
    #There are 19 gas compositions in this paper, but I only have the pages for 2 of them
    # R. C. Johnson, “Real gas effects in the flow of methane and natural gas through critical flow nozzles.” [Online]. Available: https://ntrs.nasa.gov/citations/19710011855.
    'NASA-1': {'CH4': 0.9535, 'C2H6': 0.0296, 'C3H8': 0.0046, 'nC4': 0.0006, 'iC4': 0.0007,'nC5': 0.00, 'iC5': 0.00, 'C6': 0.0, 'CO2': 0.007, 'N2': 0.004, 'O2': 0, 'neoC5': 0.0},
    
    'NASA-2': {'CH4': 0.8850, 'C2H6': 0.0795, 'C3H8': 0.011, 'nC4': 0.0017, 'iC4': 0.0007,'nC5': 0.00, 'iC5': 0.00, 'C6': 0.0, 'CO2': 0.0, 'N2': 0.0221, 'O2': 0, 'neoC5': 0.0}, 
    
    'ALGr': {'CH4': 0.8484, 'C2H6': 0.084, 'C3H8': 0.005, 'nC4': 0.00, 'nC5': 0.00, 'CO2': 0.0066, 'N2': 0.056, 'O2': 0}, # Air Liquide, Greece in 
     #M. J. Assael, N. K. Dalaouti, and V. Vesovic, “Viscosity of Natural-Gas Mixtures :,” Int. J. Thermophys., vol. 22, no. 1, pp. 61–71, 2001.
    
    
    'AHBJ': {'CH4': 0.9376, 'C2H6': 0.0314, 'C3H8': 0.0062, 'nC4': 0.002, 'nC5': 0.0007, 'CO2': 0.0018, 'N2': 0.0203, 'O2': 0}, # Abbas, Hassani, Burby, John (2021)
    
    'Tokyo': {'CH4': 0.896, 'C2H6': 0.056, 'C3H8': 0.0034, 'iC4': 0.0007, 'nC4': 0.0007, 'N2': 0.0432 }, # Tokyo town gas, with assumed missing gas all N2 http://members.igu.org/html/wgc2009/papers/docs/wgcFinal00580.pdf
    
    'Biomethane': {'CH4': 0.92,  'C3H8': 0.04, 'CO2': 0.04 }, # wobbe central, not a real natural gas  https://www.gasgovernance.co.uk/sites/default/files/ggf/Impact%20of%20Natural%20Gas%20Composition%20-%20Paper_0.pdf
 
    '10C2-10N': {'CH4': 0.80,  'C3H8': 0.1, 'N2': 0.1 }, # RH corner of allowable wobbe polygon ?
    '7C2-2N': {'CH4': 0.91,  'C3H8': 0.07, 'N2': 0.02 }, # top corner of allowable wobbe polygon ?
  
    'mix6': {'CH4': 0.8, 'C2H6': 0.05, 'C3H8': 0.03, 'CO2': 0.02, 'N2': 0.10}, # ==mix6 from      ?? Source mislaid. Looks artificial.

    'NTS79': {'CH4': 0.9363, 'C2H6': 0.0325, 'C3H8': 0.0069, 'nC4': 0.0027, 'CO2': 0.0013, 'N2': 0.0178, 'He': 0.0005, 'nC5': 0.002}, # https://en.wikipedia.org/wiki/National_Transmission_System
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
    
   # 8 gas cmpositions from Texas in https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=903432
    # A. Johnson and B. Johansen, “Comparison of Five Natural Gas Equations of State Used for Flow and Energy Measurement,” 7th Annu. Int. Symp. Fluid Flow Meas., pp. 1–19, 2009, [Online]. Available: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=903432.:
    # Gulf coast, Amarillo, Ekofisk, High N2, High N2/CO2, CEESI Iowa CEESI Colorado high ethane, CEESI Iowa high ethane.
    'Gulf': {'CH4': 0.965222, 'C2H6': 0.018186, 'C3H8':0.004596, 'nC4':0.001007, 'iC4':0.000977, 'nC5':0.000324, 'iC5': 0.000473, 'CO2':0.005956, 'N2':0.002595, 'C6': 0.000664},
    
    'Amrllo': {'CH4': 0.906724, 'C2H6': 0.045279, 'C3H8':0.00828, 'N2':0.031284, 'CO2':0.004676,'iC4':0.001037, 'nC4':0.001563, 'nC5':0.000321, 'iC5':0.000443, 'C6': 0.000393 },        

    'Ekofsk': {'CH4': 0.859063, 'C2H6': 0.084919, 'C3H8':0.023015, 'N2':0.010068, 'CO2':0.014954,'iC4':0.003486, 'nC4':0.003506, 'nC5':0.000509, 'iC5':0.00048, 'C6': 0.000 },        

    'HiN2C': {'CH4': 0.81212, 'C2H6': 0.04303, 'C3H8':0.00895, 'N2':0.05702, 'CO2':0.07585,'iC4':0.00151, 'nC4':0.00152, 'nC5':0.000, 'iC5':0.000, 'C6': 0.000 },        

    'Iowa': {'CH4': 0.95485, 'C2H6': 0.018984, 'C3H8':0.00177, 'H2': 0.001599, 'N2':0.015987, 'CO2':0.005995,'iC4':0.000154, 'nC4':0.0002013, 'nC5':0.00005, 'iC5':0.00003, 'He': 0.0003497, 'C6': 0.00003 },        

    'HiEth': {'CH4': 0.848128, 'C2H6': 0.106707, 'C3H8':0.017673, 'H2': 0.0, 'N2':0.00409, 'CO2':0.021109,'iC4':0.0008, 'nC4':0.001258, 'nC5':0.000115, 'iC5':0.0001, 'He': 0.0, 'C6': 0.00002 },        

    'IwEth': {'CH4': 0.921244, 'C2H6': 0.043547, 'C3H8':0.009299, 'H2': 0.001427, 'N2':0.011733, 'CO2':0.009663,'iC4':0.00093, 'nC4':0.001218, 'nC5':0.000259, 'iC5':0.00024, 'He': 0.000293, 'C6': 0.000147 },        


    
    'High-N2': {'CH4': 0.81441, 'C2H6': 0.033, 'C3H8':0.00605, 'N2':0.13465, 'CO2':0.00985,'iC4':0.001, 'nC4':0.00104, 'nC5':0.0, 'iC5':0.0, 'C6': 0.0 },        
    
    'O2':  { 'O2': 1.0}, # for the 'oxidiser' list, also test for gas 'mix'
    'HeOx':  { 'O2': 0.2, 'He': 0.8}, # Heliox, for testing viscosity algorithms
    'ArH2':  { 'Ar': 0.2, 'H2': 0.4, 'He': 0.4}, # Argon-H2, for testing viscosity algorithms
}

# There are 19 more natural gas compositions from Texas  in RC Johnson(1971) 
# R. C. Johnson, “Real gas effects in the flow of methane and natural gas through critical flow nozzles.” Available: https://ntrs.nasa.gov/citations/19710011855.
# but 17 are missing from the accessible copies of the paper

ng_gases = ['NG', 'Groening', 'AHBJ', 'Tokyo', 'Biomethane', '10C2-10N', '7C2-2N', 'mix6',
    'NTS79', '11D', 'Algerian', 'North Sea', 'UW', 'NoGas', 'RuGas', 'BioGas', 'NL-H', 'NL-L','NASA-1','NASA-2', 'ALGr', 'Yamal', 'Gulf', 'Amrllo', 'Ekofsk', 'High-N2', 'HiN2C', 'Iowa', 'HiEth', 'IwEth']
    
"""reduce the lower limit for Wobbe Index from 47.2 MJ/m³  to 46.50 MJ/m³ was approved by HSE. 
This shall enter into force from 6 April 2025
"Gas Ten Year Statement December 2023"
https://www.nationalgas.com/document/144896/download
"""
gas_mixture_properties = {
    'Algerian': {'Wb': 49.992, 'HHV': 39.841, 'RD': 0.6351}, #Algerian NG, Romeo 2022, C6+ BUT not according to my calcs. for Hc and wobbe.
    # 'Air': {'Hc': 0} 
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

def cp(g, T):
    """
    Compute the specific heat capacity (Cp) for a given substance and temperature.

    Parameters:
    substance (str): The name of the substance.
    T (float): The temperature in Kelvin.
    data (dict): The data dictionary containing the a, b, c, d values for each substance.

    Returns:
    float: The computed Cp value.
    """
    if g not in Cp_data:
        # use the temp-independent value
        if "Cp" in gas_data[g]:
            return gas_data[g]["Cp"]
        return 0
        
    a = Cp_data[g]["a"]
    b = Cp_data[g]["b"]
    c = Cp_data[g]["c"]
    d = Cp_data[g]["d"]

    Cp = a + b*T + c*T**2 + d*T**3
    return Cp

# We need to compute the a,b,c,d coefficients for Cp from the tabular data for each gas at
#https://webbook.nist.gov/cgi/cbook.cgi?ID=C78784&Units=SI&Mask=1#Thermo-Gas
# for iC5, neoC5, He, Ar

@memoize   
def Cp_H2(T):
    # data from https://www.engineeringtoolbox.com/hydrogen-d_976.html downloaded 29 March 2024
    temperatures = [175, 200, 225, 250, 275, 300, 325, 350, 375, 400]  # in Kelvin
    specific_heats = [13.12, 13.53, 13.83, 14.05, 14.20, 14.31, 14.38, 14.43, 14.46, 14.48]  # in kJ/(kg·K)

    cp_kg = np.interp(T, temperatures, specific_heats) # in kJ/(kg·K)
    cp = cp_kg * gas_data['H2']['Mw']
    return cp

@memoize  
def main():
    # Testing the specific heat function, for H2
    for TC in [-40, 0,15,25,100,150]:
        T = 273.15+TC
        print(f"{T:7.2f} {cp('H2', T):8.4f}  {Cp_H2(T):8.4f} {100*(cp('H2', T)- Cp_H2(T))/cp('H2', T):8.4f} %")
    for g in gas_data:
        for TC in [-40, 25,100,150]:
            T = 273.15+TC
            
            if "Cp" in gas_data[g]:
                print(f"{g:7} {T:7.2f} {cp(g, T):8.4f}  {gas_data[g]['Cp']:8.4f} {100*(cp(g, T)- gas_data[g]['Cp'])/gas_data[g]['Cp']:8.4f} %")
    for g in Cp_data:
        T = 273.15+25
        print(f"{g:7} {T:7.2f} {cp(g, T):8.4f}")
    print("\n### We need to compute the a,b,c,d coefficients for Cp from the tabular data: iC5, neoC5, He, Ar")
    print("### https://webbook.nist.gov/cgi/cbook.cgi?ID=C78784&Units=SI&Mask=1#Thermo-Gas")
    program = sys.argv[0]
    print(f"\nThis program '{program}' is not intended to be run as a standalone program.")

if __name__ == '__main__':
    sys.exit(main())  