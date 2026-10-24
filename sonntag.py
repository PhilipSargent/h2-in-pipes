import math
import sys
import CoolProp.CoolProp as CP

# Universal Gas Constant (bar * cm^3 / (mol * K))
R_GAS = 83.144626
V_LIQ = 18.07  # Molar volume of liquid water (cm^3 / mol)

NAME_MAP = {
    'H2O': 'Water',
    'CO2': 'CO2',
    'N2': 'Nitrogen',
    'O2': 'Oxygen',
    'Ar': 'Argon',
    'He': 'Helium',
}

def sonntag_vapor_pressure(T_K):
    """
    Calculates the saturation vapor pressure of water vapor using the Sonntag equation.

    Args:
        T_K: Temperature in Kelvin.

    Returns:
        Saturation vapor pressure in Pa.
    """

    # Constants for the Sonntag equation
    A1 = -6096.9385
    A2 = 21.2409642
    A3 = -2.711193e-2
    A4 = 1.673952e-5
    A5 = 2.433502

    # Calculate the logarithm of the saturation vapor pressure
    ln_P_sat = A1 / T_K + A2 + A3 * T_K + A4 * T_K**2 + A5 * math.log(T_K)

    # Convert the logarithm to saturation vapor pressure
    P_sat = math.exp(ln_P_sat)
    return P_sat

def d_sonntag_vapor_pressure_dT(T_K):
    """Calculates the analytical derivative of the Sonntag vapor pressure function.
    This is a spectacularly pointless optimization for this purpose. """

    # Constants for the Sonntag equation
    A1 = -6096.9385
    A2 = 21.2409642
    A3 = -2.711193e-2
    A4 = 1.673952e-5
    A5 = 2.433502

    # Calculate the derivative of the logarithm of the saturation vapor pressure
    ln_P_sat_prime = -A1 / T_K**2 + A3 + 2 * A4 * T_K + A5 / T_K

    # Convert the derivative back to the original units
    #P_sat = sonntag_vapor_pressure(T_K)
    dP_dT =  math.exp(ln_P_sat_prime)   # P_sat is already calculated in the original function

    return dP_dT
    
def get_gerg_B_cm3(sp1, sp2, T_K):
    """
    Extracts cross-second virial coefficient B_12(T) in cm^3/mol 
    from CoolProp HEOS/GERG-2008.
    """
    n1 = NAME_MAP.get(sp1, sp1)
    n2 = NAME_MAP.get(sp2, sp2)
    try:
        if sp1 == sp2:
            return CP.PropsSI('Bvirial', 'T', T_K, 'P', 1e5, n1) * 1e6
        else:
            mix_str = f"HEOS::{n1}[0.5]&{n2}[0.5]"
            return CP.PropsSI('Bvirial', 'T', T_K, 'P', 1e5, mix_str) * 1e6
    except ValueError:
        # Fallback for subcooled pure water below melting point (T < 273.15 K)
        return -1150.0


def get_enhancement_factor_gerg(T_K, P_bar, flue_gas):
    """
    Calculates enhancement factor f(T, P, y) using GERG-2008 virial coefficients.
    
    Args:
        T_K: Temperature in Kelvin.
        P_bar: Total system pressure in bar.
        flue_gas: Dict of mole fractions, e.g., {'H2O': 0.15, 'CO2': 0.08, 'N2': 0.74, 'O2': 0.03}
    
    Returns:
        f (dimensionless enhancement factor, typically ~ 1.00 to 1.05)
    """
    y_w = flue_gas.get('H2O', 0.0)
    if y_w <= 0.0 or y_w >= 1.0:
        return 1.0

    # Dry gas normalized fractions
    dry_flue_gas = {k: v / (1.0 - y_w) for k, v in flue_gas.items() if k != 'H2O'}

    # 1. B_gg (Dry gas matrix virial sum)
    B_gg = 0.0
    for g1, y1 in dry_flue_gas.items():
        for g2, y2 in dry_flue_gas.items():
            B_gg += y1 * y2 * get_gerg_B_cm3(g1, g2, T_K)

    # 2. B_gw (Dry gas <-> Water cross virial sum)
    B_gw = 0.0
    for g, y_g in dry_flue_gas.items():
        B_gw += y_g * get_gerg_B_cm3(g, 'H2O', T_K)

    # 3. Virial enhancement factor expression
    bracket = B_gg - 2.0 * B_gw + V_LIQ
    ln_f = ((1.0 - y_w) * P_bar / (R_GAS * T_K)) * bracket
    return math.exp(ln_f)


def get_enhanced_vapor_pressure(T_K, P_bar, flue_gas):
    """
    Returns effective saturation vapor pressure of water in flue gas mixture in Pa.
    """
    P_sat_pure = sonntag_vapor_pressure(T_K)
    f = get_enhancement_factor_gerg(T_K, P_bar, flue_gas)
    return f * P_sat_pure


def get_dew_point_enhanced(h2o_pp_Pa, P_bar, flue_gas):
    """
    Finds dew point temperature [K] where h2o_pp_Pa = f(T, P, y) * P_sat_pure(T).
    
    Args:
        h2o_pp_Pa: Partial pressure of H2O in flue gas [Pa].
        P_bar: Total flue gas pressure [bar].
        flue_gas: Flue gas mole fraction dictionary.
    """
    T = 323.15  # Initial guess ~ 50°C
    tolerance = 1e-6
    
    for i in range(50):
        P_eff = get_enhanced_vapor_pressure(T, P_bar, flue_gas)
        error = P_eff - h2o_pp_Pa
        
        if abs(error) < tolerance:
            print(f"{i:5} iterations")
            return T
        
        # Secant/Numerical derivative step
        dT = 0.01
        dP_dT = (get_enhanced_vapor_pressure(T + dT, P_bar, flue_gas) - P_eff) / dT
        
        if dP_dT == 0:
            print(f"{i:5} iterations, but dP_dT == 0")
            break
            
        T -= error / dP_dT
        
    return T
    
def get_dew_point(pressure):
    """
    Calculates the dew point temperature given a pressure using the Sonntag equation.

    Args:
        pressure: Pressure in Pa.

    Returns:
        Dew point temperature in Kelvin.
    """

    # Initial guess for temperature
    T = 273.15 + 50 # Start near 50 C

    # Iteration tolerance
    tolerance = 1e-6
    #print(f"Find dew point for p = {pressure}")
    i = 0
    T_last = T - 5
    while True:
        i += 1
        calculated_pressure = sonntag_vapor_pressure(T)
        #dP_dT = d_sonntag_vapor_pressure_dT(T)
        dP = (calculated_pressure - pressure)
        dT = T - T_last
        dP_dT = dP / dT
        error = abs(dP)
        #print(f"T={T:.2f} P={sonntag_vapor_pressure(T):.4f}  {dP_dT=:.4}")
        if error < tolerance or i>50 :
            # print(f"{i:5} iterations")
            return T

        # Adjust temperature based on error
        #T +=  (pressure - calculated_pressure) / calculated_pressure
        dT =   - 0.1 * T * dP /pressure
        T += dT
        #print(f"{dP=:.2f} {dT=:.2f} {T=:.2f}")
        

# # Example usage
# pressure_Pa = 10000  # Example pressure
# dew_point_temperature_K = get_dew_point_temperature(pressure_Pa)
# print(f"Dew point temperature at {pressure_Pa} Pa is {dew_point_temperature_K:.2f} K")    
    
def main():
    # Example usage
    T_K = 300  # Temperature in Kelvin
    atm = 101325 # Pa
    P_sat_Pa = sonntag_vapor_pressure(T_K)
    print(f"Saturation vapor pressure at {T_K:.2f} K is {P_sat_Pa:.2f} Pa is {P_sat_Pa/atm:.2f} atm")
    for TC in range(0,100,5):
        t = 273.15 + TC
        p = sonntag_vapor_pressure(t)/atm
        print(f"{TC:4}  {p:8.4f}" )

    p = 0.1661356 # partial pressure in atm
    p2 = p * atm  # pp in Pa
    dew_point = get_dew_point(p2) - 273.15
    print(f"Dew point for partial pressure of  {p2:.4f} is {dew_point:.4f} C")   
    
if __name__ == '__main__':
    sys.exit(main())  