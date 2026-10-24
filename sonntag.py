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
    Guarded against sub-zero Kelvin overshoots.
    
    Args:
        h2o_pp_Pa: Partial pressure of H2O in flue gas [Pa].
        P_bar: Total flue gas pressure [bar].
        flue_gas: Flue gas mole fraction dictionary.
    """
    if h2o_pp_Pa <= 0:
        return 273.15

    # Initial guess based on partial pressure
    T = 250.0 if h2o_pp_Pa < 1000 else 320.0
    tolerance = 1e-6

    for i in range(100):
        P_eff = get_enhanced_vapor_pressure(T, P_bar, flue_gas)
        error = P_eff - h2o_pp_Pa

        if abs(error) < tolerance:
            return T

        dT = 0.01
        P_eff_plus = get_enhanced_vapor_pressure(T + dT, P_bar, flue_gas)
        dP_dT = (P_eff_plus - P_eff) / dT

        if dP_dT == 0:
            break

        step = error / dP_dT
        T = max(180.0, T - step)

    return T

def get_dew_point(pressure):
    """
    Calculates the dew point temperature given a partial pressure in Pa 
    using a guarded Newton-Raphson solver on the Sonntag equation.
    Args:
        pressure: Pressure in Pa.

    Returns:
        Dew point temperature in Kelvin.

    """
    if pressure <= 0:
        return 273.15

    # Estimate initial guess based on order of magnitude
    if pressure < 1000:
        T = 250.0  # ~ -23 °C for low partial pressure
    else:
        T = 320.0  # ~ 47 °C

    tolerance = 1e-6
    for i in range(100):
        P_calc = sonntag_vapor_pressure(T)
        dP = P_calc - pressure
        
        if abs(dP) < tolerance:
            return T
            
        dP_dT = d_sonntag_vapor_pressure_dT(T)
        if dP_dT == 0:
            break

        # Newton-Raphson step with safety guard against T <= 0 K
        dT = dP / dP_dT
        T = max(180.0, T - dT)

    return T
        
def run_enhancement_self_test():
    """Self-test comparing pure Sonntag vs GERG-2008 enhanced water vapor 
    properties for NG Flue Gas, H2 Flue Gas, and Dry Air at 0°C and 10 atm.
    """
    T_C = 0.0
    T_K = 273.15 + T_C
    P_atm = 10.0
    P_bar = P_atm * 1.01325  # 10.1325 bar
    P_Pa = P_bar * 1e5

    # Flue gas compositions (mole fractions from published paper Tables 3 & 4)
    mixtures = {
        'NG Flue Gas (Table 3)': {
            'O2': 0.0249,
            'N2': 0.7399,
            'CO2': 0.0894,
            'H2O': 0.1661
        },
        'H2 Flue Gas (Table 4)': {
            'O2': 0.0231,
            'N2': 0.6687,
            'CO2': 0.0,
            'H2O': 0.3082
        },
        'Dry Air + Trace H2O': {
            'N2': 0.7808,
            'O2': 0.2094,
            'CO2': 0.0004,
            'Ar': 0.0094,
            'H2O': 0.0001
        }
    }

    P_sat_pure_Pa = sonntag_vapor_pressure(T_K)

    print("=" * 80)
    print(f"ENHANCEMENT FACTOR COMPARISON TEST AT T = {T_C:.1f}°C ({T_K:.2f} K), P = {P_atm:.1f} atm ({P_bar:.3f} bar)")
    print(f"Pure Water Saturation Pressure (Sonntag): {P_sat_pure_Pa:.2f} Pa ({P_sat_pure_Pa/101325:.5f} atm)")
    print("=" * 80)
    print(f"{'Gas Mixture':<25} | {'f (GERG)':<10} | {'Pure Psat (Pa)':<14} | {'Eff Psat (Pa)':<14} | {'Dew Point Shift':<15}")
    print("-" * 80)

    for name, y_dict in mixtures.items():
        # 1. Enhancement factor
        f_gerg = get_enhancement_factor_gerg(T_K, P_bar, y_dict)
        
        # 2. Enhanced vapor pressure
        P_sat_eff_Pa = get_enhanced_vapor_pressure(T_K, P_bar, y_dict)
        
        # 3. Partial pressure of water in total mixture at 10 atm
        y_w = y_dict['H2O']
        p_h2o_Pa = y_w * P_Pa
        
        # Dew point comparisons at 10 atm partial pressure
        T_dp_pure_C = get_dew_point(p_h2o_Pa) - 273.15
        T_dp_eff_C = get_dew_point_enhanced(p_h2o_Pa, P_bar, y_dict) - 273.15
        dT_dp = T_dp_eff_C - T_dp_pure_C

        print(f"{name:<25} | {f_gerg:<10.5f} | {P_sat_pure_Pa:<14.2f} | {P_sat_eff_Pa:<14.2f} | +{dT_dp:.3f} °C")

    print("=" * 80)


def main():
 

    # Legacy Sonntag table printout
    print("\nSonntag Saturation Vapor Pressure Table (0 to 95°C):")
    atm = 101325 # Pa
    for TC in range(0, 100, 5):
        t = 273.15 + TC
        p = sonntag_vapor_pressure(t) / atm
        print(f"{TC:4}°C  {p:8.5f} atm")

    p = 0.1661356 # partial pressure in atm (16.61% NG flue gas)
    p2 = p * atm  # pp in Pa
    dew_point_pure = get_dew_point(p2) - 273.15
    print(f"\nPure Sonntag Dew point for H2O partial pressure of {p:.4f} atm: {dew_point_pure:.4f} °C")   


# # Example usage
# pressure_Pa = 10000  # Example pressure
# dew_point_temperature_K = get_dew_point_temperature(pressure_Pa)
# print(f"Dew point temperature at {pressure_Pa} Pa is {dew_point_temperature_K:.2f} K")    

    # Run enhancement factor comparison tests
    run_enhancement_self_test()
    
if __name__ == '__main__':
    sys.exit(main())  