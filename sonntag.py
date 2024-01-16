import math
import sys

def sonntag_vapor_pressure(T_K):
    """
    Calculates the saturation vapor pressure of water vapor using the provided Sonntag equation.

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

if __name__ == '__main__':
    sys.exit(main())  