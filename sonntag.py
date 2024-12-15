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