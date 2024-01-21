import numpy as np
import sys
"""Converted to python from fortran by Philip Sargent, with the assistance of bard.google.com
in January 2024.
From Yang and Joseph, doi:10.1080/14685240902806491
but the forran is only on their ResearchGate version of the paper
on https://www.researchgate.net/publication/232974572_Virtual_Nikuradse

"""

# Constants and arrays (using numpy for convenience)
s = np.array([-50., -15., -5., -2.])
t = np.array([0.5, 0.5, 0.5, 0.5])
a = np.array([64., 0.000083, 0.3164, 0.1537, 0.0753])
b = np.array([-1., 0.75, -0.25, -0.185, -0.136])
r = np.array([2320., 3810., 70000., 2000000.])
m = np.array([-2., -5., -5., -5., -5.])
n = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

# Calculate arrays
# Array initializations (equivalent to Fortran's reshape)
aax = np.array([
    [0.05016, 0.0476, 0.00944, 0.02076, 0.00253],
    [0.03599, 0.0331, 0.00758, 0.02448, 0.0225],
    [0.02615, 0.0235, 0.00615, 0.02869, 0.0561],
    [0.01851, 0.0161, 0.00491, 0.03410, 0.1031],
    [0.01344, 0.0113, 0.00397, 0.04000, 0.1307],
    [0.00965, 0.0079, 0.00320, 0.04710, 0.1593]
])

bbx = np.array([
    [0.0, 0.002, 0.1229, 0.2945, 0.5435],
    [0.0, 0.002, 0.1, 0.2413, 0.2687],
    [0.0, 0.002, 0.0822, 0.2003, 0.1417],
    [0.0, 0.002, 0.0665, 0.1619, 0.0693],
    [0.0, 0.002, 0.0544, 0.1337, 0.0356],
    [0.0, 0.002, 0.0445, 0.1099, 0.0181]
])

rrx = np.array([
    [1010000., 23900., 6000., 6000., 1289.],
    [1400000., 49800., 12300., 10280., 3109.],
    [1900000., 100100., 23900., 17100., 7109.],
    [2660000., 214500., 50100., 29900., 18109.],
    [3650000., 441000., 99900., 50000., 42109.],
    [5000000., 910000., 200000., 85070., 100109.]
])
aa = np.copy(aax)  # Create a copy of aax
aa[0, :] += 0.0098
aa[1, :] += 0.011
aa[2, :] += 0.0053

bb = np.copy(bbx)  # Create a copy of bbx
bb[3, :] += 0.015
bb[4, :] -= 0.191
bb[5, :] -= 0.2032

rr = np.copy(rrx)  # Create a copy of rrx
rr[5, :] += 1891.
    
    
def ldfa(var_independent, fl, fr, power_sm, power_jtn, rcr):
    """
    Calculates the logistic dose function.

    Args:
        var_independent: The independent variable.
        fl: The left asymptote.
        fr: The right asymptote.
        power_sm: The power for the slope.
        power_jtn: The power for the transition.
        rcr: The response concentration ratio.

    Returns:
        The value of the logistic dose function.
    """
    #print(var_independent, fl, fr, power_sm, power_jtn, rcr)
    
    t1 =  (fr - fl)
    t9 = (var_independent / rcr)

    t2 = t9 ** power_sm
    t3 = ((1 + t2) ** power_jtn)
    
    k1 =  t1 / t3
    #print(power_sm, fl,k1)
    r = fl + k1
    return r

def vm(re, sigma):
    # Modify AA, BB, and RR based on Sigma
    aa[0, :] = 0.17805185 * (sigma**(-0.46785053)) + 0.0098
    aa[1, :] = 0.18954211 * (sigma**(-0.51003100)) + 0.011
    aa[2, :] = 0.02166401 * (sigma**(-0.30702955)) + 0.0053
    aa[3, :] = 0.01105244 * (sigma**(0.23275646))
    aa_5_fl = 0.00255391 * (sigma**(0.8353877)) - 0.022
    aa_5_fr = 0.92820419 * (sigma**(0.03569244)) - 1.
    aa[4, :] = ldfa(sigma, aa_5_fl, aa_5_fr, -50., 0.5, 93.)

    bb[0, :] = 0.0
    bb[1, :] = 0.002
    bb[2, :] = 0.26827956 * (sigma**(-0.28852025)) + 0.015
    bb[3, :] = 0.62935712 * (sigma**(-0.28022284)) - 0.191
    bb[4, :] = 7.3482780 * (sigma**(-0.96433953)) - 0.2032

    rr[0, :] = 295530.05 * (sigma**(0.45435343))
    rr[1, :] = 1451.4594 * (sigma**(1.0337774))
    rr[2, :] = 406.33954 * (sigma**(0.99543306))
    rr[3, :] = 783.39696 * (sigma**(0.75245644))
    rr[4, :] = 45.196502 * (sigma**(1.2369807)) + 1891.
    
    # Calculations (using numpy's vectorized operations for efficiency)

    p = a * re**b  # Vectorized multiplication and exponentiation
    f = np.zeros_like(a)  # Initialize f with zeros

    f[0] = p[0]
    # f[1:] = ldfa(re, f[:-1], p[1:], s[:-1], t[:-1], r[:-1])  # Vectorized LDFA calls
    for i in range(5):  # Iterate from 1 to 4 (length of the arrays)
        f[i] = ldfa(re, f[i - 1], p[i], s[i - 1], t[i - 1], r[i - 1])

    pp = aa * re**bb  # Vectorized multiplication and exponentiation for 2D arrays
    ff = np.zeros_like(aa)  # Initialize ff with zeros
    ff[0, :] = pp[0, :]
    
    # ff[1:, :] = ldfa(re, pp[1:, :], ff[:-1, :], m[:-1], n[:-1], rr[:-1, :])  # Vectorized LDFA calls
    for i in range(1,5):  # Iterate over rows
        for j in range(5):  # Iterate over columns (assuming 6 columns in ff)
            ff[i, j] = ldfa(re, pp[i, j], ff[i - 1, j], m[i - 1], n[i - 1], rr[i - 1, j])

    index_j = 1  # Comment: INDEX_J is roughness index and may be 1, 2, 3, 4, 5, or 6 for the six values of roughness in Nikuradse’s (1933) data. The result does not depend on INDEX_J.
    lamda_s = f[-1]  # Access the last element of f
    lamda_r = ff[-1, index_j - 1] # clever, spotted that user is asked for 1..6 but python indexing works from zero !
    lamda = ldfa(re, lamda_s, lamda_r, m[-1], n[-1], rr[-1, index_j - 1])
    
    # attempting to fix bad stuff
    lamda = max(lamda, 5e-3)
    f_laminar = 64 / re
    if lamda < f_laminar:
        lamda = f_laminar
    return lamda
    
def main():
    """
    Calculates the friction factor using the Virtual Nikuradse Correlation (VNC).
    """
    print("Welcome! This code calculates Lamda = f(Re, Sigma) for pipe flows using the Virtual Nikuradse Correlation (VNC).")
    print("Reminder: VNC requires Re > 0 and Sigma >= 15.")
    print("This DOES NOT WORK - imperfect conversion from frotran not fixed yet.")

    # Main loop
    outer_loop = False
    while not outer_loop:
        # Get input from the user
        # re = float(input("Please input the value of flow Reynolds number (Re>0) then press ENTER: "))
        # sigma = float(input("Please input the value of roughness ratio (Sigma>=15) then press ENTER: "))
        re = 3000
        sigma = 200
        if re <= 0 or sigma < 15:
            raise ValueError("Invalid input: Re must be greater than 0 and Sigma must be greater than or equal to 15")

        lamda = vm(re, sigma)


        print(f"{re=} {sigma=} : The friction factor is: Lamda = {lamda:.5f}")

        while True:
            print("[Press '1' to continue or press '0' to exit]")
            k = input()

            if k == "0":
                outer_loop = True
            elif k == "1":
                # Continue with the calculation
                pass
            else:
                print("ERROR!!! Please only input the number 1 or 0!")  
            break
    print("---------------------------------------------------------")
    print("Thank you for using the VNC friction factor calculator.")
    print("If there is any question, please feel free to contact me")
    print("at yhaoping@aem.umn.edu or Haoping.Yang@noaa.gov.")
    print("\n")  # Print two newlines to create empty lines
    

if __name__ == '__main__':
    sys.exit(main())  
