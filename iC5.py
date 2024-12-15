# Data from
#https://webbook.nist.gov/cgi/cbook.cgi?ID=C78784&Units=SI&Mask=1#Thermo-Gas
# Used copilot to parse the table of data and write the code 31/3/2024
import numpy as np
from numpy.polynomial import Polynomial

# Data
T = np.array([200., 273.15, 298.15, 300., 400., 500., 600., 700., 800., 900., 1000., 1100., 1200., 1300., 1400., 1500.])
Cp = np.array([84.94, 110.37, 118.9, 119.50, 152.88, 183.26, 210.04, 233.05, 253.13, 270.70, 286.19, 299.57, 311.29, 322.17, 330.54, 338.90])

# Fit a third order polynomial
p = Polynomial.fit(T, Cp, 3)

# Print the coefficients
print("Coefficients: ", p.coef)

# Function to evaluate the polynomial
def eval_poly(x):
    return p(x)

# Test the function
print("Value at T = 300: ", eval_poly(300))
print(p)
# Coefficients:  [262.15489124 116.06106507 -51.11614166  11.57123613]
# Value at T = 300:   120.34126989351572
# 262.15489124 + 116.06106507·T - 51.11614166·T² + 11.57123613·T³