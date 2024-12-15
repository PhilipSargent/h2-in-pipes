import matplotlib.pyplot as plt

"""
Equation and Parameters:

The paper presents two main equations:

Equation (1): Relates the friction factor (f) to the Reynolds number (Re) and the roughness parameter (k):
f = \\frac{0.8}{\\left(Re/A \\right)^{\\frac{1}{3}}} \\left( 1 + \\frac{12}{\\left(Re/B \\right)^2} \\right)^{-\\frac{1}{2}}

Equation (2): Defines the parameters A and B based on the roughness ratio ϵ=k/D:
A = 30 + 8.8 \\ epsilon + 7.1 \\ epsilon^2 + 2.4 \\ epsilon^3
B = 550 + 33 \\ epsilon^2

"""
def gioia_chakraborty_friction_factor(Re, epsilon):
    """
    Calculates the friction factor using the Gioia and Chakraborty method.

    Args:
        Re: Reynolds number
        k: Roughness height
        D: Pipe diameter

    Returns:
        Friction factor
    """

    A = 30 + 8.8 * epsilon + 7.1 * epsilon**2 + 2.4 * epsilon**3
    B = 550 + 33 * epsilon**2

    f = 0.8 / (Re / A)**(1/3) * (1 + 12 / (Re / B)**2)**(-1/2)

    return f
    

# Parameters

#r = k / D  # Calculate roughness ratio
r = 0.02
Re_values = range(1000, 10000, 10)  # Range of Reynolds numbers

plt.figure(figsize=(10, 6))

# Calculate friction factors for each Reynolds number
for r in [1e-1, 3e-2, 1e-6]:
    f_values = [gioia_chakraborty_friction_factor(Re, r) for Re in Re_values]
    plt.plot(Re_values, f_values,label=f'ε/D = {r}')

# Create the plot
plt.xlabel("Reynolds Number (Re)")
plt.ylabel("Friction Factor (f)")
plt.title("Friction Factor vs. Reynolds Number (Gioia and Chakraborty)")

plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('gcff.png')