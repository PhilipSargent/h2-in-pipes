import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

"""This script will plot the original data and the fitted parabola. The numpy.polyfit function returns the coefficients of the fitted polynomial, highest power first. In this case, it returns the coefficients a, b, and c of the parabola equation y=ax2+bx+c
"""
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# data: digitised from fitted line in Fig.2 of Mehrafarin and Nima Pourtolami (2008)
x = np.array([23.454021622906026, 27.788516804793403, 31.789589280381733, 35.709832008988485, 39.34717062315971, 45.12649753234285, 52.961931130370004, 61.30177343792644,  68.93085801841892, 77.0461646153698, 86.91547672182102, 88.54554328594958, 94.91762167299768, 106.03171188296528, 111.81103879214845, 130.03814673649532, 139.81854612126682, 146.04243663884864, 158.49021767401234, 166.78873836412146, 177.16188922675792, 189.1651066535229, 207.8367782062685, 225.17475893381794, 235.20919466624582, 240.2899216193739, 257.73904324902304, 264.2963564729039, 286.07997328444037, 298.5277543196041, 325.20157082352637,  354.0982053694421, 379.8828946565669, 423.89469188803866, 463.01628942712455])
y = np.array([46.60324506513257, 50.99987423068836, 54.4591991895596, 58.53432125275529, 62.21929265648477, 65.90935049646487, 72.02672337881052, 78.3874644673193,  81.89781621723549, 85.46338469659703, 89.15536744937835, 90.1165405747983, 90.89356370889946, 95.55937734703616, 96.53933295494095, 102.42396638040923, 104.44953462194842, 105.19887401012629, 109.75550426094892, 111.62885273139358, 113.4152573626258, 116.8587124848468, 120.48197585062378, 124.74455525531494, 126.28308555972544, 126.8836583537128, 129.7132801715379, 130.92597523632008, 134.44728238739128, 136.11810669886896, 139.29806264652,  145.05791505791507, 148.05559926249583, 154.74869606448556, 158.44067881726684])

# Plot the original data
plt.scatter(x, y, label='Digitised data')

for n in [ 2]:
    # Fit a second degree polynomial to the data
    coefficients = np.polyfit(x, y, n)

    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Generate y-values for the polynomial on a set of x-values
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = polynomial(x_fit)

 
    # Plot the fitted polynomial
    #plt.plot(x_fit, y_fit, 'r', label=f'polynomial order-{n}')



# Take the logarithm of the data
log_x = np.log(x)
log_y = np.log(y)

# Fit a line to the logarithmic data
coefficients = np.polyfit(log_x, log_y, 1)

# Create a polynomial function from the coefficients
polynomial = np.poly1d(coefficients)

# Generate y-values for the polynomial on a set of x-values
x2_fit = np.linspace(min(x), max(x), 500)
y2_fit = np.exp(polynomial(np.log(x_fit)))



# Plot the fitted power law
plt.plot(x2_fit, y2_fit, 'r', label='Fitted power law')


# Fit a cubic spline to the data
cs = CubicSpline(x, y)

# Generate y-values for the spline on a set of x-values
x_fit = np.linspace(min(x), max(x), 500)
y_fit = cs(x_fit)

# Plot the fitted cubic spline
# plt.plot(x_fit, y_fit, 'r', label='Fitted cubic spline')

plt.xlabel('(ε/D)Re^6/(8+3η)')

plt.ylabel('100f Re^(2+3η)/(8+3η)')


plt.legend()
plt.grid(True)
plt.savefig("goldenf_1.png")
plt.close()
