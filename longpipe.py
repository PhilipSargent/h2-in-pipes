import math
from scipy.optimize import newton
import matplotlib.pyplot as plt
# AI written but clearly bananas

# Constants for hydrogen gas at 1 atm and 0°C
VISCOSITY_HYDROGEN = 0.0000089  # Pa.s
DENSITY_HYDROGEN = 0.08376  # kg/m^3
SOS_HYDROGEN = 1320 # m/s speed of sound

# Relative roughness of the pipe
RELATIVE_ROUGHNESS = 1e-5
PRESSURE = 1e5 # Pa i.e. 1 bar

def colebrook(f, Re, relative_roughness):
    """
    Implicit Colebrook equation.

    Parameters:
    f (float): Darcy friction factor
    Re (float): Reynolds number
    relative_roughness (float): Relative roughness of the pipe

    Returns:
    float: Result of the Colebrook equation
    """
    return 1/f**0.5 + 2.0 * math.log10(relative_roughness/3.7 + 5.74/(Re * f**0.5))

def calculate_pressure_drop(diameter, length, flow_rate):
    """
    Calculate the pressure drop in a pipe using the Darcy-Weisbach equation.

    Parameters:
    diameter (float): Diameter of the pipe (m)
    length (float): Length of the pipe (m)
    flow_rate (float): Flow rate (m^3/s)

    Returns:
    float: Pressure drop (Pa)
    """

    # Initialize variables
    pressure_drop = 0
    current_flow_rate = flow_rate
    current_density = DENSITY_HYDROGEN

    # Calculate the velocity
    velocity_initial = current_flow_rate / (math.pi * (diameter / 2)**2)

    # Calculate the Reynolds number
    reynolds_number = (current_density * velocity_initial * diameter) / VISCOSITY_HYDROGEN

    # Solve the Colebrook equation for the friction factor using the Newton-Raphson method
    friction_factor = newton(colebrook, 0.02, args=(reynolds_number, RELATIVE_ROUGHNESS))
    print(f"Re={reynolds_number:.0f} {friction_factor=:.3f}  {velocity_initial=:.3f} m/s")

    # Divide the pipe into small segments and calculate the pressure drop for each segment
    num_segments = 1000
    segment_length = length / num_segments

    # Initialize lists to store the pressure drop and position along the pipe for plotting
    pressure_drops = []
    positions = []
    pressure = []
    velocity = []
    v = velocity_initial

    for i in range(num_segments):
        # Calculate the pressure drop for this segment using the Darcy-Weisbach equation
        segment_pressure_drop = friction_factor * (segment_length/diameter) * (0.5 * current_density * v**2)
        print(f"{segment_pressure_drop=}")
        
        pressure_drop += segment_pressure_drop

        # Update the flow rate for the next segment based on the pressure drop
        current_flow_rate = current_flow_rate * (1 + segment_pressure_drop / (current_density * current_flow_rate**2))
        v =  velocity_initial *(current_flow_rate /flow_rate)

        # Update the density for the next segment based on the conservation of mass
        current_density = DENSITY_HYDROGEN * flow_rate / current_flow_rate

        # Store the pressure drop and position along the pipe for plotting
        velocity.append(v)
        pressure_drops.append(pressure_drop)
        pressure.append(PRESSURE - pressure_drop)
        positions.append(i * segment_length)

    # Plot the pressure drop along the pipe
    plt.figure(figsize=(10, 6))
    plt.plot(positions, velocity)
    plt.xlabel('Position along the pipe (m)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity along the pipe')
    plt.grid(True)
    plt.savefig('velocity.png')
 
    plt.figure(figsize=(10, 6))
    plt.plot(positions, pressure_drops)
    plt.xlabel('Position along the pipe (m)')
    plt.ylabel('Pressure drop (Pa)')
    plt.title('Pressure drop along the pipe')
    plt.grid(True)
    plt.savefig('pressure_drop.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(positions, pressure)
    plt.xlabel('Position along the pipe (m)')
    plt.ylabel('Pressure  (Pa)')
    plt.title('Pressure  along the pipe')
    plt.grid(True)
    plt.savefig('pressure.png')

# Use the function with the necessary parameters
diameter =  35 / 1000 # mm conveted to m
flow_rate = 6 / 3600 # 6 cubic metres an hour

calculate_pressure_drop(diameter, 200, flow_rate)
