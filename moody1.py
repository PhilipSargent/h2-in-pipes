'''Bard created program
This gets it completely wrong.
'''
import matplotlib.pyplot as plt
import numpy as np

def plot_moody_diagram():
    """
    Plots a Moody diagram.
    """
    # Reynolds number range
    Re_min = 2000
    Re_max = 10^7
    Re_values = np.logspace(np.log10(Re_min), np.log10(Re_max), 100)

    # Relative roughness range
    epsilon_over_D_min = 0.001
    epsilon_over_D_max = 0.05
    epsilon_over_D_values = np.linspace(epsilon_over_D_min, epsilon_over_D_max, 100)

    # Calculate friction factor for each combination of Reynolds number and relative roughness
    f_D_values = np.zeros((len(Re_values), len(epsilon_over_D_values)))
    for i, Re in enumerate(Re_values):
        for j, epsilon_over_D in enumerate(epsilon_over_D_values):
            if Re <= 2300:
                f_D = 64/Re
            elif 2300 < Re <= 4000:
                f_D = 0.079/(Re**0.25)
            else:
                f_D = 0.251/(Re**0.25) * np.log10(epsilon_over_D/3.7) + 0.046
            f_D_values[i, j] = f_D

    # Create the plot
    fig, ax1 = plt.subplots()

    # Plot friction factor vs. Reynolds number for each roughness
    for j, epsilon_over_D in enumerate(epsilon_over_D_values):
        ax1.plot(Re_values, f_D_values[:, j], label=f'$\epsilon/D = {epsilon_over_D:.3f}$')

    # Set labels and limits
    ax1.set_xlabel('Reynolds number, $Re$')
    ax1.set_ylabel('Friction factor, $f_D$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([Re_min, Re_max])
    ax1.set_ylim([1e-3, 10])

    # Add legend
    ax1.legend()

    # Add grid
    ax1.grid(True)

    # Show the plot
    plt.savefig('moody1_diagram.png')
    #plt.show()

if __name__ == '__main__':
    plot_moody_diagram()
