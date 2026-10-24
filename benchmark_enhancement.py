import sys
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# 1. Import gas critical data from uploaded gas_data.py
try:
    from peng_utils import memoize
    from gas_data import gas_data
except ImportError:
    print("Error: Could not import gas_data.py. Ensure gas_data.py is in the working directory.")
    sys.exit(1)

# Import Sonntag pure water vapor pressure function from uploaded sonntag_2.py
try:
    from sonntag_2 import sonntag_vapor_pressure
except ImportError:
    # Fallback implementation of Sonntag (1990) if sonntag_2.py is missing
    def sonntag_vapor_pressure(T_K):
        A1, A2, A3, A4, A5 = -6096.9385, 21.2409642, -2.711193e-2, 1.673952e-5, 2.433502
        ln_P = A1 / T_K + A2 + A3 * T_K + A4 * T_K**2 + A5 * np.log(T_K)
        return np.exp(ln_P)

# --- Universal Gas Constant & Constants ---
R = 83.144626  # bar cm^3 / (mol K)

# ==============================================================================
# 2. COUTINHO - TSONOPOULOS MODEL IMPLEMENTATION
# ==============================================================================

@memoize
def coutinho_b(species):
    """
    Computes component co-volume b_i (cm^3/mol) using Peng-Robinson definition:
    b_i = 0.07780 * R * Tc / Pc
    """
    Tc = gas_data[species]['Tc']
    Pc = gas_data[species]['Pc']  # bar
    return 0.07780 * (R * Tc / Pc)

@memoize
def coutinho_k_ij(b1, b2):
    """Calculates binary interaction parameter k_ij using Coutinho et al. correlation."""
    term = 2.0 * np.sqrt(b1 * b2) / (b1 + b2)
    return 1.0 - 0.885 * (term ** -0.036)

@memoize
def tsonopoulos_B(T, Tc, Pc, omega):
    """Calculates pure second virial coefficient B(T) [cm^3/mol] via Tsonopoulos (1974)."""
    Tr = T / Tc
    f0 = 0.1445 - 0.330 / Tr - 0.1385 / (Tr**2) - 0.0121 / (Tr**3) - 0.00060 / (Tr**8)
    f1 = 0.0637 + 0.331 / (Tr**2) - 0.423 / (Tr**3) - 0.008 / (Tr**8)
    return (R * Tc / Pc) * (f0 + omega * f1)

@memoize
def cross_virial_B_gw(T, species1, species2):
    """Calculates cross-second virial coefficient B_12(T) using Coutinho k_ij."""
    p1, p2 = gas_data[species1], gas_data[species2]
    
    b1 = coutinho_b(species1)
    b2 = coutinho_b(species2)
    k_ij = coutinho_k_ij(b1, b2)
    
    Tc_12 = np.sqrt(p1['Tc'] * p2['Tc']) * (1.0 - k_ij)
    Vc1 = (R * p1['Tc'] / p1['Pc']) * 0.29
    Vc2 = (R * p2['Tc'] / p2['Pc']) * 0.29
    Vc_12 = ((Vc1**(1/3) + Vc2**(1/3)) / 2.0)**3
    omega_12 = 0.5 * (p1['omega'] + p2['omega'])
    Pc_12 = (0.29 * R * Tc_12) / Vc_12
    
    return tsonopoulos_B(T, Tc_12, Pc_12, omega_12)

def f_enhancement_coutinho(T_K, P_bar, y_dict):
    """
    Computes enhancement factor f(T, P) using Coutinho-Tsonopoulos Virial Model.
    y_dict: mole fractions of all components including H2O.
    """
    v_l = 18.07  # Molar volume of liquid water (cm^3/mol)
    y_w = y_dict.get('H2O', 0.0)
    
    # Extract dry gas composition normalized
    dry_dict = {k: v / (1.0 - y_w) for k, v in y_dict.items() if k != 'H2O'}
    
    # 1. Mixture dry-gas B_gg
    B_gg = 0.0
    for g1, y1 in dry_dict.items():
        for g2, y2 in dry_dict.items():
            if g1 == g2:
                B_12 = tsonopoulos_B(T_K, gas_data[g1]['Tc'], gas_data[g1]['Pc'], gas_data[g1]['omega'])
            else:
                B_12 = cross_virial_B_gw(T_K, g1, g2)
            B_gg += y1 * y2 * B_12

    # 2. Cross-virial B_gw (Dry Matrix <-> H2O)
    B_gw = 0.0
    for g, y_g in dry_dict.items():
        B_gw += y_g * cross_virial_B_gw(T_K, g, 'H2O')
        
    # 3. Logarithm of enhancement factor ln(f)
    term1 = ((1.0 - y_w) * P_bar) / (R * T_K)
    ln_f = term1 * (B_gg - 2.0 * B_gw + v_l)
    return np.exp(ln_f)

# ==============================================================================
# 3. COOLPROP MODEL IMPLEMENTATION
# ==============================================================================
def f_enhancement_coolprop(T_K, P_bar, y_dict):
    return f_enhancement_coolprop_gerg_virial(T_K, P_bar, y_dict)
    
def f_enhancement_coolprop_gerg_virial(T_K, P_bar, y_dict):
    """
    Computes f(T, P) using exact GERG-2008 Virial Coefficients extracted from CoolProp.
    Bypasses phase-flash instability completely.
    """
    y_w = y_dict.get('H2O', 0.0)
    if y_w <= 0:
        return 1.0

    v_l = 18.07  # Liquid water molar volume (cm^3/mol)
    dry_dict = {k: v / (1.0 - y_w) for k, v in y_dict.items() if k != 'H2O'}
    
    name_map = {'H2O': 'Water', 'CO2': 'CO2', 'N2': 'Nitrogen', 'O2': 'Oxygen'}
    
    def get_gerg_B(sp1, sp2, T):
        """Extracts B_12(T) in cm^3/mol from CoolProp HEOS/GERG2008."""
        n1, n2 = name_map.get(sp1, sp1), name_map.get(sp2, sp2)
        if sp1 == sp2:
            # Pure second virial B(T) in m^3/mol -> convert to cm^3/mol
            B_m3 = CP.PropsSI('Bvirial', 'T', T, 'P', 1e5, n1)
            return B_m3 * 1e6
        else:
            # Cross second virial B_12(T) using HEOS binary mixture
            mix_str = f"HEOS::{n1}[0.5]&{n2}[0.5]"
            B_m3 = CP.PropsSI('Bvirial', 'T', T, 'P', 1e5, mix_str)
            return B_m3 * 1e6

    try:
        # 1. GERG Dry Gas Virial B_gg
        B_gg = 0.0
        for g1, y1 in dry_dict.items():
            for g2, y2 in dry_dict.items():
                B_12 = get_gerg_B(g1, g2, T_K)
                B_gg += y1 * y2 * B_12

        # 2. GERG Cross Virial B_gw (Dry Gas Matrix <-> Water)
        B_gw = 0.0
        for g, y_g in dry_dict.items():
            B_gw += y_g * get_gerg_B(g, 'H2O', T_K)

        # 3. Enhancement Factor
        term1 = ((1.0 - y_w) * P_bar) / (R * T_K)
        ln_f = term1 * (B_gg - 2.0 * B_gw + v_l)
        return np.exp(ln_f)
        
    except Exception:
        return np.nan
        
# ==============================================================================
# 4. BENCHMARKING & PLOTTING ROUTINES
# ==============================================================================

def generate_flue_gas_composition(y_co2, y_h2o, o2_ratio=0.03):
    """Generates a balanced flue gas composition (CO2, H2O, O2, N2 remainder)."""
    y_remainder = 1.0 - y_co2 - y_h2o
    y_o2 = y_remainder * o2_ratio
    y_n2 = y_remainder * (1.0 - o2_ratio)
    return {'H2O': y_h2o, 'CO2': y_co2, 'O2': y_o2, 'N2': y_n2}

def run_benchmarking():
    # Grid Setup
    T_C_vec = np.linspace(-10, 100, 5)       # Temperature -10 C to 100 C, 45 iter
    P_bar_vec = np.linspace(0.5, 10, 5)      # Pressure 0.5 bar to 10 bar, 45 iter
    T_grid, P_grid = np.meshgrid(T_C_vec, P_bar_vec)

    # Parametric sweeps for CO2 and H2O contents
    co2_contents = [0.00, 0.05, 0.10]         # 0%, 5%, 10% CO2
    h2o_contents = [0.05, 0.15, 0.30, 0.50]   # 5%, 15%, 30%, 50% H2O

    print("Running benchmarking simulations across T, P, CO2, and H2O grids...")

    # Figure 1: Multi-panel Contour Plot of Delta f = f_Coutinho - f_CoolProp
    fig, axes = plt.subplots(len(co2_contents), len(h2o_contents), 
                             figsize=(18, 12), sharex=True, sharey=True)
    
    delta_max_overall = 0.0

    for i, y_co2 in enumerate(co2_contents):
        print(f"CO2: {y_co2}")
        for j, y_h2o in enumerate(h2o_contents):
            print(f"  H2O: {y_h2o}")
            ax = axes[i, j]
            y_dict = generate_flue_gas_composition(y_co2, y_h2o)
            
            f_coutinho = np.zeros_like(T_grid)
            f_coolprop = np.zeros_like(T_grid)
            
            for r in range(T_grid.shape[0]):
                for c in range(T_grid.shape[1]):
                    T_K = T_grid[r, c] + 273.15
                    P_bar = P_grid[r, c]
                    f_coutinho[r, c] = f_enhancement_coutinho(T_K, P_bar, y_dict)
                    f_coolprop[r, c] = f_enhancement_coolprop(T_K, P_bar, y_dict)
            
            # Difference in enhancement factor: Delta f = f_Coutinho - f_CoolProp
            delta_f = f_coutinho - f_coolprop
            print(f"    {f_coutinho=}\n    {f_coolprop=}")
            delta_max_overall = max(delta_max_overall, np.nanmax(np.abs(delta_f)))
            
            # Contour plot
            cp_plot = ax.contourf(T_grid, P_grid, delta_f, levels=15, cmap='RdBu_r')
            ax.contour(T_grid, P_grid, delta_f, levels=8, colors='k', linewidths=0.4)
            
            ax.set_title(rf"$\mathrm{{CO_2}}$={y_co2*100:.0f}%, $\mathrm{{H_2O}}$={y_h2o*100:.0f}%", fontsize=10)
            if i == len(co2_contents) - 1:
                ax.set_xlabel(r"Temperature ($^\circ$C)")
            if j == 0:
                ax.set_ylabel("Pressure (bar)")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cp_plot, cax=cbar_ax)
    cbar.set_label(r"$\Delta f = f_{\mathrm{Coutinho-Tsonopoulos}} - f_{\mathrm{CoolProp}}$", fontsize=12)
    fig.suptitle(r"Enhancement Factor Deviation ($\Delta f$) Across Composition Matrix", fontsize=16, fontweight='bold')
    plt.savefig("enhancement_factor_contour_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Complementary 1D Cross-Section Plot at P = 5 bar
    plt.figure(figsize=(10, 6))
    P_fixed = 5.0  # bar
    for y_h2o in [0.05, 0.15, 0.30, 0.50]:
        y_dict = generate_flue_gas_composition(y_co2=0.08, y_h2o=y_h2o)
        f_cout_line = [f_enhancement_coutinho(tc + 273.15, P_fixed, y_dict) for tc in T_C_vec]
        f_cp_line = [f_enhancement_coolprop(tc + 273.15, P_fixed, y_dict) for tc in T_C_vec]
        
        plt.plot(T_C_vec, f_cout_line, label=rf"Coutinho ($\mathrm{{H_2O}}$={y_h2o*100:.0f}%)", linestyle='--')
        plt.plot(T_C_vec, f_cp_line, label=rf"CoolProp ($\mathrm{{H_2O}}$={y_h2o*100:.0f}%)", linestyle='-')

    plt.title(rf"Enhancement Factor vs. Temperature at P = {P_fixed} bar ($\mathrm{{CO_2}}$ = 8%)", fontsize=12, fontweight='bold')
    plt.xlabel(r"Temperature ($^\circ$C)", fontsize=11)
    plt.ylabel(r"Enhancement Factor $f(T, P)$", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("enhancement_factor_1D_cross_section.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(rf"Benchmarking completed. Maximum absolute deviation: {delta_max_overall:.6f}")

if __name__ == '__main__':
    run_benchmarking()