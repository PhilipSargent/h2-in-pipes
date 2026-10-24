import sys
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# 1. Import gas critical data
try:
    from peng_utils import memoize
    from gas_data import gas_data
except ImportError:
    print("Error: Could not import gas_data.py. Ensure gas_data.py is in the working directory.")
    sys.exit(1)

# Import Sonntag pure water vapor pressure function
try:
    from sonntag_2 import sonntag_vapor_pressure
except ImportError:
    # Fallback implementation of Sonntag (1990) if sonntag_2.py is missing
    def sonntag_vapor_pressure(T_K):
        A1, A2, A3, A4, A5 = -6096.9385, 21.2409642, -2.711193e-2, 1.673952e-5, 2.433502
        ln_P = A1 / T_K + A2 + A3 * T_K + A4 * T_K**2 + A5 * np.log(T_K)
        return np.exp(ln_P)

sonntag_vapor_pressure_vec = np.vectorize(sonntag_vapor_pressure)

# Universal Gas Constant & Constants
R = 83.144626  # bar cm^3 / (mol K)
v_l = 18.07    # Liquid water molar volume (cm^3/mol)
name_map = {'H2O': 'Water', 'CO2': 'CO2', 'N2': 'Nitrogen', 'O2': 'Oxygen'}

# ==============================================================================
# 2. COUTINHO - TSONOPOULOS MODEL IMPLEMENTATION
#    PRE-CALCULATED VIRIAL TABLES (Executed ONCE across 1D Temperature vector)
# ==============================================================================

def coutinho_b(species):
    Tc = gas_data[species]['Tc']
    Pc = gas_data[species]['Pc']  # bar
    return 0.07780 * (R * Tc / Pc)

@memoize
def coutinho_k_ij(b1, b2):
    """Calculates binary interaction parameter k_ij using Coutinho et al. correlation."""
    term = 2.0 * np.sqrt(b1 * b2) / (b1 + b2)
    return 1.0 - 0.885 * (term ** -0.036)

def tsonopoulos_B_1D(T_vec_K, Tc, Pc, omega):
    """Calculates 1D array of pure B(T) [cm^3/mol] over 1D temperature vector."""
    Tr = T_vec_K / Tc
    f0 = 0.1445 - 0.330 / Tr - 0.1385 / (Tr**2) - 0.0121 / (Tr**3) - 0.00060 / (Tr**8)
    f1 = 0.0637 + 0.331 / (Tr**2) - 0.423 / (Tr**3) - 0.008 / (Tr**8)
    return (R * Tc / Pc) * (f0 + omega * f1)

def build_virial_tables_1D(T_vec_K, species_list):
    """
    Pre-computes B_12(T) arrays for all species pairs on 1D T_vec_K.
    Robust against CoolProp subcooled water melting line exceptions (T < 273.15 K).
    """
    B_coutinho_table = {}
    B_gerg_table = {}

    for i, sp1 in enumerate(species_list):
        for j, sp2 in enumerate(species_list[i:], i):
            n1, n2 = name_map.get(sp1, sp1), name_map.get(sp2, sp2)
            
            # --- 1. Coutinho - Tsonopoulos Virial ---
            if sp1 == sp2:
                B_cout = tsonopoulos_B_1D(T_vec_K, gas_data[sp1]['Tc'], gas_data[sp1]['Pc'], gas_data[sp1]['omega'])
            else:
                p1, p2 = gas_data[sp1], gas_data[sp2]
                b1, b2 = coutinho_b(sp1), coutinho_b(sp2)
                k_ij = coutinho_k_ij(b1, b2)
                Tc_12 = np.sqrt(p1['Tc'] * p2['Tc']) * (1.0 - k_ij)
                Vc1 = (R * p1['Tc'] / p1['Pc']) * 0.29
                Vc2 = (R * p2['Tc'] / p2['Pc']) * 0.29
                Vc_12 = ((Vc1**(1/3) + Vc2**(1/3)) / 2.0)**3
                omega_12 = 0.5 * (p1['omega'] + p2['omega'])
                Pc_12 = (0.29 * R * Tc_12) / Vc_12
                B_cout = tsonopoulos_B_1D(T_vec_K, Tc_12, Pc_12, omega_12)

            B_coutinho_table[(sp1, sp2)] = B_cout
            B_coutinho_table[(sp2, sp1)] = B_cout

            # --- 2. GERG-2008 Virial (CoolProp) with Subcooled Exception Handling ---
            B_gerg = np.zeros_like(T_vec_K)
            for k, T_scalar in enumerate(T_vec_K):
                try:
                    if sp1 == sp2:
                        B_gerg[k] = CP.PropsSI('Bvirial', 'T', T_scalar, 'P', 1e5, n1) * 1e6
                    else:
                        mix_str = f"HEOS::{n1}[0.5]&{n2}[0.5]"
                        B_gerg[k] = CP.PropsSI('Bvirial', 'T', T_scalar, 'P', 1e5, mix_str) * 1e6
                except ValueError:
                    # Fallback to Tsonopoulos correlation for subcooled water state (T < 273.15 K)
                    B_gerg[k] = B_cout[k]

            B_gerg_table[(sp1, sp2)] = B_gerg
            B_gerg_table[(sp2, sp1)] = B_gerg

    return B_coutinho_table, B_gerg_table
    
# ==============================================================================
# 3. FAST 2D MATRIX EVALUATION VIA BROADCASTING
# ==============================================================================

def compute_enhancement_grid(T_vec_K, P_vec_bar, y_dict, B_table):
    """
    Evaluates full (P x T) 2D matrix using 1D virial table and 2D pressure broadcasting.
    """
    y_w = y_dict.get('H2O', 0.0)
    dry_dict = {k: v / (1.0 - y_w) for k, v in y_dict.items() if k != 'H2O'}

    # 1. Compute 1D B_gg(T) and B_gw(T) vectors across temperature axis
    B_gg_1D = np.zeros_like(T_vec_K)
    for g1, y1 in dry_dict.items():
        for g2, y2 in dry_dict.items():
            B_gg_1D += y1 * y2 * B_table[(g1, g2)]

    B_gw_1D = np.zeros_like(T_vec_K)
    for g, y_g in dry_dict.items():
        B_gw_1D += y_g * B_table[(g, 'H2O')]

    # 2. 1D Temperature term
    bracket_term_1D = (B_gg_1D - 2.0 * B_gw_1D + v_l)  # shape (N_T,)

    # 3. Broadcast 1D Pressure vector (N_P, 1) against 1D Temperature terms (1, N_T)
    P_col = P_vec_bar[:, np.newaxis]   # shape (N_P, 1)
    T_row = T_vec_K[np.newaxis, :]     # shape (1, N_T)

    ln_f_grid = ((1.0 - y_w) * P_col / (R * T_row)) * bracket_term_1D  # shape (N_P, N_T)
    return np.exp(ln_f_grid)

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
    # 1D Grid Vectors
    T_C_vec = np.linspace(-10, 100, 45)       # 45 temperature points
    P_bar_vec = np.linspace(0.5, 10, 45)      # 45 pressure points
    T_vec_K = T_C_vec + 273.15

    # Build 2D meshgrid coordinates for contour plotting
    T_grid_C, P_grid = np.meshgrid(T_C_vec, P_bar_vec)

    # Pre-compute all virials once for species list
    all_species = ['H2O', 'CO2', 'N2', 'O2']
    print("Pre-computing 1D Virial Tables across temperature vector...")
    B_coutinho_table, B_gerg_table = build_virial_tables_1D(T_vec_K, all_species)

    # Parametric sweeps for CO2 and H2O contents
    co2_contents = [0.00, 0.05, 0.10]         # 0%, 5%, 10% CO2
    h2o_contents = [0.05, 0.15, 0.30, 0.50]   # 5%, 15%, 30%, 50% H2O

    print("Running matrix calculations...")

    fig, axes = plt.subplots(len(co2_contents), len(h2o_contents), 
                             figsize=(18, 12), sharex=True, sharey=True)
    
    ratio_pct_max_overall = 0.0

    for i, y_co2 in enumerate(co2_contents):
        for j, y_h2o in enumerate(h2o_contents):
            ax = axes[i, j]
            y_dict = generate_flue_gas_composition(y_co2, y_h2o)
            
            # Fast 2D matrix calculation via broadcasting
            f_coutinho = compute_enhancement_grid(T_vec_K, P_bar_vec, y_dict, B_coutinho_table)
            f_coolprop = compute_enhancement_grid(T_vec_K, P_bar_vec, y_dict, B_gerg_table)
            
            # Percentage Ratio: (f_Coutinho / f_CoolProp - 1) * 100 %
            pct_ratio = ((f_coutinho / f_coolprop) - 1.0) * 100.0
            ratio_pct_max_overall = max(ratio_pct_max_overall, np.nanmax(np.abs(pct_ratio)))
            
            # Contour plot for percentage ratio
            cp_plot = ax.contourf(T_grid_C, P_grid, pct_ratio, levels=15, cmap='RdBu_r')
            ax.contour(T_grid_C, P_grid, pct_ratio, levels=8, colors='k', linewidths=0.4)
            
            ax.set_title(rf"$\mathrm{{CO_2}}$={y_co2*100:.0f}%, $\mathrm{{H_2O}}$={y_h2o*100:.0f}%", fontsize=10)
            if i == len(co2_contents) - 1:
                ax.set_xlabel(r"Temperature ($^\circ$C)")
            if j == 0:
                ax.set_ylabel("Pressure (bar)")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cp_plot, cax=cbar_ax)
    cbar.set_label(r"Relative Deviation $\left(\frac{f_{\mathrm{Coutinho}}}{f_{\mathrm{GERG}}} - 1\right) \times 100$ (%)", fontsize=12)
    fig.suptitle(r"Enhancement Factor Relative Deviation (%) Across Composition Matrix", fontsize=16, fontweight='bold')
    plt.savefig("enhancement_factor_contour_matrix_pct.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Complementary 1D Cross-Section Plot at P = 5 bar
    plt.figure(figsize=(10, 6))
    P_fixed_vec = np.array([5.0])  # Single 5 bar slice

    for y_h2o in [0.05, 0.15, 0.30, 0.50]:
        y_dict = generate_flue_gas_composition(y_co2=0.08, y_h2o=y_h2o)
        
        f_cout_line = compute_enhancement_grid(T_vec_K, P_fixed_vec, y_dict, B_coutinho_table)[0]
        f_cp_line = compute_enhancement_grid(T_vec_K, P_fixed_vec, y_dict, B_gerg_table)[0]
        
        plt.plot(T_C_vec, f_cout_line, label=rf"Coutinho ($\mathrm{{H_2O}}$={y_h2o*100:.0f}%)", linestyle='--')
        plt.plot(T_C_vec, f_cp_line, label=rf"CoolProp GERG ($\mathrm{{H_2O}}$={y_h2o*100:.0f}%)", linestyle='-')

    plt.title(r"Enhancement Factor vs. Temperature at P = 5 bar ($\mathrm{CO_2}$ = 8%)", fontsize=12, fontweight='bold')
    plt.xlabel(r"Temperature ($^\circ$C)", fontsize=11)
    plt.ylabel(r"Enhancement Factor $f(T, P)$", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("enhancement_factor_1D_cross_section.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(rf"Benchmarking completed. Maximum relative deviation: {ratio_pct_max_overall:.4f}%")

if __name__ == '__main__':
    run_benchmarking()