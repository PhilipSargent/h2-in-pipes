The cross-virial coefficient $B_{ij}(T)$ and the binary interaction parameter $k_{ij}$ (described in resources like Whitson's EOS guide) reflect the exact same underlying physics: **they measure the deviation of cross-molecular interactions (molecule $i$ with molecule $j$) from simple geometric or arithmetic averages of the pure components.**

However, they are applied at different levels of thermodynamic modeling.

---

### 1. Thermodynamic Context: Virial EOS vs. Cubic EOS

* **Virial Expansion (Low to Moderate Pressure, $\le 10\text{ bar}$):**
The second virial coefficient $B_{\text{mix}}$ accounts for pairwise molecular collisions:

$$B_{\text{mix}}(T) = \sum_{i} \sum_{j} y_i y_j B_{ij}(T) = \sum_{i} y_i^2 B_{ii}(T) + 2 \sum_{i \neq j} y_i y_j B_{ij}(T)$$



Here, $B_{ij}$ is the **cross-second virial coefficient**, which directly quantifies $i\text{--}j$ interactions.
* **Cubic Equations of State (PR, SRK, etc.):**
Cubic EOS models group attractive forces into a bulk parameter $a_{\text{mix}}$ using van der Waals mixing rules:

$$a_{\text{mix}} = \sum_{i} \sum_{j} y_i y_j a_{ij}, \quad \text{where } a_{ij} = \sqrt{a_i a_j}(1 - k_{ij})$$



Here, $k_{ij}$ is the empirical **binary interaction parameter**.

---

### 2. How $B_{ij}$ and $k_{ij}$ Are Explicitly Linked

The cross-virial coefficient $B_{ij}$ can be computed directly using pseudo-critical parameters ($T_{c,ij}, P_{c,ij}, V_{c,ij}, \omega_{ij}$) via standard correlations like **Tsonopoulos (1974)** or **Hayden-O'Connell (HOC)**.

The binary interaction parameter $k_{ij}$ enters the cross-virial calculation through the **cross-critical temperature** $T_{c,ij}$:

$$T_{c,ij} = \sqrt{T_{c,i} T_{c,j}} \, (1 - k_{ij})$$

#### Corresponding Mixing/Combining Rules for Virial Terms:

1. **Cross Critical Volume ($V_{c,ij}$):**

$$V_{c,ij} = \left( \frac{V_{c,i}^{1/3} + V_{c,j}^{1/3}}{2} \right)^3$$


2. **Cross Acentric Factor ($\omega_{ij}$):**

$$\omega_{ij} = \frac{\omega_i + \omega_j}{2}$$


3. **Cross Critical Pressure ($P_{c,ij}$):**

$$P_{c,ij} = \frac{4 Z_{c,ij} R T_{c,ij}}{V_{c,ij}}, \quad \text{where } Z_{c,ij} = \frac{Z_{c,i} + Z_{c,j}}{2}$$



Once $T_{c,ij}$ and $P_{c,ij}$ are corrected using $k_{ij}$, $B_{ij}(T)$ is evaluated using reduced temperature correlations:

$$B_{ij}(T) = \frac{R T_{c,ij}}{P_{c,ij}} \left( f^{(0)}\left(T_{r,ij}\right) + \omega_{ij} f^{(1)}\left(T_{r,ij}\right) \right)$$

where $T_{r,ij} = T / T_{c,ij}$.

---

### 3. Key Differences to Keep in Mind for Your Paper

| Feature | Cross-Virial Coefficient $B_{ij}(T)$ | Binary Interaction Parameter $k_{ij}$ |
| --- | --- | --- |
| **Physical Meaning** | Actual thermodynamic property (units: $\text{cm}^3/\text{mol}$ or $\text{m}^3/\text{mol}$) representing two-body potential energy integrals. | Dimensionless fitting factor ($0.0 \text{ to } 0.5$) correcting the geometric mean rule for $T_{c,ij}$ or $a_{ij}$. |
| **Temperature Dependence** | Strongly temperature-dependent ($B_{ij}(T)$ changes sign from negative at low $T$ to positive at high $T$). | Often treated as a constant, though advanced correlations (like Coutinho et al.) make $k_{ij}(T)$ temperature-dependent. |
| **Physical Role in Enhancement Factor $f(T, P)$** | Directly enters the low-pressure expansion equation for $f(T, P)$: $\ln f \approx \frac{(1-y_w)P}{RT}\left(B_{gg} - 2 B_{gw} + v_l\right)$. | Indirectly modifies $B_{gw}$ by scaling the cross-critical temperature $T_{c,gw}$. |

---

### 4. Application to Water–Gas Systems ($1\text{ to }10\text{ bar}$)

In systems with polar or hydrogen-bonding components (such as water vapor interacting with $N_2, CO_2$, or $O_2$), simple geometric means ($k_{ij} = 0$) fail significantly:

* **Non-polar/Non-polar ($N_2 - O_2$):** $k_{ij} \approx 0.01$ to $0.03$ (geometric mean holds well).
* **Water – Nitrogen ($H_2O - N_2$):** $k_{ij} \approx 0.25 - 0.35$.
* **Water – Carbon Dioxide ($H_2O - CO_2$):** $k_{ij} \approx 0.18 - 0.22$.

If you use the **Tsonopoulos** or **Hayden-O'Connell (HOC)** model in your paper to compute $B_{gw}(T)$, published $k_{ij}$ values specifically fitted for virial cross-coefficients should be used to establish $T_{c,gw}$.

Would you like the explicit mathematical derivation showing how $B_{gw}(T)$ directly converts into the enhancement factor $f(T, P)$, or sample Python code demonstrating the Tsonopoulos mixing rule for a water–carrier gas mixture?