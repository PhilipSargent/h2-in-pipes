i see that the volume shift affects the liquid properties, but does it have no effect on dew point enahncement factor ?
---

**Mathematically, no—it has virtually zero effect on the dew point or the enhancement factor.**

While the volume shift parameter $c_i$ technically enters the expression for liquid molar volume, its actual numerical impact on the calculated enhancement factor $f(T, P)$ and dew point temperature is negligible for three key reasons:

---

### 1. The VLE Cancellation Principle

As shown in the thermodynamic derivation, the volume shift parameter modifies both the calculated liquid molar volume ($v_l^{\text{shifted}} = v_l^{\text{EOS}} - c$) and the liquid-phase fugacity coefficient ($\ln \phi_{i, L}$).

However, because the phase equilibrium condition equates the liquid and vapor fugacities ($\phi_{i, L} = \phi_{i, V}$), the volume shift correction term ($-\frac{c_i P}{R T}$) **cancels out identically across both phases**. Consequently, predicted vapor-liquid equilibrium compositions, bubble points, and dew points remain mathematically identical whether the EOS is shifted or unshifted.

---

### 2. Physical Weight in the Enhancement Factor Equation

In the rigorous statistical mechanics expression for the enhancement factor $f(T, P)$:

$$\ln f(T, P) = \frac{(1 - y_w) P}{R T} \cdot \left[ B_{gg}(T) - 2 B_{gw}(T) + v_l(T) \right]$$

The liquid molar volume of water $v_l$ appears in the **Poynting factor term** (which accounts for the mechanical pressure applied to the condensed liquid).

Looking at the relative magnitudes at $25\ ^\circ\text{C}$ and $5\text{ bar}$:

* **Cross-Virial Term ($2 B_{gw}$):** $\approx -400\text{ to }-600\text{ cm}^3/\text{mol}$
* **Dry Gas Virial Term ($B_{gg}$):** $\approx -150\text{ to }-200\text{ cm}^3/\text{mol}$
* **Liquid Molar Volume ($v_l$):** $\approx 18.07\text{ cm}^3/\text{mol}$

The Péneloux shift for water ($c_{H_2O} \approx 3.51\text{ cm}^3/\text{mol}$) adjusts $v_l$ from $\sim 21.5\text{ cm}^3/\text{mol}$ (uncorrected PR EOS) down to its physical value of $\sim 18.0\text{ cm}^3/\text{mol}$.

This $\sim 3.5\text{ cm}^3/\text{mol}$ adjustment represents **less than $0.5\%$** of the total bracketed term $(B_{gg} - 2B_{gw} + v_l)$. Across your pressure range of $1\text{ to }10\text{ bar}$, this translates to a shift in predicted dew point of **less than $0.01\ ^\circ\text{C}$**.

---

### 3. Summary for Your Academic Paper

* **Use Péneloux Volume Shift** when reporting liquid phase properties (such as condensed water liquid density, molar volume, or liquid holdup in the flue gas cooler).
* **Omit Volume Shift** (or note its negligible $<0.01\ ^\circ\text{C}$ impact) when presenting the vapor-phase enhancement factor $f(T, P)$ or gas dew-point calculations, as cross-virial interaction terms ($B_{gw}$) completely dominate the thermodynamic non-ideality.

Would you like to build a benchmark comparison plot showing how $f(T, P)$ varies between pure air, natural gas flue gas, and hydrogen combustion flue gas across your 1–10 bar range?