"""
==============================================================
  PUNTOS 4 y 5 — Parámetros de red a y c  +  error estimado
  Muestra : HAP-Mg_800_2_horas (fase β-TCP / Whitlockite)
  Sistema : Trigonal-hexagonal  |  Grupo espacial: R3c
  Radiación: Cu Kα  |  λ = 1.5406 Å
==============================================================
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. DATOS DE ENTRADA
# ──────────────────────────────────────────────────────────────

LAMBDA = 1.5406   # Å — Cu Kα

# (2θ en grados,  h,  k,  l)
peaks = [
    (17.201,  2, -1,   0),
    (26.074,  1,  0,  10),
    (27.094,  3, -1,   4),
    (27.828,  3, -1,  -2),
    (29.725,  3,  0,   0),
    (31.377,  2,  0, -10),
    (32.663,  3, -1,  -8),
    (34.764,  4, -2,   0),
    (43.066,  3,  0, -12),
    (45.378,  4, -2,  12),
]

n = len(peaks)   # número de ecuaciones
p = 2            # número de incógnitas (1/a², 1/c²)

# ──────────────────────────────────────────────────────────────
# 2. CONSTRUIR EL SISTEMA  A·x = b
# ──────────────────────────────────────────────────────────────

A_matrix = []
b_vector = []
rows_info = []

for (two_theta, h, k, l) in peaks:
    theta_rad = np.radians(two_theta / 2)
    d         = LAMBDA / (2 * np.sin(theta_rad))
    inv_d2    = 1.0 / d**2
    coef_A    = (4 / 3) * (h**2 + h*k + k**2)
    coef_C    = float(l**2)

    A_matrix.append([coef_A, coef_C])
    b_vector.append(inv_d2)
    rows_info.append((two_theta, h, k, l, d, inv_d2, coef_A, coef_C))

A_matrix = np.array(A_matrix)
b_vector = np.array(b_vector)

# ──────────────────────────────────────────────────────────────
# 3. PUNTO 4 — Mínimos cuadrados → a y c
# ──────────────────────────────────────────────────────────────

x, _, _, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
inv_a2, inv_c2 = x

a = 1.0 / np.sqrt(inv_a2)
c = 1.0 / np.sqrt(inv_c2)

# ──────────────────────────────────────────────────────────────
# 4. PUNTO 5 — Error estimado por propagación
# ──────────────────────────────────────────────────────────────

residuals  = b_vector - A_matrix @ x
sigma2     = np.dot(residuals, residuals) / (n - p)
ATA_inv    = np.linalg.inv(A_matrix.T @ A_matrix)
cov_matrix = sigma2 * ATA_inv

sigma_a = (a**3 / 2) * np.sqrt(cov_matrix[0, 0])
sigma_c = (c**3 / 2) * np.sqrt(cov_matrix[1, 1])

# ──────────────────────────────────────────────────────────────
# 5. IMPRIMIR RESULTADOS
# ──────────────────────────────────────────────────────────────

SEP  = "─" * 72
SEP2 = "═" * 72

print("\n" + SEP2)
print("  PUNTOS 4 y 5 — Parámetros de red | HAP-Mg_800_2h | β-TCP (R3c)")
print(SEP2)

# Tabla de entrada (punto 4)
print(f"\n{'#':>3}  {'2θ(°)':>8}  {'(h,k,l)':>13}  {'d_exp(Å)':>10}  "
      f"{'1/d²(Å⁻²)':>12}  {'coef_A':>8}  {'coef_C':>8}")
print(SEP)
for i, (t2, h, k, l, d, inv_d2, coef_A, coef_C) in enumerate(rows_info):
    hkl = f"({h:>2},{k:>2},{l:>3})"
    print(f"{i+1:>3}  {t2:>8.3f}  {hkl:>13}  {d:>10.4f}  "
          f"{inv_d2:>12.4f}  {coef_A:>8.3f}  {coef_C:>8.0f}")

# d_exp vs d_calc (punto 4)
print(f"\n{'#':>3}  {'d_exp(Å)':>10}  {'d_calc(Å)':>11}  {'Δd(Å)':>10}  {'Δ(%)':>8}")
print(SEP)
for i, (t2, h, k, l, d_exp, inv_d2, coef_A, coef_C) in enumerate(rows_info):
    inv_d2_calc = coef_A * inv_a2 + coef_C * inv_c2
    d_calc      = 1.0 / np.sqrt(inv_d2_calc)
    delta       = d_calc - d_exp
    delta_p     = (delta / d_exp) * 100
    flag        = "  ✓" if abs(delta_p) < 2.0 else " ⚠"
    print(f"{i+1:>3}  {d_exp:>10.5f}  {d_calc:>11.5f}  {delta:>+10.5f}  {delta_p:>+8.3f}%{flag}")

# Resultados finales puntos 4 y 5
print(f"\n{SEP}")
print(f"  PUNTO 4 — Parámetros de red (mínimos cuadrados, {n} picos):")
print(SEP)
print(f"  a  =  {a:.5f}  Å")
print(f"  c  =  {c:.5f}  Å")
print(f"  c/a = {c/a:.5f}")

print(f"\n{SEP}")
print(f"  PUNTO 5 — Valores finales con error estimado:")
print(f"  (grados de libertad = {n - p}  |  σ² = {sigma2:.2e}  Å⁻⁴)")
print(SEP)
print(f"  a  =  {a:.5f}  ±  {sigma_a:.5f}  Å")
print(f"  c  =  {c:.5f}  ±  {sigma_c:.5f}  Å")
print(f"  c/a = {c/a:.5f}  ±  {np.sqrt((sigma_a/a)**2 + (sigma_c/c)**2) * (c/a):.5f}")
print("\n" + SEP2 + "\n")