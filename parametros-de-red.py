"""
==============================================================
  PUNTO 4 — Cálculo de parámetros de red a y c
  Muestra : HAP-Mg_800_2_horas (fase β-TCP / Whitlockite)
  Sistema : Trigonal-hexagonal  |  Grupo espacial: R3c
  Radiación: Cu Kα  |  λ = 1.5406 Å
==============================================================

LÓGICA (para quien viene de JS/Go):
──────────────────────────────────
  La ecuación del sistema trigonal en notación hexagonal:

      1/d² = (4/3)·(h²+hk+k²)/a²  +  l²/c²

  Definiendo  X = 1/a²  y  Y = 1/c², queda un sistema lineal:

      1/d²  =  coef_A · X  +  coef_C · Y
       ↑            ↑               ↑
   (conocido)   (4/3)(h²+hk+k²)   l²

  Con 10 picos → 10 ecuaciones, 2 incógnitas → sistema sobredeterminado.
  Se resuelve con mínimos cuadrados: numpy.linalg.lstsq
  (equivalente al "normal equation" en regresión lineal múltiple)
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. DATOS DE ENTRADA
#    Picos indexados con QualX (tabla del punto 3)
#    Referencia: ICDD 04-010-6568  Ca₉.₅Mg(PO₄)₇  β-TCP/Whitlockite
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

# ──────────────────────────────────────────────────────────────
# 2. CONSTRUIR EL SISTEMA  A·x = b
#    A : matriz (n_picos × 2) de coeficientes
#    b : vector  (n_picos)    de valores 1/d²
#    x : [1/a², 1/c²]  — las dos incógnitas
# ──────────────────────────────────────────────────────────────

A_matrix = []
b_vector = []
rows_info = []    # guardamos todo para la impresión final

for (two_theta, h, k, l) in peaks:

    theta_rad = np.radians(two_theta / 2)           # θ en radianes
    d         = LAMBDA / (2 * np.sin(theta_rad))    # Ley de Bragg
    inv_d2    = 1.0 / d**2                          # 1/d²

    coef_A = (4 / 3) * (h**2 + h*k + k**2)         # coeficiente de 1/a²
    coef_C = float(l**2)                            # coeficiente de 1/c²

    A_matrix.append([coef_A, coef_C])
    b_vector.append(inv_d2)
    rows_info.append((two_theta, h, k, l, d, inv_d2, coef_A, coef_C))

A_matrix = np.array(A_matrix)   # shape (10, 2)
b_vector = np.array(b_vector)   # shape (10,)

# ──────────────────────────────────────────────────────────────
# 3. RESOLVER CON MÍNIMOS CUADRADOS
#    lstsq resuelve:  min ||A·x - b||²
#    Devuelve x = [1/a², 1/c²]
# ──────────────────────────────────────────────────────────────

x, _residuals, _rank, _sv = np.linalg.lstsq(A_matrix, b_vector, rcond=None)

inv_a2, inv_c2 = x
a = 1.0 / np.sqrt(inv_a2)
c = 1.0 / np.sqrt(inv_c2)

# ──────────────────────────────────────────────────────────────
# 4. CALCULAR d_calc PARA VERIFICAR EL AJUSTE
# ──────────────────────────────────────────────────────────────

d_comparisons = []
for (coef_A, coef_C), inv_d2_exp in zip(A_matrix, b_vector):
    inv_d2_calc = coef_A * inv_a2 + coef_C * inv_c2
    d_exp  = 1.0 / np.sqrt(inv_d2_exp)
    d_calc = 1.0 / np.sqrt(inv_d2_calc)
    d_comparisons.append((d_exp, d_calc))

# ──────────────────────────────────────────────────────────────
# 5. IMPRIMIR RESULTADOS
# ──────────────────────────────────────────────────────────────

SEP  = "─" * 80
SEP2 = "═" * 80

print("\n" + SEP2)
print("  PUNTO 4 — Parámetros de red | HAP-Mg_800_2h | β-TCP Whitlockite (R3c)")
print(SEP2)

# ── Tabla de entrada ──
print(f"\n{'#':>3}  {'2θ(°)':>8}  {'(h,k,l)':>13}  {'d_exp(Å)':>10}  "
      f"{'1/d²(Å⁻²)':>12}  {'coef_A':>8}  {'coef_C':>8}")
print(SEP)
for i, (t2, h, k, l, d, inv_d2, coef_A, coef_C) in enumerate(rows_info):
    hkl = f"({h:>2},{k:>2},{l:>3})"
    print(f"{i+1:>3}  {t2:>8.3f}  {hkl:>13}  {d:>10.4f}  "
          f"{inv_d2:>12.4f}  {coef_A:>8.3f}  {coef_C:>8.0f}")

# ── d_exp vs d_calc ──
print(f"\n{'#':>3}  {'d_exp(Å)':>10}  {'d_calc(Å)':>11}  {'Δd(Å)':>10}  {'Δ(%)':>8}")
print(SEP)
for i, (d_exp, d_calc) in enumerate(d_comparisons):
    delta   = d_calc - d_exp
    delta_p = (delta / d_exp) * 100
    flag    = "  ✓" if abs(delta_p) < 2.0 else " ⚠"
    print(f"{i+1:>3}  {d_exp:>10.5f}  {d_calc:>11.5f}  {delta:>+10.5f}  {delta_p:>+8.3f}%{flag}")

# ── Parámetros finales ──
V = (np.sqrt(3) / 2) * a**2 * c   # Volumen celda hexagonal

print(f"\n{SEP}")
print(f"  RESULTADOS FINALES (mínimos cuadrados, 10 picos):")
print(SEP)
print(f"  a   = {a:.5f}  Å")
print(f"  c   = {c:.5f}  Å")
print(f"  c/a = {c/a:.5f}")
print(f"  V   = {V:.3f}  Å³")

# ── Comparación con referencia β-TCP ──
# Referencia: ICDD 04-010-6568  Ca₉.₅Mg(PO₄)₇
a_ref = 10.429
c_ref = 37.380
V_ref = (np.sqrt(3) / 2) * a_ref**2 * c_ref

print(f"\n{'─'*55}")
print(f"  Comparación con β-TCP puro (ICDD 04-010-6568):")
print(f"{'─'*55}")
print(f"  {'':6}  {'Exp (Å)':>10}  {'Ref (Å)':>10}  {'Δ (Å)':>10}  {'Δ (%)':>8}")
print(f"  {'a':6}  {a:>10.5f}  {a_ref:>10.3f}  {a-a_ref:>+10.5f}  {(a-a_ref)/a_ref*100:>+8.3f}%")
print(f"  {'c':6}  {c:>10.5f}  {c_ref:>10.3f}  {c-c_ref:>+10.5f}  {(c-c_ref)/c_ref*100:>+8.3f}%")
print(f"  {'V':6}  {V:>10.3f}  {V_ref:>10.3f}  {V-V_ref:>+10.3f}  {(V-V_ref)/V_ref*100:>+8.3f}%")

print(f"\n  Interpretación:")
print(f"  La contracción de a ({(a-a_ref)/a_ref*100:+.2f}%) y c ({(c-c_ref)/c_ref*100:+.2f}%)")
print(f"  es coherente con la sustitución de Ca²⁺ (r=1.00Å) por Mg²⁺ (r=0.72Å),")
print(f"  que comprime la celda unidad al incorporarse en la estructura.")

print("\n" + SEP2 + "\n")