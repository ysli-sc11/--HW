import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# (1) ∫₀^∞ 1/(1+25x²) dx  —  variable substitution x = t/(1−t)
# ==========================================================
true_val1 = np.pi / 10  # analytic value ≈ 0.3141592653589793

def f1_transformed(t):
    """After substitution x = t/(1−t), dx = dt/(1−t)²"""
    return 1 / ((1 - t)**2 + 25 * t**2)

def composite_trapezoidal(f, a, b, n):
    """Composite trapezoidal rule"""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

# --- Convergence test for integral (1) ---
n_list1, error_list1 = [], []
target_error = 1e-10
a, b = 0.0, 1.0 - 1e-12  # avoid t=1 exactly

start_time = time.time()
for n in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840]:
    approx = composite_trapezoidal(f1_transformed, a, b, n)
    err = abs(approx - true_val1)
    n_list1.append(n)
    error_list1.append(err)
    if err < target_error:
        break
end_time = time.time()

print("(1) Integral over [0, ∞):")
print(f"  Approximation = {approx:.12f}")
print(f"  True value    = {true_val1:.12f}")
print(f"  Error (n={n}) = {err:.2e}")
print(f"  Computation time = {end_time - start_time:.3f} s\n")

plt.figure(figsize=(6,4))
plt.loglog(n_list1, error_list1, 'o-', label='Error vs n')
plt.axhline(1e-10, color='r', linestyle='--', label='Target = 1e-10')
plt.xlabel('n (number of subintervals)')
plt.ylabel('Absolute Error')
plt.title('Error Convergence: ∫₀^∞ 1/(1+25x²) dx')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()


# ==========================================================
# (2) ∫₀¹ ln(x)/(1+25x²) dx — logarithmic singularity
# ==========================================================
def f2(x):
    return np.log(x) / (1 + 25 * x**2)

def singular_part(eps):
    """Analytic correction for x∈[0,ε]"""
    term1 = eps * (np.log(eps) - 1)
    term2 = -25 * (eps**3 / 3) * (np.log(eps) - 1/3)
    return term1 + term2

def integral_with_singularity(f, eps, n):
    """Integrate f(x) from [ε,1] + analytic correction"""
    x = np.linspace(eps, 1, n + 1)
    y = f(x)
    h = (1 - eps) / n
    trap = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return trap + singular_part(eps)

eps = 1e-6
n_list2, error_list2, approx_list = [], [], []

start_time = time.time()
for n in [100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 160000]:
    approx = integral_with_singularity(f2, eps, n)
    approx_list.append(approx)
    n_list2.append(n)
    if len(approx_list) > 1:
        err = abs(approx_list[-1] - approx_list[-2])
    else:
        err = np.nan
    error_list2.append(err)
    if not np.isnan(err) and err < 1e-10:
        break
end_time = time.time()

I2_final = approx_list[-1]

print("(2) Integral with logarithmic singularity:")
print(f"  Approximation = {I2_final:.12f}")
print(f"  Estimated error < 1e-10 (n={n}, ε={eps})")
print(f"  Computation time = {end_time - start_time:.3f} s\n")

plt.figure(figsize=(6,4))
plt.loglog(n_list2[1:], error_list2[1:], 'o-', color='purple', label='|Iₙ - Iₙ₋₁|')
plt.axhline(1e-10, color='r', linestyle='--', label='Target = 1e-10')
plt.xlabel('n (number of subintervals)')
plt.ylabel('Estimated Error')
plt.title('Error Convergence: ∫₀¹ ln(x)/(1+25x²) dx')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()
