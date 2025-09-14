import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BarycentricInterpolator

# Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# 評估誤差
def max_error(f, g, xs):
    return np.max(np.abs(f(xs) - g(xs)))

#---------------------------
# 第一題: Cubic spline interpolation with uniform nodes (natural)
#---------------------------
tol = 1e-10
N_uniform = None
errors_uniform = []

# 測試點
xx = np.linspace(-1, 1, 2000)
fxx = runge(xx)

for N in range(5, 2000):
    x_nodes = np.linspace(-1, 1, N+1)
    y_nodes = runge(x_nodes)

    # natural spline
    cs = CubicSpline(x_nodes, y_nodes, bc_type="natural")
    err = max_error(runge, cs, xx)
    errors_uniform.append(err)

    if err < tol:
        N_uniform = N
        break

print(f"[Uniform natural spline] 最小 N = {N_uniform}, 誤差 = {errors_uniform[-1]:.3e}")

# 作圖 (函數 vs 插值)
plt.figure(figsize=(6,4))
plt.plot(xx, fxx, label="Runge function")
plt.plot(xx, cs(xx), "--", label=f"Spline N={N_uniform}")
plt.scatter(x_nodes, y_nodes, c="red", s=10, label="nodes")
plt.legend()
plt.title("Uniform natural spline interpolation")
plt.savefig("uniform_vs_spline.png", dpi=300)
plt.close()

# 作圖 (誤差 vs N)
plt.figure(figsize=(6,4))
plt.semilogy(range(5, 5+len(errors_uniform)), errors_uniform, marker="o")
plt.axhline(tol, color="r", linestyle="--", label="Tolerance")
plt.xlabel("N (number of intervals)")
plt.ylabel("Max error")
plt.title("Error vs N (uniform natural spline)")
plt.legend()
plt.savefig("uniform_error_vs_N.png", dpi=300)
plt.close()


#---------------------------
# 第二題: Chebyshev nodes (second kind)
#---------------------------
def chebyshev_nodes_second_kind(N):
    k = np.arange(N+1)
    x = np.cos(np.pi * k / N)
    return np.flip(x)  # 從 -1 到 1

N_cheb = None
errors_cheb = []

for N in range(5, 2000):
    x_nodes = chebyshev_nodes_second_kind(N)
    y_nodes = runge(x_nodes)

    interp = BarycentricInterpolator(x_nodes, y_nodes)
    err = max_error(runge, interp, xx)
    errors_cheb.append(err)

    if err < tol:
        N_cheb = N
        break

print(f"[Chebyshev second kind] 最小 N = {N_cheb}, 誤差 = {errors_cheb[-1]:.3e}")

# 作圖 (函數 vs 插值)
plt.figure(figsize=(6,4))
plt.plot(xx, fxx, label="Runge function")
plt.plot(xx, interp(xx), "--", label=f"Chebyshev N={N_cheb}")
plt.scatter(x_nodes, y_nodes, c="red", s=10, label="nodes")
plt.legend()
plt.title("Chebyshev interpolation (2nd kind)")
plt.savefig("chebyshev_vs_function.png", dpi=300)
plt.close()

# 作圖 (誤差 vs N)
plt.figure(figsize=(6,4))
plt.semilogy(range(5, 5+len(errors_cheb)), errors_cheb, marker="o")
plt.axhline(tol, color="r", linestyle="--", label="Tolerance")
plt.xlabel("N (number of intervals)")
plt.ylabel("Max error")
plt.title("Error vs N (Chebyshev 2nd kind)")
plt.legend()
plt.savefig("chebyshev_error_vs_N.png", dpi=300)
plt.close()
