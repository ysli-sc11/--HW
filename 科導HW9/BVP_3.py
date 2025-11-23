"""
Consider the boundary value problem:
u''(x) = sin(2\pi x), u'(0)=0, u'(1)=0
(1) Show that the consistency condition is satisfied so that
    the solution of the problem exists.
(2) Develop a 2nd-order finite difference method.
(3) Solve the problem and check the accuracy of your solutions.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi
from numpy.linalg import solve

def f(x):
    """右端項 f(x)=sin(2πx)"""
    return np.sin(2*pi*x)

def analytic_u(x):
    """解析解（平均值為 0 的唯一化）"""
    return -1.0/(4*pi**2)*np.sin(2*pi*x) + (1.0/(2*pi))*x - 1.0/(4*pi)

# --- (1) 一致性條件檢查 ---
"""
Since u'(1)-u'(0) = 0 = \int_{0}^{1} sin(2\pi x) dx,
the consistency condition is satisfied.
And so the solution of the problem exists.
"""
x_fine = np.linspace(0,1,10001)
I = np.trapezoid(f(x_fine), x_fine)
print(f"一致性檢查 ∫_0^1 f(x) dx ≈ {I:.3e} (理論上應為 0)")

# --- (2) 建矩陣 (包含 Neumann 邊界 ghost-elimination) ---
def build_matrix_and_rhs_with_mean0(N):
    """
    x_j = jh, j = 0,...,N.
    N: number of subintervals
    Unknowns: u_0,...,u_N (N+1 unknowns)
    Equations:
      j = 0:  2*(u1 - u0)/h^2 = f(x0)
            (from ghost elimination:
             0 = u'(0) ~= (u_1 - u_{-1}) / (2h) -> u_{-1} = u_1.
             f(x_0) = u''(0) ~= (u_1 - 2u_0 + u_{-1}) / (h^2)
                    = 2(u_1 - u_0) / (h^2))
      j = 1,...,N-1: (u_{j-1} - 2u_j + u_{j+1}) / (h^2) = f(x_j)
      j = N:  2*(u_{N-1} - u_N)/h^2 = f(xN)
            (from ghost elimination: similar to j = 0)
    """
    h = 1.0 / N
    x = np.linspace(0,1,N+1)
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)

    # j=0 (Neumann)
    A[0,0] = -2.0/(h*h)
    A[0,1] = 2.0/(h*h)
    b[0] = f(x[0])

    # interior nodes
    for j in range(1, N):
        A[j, j-1] = 1.0/(h*h)
        A[j, j]   = -2.0/(h*h)
        A[j, j+1] = 1.0/(h*h)
        b[j] = f(x[j])

    # j = N (Neumann)
    A[N, N]   = -2.0/(h*h)
    A[N, N-1] = 2.0/(h*h)
    b[N] = f(x[N])

    # replace last row by mean=0 constraint: h * sum_j u_j = 0
    A[N, :] = h
    b[N] = 0.0

    return A, b, x

def solve_fd_mean0(N):
    A, b, x = build_matrix_and_rhs_with_mean0(N)
    u = solve(A, b)   # solve linear system
    return x, u

# --- (3) 求解、誤差與收斂分析 ---
"""
u(x) = - (sin(2\pi x)) / (4\pi ^2) + (x) / (2\pi) + C
Substitute \int_{0}^{1} u(x) dx = 0, then C = -1/(4\pi).
So u(x) = - (sin(2\pi x)) / (4\pi ^2) + (x) / (2\pi) - 1/(4\pi).
"""
Ns = [10, 20, 40, 80, 160]   # 可再加更細的網格
hs = []
errors = []

for N in Ns:
    x, u_h = solve_fd_mean0(N)
    u_ex = analytic_u(x)
    # 我們已在系統中以平均為0做唯一化，解析解也是 average=0，可以直接比較
    err_max = np.max(np.abs(u_h - u_ex))
    hs.append(1.0/N)
    errors.append(err_max)
    print(f"N={N:4d}, h={1.0/N:.5f}, max error = {err_max:.3e}")

# 計算相鄰網格的收斂階（log-log slope）
orders = []
for k in range(1, len(Ns)):
    rate = np.log(errors[k-1]/errors[k]) / np.log(hs[k-1]/hs[k])
    orders.append(rate)
print("相鄰網格估計收斂階（應接近 2）:", ["{:.4f}".format(r) for r in orders])

# 畫圖：數值解 vs 解析解（在最細網格）
Nplot = Ns[-1]
x, u_h = solve_fd_mean0(Nplot)
u_ex = analytic_u(x)
plt.figure(figsize=(8,4))
plt.plot(x, u_ex, label='analytic')
plt.plot(x, u_h, '--', label=f'FD N={Nplot}')
plt.xlabel('x'); plt.ylabel('u(x)')
plt.title('Finite Difference vs Analytic Solution (mean=0)')
plt.legend(); plt.grid(True)
plt.show()

# 畫收斂圖（log-log）
plt.figure(figsize=(6,4))
plt.loglog(hs, errors, marker='o')
plt.xlabel('h'); plt.ylabel('max error (infinity norm)')
plt.title('Convergence plot (expected slope ≈ 2)')
plt.grid(True); plt.show()
