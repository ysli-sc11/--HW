"""
Consider the linear boundary value problem:
\epsilon u'' + (1+\epsilon) u' + u = 0, u(0) = 0, u(1) = 1.
Solve the problem and check the accuracy of your solutions.
Choose \epsilon=0.01.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# 解析解（closed-form）
# -----------------------
"""
\epsilon u'' + (1+\epsilon) u' + u = 0, u(0) = 0, u(1) = 1.
-> u(x) = A*e^{-x} + B*e^{-100*x}.
Substitute u(0) = 0, then A = -B.
Substitute u(1) = 1, then A*e^{-1} + B*e^{-100} = 1.
-> A = 1 / (e^{-1} - e^{-100}), B = -A.
-> u(x) = (1 / (e^{-1} - e^{-100}))*e^{-x} + (-1 / (e^{-1} - e^{-100}))*e^{-100*x}.
"""
def exact_solution(x, eps):
    # roots computed analytically for constant-coefficient ODE
    a = eps
    b = 1.0 + eps
    c = 1.0
    disc = b*b - 4*a*c
    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)
    denom = np.exp(r1) - np.exp(r2)
    A = 1.0 / denom
    B = -A
    return A * np.exp(r1 * x) + B * np.exp(r2 * x)

# -----------------------
# 有限差分組裝
# -----------------------
def build_fd_matrix_and_rhs(N, eps):
    """
    x_j = jh, j = 0,...,N.
    N: number of subintervals (so number of nodes = N+1)
    interior indices j = 1,...,N-1  (count = N-1)
    Use central differences:
      u'' ≈ (u_{j-1} - 2 u_j + u_{j+1}) / (h^2)
      u'  ≈ (u_{j+1} - u_{j-1}) / (2h)
    The coefficient for u_{j-1}, u_j, u_{j+1} derived accordingly.
    """
    h = 1.0 / N
    # coefficients (same at every interior node because constant coefficients)
    a_coef = eps / h**2 - (1.0 + eps) / (2.0*h)   # sub-diagonal
    b_coef = -2.0 * eps / h**2 + 1.0              # main diagonal
    c_coef = eps / h**2 + (1.0 + eps) / (2.0*h)   # super-diagonal

    n_interior = N - 1
    if n_interior <= 0:
        raise ValueError("N must be >= 2")

    # build arrays for Thomas solver
    a = np.full(n_interior-1, a_coef) if n_interior-1 > 0 else np.array([])
    b = np.full(n_interior, b_coef)
    c = np.full(n_interior-1, c_coef) if n_interior-1 > 0 else np.array([])

    # RHS initially zero (homogeneous ODE); boundary u(1)=1 contributes to last eq
    rhs = np.zeros(n_interior)
    rhs[-1] -= c_coef * 1.0  # move c_coef * u_N (u_N=1) to RHS with negative sign

    return a, b, c, rhs, h

# -----------------------
# Thomas algorithm
# -----------------------
def thomas_solve(a, b, c, d):
    """
    Solve tridiagonal system with Thomas algorithm.
    a: sub-diagonal (len n-1)
    b: main diagonal (len n)
    c: super-diagonal (len n-1)
    d: RHS (len n)
    returns x of length n
    """
    n = len(b)
    # copy to avoid modifying inputs
    ac = a.copy() if len(a) > 0 else np.array([])
    bc = b.copy()
    cc = c.copy() if len(c) > 0 else np.array([])
    dc = d.copy()

    # forward elimination
    for i in range(1, n):
        if i-1 < len(ac):
            m = ac[i-1] / bc[i-1]
            bc[i] = bc[i] - m * cc[i-1]
            dc[i] = dc[i] - m * dc[i-1]

    # back substitution
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - (cc[i] * x[i+1] if i < len(cc) else 0.0)) / bc[i]
    return x

# -----------------------
# Solve BVP with FD
# -----------------------
def solve_bvp_fd(N, eps):
    a, b, c, rhs, h = build_fd_matrix_and_rhs(N, eps)
    u_interior = thomas_solve(a, b, c, rhs)
    # full solution includes boundaries u(0)=0 and u(1)=1
    u = np.concatenate(([0.0], u_interior, [1.0]))
    x = np.linspace(0.0, 1.0, N+1)
    return x, u

# -----------------------
# Error metrics
# -----------------------
def compute_error_metrics(x, u_num, eps):
    u_ex = exact_solution(x, eps)
    abs_err = np.abs(u_num - u_ex)
    max_err = np.max(abs_err)
    l2_err = np.sqrt(np.sum(abs_err**2) / len(abs_err))
    return u_ex, abs_err, max_err, l2_err

# -----------------------
# 主程序：執行一次並做收斂試驗
# -----------------------
if __name__ == "__main__":
    eps = 0.01

    # --- 單次測試 ---
    N = 200  # 建議起始值；若要解析 boundary layer（寬度 ~0.01），N 應至少 ~100 以上
    x, u_num = solve_bvp_fd(N, eps)
    u_ex, abs_err, max_err, l2_err = compute_error_metrics(x, u_num, eps)
    print(f"eps = {eps}, N = {N}, h = {1.0/N:.5e}")
    print(f"max error = {max_err:.3e}, L2 error = {l2_err:.3e}")

    # 畫圖：數值解 vs 解析解
    plt.figure(figsize=(8,4))
    plt.plot(x, u_num, marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.plot(x, u_ex, linestyle='--')
    plt.title('Numerical (FD) vs Exact')
    plt.xlabel('x'); plt.ylabel('u(x)')
    plt.grid(True)
    plt.show()

    # 畫圖：絕對誤差
    plt.figure(figsize=(8,4))
    plt.plot(x, abs_err, marker='o', linestyle='-')
    plt.title('Absolute error |u_num - u_exact|')
    plt.xlabel('x'); plt.ylabel('abs error')
    plt.grid(True)
    plt.show()

    # --- 收斂性研究（不同 N） ---
    Ns = [20, 40, 80, 160, 320, 640]
    results = []
    for Ntest in Ns:
        xt, ut = solve_bvp_fd(Ntest, eps)
        _, _, maxe, l2e = compute_error_metrics(xt, ut, eps)
        results.append({'N': Ntest, 'h': 1.0/Ntest, 'max_error': maxe, 'l2_error': l2e})

    df = pd.DataFrame(results)
    print("\nConvergence table:")
    print(df.to_string(index=False))

    # 估算收斂階 p（基於 max_error）
    print("\nEstimated orders (from successive max_error):")
    for i in range(1, len(results)):
        e1 = results[i-1]['max_error']; e2 = results[i]['max_error']
        h1 = results[i-1]['h']; h2 = results[i]['h']
        p = np.log(e2/e1) / np.log(h2/h1)
        print(f"{results[i-1]['N']} -> {results[i]['N']}: estimated order p = {p:.3f}")

    # log-log 圖（max error vs h）
    hs = df['h'].values
    maxes = df['max_error'].values
    plt.figure(figsize=(6,4))
    plt.loglog(hs, maxes, marker='o', linestyle='-')
    plt.gca().invert_xaxis()
    plt.title('Convergence: max error vs h (log-log)')
    plt.xlabel('h')
    plt.ylabel('max error')
    plt.grid(True, which='both', ls='--')
    plt.show()
