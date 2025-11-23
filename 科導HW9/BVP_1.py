"""
Consider the boundary value problem:
u'' = f, u(0) = 0, u(1) = 0,
where f(x)= 1 if 0.4 <= x<= 0.6,
            0 otherwise.
(1)Find the exact solution of this problem.
(2)Solve the problem using finite difference method and
   check the accuracy of your solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 參數 ---
a = 0.4
b = 0.6

"""
(i)    If 0 <= x <= 0.4, then u'' = 0.
       -> u_1(x) = A_1*x + B_1.
       -> u_1'(x) = A_1.
(ii)   If 0.4 <= x <= 0.6, then u'' = 1.
       -> u_2(x) = (x^2) / 2 + A_2*x + B_2.
       -> u_2'(x) = x + A_2.
(iii)  If 0.6 <= x <= 1, then u'' = 0.
       -> u_3(x) = A_3*x + B_3.
       -> u_3'(x) = A_3.
 
Substitute u(0) = 0, then B_1 = 0.
By continuity,
0.4*A_1 = u_1(0.4) = u_2(0.4) = (0.08) + 0.4*A_2 + B_2.
0.6*A_3 + B_3 = u_3(0.6) = u_2(0.6) = 0.18 + 0.6*A_2 + B_2.
A_1 = u_1'(0.4) = u_2'(0.4) = 0.4 + A_2.
A_3 = u_3'(0.6) = u_2'(0.6) = 0.6 + A_2.
Substitute u(1) = 0, then A_3 + B_3 = 0.
So A_1 = -0.1, A_2 = -0.5, A_3 = 0.1, B_1 = 0, B_2 = 0.08, B_3 = -0.1.
Thus u(x) = -0.1*x if 0 <= x <= 0.4,
            (x^2) / 2 - 0.5*x + 0.08 if 0.4 <= x <= 0.6,
            0.1*x - 0.1 if 0.6 <= x <= 1.
"""

# --- 解析解常數（由上面的解析推導得到） ---
A1 = -0.1
B1 = 0.0
A2 = -0.5
B2 = 0.08
A3 = 0.1
B3 = -0.1

def exact_u(x):
    xs = np.array(x, copy=False)
    scalar = False
    if xs.shape == ():
        xs = xs[np.newaxis]
        scalar = True
    u = np.empty_like(xs, dtype=float)
    for i, xv in enumerate(xs):
        if xv <= a:
            u[i] = A1 * xv + B1
        elif xv <= b:
            u[i] = 0.5 * xv**2 + A2 * xv + B2
        else:
            u[i] = A3 * xv + B3
    return u[0] if scalar else u

# --- Thomas 三對角解法 ---
def thomas_solver(rhs):
    """
    rhs: right hand side.
    Note that u''(x_j) ~= (u_{j-1} - 2u_j + u{j+1}) / (h^2)
    解系統 A u = rhs，其中 A 為 n x n 三對角：
    主對角 = -2, 上 / 下 對角 = 1.
    u = [u_1,...u_{N-1}]
    rhs = [f_1,...,f_{N-1}] 長度為 n。
    返回解向量 length n.
    """
    n = len(rhs)
    if n == 0:
        return np.array([], dtype=float)
    a_diag = 1.0  # 下對角 (a_1..a_{n-1})
    b_diag = -2.0 # 主對角
    c_diag = 1.0  # 上對角
    # copy rhs
    d = rhs.astype(float).copy()
    # modified coefficients
    cp = np.zeros(n-1, dtype=float)
    dp = np.zeros(n, dtype=float)
    cp[0] = c_diag / b_diag
    dp[0] = d[0] / b_diag
    for i in range(1, n-1):
        denom = b_diag - a_diag * cp[i-1]
        cp[i] = c_diag / denom
        dp[i] = (d[i] - a_diag * dp[i-1]) / denom
    if n >= 2:
        denom_last = b_diag - a_diag * cp[n-2]
        dp[n-1] = (d[n-1] - a_diag * dp[n-2]) / denom_last
    # back substitution
    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def solve_fd(N):
    """
    fd: finite difference.
    x_j = jh, j = 0,...,N.
    用 N 個 interval (h = 1/N)，做二階中心差分。
    回傳格點 x (長度 N+1) 與數值解 u (長度 N+1)，包含端點 (0,1)。
    """
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N+1)
    # 給定 f(x)
    f = np.zeros_like(x)
    f[(x >= a) & (x <= b)] = 1.0
    # interior RHS = h^2 * f_i for i=1..N-1
    rhs = (h**2) * f[1:-1]
    u = np.zeros(N+1, dtype=float)
    if len(rhs) > 0:
        u_interior = thomas_solver(rhs)
        u[1:-1] = u_interior
    # u[0]=u[N]=0 (Dirichlet)
    return x, u

# --- 收斂性檢驗與誤差計算 ---
def compute_errors(N_list):
    rows = []
    for N in N_list:
        x, u_num = solve_fd(N)
        u_ex = exact_u(x)
        err = u_num - u_ex
        L_inf = np.max(np.abs(err))
        L2 = np.sqrt(np.sum(err**2) * (1.0 / len(x)))  # discrete L2 (normalized)
        rows.append({"N":N, "h":1.0/N, "L_inf":L_inf, "L2":L2})
    return pd.DataFrame(rows)

N_list = [10, 20, 40, 80, 160, 320]
df = compute_errors(N_list)
print(df)

# 估計收斂階（用 max norm 做 log-log 擬合）
h = df["h"].values
E = df["L_inf"].values
p = np.polyfit(np.log(h), np.log(E), 1)
order_est = -p[0]
print("估計收斂階 (max norm): {:.3f}".format(order_est))

# --- 繪圖 ---
# 用最細網格畫解析解與數值解比較，以及誤差
N_plot = N_list[-1]
x, u_num = solve_fd(N_plot)
u_ex = exact_u(x)
err = u_num - u_ex

plt.figure(figsize=(8,4))
plt.plot(x, u_ex, label="exact")
plt.plot(x, u_num, label="fd numeric", linestyle='--')
plt.title(f"Exact vs FD (N={N_plot})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x, err)
plt.title(f"Pointwise error (numeric - exact), N={N_plot}")
plt.xlabel("x")
plt.ylabel("error")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.loglog(h, E, marker='o')
plt.gca().invert_xaxis()
plt.title("Convergence (max norm)")
plt.xlabel("h")
plt.ylabel("max error")
plt.grid(True, which="both", ls=":")
plt.show()
