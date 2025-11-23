"""
Consider the boundary value problem:
u'' - 2u' + u = 1, \quad u(0)=0, \quad u'(1)=1.
(1)Show that the solution is unique by considering the homogeneous problem.
(2)Develop a 2nd-order finite difference method.
(3)Solve the problem and check the accuracy of your solutions.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- 證明唯一解 ----------
"""
Suppose u_1, u_2 are solutions.
Then u_1'' - 2u_1' + u_1 = 1, u_2'' - 2u_2' + u_2 = 1,
u_1(0) = 0, u_1'(1) = 1, u_2(0) = 0, u_2'(1) = 1.
Let u = u_1 - u_2. Then u'' - 2u' + u = 0, u(0) = 0, u'(1) = 0.
So u(x) = (c_1​ + c_2​x)*e^x. Substitute u(0)=0, u'(1) = 0,
then c_1 = 0, c_2 = 0. So 0 = u = u_1 - u_2. Thus u_1 = u_2.

"""

# ---------- 解析解函數 ----------
def exact_solution(x):
    # 解析解：u(x) = (A + B x) e^x + 1, A=-1, B=(1 + 1/e)/2
    e = np.e
    B = (1 + 1/e) / 2.0
    A = -1.0
    return (A + B * x) * np.exp(x) + 1.0

# ---------- 差分求解器 ----------
def solve_fd(N):
    """
    fd: finite difference.
    設網格x_i = ih, i = 0...N.
    使用 N 等分 (h=1/N)，未知為 u_1...u_N（u_0 = 0 已知）。
    回傳 x 節點 (0...1) 與 u 數值 (長度 N+1 包含 u_0 = 0)
    """
    h = 1.0 / N
    M = N  # unknowns: u_1..u_N
    A = np.zeros((M, M))
    b = np.zeros(M)

    """
    For 1 <= i <= N-1
    u_i'' ~= (u_{i-1} - 2u_i + u_{i-1}) / (h^2)
    u_i' ~= (u_{i+1} - u_{i-1}) / (2h)
    """
    # interior coefficients for i = 1..N-1
    coef_im1 = 1.0 / h**2 + 1.0 / h
    coef_i   = -2.0 / h**2 + 1.0
    coef_ip1 = 1.0 / h**2 - 1.0 / h

    # fill interior rows
    for i in range(1, N):   # i=1..N-1
        k = i - 1           # maps u_i -> index k (u_1 -> k=0)
        if k - 1 >= 0:
            A[k, k-1] = coef_im1
        # if k-1 < 0 then u_0 is known (u_0=0), contribution goes to RHS but it's zero here
        A[k, k]   = coef_i
        A[k, k+1] = coef_ip1
        b[k] += 1.0   # RHS f=1

    # Neumann BC(boundary condition) at x=1 (2nd-order backward):
    # u'(1) ~= (3 u_N - 4 u_{N-1} + u_{N-2})/(2h) = 1
    k = M - 1  # last unknown index (u_N)
    if N >= 2:
        A[k, k]     = 3.0 / (2.0 * h)      # coefficient for u_N
        A[k, k-1]   = -4.0 / (2.0 * h)     # coefficient for u_{N-1}
        A[k, k-2]   = 1.0 / (2.0 * h)      # coefficient for u_{N-2}
        b[k] = 1.0
    else:
        # N = 1 的特殊情況（通常不使用）
        A[k, k] = 1.0
        b[k] = h * 1.0 + 0.0

    # solve linear system for u_1..u_N
    u_inner = np.linalg.solve(A, b)

    u = np.zeros(N+1)
    u[0] = 0.0
    u[1:] = u_inner
    x = np.linspace(0, 1, N+1)
    return x, u

# ---------- 主流程：收斂實驗與繪圖 ----------
if __name__ == "__main__":
    print("唯一性（簡短）：齊次解 (A + B x) e^x，u(0)=0 => A=0，u'(1)=0 => B=0，故唯一解。")

    Ns = [20, 40, 80, 160]  # 可以自行增加到 320, 640 看收斂
    errors_max = []

    for N in Ns:
        x, u_num = solve_fd(N)
        u_ex = exact_solution(x)
        err = np.abs(u_num - u_ex)
        errors_max.append(np.max(err))
        print(f"N={N:4d}, h={1/N:.5e}, max error = {errors_max[-1]:.5e}")

    # empirical convergence order (max-norm)
    for i in range(1, len(Ns)):
        rate = np.log(errors_max[i-1] / errors_max[i]) / np.log(2.0)
        print(f"empirical order between N={Ns[i-1]} and N={Ns[i]}: {rate:.3f}")

    # draw solution and error for finest grid
    Nplot = Ns[-1]
    x, u_num = solve_fd(Nplot)
    u_ex = exact_solution(x)
    err = u_num - u_ex

    plt.figure()
    plt.plot(x, u_num, label='numerical (FD)')
    plt.plot(x, u_ex, label='exact', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Solution, N={Nplot}')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(x, err)
    plt.xlabel('x')
    plt.ylabel('numerical - exact')
    plt.title(f'Error (N={Nplot}), max error={np.max(np.abs(err)):.2e}')
    plt.grid(True)
    plt.show()

    print("\n說明：若是二階格式，當 h-> h/2 時，誤差應該約縮小 2^2=4 倍（empirical order 約為 2）。")
