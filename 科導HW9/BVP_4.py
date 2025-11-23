"""
Consider the boundary value problem:
u'' = e^{sin(x)}, u'(0) = 0, u'(1) = \alpha.
(1)Determine \alpha such that the problem has at least one solution.
(2)Solve the problem by finding one of its solution.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# ---- composite trapezoid for whole-interval integral ----
def trapezoid_integral(fvals, h):
    # fvals: array of f(x) at equally spaced points, h spacing
    return h * (0.5 * fvals[0] + fvals[1:-1].sum() + 0.5 * fvals[-1])

# ---- cumulative trapezoid: returns integral from 0 up to each grid point ----
def cumulative_trapezoid(fvals, h):
    n = len(fvals)
    I = np.zeros(n)
    for k in range(1, n):
        I[k] = I[k-1] + 0.5 * h * (fvals[k-1] + fvals[k])
    return I

# ---- main parameters ----
N = 2000          # number of subintervals (可以改大以檢驗收斂)
x = np.linspace(0.0, 1.0, N+1)
h = x[1] - x[0]

# right-hand side f(x) = exp(sin x)
f = np.exp(np.sin(x))

# (1) compute alpha from compatibility condition
"""
\alpha = u'(1) - u'(0) = \int_{0}^{1} e^{sin(x)} dx
-> \alpha = \int_{0}^{1} e^{sin(x)} dx

u''(x) = e^{sin(x)}
-> u'(x) = \int e^{sin(x)} dx + C_1
Define A(x) = \int_{0}^{x} e^{sin(s)} ds.
Then A'(x) = e^{sin(x)} and A(0) = 0.
So u'(x) = A(x) + C_1.
-> u(x) = \int_{0}^{x} u'(t) dt + C_2 = \int_{0}^{x} (A(x) + C_1) dt + C_2
Define B(x) = \int_{0}^{x} A(t) dt = \int_{0}^{x}\int_{0}^{t} e^{sin(s)} ds dt.
Then u(x) = B(x) + C_1*x + C_2.
Substitute u'(0) = 0, then C_1 = 0.
Substitute u'(1) = \alpha, then \alpha = A(1) = \int_{0}^{1} e^{sin(s)} ds.
So u(x) = B(x) + C_2.
"""
alpha = trapezoid_integral(f, h)
print(f"Computed alpha = ∫_0^1 e^(sin x) dx ≈ {alpha:.12f}")

# (2) construct one solution with u(0)=0
"""
Choose u(0)=0. Then 0 = u(0) = B(0) + C_2 = C_2.
So u(x) = B(x) = int_{0}^{x}\int_{0}^{t} e^{sin(s)} ds dt.
"""
# u'(x) = \int_{0}^{x} f(s) ds with u'(0)=0
u_prime = cumulative_trapezoid(f, h)

# u(x) = \int_{0}^{x} u'(t) dt with u(0)=0
u = cumulative_trapezoid(u_prime, h)

# verification
u_prime_at_1 = u_prime[-1]
print(f"Numeric u'(1) = {u_prime_at_1:.12f}  (should equal alpha)")
print(f"Difference |u'(1)-alpha| = {abs(u_prime_at_1 - alpha):.3e}")

# approximate second derivative from computed u (central difference)
"""
x_j = jh, j = 0,...,N.
N: number of subintervals
Unknowns: u_0,...,u_N (N+1 unknowns)
Equations:
    j = 0: u''(0) ~= (u_2 - 2u_1 + u_0) / (h^2)
    j = 1,...,N-1: u''_j ~= (u_{j+1} -2u_{j} + u_{j-1}) / (h^2)
    j = N: u''(1) ~= (u_N - 2u_{N-1} + u_{N-2}) / (h^2)
"""
u_dd = np.zeros_like(u)
for i in range(1, len(u)-1):
    u_dd[i] = (u[i+1] - 2*u[i] + u[i-1]) / (h*h)
# endpoints (one-sided) - lower accuracy but okay for residual check
u_dd[0] = (u[2] - 2*u[1] + u[0]) / (h*h)
u_dd[-1] = (u[-1] - 2*u[-2] + u[-3]) / (h*h)

residual = u_dd - f
max_residual = np.max(np.abs(residual))
print(f"Max |u'' - f| over grid ≈ {max_residual:.3e}")

# check u'(0) ~ 0
print(f"u'(0) numeric = {u_prime[0]:.3e} (should be 0)")

# ---- plotting ----
# Plot 1: u(x)
plt.figure(figsize=(8,4))
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title("Numerical solution u(x) of u'' = exp(sin x), with u(0)=0, u'(0)=0")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: residual u'' - f
plt.figure(figsize=(8,4))
plt.plot(x, residual)
plt.xlabel('x')
plt.ylabel('u\"(x) - exp(sin x)')
plt.title("Residual (should be approx. zero)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- optional: mesh refinement test ----
def mesh_test(N):
    x = np.linspace(0,1,N+1)
    h = x[1]-x[0]
    f = np.exp(np.sin(x))
    alpha = trapezoid_integral(f, h)
    u_prime = cumulative_trapezoid(f, h)
    u = cumulative_trapezoid(u_prime, h)
    # compute u'' residual
    u_dd = np.zeros_like(u)
    for i in range(1,len(u)-1):
        u_dd[i] = (u[i+1] - 2*u[i] + u[i-1]) / (h*h)
    u_dd[0] = (u[2] - 2*u[1] + u[0]) / (h*h)
    u_dd[-1] = (u[-1] - 2*u[-2] + u[-3]) / (h*h)
    residual = u_dd - f
    return alpha, np.max(np.abs(residual))

for testN in [500, 1000, 2000, 4000]:
    a, mr = mesh_test(testN)
    print(f"N={testN:5d}  alpha≈{a:.12f}  max_residual={mr:.3e}")
