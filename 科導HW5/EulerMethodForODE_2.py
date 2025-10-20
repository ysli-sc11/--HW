# ===============================================================
#  Forward Euler Method for Logistic Equation
#  y' = y(1 - y),   y(0) = y0
#  Goal: Observe how the choice of step size h affects
#        the qualitative correctness of the numerical solution.
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt

# ====== Define problem ======
def f(y):
    """Derivative function f(y) = y(1 - y) for logistic growth."""
    return y * (1 - y)

# Initial condition and simulation settings
y0 = 0.1          # Initial value (between 0 and 1)
t_end = 10        # Simulation time
hs = [0.1, 0.5, 1.0, 1.5, 2.5]  # Different step sizes to test

# ====== Forward Euler implementation ======
def forward_euler(h):
    """
    Perform Forward Euler method with step size h.
    Returns time array t and numerical solution y.
    """
    t = np.arange(0, t_end + h, h)  # Time steps from 0 to t_end
    y = np.zeros_like(t)            # Initialize y array
    y[0] = y0                       # Set initial condition
    for n in range(len(t)-1):       # Forward Euler iteration
        y[n+1] = y[n] + h * f(y[n])
    return t, y

# ====== Exact (analytical) solution ======
# Logistic equation exact form: y(t) = y0 e^t / (1 + y0 (e^t - 1))
t_exact = np.linspace(0, t_end, 500)
y_exact = y0 * np.exp(t_exact) / (1 + y0 * (np.exp(t_exact) - 1))

# ===============================================================
#  Figure 1: Numerical Solutions for Different Step Sizes
# ===============================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for h in hs:
    t, y = forward_euler(h)
    plt.plot(t, y, 'o-', label=f'h={h}')
# Plot the exact (true) solution for reference
plt.plot(t_exact, y_exact, 'k--', linewidth=2, label='Exact solution')

plt.title("Forward Euler for y' = y(1 - y)")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axhline(1, color='gray', linestyle=':', label='Equilibrium y=1')
plt.text(7, 1.05, "Stable equilibrium (y*=1)", fontsize=9, color='gray')

# ===============================================================
#  Figure 2: Error Plot |y - 1| for Different Step Sizes
# ===============================================================
plt.subplot(1,2,2)
for h in hs:
    t, y = forward_euler(h)
    error = np.abs(y - 1)   # Absolute error from equilibrium y*=1
    plt.semilogy(t, error, 'o-', label=f'h={h}')  # semilog-y to emphasize oscillations

plt.title("Error |y - 1| vs t")
plt.xlabel('t')
plt.ylabel('|y - 1| (log scale)')
plt.legend()
plt.grid(True, which='both')

plt.tight_layout()
plt.show()

# ===============================================================
#  Stability and Qualitative Correctness Explanation
# ===============================================================
print("Theoretical stability condition near y* = 1:")
print("  Linearization gives η_{n+1} = (1 + h f'(1)) η_n = (1 - h) η_n")
print("  For stability (no divergence or oscillation): |1 - h| < 1")
print("  ⇒  0 < h < 2  ensures qualitatively correct behavior.\n")

print("Interpretation:")
print("  0 < h < 1   → smooth convergence to y=1 (qualitatively correct).")
print("  1 < h < 2   → oscillatory convergence (still qualitatively correct).")
print("  h > 2       → divergence, solution oscillates wildly or blows up.")
