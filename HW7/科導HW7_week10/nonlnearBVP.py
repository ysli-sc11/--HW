import numpy as np
import matplotlib.pyplot as plt

def solve_nonlinear_bvp(n, tol=1e-8, max_iter=20):
    """
    Solve -u'' + sin(u) = 0, u(0)=1, u(1)=1 using finite difference + Newton's method.
    n: number of intervals (so we have n-1 internal unknowns)
    """
    h = 1.0 / n
    x = np.linspace(0, 1, n + 1)

    # initial guess (linear interpolation)
    u = np.linspace(1, 1, n + 1)

    # Apply boundary conditions
    u[0], u[-1] = 1.0, 1.0

    for k in range(max_iter):
        # Construct residual F(u)
        F = np.zeros(n - 1)
        for i in range(1, n):
            if i == n:  # skip last boundary
                continue
            F[i - 1] = -(u[i + 1] - 2 * u[i] + u[i - 1]) / h**2 + np.sin(u[i])

        # Construct Jacobian matrix J(u)
        J = np.zeros((n - 1, n - 1))
        for i in range(n - 1):
            J[i, i] = 2 / h**2 + np.cos(u[i + 1])
            if i > 0:
                J[i, i - 1] = -1 / h**2
            if i < n - 2:
                J[i, i + 1] = -1 / h**2

        # Solve for correction: J * delta = -F
        delta = np.linalg.solve(J, -F)

        # Update u (internal points)
        u[1:-1] += delta

        # Check convergence
        if np.linalg.norm(delta, np.inf) < tol:
            print(f"Converged in {k+1} iterations for n={n}")
            break
    else:
        print("Warning: Newton did not converge.")

    return x, u


def convergence_test():
    ns = [10, 20, 40, 80]
    solutions = []

    for n in ns:
        x, u = solve_nonlinear_bvp(n)
        solutions.append((x, u))

    # Compare successive grid solutions (interpolate coarse to fine grid)
    errors = []
    for i in range(len(ns) - 1):
        x_coarse, u_coarse = solutions[i]
        x_fine, u_fine = solutions[i + 1]

        # interpolate coarse solution onto fine grid
        u_interp = np.interp(x_fine, x_coarse, u_coarse)
        err = np.max(np.abs(u_fine - u_interp))
        errors.append(err)
        print(f"Error between n={ns[i]} and n={ns[i+1]}: {err:.3e}")

    # Plot solution for largest n
    plt.figure(figsize=(6, 4))
    plt.plot(solutions[-1][0], solutions[-1][1], 'b-', label=f'n={ns[-1]}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution of -u"" + sin(u)=0, u(0)=u(1)=1')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot error decay
    plt.figure(figsize=(6, 4))
    plt.loglog(ns[1:], errors, 'o-')
    plt.xlabel('Grid size n')
    plt.ylabel('Max error between consecutive grids')
    plt.title('Convergence test (self-consistency)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    convergence_test()
