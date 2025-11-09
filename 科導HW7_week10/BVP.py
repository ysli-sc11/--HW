import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(n):
    """
    Solve -u'' = e^{sin(x)}, u(0)=u(1)=0
    using central finite difference with n intervals.
    """
    h = 1.0 / n
    x = np.linspace(0, 1, n + 1)
    f = np.exp(np.sin(x[1:-1]))  # interior points only

    # Construct tridiagonal matrix A
    main_diag = 2 * np.ones(n - 1)
    off_diag = -1 * np.ones(n - 2)
    A = (1 / h**2) * (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))

    # Solve A u = f
    u_inner = np.linalg.solve(A, f)
    u = np.zeros_like(x)
    u[1:-1] = u_inner

    return x, u

def estimate_error():
    ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    errors = []

    prev_x, prev_u = solve_bvp(ns[0])
    for n in ns[1:]:
        x, u = solve_bvp(n)
        # Compare fine grid with coarse grid on common nodes
        u_coarse_interp = u[::2]
        err = np.max(np.abs(u_coarse_interp - prev_u))
        errors.append(err)
        prev_x, prev_u = x, u

    # Print table
    print(" n\tMax error vs previous")
    for i, n in enumerate(ns[1:]):
        print(f"{n:4d}\t{errors[i]:.3e}")
    
    # Plot convergence
    plt.loglog(ns[1:], errors, 'o-', label='Error estimate')
    plt.xlabel('n (number of subintervals)')
    plt.ylabel('Estimated max error')
    plt.title('Convergence of finite difference solution')
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

estimate_error()


"""
由BVP_Figure得知:
當n增至約32768時達到最小數值誤差(約2.7*10^(-11))，再增加n反而使誤差增加，
應是浮點捨入誤差(rounding error)開始主導，限制了可獲得的最小誤差。

"""
