import numpy as np
import matplotlib.pyplot as plt

def aaa(F, Z, tol=1e-13, mmax=20):
    """
    Simplified version of AAA algorithm.
    Demonstrates adaptive rational approximation with stability improvements.
    """
    M = len(Z)
    F = F.astype(complex)
    Z = Z.astype(complex)

    Fdiag = np.diag(F)
    J = np.arange(M)
    z, f = [], []
    C = np.zeros((M, 0), dtype=complex)
    errvec = []

    # Initial approximation = constant mean value
    R = np.mean(F)

    for m in range(1, mmax + 1):
        # --- Step 1: pick next support point (max residual) ---
        j = np.argmax(np.abs(F - R))
        z.append(Z[j])
        f.append(F[j])
        J = J[J != j]

        # --- Step 2: update Cauchy matrix ---
        diff = Z - Z[j]
        diff[diff == 0] = np.inf  # temporarily avoid divide-by-zero
        c_new = 1.0 / diff
        C = np.column_stack((C, c_new))

        # --- Step 3: form Loewner matrix ---
        fdiag = np.diag(f)
        A = Fdiag @ C - C @ fdiag

        # --- Step 4: compute right singular vector ---
        try:
            U, s, Vh = np.linalg.svd(A[J, :], full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"⚠️ SVD failed at step {m}, stopping.")
            break
        w = Vh[-1, :]

        # --- Step 5: build rational approximation ---
        N = C @ (w * f)
        D = C @ w
        R = F.copy()
        mask = np.abs(D[J]) > 1e-14  # avoid division by near-zero
        R[J[mask]] = N[J[mask]] / D[J[mask]]

        # --- Step 6: convergence check ---
        err = np.linalg.norm(F - R, np.inf)
        errvec.append(err)
        if err <= tol * np.linalg.norm(F, np.inf):
            break

    # --- Step 7: define final rational approximant function ---
    def r(zz):
        zz = np.array(zz, ndmin=1, dtype=complex)
        Zsup = np.array(z, dtype=complex)

        # Build Cauchy matrix safely
        diff = zz[:, None] - Zsup[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            Cmat = np.where(np.abs(diff) < 1e-14, 0, 1.0 / diff)

        # Compute rational approximation
        num = Cmat @ (w * f)
        den = Cmat @ w

        # Avoid division by very small denominator
        rvals = np.empty_like(num)
        small_den = np.abs(den) < 1e-14
        rvals[~small_den] = num[~small_den] / den[~small_den]
        rvals[small_den] = np.nan  # placeholder

        # Fix NaN (at support points)
        for k, val in enumerate(zz):
            if np.isnan(rvals[k]):
                match = np.where(np.isclose(val, Zsup, atol=1e-14))[0]
                if len(match) > 0:
                    rvals[k] = f[match[0]]

        return rvals

    return r, np.array(z), np.array(f), w, np.array(errvec)


# === Demonstration ===
if __name__ == "__main__":
    Z = np.linspace(-1, 1, 200)
    F = np.exp(Z)

    r, z, f, w, errvec = aaa(F, Z, tol=1e-10, mmax=20)

    # --- Plot 1: function vs AAA approximation ---
    plt.figure(figsize=(8, 4))
    plt.plot(Z, F.real, label='f(z) = exp(z)', color='black')
    rvals = r(Z)
    plt.plot(Z, np.real(rvals), '--', label='Re(AAA approx)', color='red')
    plt.plot(Z, np.imag(rvals), '--', label='Im(AAA approx)', color='orange')
    plt.scatter(z.real, np.real(f), color='blue', zorder=5, label='support points')
    plt.legend()
    plt.title('AAA Approximation of exp(z)')
    plt.grid(True)

    # --- Plot 2: Convergence of error ---
    plt.figure()
    plt.semilogy(errvec, 'o-')
    plt.title('Convergence of AAA Algorithm')
    plt.xlabel('Iteration m')
    plt.ylabel('Error ||f - r||∞')
    plt.grid(True)

    plt.show()

