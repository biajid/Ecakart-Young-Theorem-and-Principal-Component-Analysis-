from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# No forced 'Agg' backend: forcing it would HIDE figures in Jupyter.
np.set_printoptions(precision=5, suppress=True)
OUTPUT_DIR = (Path(__file__).resolve().parent
              if '__file__' in globals() else Path.cwd())


# ===============================================================
# IMPLEMENTATION 1: SMALL, HAND-ENTERED DATASET
# ===============================================================
def run_small_example():
    # ---------------------------------------------------------------
    # Step 1. Enter observations as COLUMNS; features are ROWS.
    # ---------------------------------------------------------------
    X = np.array([
        [8, 9, 11, 12],  # Feature 1, measured for four observations
        [4, 4,  6,  6],  # Feature 2, measured for four observations
    ], dtype=float)

    m, n = X.shape
    assert (m, n) == (2, 4)

    mean = X.mean(axis=1, keepdims=True)  # (m, 1): mean of each ROW
    A = X - mean                          # (m, n): centered data

    print("1. DATA AND CENTERING")
    print("X =\n", X)
    print("Feature means =\n", mean)
    print("A = X - mean =\n", A)
    print("Shape (features, observations):", A.shape)
    assert np.allclose(A.sum(axis=1), 0.0)

    # ---------------------------------------------------------------
    # Step 2. Build the sample covariance matrix: C = A A^T/(n-1).
    # ---------------------------------------------------------------
    C = (A @ A.T) / (n - 1)
    print("\n2. SAMPLE COVARIANCE")
    print("C = A @ A.T / (n - 1) =\n", C)
    print("Individual feature variances:", np.diag(C))
    print("Total sample variance = tr(C):", np.trace(C))
    assert np.allclose(C, np.cov(X, rowvar=True, ddof=1))

    # ---------------------------------------------------------------
    # Step 3. SVD of CENTERED data, not of the covariance matrix.
    # ---------------------------------------------------------------
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    pc_variances = s**2 / (n - 1)
    explained_ratios = s**2 / np.sum(s**2)

    print("\n3. SINGULAR VALUE DECOMPOSITION")
    print("U (principal directions) =\n", U)
    print("Singular values =", s)
    print("V.T =\n", Vt)
    print("Variance along principal directions =", pc_variances)
    print("Explained-variance ratios =", explained_ratios)
    assert np.allclose(A, U @ np.diag(s) @ Vt)
    assert np.allclose(U.T @ C @ U, np.diag(pc_variances))
    # Singular vector signs may differ from other computers; that is normal.

    # ---------------------------------------------------------------
    # Step 4. Keep k=1 principal component and reconstruct.
    # ---------------------------------------------------------------
    k = 1
    Uk = U[:, :k]       # (m, k): first k LEFT singular vectors
    Z = Uk.T @ A        # (k, n): one reduced coordinate per observation
    Ahat = Uk @ Z       # (m, n): centered rank-k reconstruction
    Xhat = Ahat + mean  # (m, n): restore the means of the features
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]  # Same truncated SVD

    print("\n4. PROJECTION AND RECONSTRUCTION")
    print("U_k shape:", Uk.shape, " Z shape:", Z.shape)
    print("Principal direction U_k =\n", Uk)
    print("Scores Z = U_k.T @ A =\n", Z)
    print("Reconstructed centered data Ahat =\n", Ahat)
    print("Reconstructed raw data Xhat =\n", Xhat)
    assert np.allclose(Ahat, Ak)
    assert np.linalg.matrix_rank(Ahat) == k

    # ---------------------------------------------------------------
    # Step 5. Verify BOTH Eckart--Young error formulas.
    # ---------------------------------------------------------------
    E = A - Ahat
    fro_squared = np.linalg.norm(E, ord="fro")**2
    spectral_error = np.linalg.norm(E, ord=2)
    fro_prediction = np.sum(s[k:]**2)
    spectral_prediction = s[k]  # k < rank(A) in this example
    retained_variance = np.sum(s[:k]**2) / np.sum(s**2)

    print("\n5. ECKART--YOUNG VERIFICATION")
    print(f"Variance retained: {retained_variance:.4%}")
    print(f"Frobenius error squared:   {fro_squared:.8f}")
    print(f"Sum of discarded sigma^2: {fro_prediction:.8f}")
    print(f"Spectral error:           {spectral_error:.8f}")
    print(f"Next singular value:      {spectral_prediction:.8f}")
    print(f"Relative squared error:   {fro_squared / np.sum(s**2):.4%}")
    print(f"1 - explained variance:   {1.0 - retained_variance:.4%}")
    assert np.isclose(fro_squared, fro_prediction)
    assert np.isclose(spectral_error, spectral_prediction)
    assert np.isclose(fro_squared / np.sum(s**2), 1.0 - retained_variance)

    # Compare two other one-dimensional subspaces with PCA's optimal one.
    horizontal = np.array([[1.0], [0.0]])
    diagonal = np.array([[1.0], [1.0]]) / np.sqrt(2.0)
    print("\n6. OTHER ONE-DIMENSIONAL PROJECTIONS")
    for name, q in [("horizontal", horizontal),
                    ("45-degree diagonal", diagonal),
                    ("optimal PCA direction", Uk)]:
        B = q @ (q.T @ A)
        error_sq = np.linalg.norm(A - B, ord="fro")**2
        print(f"{name:>22}: squared Frobenius error = {error_sq:.6f}")
        assert error_sq + 1e-10 >= fro_squared

    # ---------------------------------------------------------------
    # Step 6. Plot the points and their orthogonal projections.
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(A[0], A[1], s=75, label="Original centered observations")
    ax.scatter(Ahat[0], Ahat[1], s=85, marker="x",
               label="Rank-1 reconstruction")
    for j in range(n):
        ax.plot([A[0, j], Ahat[0, j]], [A[1, j], Ahat[1, j]],
                linestyle="--", linewidth=1)
        ax.annotate(str(j + 1), (A[0, j], A[1, j]),
                    xytext=(5, 5), textcoords="offset points")
    t = np.linspace(-3, 3, 101)
    ax.plot(t * Uk[0, 0], t * Uk[1, 0], label="First principal axis")
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    ax.set(xlabel="Centered feature 1", ylabel="Centered feature 2",
           title="Implementation 1: PCA with observations as columns")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    destination = OUTPUT_DIR / "implementation1_projection.png"
    fig.savefig(destination, dpi=250)
    plt.show()  # Display immediately in Jupyter / an interactive session.
    plt.close(fig)
    print("\nFigure saved:", destination)
    print("All Implementation 1 mathematical checks passed.")


# ===============================================================
# IMPLEMENTATION 2: IMPORTED HANDWRITTEN-DIGITS DATASET
# ===============================================================
def run_digits_example():
    # ---------------------------------------------------------------
    # Step 1. Import a real dataset and TRANSPOSE to our convention.
    # ---------------------------------------------------------------
    digits = load_digits()
    # sklearn provides digits.data as (observations, features) = (1797, 64).
    # We TRANSPOSE it: X is (64 features, 1797 observations).
    X = digits.data.T.astype(float)
    labels = digits.target  # One digit label for each COLUMN of X.
    m, n = X.shape
    assert m == 64 and n == len(labels)

    print("1. LOADING THE HANDWRITTEN-DIGITS DATASET")
    print("Shape of X (features, observations):", X.shape)
    print("Each column reshapes to an 8 x 8 handwritten-digit image.")
    print("The labels are used only for visualizing the results.")

    # ---------------------------------------------------------------
    # Step 2. Center EACH FEATURE (each row), then compute covariance.
    # ---------------------------------------------------------------
    mean = X.mean(axis=1, keepdims=True)  # (64, 1)
    A = X - mean                          # (64, 1797)
    C = (A @ A.T) / (n - 1)                # (64, 64)

    print("\n2. CENTERING AND COVARIANCE")
    print("Shape of centered matrix A:", A.shape)
    print("Shape of covariance matrix C:", C.shape)
    print("Total sample variance, trace(C):", float(np.trace(C)))
    assert np.allclose(A.mean(axis=1), 0.0)
    assert np.allclose(C, C.T)

    # ---------------------------------------------------------------
    # Step 3. Compute PCA from the SVD: A = U Sigma V.T.
    # ---------------------------------------------------------------
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    variances = s**2 / (n - 1)
    total_energy = np.sum(s**2)
    ratios = s**2 / total_energy
    cumulative_ratios = np.cumsum(ratios)

    # Choose the SMALLEST k that retains at least 90% of total variance.
    variance_target = 0.90
    k = int(np.searchsorted(cumulative_ratios, variance_target) + 1)
    r = int(np.linalg.matrix_rank(A))
    assert 1 <= k < r

    print("\n3. SVD AND COMPONENT SELECTION")
    print("Numerical rank of A:", r)
    print("First ten singular values:", s[:10])
    print("First ten principal-component variances:", variances[:10])
    print(f"Components required for {variance_target:.0%} variance: {k}")
    print(f"Cumulative explained variance: {cumulative_ratios[k-1]:.4%}")
    assert np.isclose(np.sum(variances), np.trace(C))
    assert np.allclose((U * s) @ Vt, A)

    # ---------------------------------------------------------------
    # Step 4. Reduce from 64 original features to k principal scores.
    # ---------------------------------------------------------------
    Uk = U[:, :k]      # (64, k): LEFT singular vectors
    Z = Uk.T @ A       # (k, 1797): reduced observations in COLUMNS

    # Reconstruct the centered data and restore feature means.
    Ahat = Uk @ Z      # (64, 1797): rank-k approximation to A
    Xhat = Ahat + mean # (64, 1797): reconstructed original digits
    Ak_svd = (U[:, :k] * s[:k]) @ Vt[:k, :]

    print("\n4. PROJECTION AND RECONSTRUCTION")
    print("U_k shape:", Uk.shape)
    print("Z shape (k, observations):", Z.shape)
    print("Reconstructed X shape:", Xhat.shape)
    assert np.allclose(Ahat, Ak_svd)
    assert np.linalg.matrix_rank(Ahat) == k
    # Optional covariance of PCA scores: diagonal with retained variances.
    assert np.allclose((Z @ Z.T) / (n - 1), np.diag(variances[:k]))

    # ---------------------------------------------------------------
    # Step 5. Verify Eckart--Young in both matrix norms.
    # ---------------------------------------------------------------
    E = A - Ahat
    fro_sq = np.linalg.norm(E, ord="fro")**2
    fro_prediction = np.sum(s[k:]**2)
    spectral_error = np.linalg.norm(E, ord=2)
    spectral_prediction = s[k]
    fraction_discarded = fro_sq / total_energy

    print("\n5. ECKART--YOUNG VERIFICATION")
    print(f"Actual squared Frobenius error: {fro_sq:.8f}")
    print(f"Sum of discarded singular values squared: {fro_prediction:.8f}")
    print(f"Actual spectral error: {spectral_error:.8f}")
    print(f"Next singular value sigma_(k+1): {spectral_prediction:.8f}")
    print(f"Fraction of squared error: {fraction_discarded:.4%}")
    print(f"1 - explained-variance ratio: {1 - cumulative_ratios[k-1]:.4%}")
    assert np.isclose(fro_sq, fro_prediction)
    assert np.isclose(spectral_error, spectral_prediction)
    assert np.isclose(fraction_discarded, 1 - cumulative_ratios[k-1])

    # For comparison, a RANDOM k-dimensional subspace retains less variance.
    rng = np.random.default_rng(2026)
    Q_random, _ = np.linalg.qr(rng.standard_normal((m, k)))
    B_random = Q_random @ (Q_random.T @ A)
    random_fro_sq = np.linalg.norm(A - B_random, ord="fro")**2
    print(f"Random subspace squared Frobenius error: {random_fro_sq:.8f}")
    assert random_fro_sq + 1e-6 >= fro_sq

    # ---------------------------------------------------------------
    # Step 6. Visualize individual and cumulative explained variance.
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    pcs = np.arange(1, len(s) + 1)
    ax.plot(pcs, cumulative_ratios, label="Cumulative explained variance")
    ax.axhline(variance_target, linestyle="--",
               label=f"{variance_target:.0%} target")
    ax.axvline(k, linestyle=":", label=f"Selected k = {k}")
    ax.set(xlabel="Number of principal components retained",
           ylabel="Fraction of total variance retained",
           title="Implementation 2: explained variance for digits")
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    destination = OUTPUT_DIR / "implementation2_variance.png"
    fig.savefig(destination, dpi=250)
    plt.show()  # Display each figure now; do not wait until the end.
    plt.close(fig)
    print("\nExplained-variance figure saved:", destination)

    # ---------------------------------------------------------------
    # Step 7. Display a 2D PCA projection (for visualization only).
    # The plotted 2D coordinates use the first TWO components; this
    # is separate from the k-component reconstruction above.
    # ---------------------------------------------------------------
    Z2 = U[:, :2].T @ A  # (2, 1797): each observation is a column
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(Z2[0], Z2[1], c=labels, cmap="tab10",
                         s=13, alpha=0.65)
    ax.set(xlabel="Principal component 1 score",
           ylabel="Principal component 2 score",
           title="Digits dataset: first two PCA score coordinates")
    fig.colorbar(scatter, ax=ax, label="Actual digit label")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    destination = OUTPUT_DIR / "implementation2_scores.png"
    fig.savefig(destination, dpi=250)
    plt.show()  # Display each figure now; do not wait until the end.
    plt.close(fig)
    print("2D score figure saved:", destination)

    # ---------------------------------------------------------------
    # Step 8. Compare original digit images to reconstructed images.
    # We plot one sample for each digit class (0 through 9).
    # Clipping is only for drawing; errors above use UNCLIPPED Xhat.
    # ---------------------------------------------------------------
    chosen = [int(np.flatnonzero(labels == d)[0]) for d in range(10)]
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.3))
    for d, j in enumerate(chosen):
        axes[0, d].imshow(X[:, j].reshape(8, 8), cmap="gray_r",
                          vmin=0, vmax=16)
        axes[1, d].imshow(np.clip(Xhat[:, j], 0, 16).reshape(8, 8),
                          cmap="gray_r", vmin=0, vmax=16)
        axes[0, d].set_title(str(d))
        for ax in (axes[0, d], axes[1, d]):
            ax.axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel(f"Rank-{k}")
    fig.suptitle("Original and PCA-reconstructed handwritten digits")
    fig.tight_layout()
    destination = OUTPUT_DIR / "implementation2_reconstructions.png"
    fig.savefig(destination, dpi=250)
    plt.show()  # Display each figure now; do not wait until the end.
    plt.close(fig)
    print("Original/reconstructed digit figure saved:", destination)
    print("All Implementation 2 mathematical checks passed.")


# ===============================================================
# RUN BOTH IMPLEMENTATIONS, IN ORDER
# ===============================================================
if __name__ == "__main__":
    print("\n" + "=" * 67)
    print("PCA FROM SCRATCH: FEATURES IN ROWS, OBSERVATIONS IN COLUMNS")
    print("=" * 67)
    print("\n" + "-" * 67)
    print("IMPLEMENTATION 1: SMALL DATASET")
    print("-" * 67)
    run_small_example()

    print("\n" + "-" * 67)
    print("IMPLEMENTATION 2: IMPORTED DIGITS DATASET")
    print("-" * 67)
    run_digits_example()
    print("\nBoth implementations completed successfully.")
