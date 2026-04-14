"""
Gaussian Mixture Model (GMM) — implemented from scratch using NumPy.
Uses the Expectation-Maximization (EM) algorithm.
"""

import numpy as np


class GaussianMixtureModel:
    """
    Gaussian Mixture Model fitted via the EM algorithm.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components (clusters).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold on log-likelihood improvement.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_components: int = 2, max_iter: int = 100,
                 tol: float = 1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

        # Parameters learned during fit
        self.weights_ = None    # shape (K,)
        self.means_ = None      # shape (K, D)
        self.covariances_ = None  # shape (K, D, D)
        self.log_likelihoods_ = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """Fit the GMM to data X using EM."""
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        K = self.n_components

        # --- Initialisation (K-Means++ style for means) ---
        self.weights_ = np.ones(K) / K
        self.means_ = self._kmeans_plusplus_init(X, K)
        self.covariances_ = np.array([np.eye(n_features) * X.var() for _ in range(K)])

        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._e_step(X)

            # M-step: update parameters
            self._m_step(X, responsibilities)

            # Check convergence
            log_likelihood = self._total_log_likelihood(X)
            self.log_likelihoods_.append(log_likelihood)

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break
            prev_log_likelihood = log_likelihood
        else:
            print(f"Reached max_iter={self.max_iter} without full convergence.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most likely component index for each sample."""
        responsibilities = self._e_step(np.asarray(X, dtype=float))
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return responsibility matrix (soft assignments)."""
        return self._e_step(np.asarray(X, dtype=float))

    def score(self, X: np.ndarray) -> float:
        """Return mean log-likelihood per sample."""
        return self._total_log_likelihood(np.asarray(X, dtype=float)) / len(X)

    def sample(self, n_samples: int = 100):
        """Draw random samples from the fitted GMM."""
        if self.means_ is None:
            raise RuntimeError("Model must be fitted before sampling.")
        component_indices = self.rng.choice(
            self.n_components, size=n_samples, p=self.weights_
        )
        samples = np.array([
            self.rng.multivariate_normal(self.means_[k], self.covariances_[k])
            for k in component_indices
        ])
        return samples, component_indices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Compute the responsibility of each component for each sample."""
        n_samples = X.shape[0]
        K = self.n_components
        weighted_likelihoods = np.zeros((n_samples, K))

        for k in range(K):
            weighted_likelihoods[:, k] = (
                self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
            )

        total = weighted_likelihoods.sum(axis=1, keepdims=True)
        # Avoid division by zero
        total = np.where(total == 0, 1e-300, total)
        return weighted_likelihoods / total

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """Update mixture parameters from current responsibilities."""
        n_samples, n_features = X.shape
        K = self.n_components
        Nk = responsibilities.sum(axis=0)  # effective number of points per component

        self.weights_ = Nk / n_samples

        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]

        for k in range(K):
            diff = X - self.means_[k]                          # (N, D)
            weighted_diff = responsibilities[:, k:k+1] * diff  # (N, D)
            self.covariances_[k] = (weighted_diff.T @ diff) / Nk[k]
            # Regularisation: keep covariance matrix positive-definite
            self.covariances_[k] += np.eye(n_features) * 1e-6

    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Multivariate Gaussian probability density function."""
        n_features = X.shape[1]
        diff = X - mean
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(n_features)
            cov_det = 1.0

        cov_det = max(cov_det, 1e-300)
        norm = 1.0 / (np.sqrt((2 * np.pi) ** n_features * cov_det))
        exponent = -0.5 * np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        return norm * np.exp(exponent)

    def _total_log_likelihood(self, X: np.ndarray) -> float:
        """Compute total log-likelihood of the data under the model."""
        K = self.n_components
        likelihood = np.zeros(X.shape[0])
        for k in range(K):
            likelihood += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(likelihood + 1e-300))

    def _kmeans_plusplus_init(self, X: np.ndarray, K: int) -> np.ndarray:
        """K-Means++ initialisation for better starting means."""
        idx = self.rng.integers(0, len(X))
        centers = [X[idx]]

        for _ in range(1, K):
            dists = np.array([min(np.linalg.norm(x - c) ** 2 for c in centers) for x in X])
            probs = dists / dists.sum()
            idx = self.rng.choice(len(X), p=probs)
            centers.append(X[idx])

        return np.array(centers)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Generating synthetic data from 3 Gaussians...")
    rng = np.random.default_rng(42)
    X1 = rng.multivariate_normal([0, 0],   [[1, 0.5], [0.5, 1]],  150)
    X2 = rng.multivariate_normal([5, 5],   [[1, -0.3], [-0.3, 1]], 100)
    X3 = rng.multivariate_normal([0, 6],   [[0.5, 0], [0, 2]],     80)
    X = np.vstack([X1, X2, X3])
    rng.shuffle(X)

    print("Fitting GMM with 3 components...")
    gmm = GaussianMixtureModel(n_components=3, max_iter=200, tol=1e-5, random_state=0)
    gmm.fit(X)

    labels = gmm.predict(X)

    print("\nLearned parameters:")
    for k in range(gmm.n_components):
        print(f"  Component {k}: weight={gmm.weights_[k]:.3f}, mean={gmm.means_[k]}")

    print(f"\nMean log-likelihood per sample: {gmm.score(X):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw data
    axes[0].scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, color="steelblue")
    axes[0].set_title("Original Data")

    # Right: GMM clusters
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for k in range(gmm.n_components):
        mask = labels == k
        axes[1].scatter(X[mask, 0], X[mask, 1], s=10, alpha=0.5,
                        color=colors[k % len(colors)], label=f"Component {k}")
        axes[1].scatter(*gmm.means_[k], marker="X", s=200, color="black", zorder=5)
    axes[1].set_title("GMM Clustering (EM)")
    axes[1].legend()

    plt.suptitle("Gaussian Mixture Model — implemented from scratch", fontweight="bold")
    plt.tight_layout()
    plt.savefig("gmm_result.png", dpi=150)
    print("\nPlot saved to gmm_result.png")
    plt.show()