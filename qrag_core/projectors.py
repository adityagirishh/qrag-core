"""
Projection primitives for QRAG.
"""

import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA


class PCAProjector:
    """PCA-based projector for dimensionality reduction."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = PCA(n_components=output_dim, random_state=0)
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit PCA on embeddings."""
        self.model.fit(X)
        self.fitted = True
        var_explained = np.sum(self.model.explained_variance_ratio_)
        print(f"PCA explains {var_explained:.3f} of variance")

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project embeddings into lower-dimensional space.

        Returns
        -------
        normalized : np.ndarray  — L2-normalized projections (for index search)
        raw        : np.ndarray  — raw projections (for reconstruction)
        """
        if not self.fitted:
            raise RuntimeError("PCAProjector is not fitted yet. Call fit() first.")
        raw = self.model.transform(X).astype("float32")
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / (norms + 1e-12)
        return normalized, raw

    def inverse_transform(self, raw: np.ndarray) -> np.ndarray:
        """Reconstruct embeddings from raw projections."""
        if not self.fitted:
            raise RuntimeError("PCAProjector is not fitted yet. Call fit() first.")
        return self.model.inverse_transform(raw).astype("float32")

    @property
    def out_dim(self) -> int:
        return self.output_dim