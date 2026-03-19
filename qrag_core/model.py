"""
Core QRAG model: 4-bit residuals + 8-bit E1 quantization.
"""

import os
import shutil
import tempfile
from typing import Optional, Tuple, List

import numpy as np

from .quantizers import FourBitQuantizer, EightBitQuantizer
from .projectors import PCAProjector
from .index import build_index, search_index


class FourBitEightBitQRAG:
    """
    QRAG with 4-bit residuals and 8-bit E1 quantization.

    Achieves ~2.3x memory reduction with ~0.695 recall@10.

    Parameters
    ----------
    temp_dir : str, optional
        Directory for memory-mapped files. Uses a temp dir if not provided.

    Example
    -------
    >>> model = FourBitEightBitQRAG()
    >>> model.fit(embeddings, projection_dim=192)
    >>> ids, scores = model.search(query, k1=100, k_final=10)
    >>> model.cleanup()
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.projector = None
        self.index = None
        self.fitted = False
        self.residual_bits = 4
        self.e1_bits = 8

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, projection_dim: int = 192) -> None:
        """
        Fit the QRAG model.

        Parameters
        ----------
        embeddings     : np.ndarray, shape (N, D) — L2-normalized corpus embeddings
        projection_dim : int — PCA output dimensionality
        """
        print(f"Fitting 4-bit/8-bit QRAG on {embeddings.shape}...")
        N, D = embeddings.shape

        # 1. Fit PCA projector
        self.projector = PCAProjector(D, projection_dim)
        self.projector.fit(embeddings)

        # 2. Get E1 projections
        e1_normalized, e1_raw = self.projector.transform(embeddings)

        # 3. Build search index
        print("Building index...")
        self.index = build_index(e1_normalized)

        # 4. Compute residuals
        print("Computing residuals...")
        base_reconstructions = self.projector.inverse_transform(e1_raw)
        residuals = embeddings - base_reconstructions
        print(f"Residual stats: mean={residuals.mean():.6f}, std={residuals.std():.6f}")

        # 5. Store quantized data
        self._store_quantized_data(residuals, e1_raw)

        self.corpus_size = N
        self.embedding_dim = D
        self.fitted = True
        print("Fitting complete!")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, k1: int = 200, k_final: int = 10
    ) -> Tuple[List[int], np.ndarray]:
        """
        Two-stage search with on-demand reconstruction.

        Parameters
        ----------
        query   : np.ndarray, shape (D,)
        k1      : int — number of candidates retrieved from the index
        k_final : int — number of final results after reranking

        Returns
        -------
        ids    : List[int]
        scores : np.ndarray, shape (k_final,)
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # Stage 1: approximate search on E1 index
        q_normalized, _ = self.projector.transform(query.reshape(1, -1))
        candidate_ids = search_index(self.index, q_normalized, k1)

        # Stage 2: on-demand reconstruction + exact rerank
        packed_e1 = self.e1_raw_mm[candidate_ids]
        e1_raw_candidates = self.e1_quantizer.dequantize(
            packed_e1, self.e1_mins, self.e1_maxs,
            (len(candidate_ids), self.projector.out_dim),
        )
        base_recons = self.projector.inverse_transform(e1_raw_candidates)

        packed_residuals = self.residuals_mm[candidate_ids]
        residuals = self.residual_quantizer.dequantize(
            packed_residuals, self.residual_mins, self.residual_maxs,
            (len(candidate_ids), self.embedding_dim),
        )

        reconstructed = base_recons + residuals
        norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        reconstructed /= norms + 1e-12

        query_norm = query / (np.linalg.norm(query) + 1e-12)
        scores = reconstructed.dot(query_norm)

        top_indices = np.argsort(-scores)[:k_final]
        return candidate_ids[top_indices].tolist(), scores[top_indices]

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> dict:
        """Return memory usage breakdown in MB."""
        stats = {}

        if hasattr(self, "residuals_mm"):
            stats["residuals_mb"] = self.residuals_mm.nbytes / (1024 ** 2)

        if hasattr(self, "e1_raw_mm"):
            stats["e1_projections_mb"] = self.e1_raw_mm.nbytes / (1024 ** 2)

        if self.index is not None and hasattr(self.index, "ntotal"):
            faiss_bytes = self.index.ntotal * self.projector.out_dim * 4
            stats["faiss_index_mb"] = faiss_bytes / (1024 ** 2)

        if hasattr(self, "residual_mins"):
            metadata_bytes = (
                self.residual_mins.nbytes + self.residual_maxs.nbytes
                + self.e1_mins.nbytes + self.e1_maxs.nbytes
            )
            stats["metadata_mb"] = metadata_bytes / (1024 ** 2)

        stats["total_mb"] = sum(stats.values())
        return stats

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove memory-mapped temp files."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store_quantized_data(self, residuals: np.ndarray, e1_raw: np.ndarray) -> None:
        N, D = residuals.shape

        # 4-bit residuals
        self.residual_quantizer = FourBitQuantizer()
        packed_residuals, self.residual_mins, self.residual_maxs = \
            self.residual_quantizer.quantize(residuals)

        self.residuals_path = os.path.join(self.temp_dir, "residuals_4bit.dat")
        self.residuals_mm = np.memmap(
            self.residuals_path, dtype="uint8", mode="w+", shape=packed_residuals.shape
        )
        self.residuals_mm[:] = packed_residuals
        self.residuals_mm.flush()

        # 8-bit E1 projections
        self.e1_quantizer = EightBitQuantizer()
        packed_e1, self.e1_mins, self.e1_maxs = self.e1_quantizer.quantize(e1_raw)

        self.e1_raw_path = os.path.join(self.temp_dir, "e1_raw_8bit.dat")
        self.e1_raw_mm = np.memmap(
            self.e1_raw_path, dtype="uint8", mode="w+", shape=packed_e1.shape
        )
        self.e1_raw_mm[:] = packed_e1
        self.e1_raw_mm.flush()

        print(f"Stored {N} vectors:")
        print(f"  Residuals:      {packed_residuals.nbytes / 1024**2:.1f}MB (4-bit)")
        print(f"  E1 projections: {packed_e1.nbytes / 1024**2:.1f}MB (8-bit)")