"""
Index building and search for QRAG.
"""

import numpy as np
from typing import Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    print("[WARN] FAISS not available. Falling back to NumPy-based search.")


def build_index(vectors: np.ndarray):
    """
    Build a search index from normalized projection vectors.

    Uses FAISS if available, otherwise stores vectors for NumPy dot-product search.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, D) — L2-normalized

    Returns
    -------
    index : faiss.Index or np.ndarray
    """
    if not FAISS_AVAILABLE:
        return vectors.astype("float32")

    n, d = vectors.shape

    if n > 100_000:
        m = d // 8
        index = faiss.IndexPQ(d, m, 8)
        index.train(vectors)
    elif n > 30_000:
        nlist = int(4 * np.sqrt(n))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(vectors)
    else:
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200

    index.add(vectors)
    return index


def search_index(index, query: np.ndarray, k: int) -> np.ndarray:
    """
    Search the index for the top-k nearest neighbors.

    Parameters
    ----------
    index  : faiss.Index or np.ndarray (NumPy fallback)
    query  : np.ndarray, shape (1, D) — L2-normalized
    k      : int — number of candidates to retrieve

    Returns
    -------
    candidate_ids : np.ndarray, shape (k,)
    """
    if not FAISS_AVAILABLE:
        # NumPy fallback: dot-product search
        scores = (index @ query.reshape(-1)).astype("float32")
        return np.argsort(-scores)[:k]

    distances, indices = index.search(query, k)
    return indices[0]