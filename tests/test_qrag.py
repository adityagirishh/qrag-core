"""
Tests for qrag-core.
"""

import numpy as np
import pytest
from qrag_core import FourBitEightBitQRAG, FourBitQuantizer, EightBitQuantizer, PCAProjector


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def small_embeddings():
    """Small L2-normalized embedding matrix for fast tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 64).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

@pytest.fixture
def fitted_model(small_embeddings):
    """Pre-fitted QRAG model."""
    model = FourBitEightBitQRAG()
    model.fit(small_embeddings, projection_dim=32)
    yield model
    model.cleanup()


# ------------------------------------------------------------------
# FourBitQuantizer
# ------------------------------------------------------------------

class TestFourBitQuantizer:

    def test_quantize_shape(self):
        q = FourBitQuantizer()
        data = np.random.randn(100, 64).astype("float32")
        packed, mins, maxs = q.quantize(data)
        assert packed.shape == (100, 32)   # D/2 packed bytes
        assert mins.shape == (64,)
        assert maxs.shape == (64,)

    def test_roundtrip_error(self):
        q = FourBitQuantizer()
        data = np.random.randn(200, 64).astype("float32")
        packed, mins, maxs = q.quantize(data)
        recovered = q.dequantize(packed, mins, maxs, (200, 64))
        # 4-bit is lossy — just check error is reasonable
        mae = np.abs(data - recovered).mean()
        assert mae < 0.5

    def test_output_dtype(self):
        q = FourBitQuantizer()
        data = np.random.randn(50, 16).astype("float32")
        packed, _, _ = q.quantize(data)
        assert packed.dtype == np.uint8

    def test_dequantize_dtype(self):
        q = FourBitQuantizer()
        data = np.random.randn(50, 16).astype("float32")
        packed, mins, maxs = q.quantize(data)
        recovered = q.dequantize(packed, mins, maxs, (50, 16))
        assert recovered.dtype == np.float32


# ------------------------------------------------------------------
# EightBitQuantizer
# ------------------------------------------------------------------

class TestEightBitQuantizer:

    def test_quantize_shape(self):
        q = EightBitQuantizer()
        data = np.random.randn(100, 64).astype("float32")
        packed, mins, maxs = q.quantize(data)
        assert packed.shape == (100, 64)   # no packing — same shape
        assert packed.dtype == np.uint8

    def test_roundtrip_error(self):
        q = EightBitQuantizer()
        data = np.random.randn(200, 64).astype("float32")
        packed, mins, maxs = q.quantize(data)
        recovered = q.dequantize(packed, mins, maxs, (200, 64))
        mae = np.abs(data - recovered).mean()
        assert mae < 0.05   # 8-bit should be much tighter

    def test_dequantize_dtype(self):
        q = EightBitQuantizer()
        data = np.random.randn(50, 16).astype("float32")
        packed, mins, maxs = q.quantize(data)
        recovered = q.dequantize(packed, mins, maxs, (50, 16))
        assert recovered.dtype == np.float32


# ------------------------------------------------------------------
# PCAProjector
# ------------------------------------------------------------------

class TestPCAProjector:

    def test_transform_shapes(self, small_embeddings):
        proj = PCAProjector(64, 32)
        proj.fit(small_embeddings)
        normalized, raw = proj.transform(small_embeddings[:10])
        assert normalized.shape == (10, 32)
        assert raw.shape == (10, 32)

    def test_normalized_is_unit(self, small_embeddings):
        proj = PCAProjector(64, 32)
        proj.fit(small_embeddings)
        normalized, _ = proj.transform(small_embeddings[:20])
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)

    def test_inverse_transform_shape(self, small_embeddings):
        proj = PCAProjector(64, 32)
        proj.fit(small_embeddings)
        _, raw = proj.transform(small_embeddings[:10])
        recon = proj.inverse_transform(raw)
        assert recon.shape == (10, 64)

    def test_not_fitted_raises(self):
        proj = PCAProjector(64, 32)
        with pytest.raises(RuntimeError):
            proj.transform(np.random.randn(5, 64).astype("float32"))


# ------------------------------------------------------------------
# FourBitEightBitQRAG
# ------------------------------------------------------------------

class TestFourBitEightBitQRAG:

    def test_fit_sets_fitted_flag(self, fitted_model):
        assert fitted_model.fitted is True

    def test_fit_sets_dimensions(self, fitted_model, small_embeddings):
        assert fitted_model.corpus_size == len(small_embeddings)
        assert fitted_model.embedding_dim == small_embeddings.shape[1]

    def test_search_returns_correct_count(self, fitted_model):
        query = np.random.randn(64).astype("float32")
        query /= np.linalg.norm(query)
        ids, scores = fitted_model.search(query, k1=50, k_final=10)
        assert len(ids) == 10
        assert len(scores) == 10

    def test_search_ids_in_corpus_range(self, fitted_model):
        query = np.random.randn(64).astype("float32")
        query /= np.linalg.norm(query)
        ids, _ = fitted_model.search(query, k1=50, k_final=10)
        assert all(0 <= i < 500 for i in ids)

    def test_search_scores_bounded(self, fitted_model):
        query = np.random.randn(64).astype("float32")
        query /= np.linalg.norm(query)
        _, scores = fitted_model.search(query, k1=50, k_final=10)
        assert np.all(scores >= -1.01) and np.all(scores <= 1.01)

    def test_scores_descending(self, fitted_model):
        query = np.random.randn(64).astype("float32")
        query /= np.linalg.norm(query)
        _, scores = fitted_model.search(query, k1=50, k_final=10)
        assert list(scores) == sorted(scores, reverse=True)

    def test_search_before_fit_raises(self):
        model = FourBitEightBitQRAG()
        with pytest.raises(RuntimeError):
            model.search(np.random.randn(64).astype("float32"))

    def test_memory_usage_keys(self, fitted_model):
        stats = fitted_model.get_memory_usage()
        assert "residuals_mb" in stats
        assert "e1_projections_mb" in stats
        assert "total_mb" in stats

    def test_memory_usage_positive(self, fitted_model):
        stats = fitted_model.get_memory_usage()
        assert all(v > 0 for v in stats.values())

    def test_cleanup_removes_temp_files(self, small_embeddings):
        import os
        model = FourBitEightBitQRAG()
        model.fit(small_embeddings, projection_dim=32)
        temp_dir = model.temp_dir
        assert os.path.exists(temp_dir)
        model.cleanup()
        assert not os.path.exists(temp_dir)