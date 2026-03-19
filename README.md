
# qrag-core

Memory-efficient retrieval-augmented generation (RAG) with **4-bit residual quantization** and **8-bit E1 projection quantization**.

| Metric | Value |
|---|---|
| Memory reduction | ~2.3x |
| Recall@10 | ~0.695 |
| Residual precision | 4-bit (packed) |
| E1 projection precision | 8-bit |

---

## Installation

```bash
git clone https://github.com/adityagirishh/qrag-core.git
cd qrag-core
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

With FAISS (recommended for large corpora):
```bash
pip install faiss-cpu
```

---

## Quickstart

```python
import numpy as np
from qrag_core import FourBitEightBitQRAG

# Your corpus embeddings — shape (N, D), L2-normalized
embeddings = np.random.randn(100_000, 768).astype("float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# Fit
model = FourBitEightBitQRAG()
model.fit(embeddings, projection_dim=192)

# Search
query = np.random.randn(768).astype("float32")
ids, scores = model.search(query, k1=100, k_final=10)
print(ids)    # top-10 corpus indices
print(scores) # corresponding similarity scores

# Memory stats
print(model.get_memory_usage())

# Cleanup temp files when done
model.cleanup()
```

---

## How it works

QRAG uses a two-stage search pipeline:

```
Query
  │
  ▼
[Stage 1] PCA projection → search E1 index → top-k1 candidates
  │
  ▼
[Stage 2] On-demand reconstruction (dequantize E1 + residuals)
          → exact rerank → top-k final results
```

- **E1 projections** are stored at 8-bit precision (~4x smaller than float32)
- **Residuals** are stored at 4-bit precision (~8x smaller than float32)
- Reconstruction happens only for the small candidate set, not the full corpus

---

## API

### `FourBitEightBitQRAG`

```python
model = FourBitEightBitQRAG(temp_dir=None)
```

| Method | Description |
|---|---|
| `fit(embeddings, projection_dim=192)` | Fit model on corpus embeddings |
| `search(query, k1=200, k_final=10)` | Search and return `(ids, scores)` |
| `get_memory_usage()` | Returns memory breakdown dict in MB |
| `cleanup()` | Deletes memory-mapped temp files |

---

## Requirements

- Python >= 3.9
- numpy >= 1.24
- scikit-learn >= 1.3
- faiss-cpu >= 1.7 *(optional but recommended)*

---

## License
MIT — see [LICENSE](LICENSE) for details.
