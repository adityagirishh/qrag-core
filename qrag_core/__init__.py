"""
qrag-core: Memory-efficient RAG with 4-bit residuals and 8-bit E1 quantization.
"""

from .model import FourBitEightBitQRAG
from .quantizers import FourBitQuantizer, EightBitQuantizer
from .projectors import PCAProjector
from .index import build_index, search_index

__version__ = "0.1.0"
__all__ = [
    "FourBitEightBitQRAG",
    "FourBitQuantizer",
    "EightBitQuantizer",
    "PCAProjector",
    "build_index",
    "search_index",
]