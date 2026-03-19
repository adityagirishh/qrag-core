"""
Quantization primitives for QRAG.
"""

import numpy as np
from typing import Tuple


class FourBitQuantizer:
    """4-bit quantizer with bit packing and percentile-based clipping."""

    def __init__(self):
        self.bits = 4
        self.levels = 15  # 2^4 - 1

    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mins = np.percentile(data, 1, axis=0)
        maxs = np.percentile(data, 99, axis=0)
        ranges = np.maximum(maxs - mins, 1e-8)

        scaled = np.clip((data - mins) / ranges, 0, 1)
        quantized = np.round(scaled * self.levels).astype(np.uint8)
        packed = self._pack_4bit(quantized)
        return packed, mins, maxs

    def dequantize(self, packed: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                   original_shape: Tuple[int, int]) -> np.ndarray:
        unpacked = self._unpack_4bit(packed, original_shape)
        ranges = maxs - mins
        return mins + (unpacked.astype("float32") / self.levels) * ranges

    def _pack_4bit(self, data: np.ndarray) -> np.ndarray:
        N, D = data.shape
        packed_width = (D + 1) // 2
        packed = np.zeros((N, packed_width), dtype=np.uint8)
        for d in range(D):
            byte_idx = d // 2
            if d % 2 == 0:
                packed[:, byte_idx] |= data[:, d]
            else:
                packed[:, byte_idx] |= (data[:, d] << 4)
        return packed

    def _unpack_4bit(self, packed: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        N, D = original_shape
        unpacked = np.zeros((N, D), dtype=np.uint8)
        for d in range(D):
            byte_idx = d // 2
            if byte_idx < packed.shape[1]:
                if d % 2 == 0:
                    unpacked[:, d] = packed[:, byte_idx] & 0x0F
                else:
                    unpacked[:, d] = (packed[:, byte_idx] >> 4) & 0x0F
        return unpacked


class EightBitQuantizer:
    """8-bit quantizer (standard uint8)."""

    def __init__(self):
        self.bits = 8
        self.levels = 255  # 2^8 - 1

    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mins = np.percentile(data, 1, axis=0)
        maxs = np.percentile(data, 99, axis=0)
        ranges = np.maximum(maxs - mins, 1e-8)

        scaled = np.clip((data - mins) / ranges, 0, 1)
        quantized = np.round(scaled * self.levels).astype(np.uint8)
        return quantized, mins, maxs

    def dequantize(self, quantized: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                   original_shape: Tuple[int, int]) -> np.ndarray:
        ranges = maxs - mins
        return mins + (quantized.astype("float32") / self.levels) * ranges