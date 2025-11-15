"""Comparison tests between evlib and legacy loaders."""

from __future__ import annotations

import numpy as np
import pytest


def decode_legacy_events(event_words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode packed uint32 event_words into x, y, polarity arrays.

    Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    See: evio/src/evio/core/mmap.py:151-154

    Args:
        event_words: Packed uint32 events from legacy loader

    Returns:
        Tuple of (x, y, polarity) numpy arrays
    """
    x = (event_words & 0x3FFF).astype(np.uint16)
    y = ((event_words >> 14) & 0x3FFF).astype(np.uint16)
    raw_polarity = ((event_words >> 28) & 0xF).astype(np.uint8)
    polarity = (raw_polarity > 0).astype(np.int8)
    return x, y, polarity


def test_decode_legacy_events():
    """Test decoding of packed uint32 event words."""
    # Create test event: x=100, y=200, polarity=1
    # Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    # polarity=1 -> bits [31:28] = 0x1
    # y=200 -> bits [27:14] = 200 = 0xC8
    # x=100 -> bits [13:0] = 100 = 0x64
    event_word = (1 << 28) | (200 << 14) | 100
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 100
    assert y[0] == 200
    assert polarity[0] == 1


def test_decode_legacy_events_polarity_zero():
    """Test decoding of polarity=0 events."""
    # polarity=0 -> bits [31:28] = 0x0
    event_word = (0 << 28) | (150 << 14) | 50
    event_words = np.array([event_word], dtype=np.uint32)

    x, y, polarity = decode_legacy_events(event_words)

    assert x[0] == 50
    assert y[0] == 150
    assert polarity[0] == 0
