"""
Corrections for testing many metrics at once.

Check 20 metrics at alpha 0.05 and you expect one false "win" even when
nothing changed. Holm controls the chance of any false positive.
Benjamini-Hochberg controls the share of false positives among the wins,
which is usually the better trade for a dashboard of metrics.
"""

from __future__ import annotations

import numpy as np


def holm(p_values) -> np.ndarray:
    """Holm adjusted p values (family wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj
