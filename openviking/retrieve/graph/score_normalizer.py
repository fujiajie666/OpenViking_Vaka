# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Score normalization utilities for graph-based retrieval."""

import statistics
from typing import Dict


def zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Z-score normalization. Returns all zeros when variance is zero."""
    if not scores:
        return {}
    values = list(scores.values())
    if len(values) == 1:
        return {k: 0.0 for k in scores}
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - mean) / std for k, v in scores.items()}


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalization to [0, 1]. Returns 0.5 when range is zero."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
