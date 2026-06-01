# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for score normalization utilities."""

from openviking.retrieve.graph.score_normalizer import minmax_normalize, zscore_normalize


class TestZscoreNormalize:
    def test_empty(self):
        assert zscore_normalize({}) == {}

    def test_single_value(self):
        result = zscore_normalize({"a": 5.0})
        assert result == {"a": 0.0}

    def test_constant_values(self):
        result = zscore_normalize({"a": 3.0, "b": 3.0, "c": 3.0})
        assert all(v == 0.0 for v in result.values())

    def test_basic(self):
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = zscore_normalize(scores)
        # Mean = 2.0, stdev = 1.0
        assert abs(result["a"] + 1.0) < 1e-6
        assert abs(result["b"]) < 1e-6
        assert abs(result["c"] - 1.0) < 1e-6

    def test_preserves_keys(self):
        scores = {"x": 10.0, "y": 20.0}
        result = zscore_normalize(scores)
        assert set(result.keys()) == {"x", "y"}


class TestMinmaxNormalize:
    def test_empty(self):
        assert minmax_normalize({}) == {}

    def test_constant_values(self):
        result = minmax_normalize({"a": 5.0, "b": 5.0})
        assert all(v == 0.5 for v in result.values())

    def test_basic(self):
        scores = {"a": 0.0, "b": 5.0, "c": 10.0}
        result = minmax_normalize(scores)
        assert abs(result["a"]) < 1e-6
        assert abs(result["b"] - 0.5) < 1e-6
        assert abs(result["c"] - 1.0) < 1e-6

    def test_negative_values(self):
        scores = {"a": -2.0, "b": 2.0}
        result = minmax_normalize(scores)
        assert abs(result["a"]) < 1e-6
        assert abs(result["b"] - 1.0) < 1e-6
