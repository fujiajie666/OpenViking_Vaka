# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for Typed Weighted PPR algorithm."""

from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking.retrieve.graph.ppr import TypedWeightedPPR


def _build_simple_graph() -> GraphIndex:
    """Build a simple test graph: A -> B -> C, A -> C."""
    index = GraphIndex()
    index._nodes = {
        "viking://a": GraphNode(uri="viking://a"),
        "viking://b": GraphNode(uri="viking://b"),
        "viking://c": GraphNode(uri="viking://c"),
    }
    index._forward_edges = {
        "viking://a": [
            GraphEdge(from_uri="viking://a", to_uri="viking://b", link_type="related_to", weight=0.8),
            GraphEdge(from_uri="viking://a", to_uri="viking://c", link_type="belongs_to", weight=0.6),
        ],
        "viking://b": [
            GraphEdge(from_uri="viking://b", to_uri="viking://c", link_type="caused_by", weight=0.9),
        ],
    }
    index._reverse_edges = {
        "viking://b": [
            GraphEdge(from_uri="viking://a", to_uri="viking://b", link_type="related_to", weight=0.8),
        ],
        "viking://c": [
            GraphEdge(from_uri="viking://a", to_uri="viking://c", link_type="belongs_to", weight=0.6),
            GraphEdge(from_uri="viking://b", to_uri="viking://c", link_type="caused_by", weight=0.9),
        ],
    }
    index._space_key = "test"
    index._built_at = 999999.0  # Far future so is_fresh returns True
    return index


class TestTypedWeightedPPR:
    def test_empty_seeds(self):
        index = _build_simple_graph()
        ppr = TypedWeightedPPR(index, type_weights={"related_to": 1.0})
        result = ppr.run({})
        assert result == {}

    def test_convergence(self):
        """PPR should converge and return scores for reachable nodes."""
        index = _build_simple_graph()
        ppr = TypedWeightedPPR(
            index,
            type_weights={"related_to": 1.0, "belongs_to": 1.0, "caused_by": 1.0},
            restart=0.15,
            max_iter=100,
            tolerance=1e-6,
        )
        seeds = {"viking://a": 1.0}
        result = ppr.run(seeds)

        # Seed node should have the highest score
        assert "viking://a" in result
        assert result["viking://a"] > 0
        # All nodes should be present
        assert "viking://b" in result
        assert "viking://c" in result
        # Scores should sum approximately to 1
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_type_weights_affect_scores(self):
        """Higher type weight should increase flow through that edge type."""
        index = _build_simple_graph()

        # Equal weights
        ppr_equal = TypedWeightedPPR(
            index,
            type_weights={"related_to": 1.0, "belongs_to": 1.0, "caused_by": 1.0},
            restart=0.15,
        )
        result_equal = ppr_equal.run({"viking://a": 1.0})

        # Boost belongs_to weight (A->C edge)
        ppr_boosted = TypedWeightedPPR(
            index,
            type_weights={"related_to": 1.0, "belongs_to": 5.0, "caused_by": 1.0},
            restart=0.15,
        )
        result_boosted = ppr_boosted.run({"viking://a": 1.0})

        # With boosted belongs_to, C should get relatively more score
        ratio_equal = result_equal["viking://c"] / result_equal["viking://b"]
        ratio_boosted = result_boosted["viking://c"] / result_boosted["viking://b"]
        assert ratio_boosted > ratio_equal

    def test_dangling_node(self):
        """Nodes with no outgoing edges should redistribute mass to seeds."""
        index = GraphIndex()
        index._nodes = {
            "viking://seed": GraphNode(uri="viking://seed"),
            "viking://sink": GraphNode(uri="viking://sink"),
        }
        index._forward_edges = {
            "viking://seed": [
                GraphEdge(from_uri="viking://seed", to_uri="viking://sink", link_type="related_to", weight=1.0),
            ],
        }
        index._reverse_edges = {
            "viking://sink": [
                GraphEdge(from_uri="viking://seed", to_uri="viking://sink", link_type="related_to", weight=1.0),
            ],
        }
        index._space_key = "test"
        index._built_at = 999999.0

        ppr = TypedWeightedPPR(
            index, type_weights={"related_to": 1.0}, restart=0.15, max_iter=100
        )
        result = ppr.run({"viking://seed": 1.0})

        assert "viking://sink" in result
        assert result["viking://sink"] > 0
        assert result["viking://seed"] > result["viking://sink"]

    def test_multiple_seeds(self):
        """Multiple seeds should spread PPR mass accordingly."""
        index = _build_simple_graph()
        ppr = TypedWeightedPPR(
            index,
            type_weights={"related_to": 1.0, "belongs_to": 1.0, "caused_by": 1.0},
            restart=0.15,
        )
        seeds = {"viking://a": 0.7, "viking://c": 0.3}
        result = ppr.run(seeds)

        total = sum(result.values())
        assert abs(total - 1.0) < 0.01
