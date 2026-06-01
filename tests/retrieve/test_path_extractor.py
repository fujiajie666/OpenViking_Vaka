# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for path extraction."""

from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking.retrieve.graph.path_extractor import PathExtractor


def _build_chain_graph() -> GraphIndex:
    """Build A -> B -> C -> D chain."""
    index = GraphIndex()
    index._nodes = {
        "viking://a": GraphNode(uri="viking://a"),
        "viking://b": GraphNode(uri="viking://b"),
        "viking://c": GraphNode(uri="viking://c"),
        "viking://d": GraphNode(uri="viking://d"),
    }
    index._forward_edges = {
        "viking://a": [GraphEdge(from_uri="viking://a", to_uri="viking://b", link_type="related_to", weight=0.8)],
        "viking://b": [GraphEdge(from_uri="viking://b", to_uri="viking://c", link_type="caused_by", weight=0.9)],
        "viking://c": [GraphEdge(from_uri="viking://c", to_uri="viking://d", link_type="derived_from", weight=0.7)],
    }
    index._reverse_edges = {
        "viking://b": [GraphEdge(from_uri="viking://a", to_uri="viking://b", link_type="related_to", weight=0.8)],
        "viking://c": [GraphEdge(from_uri="viking://b", to_uri="viking://c", link_type="caused_by", weight=0.9)],
        "viking://d": [GraphEdge(from_uri="viking://c", to_uri="viking://d", link_type="derived_from", weight=0.7)],
    }
    index._space_key = "test"
    index._built_at = 999999.0
    return index


class TestPathExtractor:
    def test_empty_seeds(self):
        index = _build_chain_graph()
        extractor = PathExtractor(index)
        result = extractor.extract({}, {"viking://d": 0.5})
        assert result == []

    def test_empty_ppr(self):
        index = _build_chain_graph()
        extractor = PathExtractor(index)
        result = extractor.extract({"viking://a": 1.0}, {})
        assert result == []

    def test_path_from_seed_to_ppr_node(self):
        """Should find paths connecting seeds to high-PPR nodes."""
        index = _build_chain_graph()
        extractor = PathExtractor(index, max_paths=3)
        seeds = {"viking://a": 1.0}
        ppr_scores = {"viking://d": 0.5, "viking://c": 0.3, "viking://b": 0.2}

        paths = extractor.extract(seeds, ppr_scores, top_k=5)

        assert len(paths) >= 1
        # Each path should contain at least one seed node
        for path in paths:
            assert any(n in seeds for n in path.nodes)
            assert path.path_score > 0

    def test_max_paths_limit(self):
        index = _build_chain_graph()
        extractor = PathExtractor(index, max_paths=1)
        seeds = {"viking://a": 1.0}
        ppr_scores = {"viking://d": 0.5, "viking://c": 0.3, "viking://b": 0.2}

        paths = extractor.extract(seeds, ppr_scores, top_k=5)
        assert len(paths) <= 1

    def test_no_path_when_disconnected(self):
        """Should return no paths when seeds and PPR nodes are disconnected."""
        index = GraphIndex()
        index._nodes = {
            "viking://a": GraphNode(uri="viking://a"),
            "viking://d": GraphNode(uri="viking://d"),
        }
        index._forward_edges = {}
        index._reverse_edges = {}
        index._space_key = "test"
        index._built_at = 999999.0

        extractor = PathExtractor(index, max_paths=3)
        seeds = {"viking://a": 1.0}
        ppr_scores = {"viking://d": 0.5}
        paths = extractor.extract(seeds, ppr_scores, top_k=5)
        assert len(paths) == 0

    def test_tied_heap_items_do_not_compare_edges(self):
        """Tied queue items should not require GraphEdge to be orderable."""
        edge_a = GraphEdge(
            from_uri="viking://a",
            to_uri="viking://b",
            link_type="related_to",
            weight=0.5,
        )
        edge_b = GraphEdge(
            from_uri="viking://a",
            to_uri="viking://b",
            link_type="derived_from",
            weight=0.5,
        )
        index = GraphIndex()
        index._nodes = {
            "viking://a": GraphNode(uri="viking://a"),
            "viking://b": GraphNode(uri="viking://b"),
        }
        index._forward_edges = {"viking://a": [edge_a, edge_b]}
        index._reverse_edges = {"viking://b": [edge_a, edge_b]}
        index._space_key = "test"
        index._built_at = 999999.0

        extractor = PathExtractor(index, max_paths=3)
        paths = extractor.extract({"viking://a": 1.0}, {"viking://b": 0.5}, top_k=5)

        assert len(paths) == 1
        assert paths[0].nodes == ["viking://b", "viking://a"]

    def test_path_length_limit(self):
        """Paths should not exceed max_path_length."""
        index = _build_chain_graph()
        extractor = PathExtractor(index, max_paths=3, max_path_length=2)
        seeds = {"viking://a": 1.0}
        ppr_scores = {"viking://d": 0.5}

        paths = extractor.extract(seeds, ppr_scores, top_k=5)
        # D is 3 hops from A, so with max_path_length=2, no path should reach D
        for path in paths:
            assert len(path.nodes) <= 2
