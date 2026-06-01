# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for evidence-gated graph retrieval."""

from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking.retrieve.graph.graph_retriever import GraphRetriever
from openviking_cli.utils.config import RetrievalConfig


def _build_test_index() -> GraphIndex:
    index = GraphIndex()
    index._nodes = {
        "viking://seed1": GraphNode(
            uri="viking://seed1",
            memory_type="events",
            category="conversation",
        ),
        "viking://node2": GraphNode(
            uri="viking://node2",
            memory_type="events",
            category="activity",
        ),
        "viking://node3": GraphNode(
            uri="viking://node3",
            memory_type="entities",
            category="thing",
        ),
        "viking://summary1": GraphNode(uri="viking://summary1", is_summary=True),
    }
    index._forward_edges = {
        "viking://seed1": [
            GraphEdge(
                from_uri="viking://seed1",
                to_uri="viking://node2",
                link_type="related_to",
                weight=0.9,
                description="Andrew described autumn colors as beautiful.",
            ),
        ],
        "viking://node2": [
            GraphEdge(
                from_uri="viking://node2",
                to_uri="viking://node3",
                link_type="related_to",
                weight=0.8,
            ),
        ],
    }
    index._reverse_edges = {
        "viking://node2": [
            GraphEdge(
                from_uri="viking://seed1",
                to_uri="viking://node2",
                link_type="related_to",
                weight=0.9,
                description="Andrew described autumn colors as beautiful.",
            ),
        ],
        "viking://node3": [
            GraphEdge(
                from_uri="viking://node2",
                to_uri="viking://node3",
                link_type="related_to",
                weight=0.8,
            ),
        ],
    }
    return index


def test_build_seeds_normalizes_semantic_candidates():
    retriever = GraphRetriever(
        _build_test_index(),
        RetrievalConfig(graph_alpha=0.4, graph_seed_include_summaries=False),
    )

    seeds = retriever._build_seeds(
        [{"uri": "viking://seed1", "_final_score": 0.8, "_score": 0.8}]
    )

    assert seeds == {"viking://seed1": 1.0}


def test_build_seeds_include_summaries_only_when_enabled():
    index = _build_test_index()
    candidates = [{"uri": "viking://seed1", "_final_score": 0.8}]

    disabled = GraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_seed_include_summaries=False),
    )
    enabled = GraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_seed_include_summaries=True),
    )

    assert "viking://summary1" not in disabled._build_seeds(candidates)
    assert "viking://summary1" in enabled._build_seeds(candidates)


def test_direct_seed_support_scores_one_hop_neighbors():
    retriever = GraphRetriever(_build_test_index(), RetrievalConfig(graph_alpha=0.4))

    support = retriever._compute_direct_seed_support({"viking://seed1": 1.0})

    assert support["viking://node2"] > 0
    assert "viking://node3" not in support


def test_merge_adds_only_directly_supported_graph_nodes():
    index = _build_test_index()
    retriever = GraphRetriever(index, RetrievalConfig(graph_alpha=0.4))

    result = retriever._merge_candidates_with_ppr(
        candidates=[],
        ppr_scores={"viking://node3": 0.9, "viking://node2": 0.1},
        support_scores={"viking://node2": 0.8},
    )

    by_uri = {candidate["uri"]: candidate for candidate in result}
    assert "viking://node2" in by_uri
    assert "viking://node3" not in by_uri
    assert by_uri["viking://node2"]["_from_graph"] is True
    assert by_uri["viking://node2"]["_graph_support"] == 0.8


def test_merge_filters_graph_nodes_by_scope_level_and_summary():
    index = _build_test_index()
    in_scope = "viking://user/u/memories/events/in_scope.md"
    out_scope = "viking://user/u/memories/preferences/out_scope.md"
    index._nodes[in_scope] = GraphNode(uri=in_scope)
    index._nodes[out_scope] = GraphNode(uri=out_scope)
    retriever = GraphRetriever(index, RetrievalConfig(graph_alpha=0.4))

    scoped = retriever._merge_candidates_with_ppr(
        candidates=[],
        ppr_scores={in_scope: 0.7, out_scope: 0.6},
        support_scores={in_scope: 0.7, out_scope: 0.6},
        target_dirs=["viking://user/u/memories/events"],
        level=[2],
    )
    blocked_by_level = retriever._merge_candidates_with_ppr(
        candidates=[],
        ppr_scores={in_scope: 0.7},
        support_scores={in_scope: 0.7},
        level=[0],
    )
    summary = retriever._merge_candidates_with_ppr(
        candidates=[],
        ppr_scores={"viking://summary1": 0.7},
        support_scores={"viking://summary1": 0.7},
    )

    assert [candidate["uri"] for candidate in scoped] == [in_scope]
    assert blocked_by_level == []
    assert summary == []


def test_score_graph_candidates_keeps_semantic_scores_unchanged():
    retriever = GraphRetriever(_build_test_index(), RetrievalConfig(graph_alpha=0.4))
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
    ]

    result = retriever._score_graph_candidates(
        candidates,
        query_text="What aspect of autumn did Andrew find beautiful?",
        limit=2,
    )

    assert [candidate["_final_score"] for candidate in result] == [0.9, 0.6]


def test_unaligned_graph_candidate_is_filtered():
    retriever = GraphRetriever(_build_test_index(), RetrievalConfig(graph_alpha=0.4))
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": "viking://unrelated_graph",
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": 1.0,
            "abstract": "A weekend meal plan and grocery reminder.",
        },
    ]

    scored = retriever._score_graph_candidates(
        candidates,
        query_text="What aspect of autumn did Andrew find beautiful?",
        limit=2,
    )
    filtered = retriever._filter_unaccepted_graph_nodes(scored)

    assert [candidate["uri"] for candidate in filtered] == [
        "viking://top",
        "viking://weak",
    ]


def test_query_aligned_graph_candidate_can_enter_below_semantic_ceiling():
    retriever = GraphRetriever(_build_test_index(), RetrievalConfig(graph_alpha=0.4))
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": "viking://node2",
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": 1.0,
            "abstract": "Andrew described the autumn colors as beautiful.",
        },
    ]

    result = retriever._score_graph_candidates(
        candidates,
        query_text="What aspect of autumn did Andrew find beautiful?",
        limit=2,
    )
    by_uri = {candidate["uri"]: candidate for candidate in result}

    assert by_uri["viking://node2"]["_graph_accepted"] is True
    assert by_uri["viking://node2"]["_final_score"] > by_uri["viking://weak"][
        "_final_score"
    ]
    assert by_uri["viking://node2"]["_final_score"] < by_uri["viking://top"][
        "_final_score"
    ]


def test_degree_penalty_prefers_specific_node_over_hub():
    index = _build_test_index()
    specific = "viking://specific_event"
    hub = "viking://hub_entity"
    index._nodes[specific] = GraphNode(uri=specific)
    index._nodes[hub] = GraphNode(uri=hub)
    index._reverse_edges[hub] = [
        GraphEdge(
            from_uri=f"viking://source_{idx}",
            to_uri=hub,
            link_type="related_to",
            weight=1.0,
        )
        for idx in range(80)
    ]
    retriever = GraphRetriever(index, RetrievalConfig(graph_alpha=0.4))
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": specific,
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": retriever._degree_specificity(0),
            "abstract": "Andrew described autumn colors as beautiful.",
        },
        {
            "uri": hub,
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": retriever._degree_specificity(80),
            "abstract": "Andrew described autumn colors as beautiful.",
        },
    ]

    result = retriever._score_graph_candidates(
        candidates,
        query_text="What aspect of autumn did Andrew find beautiful?",
        limit=2,
    )
    by_uri = {candidate["uri"]: candidate for candidate in result}

    assert by_uri[specific]["_final_score"] > by_uri[hub]["_final_score"]


def test_select_expanded_candidates_preserves_semantic_then_appends_graph():
    retriever = GraphRetriever(
        _build_test_index(),
        RetrievalConfig(graph_alpha=0.4, graph_expansion_topk=5),
    )
    candidates = [
        {"uri": "viking://graph_b", "_final_score": 0.8, "_from_graph": True},
        {"uri": "viking://semantic_low", "_final_score": 0.5},
        {"uri": "viking://graph_a", "_final_score": 0.9, "_from_graph": True},
        {"uri": "viking://semantic_high", "_final_score": 1.0},
    ]

    selected = retriever._select_expanded_candidates(candidates, limit=1)

    assert [candidate["uri"] for candidate in selected] == [
        "viking://semantic_high",
        "viking://semantic_low",
        "viking://graph_a",
        "viking://graph_b",
    ]
