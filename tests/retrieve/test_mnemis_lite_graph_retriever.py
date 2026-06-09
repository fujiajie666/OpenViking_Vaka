# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the experimental Mnemis-lite graph strategy."""

import pytest
from pydantic import ValidationError

from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking.retrieve.graph.mnemis_lite.query_planner import RuleBasedGraphQueryPlanner
from openviking.retrieve.graph.mnemis_lite.retriever import MnemisLiteGraphRetriever
from openviking.retrieve.graph.mnemis_lite.scorer import MnemisLiteSlotScorer
from openviking_cli.utils.config import RetrievalConfig


def _build_mnemis_index() -> GraphIndex:
    index = GraphIndex()
    index._nodes = {
        "viking://seed": GraphNode(
            uri="viking://seed",
            memory_type="events",
            category="conversation",
        ),
        "viking://direct": GraphNode(
            uri="viking://direct",
            memory_type="events",
            category="activity",
        ),
        "viking://ppr_only": GraphNode(
            uri="viking://ppr_only",
            memory_type="events",
            category="activity",
        ),
    }
    edge_1 = GraphEdge(
        from_uri="viking://seed",
        to_uri="viking://direct",
        link_type="related_to",
        weight=0.9,
        description="Dave participated in activities with friends.",
    )
    edge_2 = GraphEdge(
        from_uri="viking://direct",
        to_uri="viking://ppr_only",
        link_type="related_to",
        weight=0.8,
        description="Dave joined another friend activity.",
    )
    index._forward_edges = {
        "viking://seed": [edge_1],
        "viking://direct": [edge_2],
    }
    index._reverse_edges = {
        "viking://direct": [edge_1],
        "viking://ppr_only": [edge_2],
    }
    return index


def test_retrieval_config_defaults_to_evidence_graph_strategy():
    assert RetrievalConfig().graph_strategy == "evidence_graph"
    assert RetrievalConfig().graph_edge_selector == "embedding"
    assert RetrievalConfig().graph_intent_source == "fallback"
    assert RetrievalConfig(graph_strategy="mnemis_lite").graph_strategy == "mnemis_lite"


def test_retrieval_config_rejects_unknown_graph_strategy():
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_strategy="not_a_strategy")


def test_retrieval_config_rejects_unknown_graph_intent_source():
    with pytest.raises(ValidationError):
        RetrievalConfig(graph_intent_source="not_a_source")


def test_query_planner_uses_rule_based_slots_without_llm():
    plan = RuleBasedGraphQueryPlanner().plan(
        "When did Calvin and Frank Ocean start collaborating?"
    )

    assert plan.query_type == "time"
    assert plan.required_slots == {"date"}
    assert "calvin" in plan.anchors
    assert "frank" in plan.anchors


def test_query_planner_separates_answer_type_from_constraints():
    planner = RuleBasedGraphQueryPlanner()
    count_plan = planner.plan(
        "After how many weeks did Tim reconnect with the fellow Harry Potter fan?"
    )
    activity_plan = planner.plan(
        "What activity does Dave find fulfilling, similar to Calvin's passion for music festivals?"
    )

    assert count_plan.query_type == "count"
    assert count_plan.constraints == {"temporal"}
    assert activity_plan.query_type == "attribute"
    assert activity_plan.constraints == set()


def test_query_planner_treats_recommendations_as_coverage_query():
    plan = RuleBasedGraphQueryPlanner().plan(
        "What recommendations has Nate received from Joanna?"
    )

    assert plan.query_type == "list_or_set"
    assert plan.coverage_mode is True
    assert "recommendations" in plan.anchors
    assert "nate" in plan.anchors
    assert "joanna" in plan.anchors


def test_mnemis_lite_rejects_time_candidate_without_date_slot():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite"),
    )
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": "viking://direct",
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": 1.0,
            "memory_type": "events",
            "abstract": "Andrew described the autumn colors as beautiful.",
        },
    ]
    retriever._current_plan = RuleBasedGraphQueryPlanner().plan(
        "When did Andrew describe autumn colors as beautiful?"
    )

    scored = retriever._score_graph_candidates(
        candidates,
        query_text="When did Andrew describe autumn colors as beautiful?",
        limit=2,
    )
    by_uri = {candidate["uri"]: candidate for candidate in scored}

    assert by_uri["viking://direct"]["_graph_accepted"] is False
    assert "time_requires_date" in by_uri["viking://direct"]["_graph_accept_reason"]


def test_mnemis_lite_accepts_time_candidate_with_date_slot():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite"),
    )
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": "viking://direct",
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": 1.0,
            "memory_type": "events",
            "abstract": "On 2023-10-28, Andrew described the autumn colors as beautiful.",
        },
    ]
    retriever._current_plan = RuleBasedGraphQueryPlanner().plan(
        "When did Andrew describe autumn colors as beautiful?"
    )

    scored = retriever._score_graph_candidates(
        candidates,
        query_text="When did Andrew describe autumn colors as beautiful?",
        limit=2,
    )
    by_uri = {candidate["uri"]: candidate for candidate in scored}

    assert by_uri["viking://direct"]["_graph_accepted"] is True
    assert by_uri["viking://direct"]["_graph_accept_reason"] == "slot:time:date_present"


def test_mnemis_lite_does_not_accept_time_slot_from_uri_date_only():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite"),
    )
    graph_uri = "viking://user/u/memories/events/2023/10/28/autumn_walk.md"
    candidates = [
        {"uri": "viking://top", "_final_score": 0.9, "_score": 0.9},
        {"uri": "viking://weak", "_final_score": 0.6, "_score": 0.6},
        {
            "uri": graph_uri,
            "_final_score": 0.0,
            "_score": 0.0,
            "_from_graph": True,
            "_ppr_score": 1.0,
            "_graph_support": 1.0,
            "_norm_graph_support": 1.0,
            "_graph_specificity": 1.0,
            "memory_type": "events",
            "abstract": "Andrew described the autumn colors as beautiful.",
        },
    ]
    retriever._current_plan = RuleBasedGraphQueryPlanner().plan(
        "When did Andrew describe autumn colors as beautiful?"
    )

    scored = retriever._score_graph_candidates(
        candidates,
        query_text="When did Andrew describe autumn colors as beautiful?",
        limit=2,
    )
    by_uri = {candidate["uri"]: candidate for candidate in scored}

    assert by_uri[graph_uri]["_graph_accepted"] is False
    assert "time_requires_date" in by_uri[graph_uri]["_graph_accept_reason"]


@pytest.mark.asyncio
async def test_mnemis_lite_clears_query_plan_on_early_return():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite"),
    )

    result = await retriever.expand(
        candidates=[],
        ctx=None,
        limit=2,
        query_text="When did Andrew describe autumn colors as beautiful?",
    )

    assert result == []
    assert retriever._current_plan is None


def test_mnemis_lite_adds_ppr_only_graph_route_candidate():
    index = _build_mnemis_index()
    retriever = MnemisLiteGraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite"),
    )
    plan = RuleBasedGraphQueryPlanner().plan(
        "What activities has Dave participated in with his friends?"
    )
    direct_support = {"viking://direct": 0.8}
    support_scores = {"viking://direct": 0.8, "viking://ppr_only": 0.4}

    merged = retriever._merge_mnemis_candidates(
        candidates=[{"uri": "viking://seed", "_final_score": 0.9, "_score": 0.9}],
        ppr_scores={"viking://direct": 0.3, "viking://ppr_only": 0.6},
        support_scores=support_scores,
        direct_support=direct_support,
        target_dirs=None,
        level=[2],
        plan=plan,
    )
    by_uri = {candidate["uri"]: candidate for candidate in merged}

    assert "viking://direct" in by_uri
    assert "viking://ppr_only" in by_uri
    assert by_uri["viking://direct"]["_graph_route"] == "direct"
    assert by_uri["viking://ppr_only"]["_graph_route"] == "ppr_or_two_hop"


def test_mnemis_lite_overfetches_when_direct_nodes_fill_rank_window():
    index = _build_mnemis_index()
    direct_support = {}
    support_scores = {"viking://ppr_only": 0.1}
    ppr_scores = {"viking://ppr_only": 0.1}
    for idx in range(10):
        uri = f"viking://direct_{idx}"
        index._nodes[uri] = GraphNode(uri=uri, memory_type="events", category="activity")
        direct_support[uri] = 1.0 - idx * 0.01
        support_scores[uri] = direct_support[uri]
        ppr_scores[uri] = direct_support[uri]

    retriever = MnemisLiteGraphRetriever(
        index,
        RetrievalConfig(
            graph_alpha=0.4,
            graph_strategy="mnemis_lite",
            graph_expansion_topk=5,
        ),
    )
    plan = RuleBasedGraphQueryPlanner().plan(
        "What activities has Dave participated in with his friends?"
    )

    merged = retriever._merge_mnemis_candidates(
        candidates=[{"uri": "viking://seed", "_final_score": 0.9, "_score": 0.9}],
        ppr_scores=ppr_scores,
        support_scores=support_scores,
        direct_support=direct_support,
        target_dirs=None,
        level=[2],
        plan=plan,
    )
    by_uri = {candidate["uri"]: candidate for candidate in merged}

    assert "viking://ppr_only" in by_uri
    assert by_uri["viking://ppr_only"]["_graph_route"] == "ppr_or_two_hop"


def test_mnemis_lite_coverage_selection_diversifies_graph_groups():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite", graph_expansion_topk=5),
    )
    retriever._current_plan = RuleBasedGraphQueryPlanner().plan(
        "What activities has Dave participated in with his friends?"
    )
    candidates = [
        {"uri": "viking://semantic", "_final_score": 1.0},
        {
            "uri": "viking://graph_a",
            "_final_score": 0.9,
            "_from_graph": True,
            "_graph_coverage_group": "same",
        },
        {
            "uri": "viking://graph_b",
            "_final_score": 0.8,
            "_from_graph": True,
            "_graph_coverage_group": "same",
        },
        {
            "uri": "viking://graph_c",
            "_final_score": 0.7,
            "_from_graph": True,
            "_graph_coverage_group": "different",
        },
    ]

    selected = retriever._select_expanded_candidates(candidates, limit=1)

    assert [candidate["uri"] for candidate in selected[:3]] == [
        "viking://semantic",
        "viking://graph_a",
        "viking://graph_c",
    ]


def test_mnemis_lite_selection_keeps_direct_route_before_ppr_route():
    retriever = MnemisLiteGraphRetriever(
        _build_mnemis_index(),
        RetrievalConfig(graph_alpha=0.4, graph_strategy="mnemis_lite", graph_expansion_topk=5),
    )
    retriever._current_plan = RuleBasedGraphQueryPlanner().plan(
        "What activities has Dave participated in with his friends?"
    )
    candidates = [
        {"uri": "viking://semantic", "_final_score": 1.0},
        {
            "uri": "viking://ppr",
            "_final_score": 0.95,
            "_from_graph": True,
            "_graph_route": "ppr_or_two_hop",
            "_graph_coverage_group": "ppr",
        },
        {
            "uri": "viking://direct",
            "_final_score": 0.5,
            "_from_graph": True,
            "_graph_route": "direct",
            "_graph_coverage_group": "direct",
        },
    ]

    selected = retriever._select_expanded_candidates(candidates, limit=1)

    assert [candidate["uri"] for candidate in selected[:3]] == [
        "viking://semantic",
        "viking://direct",
        "viking://ppr",
    ]


def test_mnemis_lite_place_slot_rejects_non_place_entity_anchor():
    plan = RuleBasedGraphQueryPlanner().plan("Which cities has John been to?")
    slot = MnemisLiteSlotScorer().slot_match(
        {
            "uri": "viking://user/u/memories/entities/organization/minnesota_wolves.md",
            "memory_type": "entities",
            "category": "organization",
            "abstract": "John discussed the Minnesota Wolves with friends.",
        },
        plan,
    )

    assert slot.matched is False
    assert slot.reason == "slot_mismatch:place_requires_place_signal"


def test_mnemis_lite_place_slot_accepts_place_like_entity():
    plan = RuleBasedGraphQueryPlanner().plan("Where has Maria made friends?")
    slot = MnemisLiteSlotScorer().slot_match(
        {
            "uri": "viking://user/u/memories/entities/organization/homeless_shelter.md",
            "memory_type": "entities",
            "category": "organization",
            "abstract": "Maria met other volunteers there.",
        },
        plan,
    )

    assert slot.matched is True
    assert slot.reason == "slot:place:entity_anchor"


def test_mnemis_lite_list_slot_rejects_entity_with_only_person_anchor():
    plan = RuleBasedGraphQueryPlanner().plan(
        "Which outdoor gear company likely signed up John for an endorsement deal?"
    )
    slot = MnemisLiteSlotScorer().slot_match(
        {
            "uri": "viking://user/u/memories/entities/object/john_signed_basketball.md",
            "memory_type": "entities",
            "category": "object",
            "abstract": "John keeps a signed basketball from a game.",
        },
        plan,
    )

    assert slot.matched is False
    assert slot.reason == "slot_mismatch:list_or_set_requires_item"


def test_mnemis_lite_list_slot_accepts_entity_matching_query_target():
    plan = RuleBasedGraphQueryPlanner().plan("What books has Tim read?")
    slot = MnemisLiteSlotScorer().slot_match(
        {
            "uri": "viking://user/u/memories/entities/book/the_hobbit.md",
            "memory_type": "entities",
            "category": "book",
            "abstract": "Tim enjoyed this fantasy story.",
        },
        plan,
    )

    assert slot.matched is True
    assert slot.reason == "slot:list_or_set:coverage_item"
