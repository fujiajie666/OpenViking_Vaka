from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openviking.retrieve.graph import evidence_graph as evidence_graph_module
from openviking.retrieve.graph.evidence_graph import (
    EmbeddingEdgeSelector,
    EvidenceGraphRetriever,
    EvidenceQueryPlanner,
    LLMEdgeSelector,
)
from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking_cli.utils.config import RetrievalConfig


def _index_with_edges(edges):
    index = GraphIndex()
    uris = {edge.from_uri for edge in edges} | {edge.to_uri for edge in edges}
    index._nodes = {
        uri: GraphNode(
            uri=uri,
            memory_type="profile" if uri.endswith("/profile.md") else "events",
        )
        for uri in uris
    }
    index._forward_edges = defaultdict(list)
    index._reverse_edges = defaultdict(list)
    for edge in edges:
        index._forward_edges[edge.from_uri].append(edge)
        index._reverse_edges[edge.to_uri].append(edge)
    return index


@pytest.mark.asyncio
async def test_embedding_edge_selector_uses_query_vector(monkeypatch):
    london = GraphEdge(
        from_uri="viking://user/a/memories/entities/person/tim.md",
        to_uri="viking://user/a/memories/entities/location/london.md",
        link_type="evidence_for",
        weight=0.9,
        subject="Tim",
        relation_slot="visited_location",
        answer_value=["London"],
        evidence_role="direct",
        source_span="Tim went to a place in London.",
    )
    book = GraphEdge(
        from_uri="viking://user/a/memories/entities/person/tim.md",
        to_uri="viking://user/a/memories/events/book.md",
        link_type="evidence_for",
        weight=0.9,
        subject="Tim",
        relation_slot="read_book",
        answer_value=["Dune"],
        evidence_role="direct",
        source_span="Tim read Dune.",
    )
    index = _index_with_edges([book, london])

    async def fake_embed_compat(embedder, text, is_query=False):
        return SimpleNamespace(
            dense_vector=[1.0, 0.0] if "London" in text else [0.0, 1.0],
            sparse_vector=None,
        )

    monkeypatch.setattr(evidence_graph_module, "embed_compat", fake_embed_compat)
    plan = await EvidenceQueryPlanner().plan("Which geographical locations has Tim been to?")
    selector = EmbeddingEdgeSelector(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="evidence_graph"),
        embedder=object(),
    )

    selected = await selector.select(
        "Which geographical locations has Tim been to?",
        query_vector=[1.0, 0.0],
        plan=plan,
        edges=[book, london],
        semantic_candidates=[],
    )

    assert selected[0].edge.to_uri == london.to_uri


@pytest.mark.asyncio
async def test_llm_edge_selector_falls_back_on_invalid_response(monkeypatch):
    edge = GraphEdge(
        from_uri="viking://user/a/memories/entities/person/tim.md",
        to_uri="viking://user/a/memories/entities/location/london.md",
        link_type="evidence_for",
        weight=0.9,
        subject="Tim",
        relation_slot="visited_location",
        answer_value=["London"],
        evidence_role="direct",
        source_span="Tim went to a place in London.",
    )
    index = _index_with_edges([edge])

    class FakeVLM:
        async def get_completion_async(self, prompt):
            return "not json"

    import openviking_cli.utils.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_openviking_config",
        lambda: SimpleNamespace(vlm=FakeVLM()),
    )
    selector = LLMEdgeSelector(
        index,
        RetrievalConfig(
            graph_alpha=0.4,
            graph_strategy="evidence_graph",
            graph_edge_selector="llm",
        ),
    )
    plan = await EvidenceQueryPlanner().plan("Which geographical locations has Tim been to?")

    selected = await selector.select(
        "Which geographical locations has Tim been to?",
        query_vector=None,
        plan=plan,
        edges=[edge],
        semantic_candidates=[],
    )

    assert selected
    assert "fallback_llm_error" in selected[0].reason


@pytest.mark.asyncio
async def test_evidence_graph_rejects_subject_mismatch():
    seed = "viking://user/a/memories/entities/person/Melanie.md"
    caroline_book = "viking://user/a/memories/events/2023/01/01/caroline_book.md"
    index = _index_with_edges(
        [
            GraphEdge(
                from_uri=seed,
                to_uri=caroline_book,
                link_type="evidence_for",
                weight=0.98,
                subject="Caroline",
                relation_slot="read_book",
                answer_value=["The Alchemist"],
                evidence_role="list_member",
                source_span="Caroline read The Alchemist.",
            )
        ]
    )
    retriever = EvidenceGraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="evidence_graph"),
    )

    result = await retriever.expand(
        [{"uri": seed, "_score": 1.0, "_final_score": 1.0}],
        ctx=Mock(),
        limit=5,
        query_text="What books has Melanie read?",
    )

    assert [candidate["uri"] for candidate in result] == [seed]


@pytest.mark.asyncio
async def test_evidence_graph_rejects_non_evidence_link_type():
    seed = "viking://user/a/memories/entities/person/John.md"
    book = "viking://user/a/memories/events/2023/01/01/alchemist.md"
    index = _index_with_edges(
        [
            GraphEdge(
                from_uri=seed,
                to_uri=book,
                link_type="related_to",
                weight=0.99,
                subject="John",
                relation_slot="read_book",
                answer_value=["The Alchemist"],
                evidence_role="direct",
                source_span="John read The Alchemist.",
            )
        ]
    )
    retriever = EvidenceGraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="evidence_graph"),
    )

    result = await retriever.expand(
        [{"uri": seed, "_score": 1.0, "_final_score": 1.0}],
        ctx=Mock(),
        limit=5,
        query_text="What books has John read?",
    )

    assert [candidate["uri"] for candidate in result] == [seed]


@pytest.mark.asyncio
async def test_evidence_graph_completes_same_aggregation_key_members():
    seed = "viking://user/a/memories/entities/person/John.md"
    book_1 = "viking://user/a/memories/events/2023/01/01/alchemist.md"
    book_2 = "viking://user/a/memories/events/2023/02/01/dune.md"
    index = _index_with_edges(
        [
            GraphEdge(
                from_uri=seed,
                to_uri=book_1,
                link_type="evidence_for",
                weight=0.98,
                subject="John",
                relation_slot="read_book",
                answer_value=["The Alchemist"],
                evidence_role="list_member",
                source_span="John read The Alchemist.",
            ),
            GraphEdge(
                from_uri=seed,
                to_uri=book_2,
                link_type="evidence_for",
                weight=0.94,
                subject="John",
                relation_slot="read_book",
                answer_value=["Dune"],
                evidence_role="list_member",
                source_span="John read Dune.",
            ),
        ]
    )
    retriever = EvidenceGraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="evidence_graph"),
    )

    result = await retriever.expand(
        [{"uri": seed, "_score": 1.0, "_final_score": 1.0}],
        ctx=Mock(),
        limit=5,
        query_text="What books has John read?",
    )

    uris = [candidate["uri"] for candidate in result]
    assert book_1 in uris
    assert book_2 in uris


@pytest.mark.asyncio
async def test_evidence_graph_completes_place_members_across_relation_slots():
    seed = "viking://user/a/memories/entities/person/tim.md"
    london = "viking://user/a/memories/entities/location/london.md"
    california = "viking://user/a/memories/entities/location/california.md"
    smoky = "viking://user/a/memories/entities/location/smoky_mountains.md"
    index = _index_with_edges(
        [
            GraphEdge(
                from_uri=seed,
                to_uri=london,
                link_type="evidence_for",
                weight=0.9,
                subject="Tim",
                relation_slot="visited_location",
                answer_value=["Harry Potter related location in London"],
                evidence_role="direct",
                source_span="I went to a place in London a few years ago.",
            ),
            GraphEdge(
                from_uri=seed,
                to_uri=california,
                link_type="evidence_for",
                weight=0.9,
                subject="Tim",
                relation_slot="met_fan_in_location",
                answer_value=["california"],
                evidence_role="direct",
                source_span="that Harry Potter fan I met in CA",
            ),
            GraphEdge(
                from_uri=seed,
                to_uri=smoky,
                link_type="evidence_for",
                weight=0.9,
                subject="Tim",
                relation_slot="visited_place",
                answer_value=["Smoky Mountains"],
                evidence_role="direct",
                source_span="I snapped that pic on my trip to the Smoky Mountains last year.",
            ),
        ]
    )
    retriever = EvidenceGraphRetriever(
        index,
        RetrievalConfig(graph_alpha=0.4, graph_strategy="evidence_graph"),
    )

    result = await retriever.expand(
        [{"uri": seed, "_score": 1.0, "_final_score": 1.0}],
        ctx=Mock(),
        limit=5,
        query_text="Which geographical locations has Tim been to?",
    )

    uris = [candidate["uri"] for candidate in result]
    assert london in uris
    assert california in uris
    assert smoky in uris


@pytest.mark.asyncio
async def test_graph_index_reads_evidence_link_metadata(monkeypatch):
    from openviking.retrieve.graph import graph_index as graph_index_module

    from_uri = "viking://user/a/memories/profile.md"
    to_uri = "viking://user/a/memories/events/2023/01/01/london.md"
    fake_vfs = Mock()
    fake_vfs.tree = AsyncMock(
        return_value=[
            {"uri": from_uri, "rel_path": "profile.md", "isDir": False},
            {"uri": to_uri, "rel_path": "events/2023/01/01/london.md", "isDir": False},
        ]
    )
    fake_vfs.read_file = AsyncMock(
        side_effect=lambda uri, ctx=None: (
            "Profile\n\n<!-- MEMORY_FIELDS\n"
            '{"memory_type":"profile","links":[{"to_uri":"'
            + to_uri
            + '","link_type":"evidence_for","weight":0.97,"match_text":"London",'
            '"subject":"Tim","relation_slot":"visited_place",'
            '"answer_value":["London"],"evidence_role":"list_member",'
            '"source_span":"Tim visited London."}]}\n-->'
            if uri == from_uri
            else "London visit\n\n<!-- MEMORY_FIELDS\n"
            '{"memory_type":"events","links":[]}\n-->'
        )
    )
    monkeypatch.setattr(graph_index_module, "get_viking_fs", lambda: fake_vfs)

    index = GraphIndex()
    await index.build(["viking://user/a/memories"], ctx=Mock())

    edge = index.get_forward_edges(from_uri)[0]
    assert edge.link_type == "evidence_for"
    assert edge.subject == "Tim"
    assert edge.relation_slot == "visited_place"
    assert edge.answer_value == ["London"]
    assert edge.evidence_role == "list_member"
    assert edge.source_span == "Tim visited London."
