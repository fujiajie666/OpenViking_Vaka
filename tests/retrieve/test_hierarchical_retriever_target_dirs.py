# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Hierarchical retriever target_directories tests."""

import pytest

from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
from openviking.server.identity import RequestContext, Role
from openviking_cli.retrieve.types import ContextType, MatchedContext, TypedQuery
from openviking_cli.session.user_id import UserIdentifier


class DummyStorage:
    """Minimal storage stub to capture search filters."""

    def __init__(self) -> None:
        self.collection_name = "context"
        self.global_search_calls = []
        self.child_search_calls = []

    async def collection_exists_bound(self) -> bool:
        return True

    async def search_global_roots_in_tenant(
        self,
        ctx,
        query_vector=None,
        sparse_query_vector=None,
        context_type=None,
        target_directories=None,
        extra_filter=None,
        limit: int = 10,
    ):
        self.global_search_calls.append(
            {
                "ctx": ctx,
                "query_vector": query_vector,
                "sparse_query_vector": sparse_query_vector,
                "context_type": context_type,
                "target_directories": target_directories,
                "extra_filter": extra_filter,
                "limit": limit,
            }
        )
        return []

    async def search_children_in_tenant(
        self,
        ctx,
        parent_uri: str,
        query_vector=None,
        sparse_query_vector=None,
        context_type=None,
        target_directories=None,
        extra_filter=None,
        limit: int = 10,
    ):
        self.child_search_calls.append(
            {
                "ctx": ctx,
                "parent_uri": parent_uri,
                "query_vector": query_vector,
                "sparse_query_vector": sparse_query_vector,
                "context_type": context_type,
                "target_directories": target_directories,
                "extra_filter": extra_filter,
                "limit": limit,
            }
        )
        return []


@pytest.mark.asyncio
async def test_retrieve_honors_target_directories_scope_filter():
    target_uri = "viking://resources/foo"
    storage = DummyStorage()
    retriever = HierarchicalRetriever(storage=storage, embedder=None, rerank_config=None)
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)

    query = TypedQuery(
        query="test",
        context_type=ContextType.RESOURCE,
        intent="",
        target_directories=[target_uri],
    )

    result = await retriever.retrieve(query, ctx=ctx, limit=3)

    assert result.searched_directories == [target_uri]
    assert storage.global_search_calls
    assert storage.global_search_calls[0]["target_directories"] == [target_uri]
    assert storage.child_search_calls
    assert storage.child_search_calls[0]["target_directories"] == [target_uri]
    assert storage.child_search_calls[0]["parent_uri"] == target_uri


def test_graph_space_uris_prefer_target_memory_dirs_for_root_ctx():
    retriever = HierarchicalRetriever(storage=DummyStorage(), embedder=None, rerank_config=None)
    ctx = RequestContext(user=UserIdentifier("acc1", "sample_0", "agent1"), role=Role.ROOT)

    space_uris = retriever._get_graph_space_uris(
        ctx=ctx,
        target_dirs=["viking://user/sample_0/memories/events"],
        candidates=[],
    )

    assert space_uris == ["viking://user/sample_0/memories"]


def test_graph_space_uris_fall_back_to_candidate_memory_space_for_root_ctx():
    retriever = HierarchicalRetriever(storage=DummyStorage(), embedder=None, rerank_config=None)
    ctx = RequestContext(user=UserIdentifier("acc1", "sample_0", "agent1"), role=Role.ROOT)

    space_uris = retriever._get_graph_space_uris(
        ctx=ctx,
        target_dirs=[],
        candidates=[
            {"uri": "viking://agent/agent1/memories/experiences/test.md"},
            {"uri": "viking://resources/docs/test.md"},
        ],
    )

    assert space_uris == ["viking://agent/agent1/memories"]


def test_graph_final_contexts_preserve_semantic_topk_and_append_graph_auxiliary():
    retriever = HierarchicalRetriever(storage=DummyStorage(), embedder=None, rerank_config=None)
    semantic = [
        MatchedContext(
            uri=f"viking://user/u/memories/events/{idx}.md",
            context_type=ContextType.MEMORY,
            level=2,
            abstract="",
            category="",
            score=1.0 - idx * 0.1,
        )
        for idx in range(3)
    ]
    graph = [
        MatchedContext(
            uri=f"viking://user/u/memories/events/graph-{idx}.md",
            context_type=ContextType.MEMORY,
            level=2,
            abstract="",
            category="",
            score=0.99 - idx * 0.01,
            match_reason="Discovered via graph expansion",
        )
        for idx in range(4)
    ]

    selected = retriever._select_final_contexts(
        [graph[0], *semantic, *graph[1:]],
        limit=2,
        graph_expanded=True,
    )

    assert [context.uri for context in selected[:2]] == [
        "viking://user/u/memories/events/0.md",
        "viking://user/u/memories/events/1.md",
    ]
    assert [context.uri for context in selected[2:]] == [
        "viking://user/u/memories/events/graph-0.md",
        "viking://user/u/memories/events/graph-1.md",
        "viking://user/u/memories/events/graph-2.md",
    ]


@pytest.mark.asyncio
async def test_convert_to_matched_contexts_preserves_graph_candidate_order(monkeypatch):
    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.get_viking_fs",
        lambda: None,
    )
    retriever = HierarchicalRetriever(storage=DummyStorage(), embedder=None, rerank_config=None)
    ctx = RequestContext(user=UserIdentifier("acc1", "sample_0", "agent1"), role=Role.ROOT)
    candidates = [
        {
            "uri": "viking://user/u/memories/events/semantic-low.md",
            "context_type": "memory",
            "level": 2,
            "_final_score": 0.1,
        },
        {
            "uri": "viking://user/u/memories/events/semantic-high.md",
            "context_type": "memory",
            "level": 2,
            "_final_score": 0.9,
        },
        {
            "uri": "viking://user/u/memories/events/graph-first.md",
            "context_type": "memory",
            "level": 2,
            "_final_score": 0.1,
            "_from_graph": True,
        },
        {
            "uri": "viking://user/u/memories/events/graph-second.md",
            "context_type": "memory",
            "level": 2,
            "_final_score": 0.9,
            "_from_graph": True,
        },
    ]

    matched = await retriever._convert_to_matched_contexts(candidates, ctx=ctx)

    semantic = [context.uri for context in matched if not context.match_reason]
    graph = [context.uri for context in matched if context.match_reason]
    assert semantic == [
        "viking://user/u/memories/events/semantic-high.md",
        "viking://user/u/memories/events/semantic-low.md",
    ]
    assert graph == [
        "viking://user/u/memories/events/graph-first.md",
        "viking://user/u/memories/events/graph-second.md",
    ]


@pytest.mark.asyncio
async def test_graph_expand_skips_empty_candidates_before_building_index():
    retriever = HierarchicalRetriever(storage=DummyStorage(), embedder=None, rerank_config=None)
    ctx = RequestContext(user=UserIdentifier("acc1", "sample_0", "agent1"), role=Role.ROOT)

    candidates = await retriever._graph_expand(
        candidates=[],
        ctx=ctx,
        limit=3,
        target_dirs=["viking://agent/shared/memories"],
    )

    assert candidates == []
