# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Hierarchical retriever for OpenViking.

Implements directory-based hierarchical retrieval with recursive search
and rerank-based relevance scoring.
"""

import asyncio
import heapq
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openviking.core.namespace import canonical_agent_root, canonical_user_root
from openviking.models.embedder.base import EmbedResult, embed_compat
from openviking.models.rerank import RerankClient
from openviking.retrieve.memory_lifecycle import hotness_score
from openviking.retrieve.retrieval_stats import get_stats_collector
from openviking.server.identity import RequestContext, Role
from openviking.storage import VikingDBManager, VikingDBManagerProxy
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry import get_current_telemetry
from openviking.utils.time_utils import parse_iso_datetime
from openviking_cli.retrieve.types import (
    ContextType,
    MatchedContext,
    QueryResult,
    RelatedContext,
    TypedQuery,
)
from openviking_cli.utils.config import RerankConfig, RetrievalConfig
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverMode(str):
    THINKING = "thinking"
    QUICK = "quick"


class HierarchicalRetriever:
    """Hierarchical retriever with dense and sparse vector support."""

    MAX_CONVERGENCE_ROUNDS = 3  # Stop after multiple rounds with unchanged topk
    MAX_RELATIONS = 5  # Maximum relations per resource
    GRAPH_AUXILIARY_RESULT_LIMIT = 3  # Graph recalls are appended, not competing.
    GRAPH_COVERAGE_AUXILIARY_RESULT_LIMIT = 8
    GRAPH_COVERAGE_QUERY_TYPES = {"count", "list_or_set", "multi_hop"}
    DIRECTORY_DOMINANCE_RATIO = 1.2  # Directory score must exceed max child score
    GLOBAL_SEARCH_TOPK = 10  # Global retrieval count (more candidates = better rerank precision)
    MAX_PARALLEL_CHILD_SEARCHES = 4  # Limit per-request fan-out against remote vector stores
    LEVEL_URI_SUFFIX = {0: ".abstract.md", 1: ".overview.md"}

    def __init__(
        self,
        storage: VikingDBManager,
        embedder: Optional[Any],
        rerank_config: Optional[RerankConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        memory_config: Optional[Any] = None,
    ):
        """Initialize hierarchical retriever with rerank_config.

        Args:
            storage: VikingVectorIndexBackend instance
            embedder: Embedder instance (supports dense/sparse/hybrid)
            rerank_config: Rerank configuration (optional, will fallback to vector search only)
            retrieval_config: Retrieval ranking configuration.
            memory_config: Memory configuration (used for link_enabled gate).
        """
        self.vector_store = storage
        self.embedder = embedder
        self.rerank_config = rerank_config
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.hotness_alpha = self.retrieval_config.hotness_alpha
        self.score_propagation_alpha = self.retrieval_config.score_propagation_alpha
        self._link_enabled = memory_config.link_enabled if memory_config else False

        # Use rerank threshold if available, otherwise use a default
        self.threshold = rerank_config.threshold if rerank_config else 0

        # Initialize rerank client — all providers go through unified dispatch
        if rerank_config and rerank_config.is_available():
            self._rerank_client = RerankClient.from_config(rerank_config)
            provider = rerank_config._effective_provider()
            logger.info(
                f"[HierarchicalRetriever] Rerank enabled (provider={provider}), threshold={self.threshold}"
            )
        else:
            self._rerank_client = None
            logger.info(
                f"[HierarchicalRetriever] Rerank not configured, using vector search only with threshold={self.threshold}"
            )

    async def retrieve(
        self,
        query: TypedQuery,
        ctx: RequestContext,
        limit: int = 5,
        mode: str = RetrieverMode.THINKING,
        score_threshold: Optional[float] = None,
        score_gte: bool = False,
        scope_dsl: Optional[Dict[str, Any]] = None,
        level: Optional[List[int]] = None,
    ) -> QueryResult:
        """
        Execute hierarchical retrieval.

        Args:
            user: User ID (for permission filtering)
            score_threshold: Custom score threshold (overrides config)
            score_gte: True uses >=, False uses >
            grep_patterns: Keyword match pattern list
            scope_dsl: Additional scope constraints passed from public find/search filter
        """
        t0 = time.monotonic()
        # Use custom threshold or default threshold
        effective_threshold = score_threshold if score_threshold is not None else self.threshold

        # 创建 proxy 包装器，绑定当前 ctx
        vector_proxy = VikingDBManagerProxy(self.vector_store, ctx)

        target_dirs = [d for d in (query.target_directories or []) if d]

        if not await vector_proxy.collection_exists_bound():
            logger.warning(
                "[RecursiveSearch] Collection %s does not exist",
                vector_proxy.collection_name,
            )
            return QueryResult(
                query=query,
                matched_contexts=[],
                searched_directories=[],
            )

        # Generate query vectors once to avoid duplicate embedding calls
        query_vector = None
        sparse_query_vector = None
        if self.embedder:
            result: EmbedResult = await embed_compat(self.embedder, query.query, is_query=True)
            query_vector = result.dense_vector
            sparse_query_vector = result.sparse_vector

        # Step 1: Determine starting directories based on target_directories or context_type
        if target_dirs:
            root_uris = target_dirs
        else:
            root_uris = self._get_root_uris_for_type(query.context_type, ctx=ctx)

        # Step 2: Global vector search to supplement starting points
        global_results = await self._global_vector_search(
            vector_proxy=vector_proxy,
            query_vector=query_vector,
            sparse_query_vector=sparse_query_vector,
            context_type=query.context_type.value if query.context_type else None,
            target_dirs=target_dirs,
            scope_dsl=scope_dsl,
            limit=max(limit, self.GLOBAL_SEARCH_TOPK),
        )

        # Debug: Print all URIs in global_results
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[retrieve] target_dirs: {target_dirs}")
            logger.debug(f"[retrieve] root_uris: {root_uris}")
            logger.debug(f"[retrieve] scope_dsl: {scope_dsl}")
            logger.debug(
                f"[retrieve] Step 2 completed, global_results contains {len(global_results)} items:"
            )
            for i, r in enumerate(global_results):
                uri = r.get("uri", "UNKNOWN_URI")
                score = r.get("_score", 0.0)
                result_level = r.get("level", "UNKNOWN_LEVEL")
                account_id = r.get("account_id", "UNKNOWN_ACCOUNT_ID")
                logger.debug(
                    f"  [{i}] URI: {uri}, score: {score:.4f}, level: {result_level}, account_id: {account_id}"
                )

        # Step 3: Merge starting points
        starting_points = self._merge_starting_points(
            query.query,
            root_uris,
            global_results,
            mode=mode,
        )

        # Add global hits to the result pool only when they match the requested level.
        if level is not None:
            initial_candidates = [r for r in global_results if r.get("level", 2) in level]
        else:
            initial_candidates = [r for r in global_results if r.get("level", 2) == 2]

        initial_candidates = self._prepare_initial_candidates(
            query.query,
            initial_candidates,
            mode=mode,
        )

        # Step 4: Recursive search
        candidates = await self._recursive_search(
            vector_proxy=vector_proxy,
            query=query.query,
            query_vector=query_vector,
            sparse_query_vector=sparse_query_vector,
            starting_points=starting_points,
            limit=limit,
            mode=mode,
            threshold=effective_threshold,
            score_gte=score_gte,
            context_type=query.context_type.value if query.context_type else None,
            target_dirs=target_dirs,
            scope_dsl=scope_dsl,
            initial_candidates=initial_candidates,
            level=level,
        )

        graph_expanded = False
        coverage_graph_expanded = False

        logger.info(f"target_dirs: {target_dirs}, level: {level}, query={query.query}")

        # Step 5: Graph expansion (conditional on link_enabled and graph_alpha)
        if self._link_enabled and self.retrieval_config.graph_alpha > 0:
            candidates = await self._graph_expand(
                candidates,
                ctx,
                limit,
                target_dirs=target_dirs,
                level=level,
                query_text=query.query,
            )
            graph_expanded = True
            coverage_graph_expanded = self._has_coverage_graph_candidates(candidates)

        # Step 6: Convert results
        matched = await self._convert_to_matched_contexts(candidates, ctx=ctx)

        final = self._select_final_contexts(
            matched,
            limit,
            graph_expanded=graph_expanded,
            coverage_graph_expanded=coverage_graph_expanded,
        )

        # Record retrieval stats for the observer.
        elapsed_ms = (time.monotonic() - t0) * 1000
        get_stats_collector().record_query(
            context_type=query.context_type.value if query.context_type else "unknown",
            result_count=len(final),
            scores=[m.score for m in final],
            latency_ms=elapsed_ms,
            rerank_used=self._rerank_client is not None and mode == RetrieverMode.THINKING,
        )

        return QueryResult(
            query=query,
            matched_contexts=final,
            searched_directories=root_uris,
        )

    def _select_final_contexts(
        self,
        matched: List[MatchedContext],
        limit: int,
        *,
        graph_expanded: bool,
        coverage_graph_expanded: bool = False,
    ) -> List[MatchedContext]:
        """Return semantic top-k plus bounded auxiliary graph recalls.

        Graph memories should add evidence, not take slots away from the
        semantic baseline.  This keeps graph_alpha>0 changes scoped to extra
        graph-discovered contexts while preserving the normal top-k memories.
        """
        if not graph_expanded or limit <= 0:
            return matched[:limit]

        semantic = [context for context in matched if not context.match_reason]
        graph = [context for context in matched if context.match_reason]
        return semantic[:limit] + graph[
            : self._graph_auxiliary_result_limit(coverage_graph_expanded)
        ]

    @classmethod
    def _graph_auxiliary_result_limit(cls, coverage_graph_expanded: bool) -> int:
        """Use extra graph evidence only for coverage-style queries."""
        if coverage_graph_expanded:
            return cls.GRAPH_COVERAGE_AUXILIARY_RESULT_LIMIT
        return cls.GRAPH_AUXILIARY_RESULT_LIMIT

    @classmethod
    def _has_coverage_graph_candidates(cls, candidates: List[Dict[str, Any]]) -> bool:
        for candidate in candidates:
            if not candidate.get("_from_graph"):
                continue
            if candidate.get("_mnemis_coverage_mode") is True:
                return True
            if candidate.get("_mnemis_query_type") in cls.GRAPH_COVERAGE_QUERY_TYPES:
                return True
        return False

    async def _global_vector_search(
        self,
        vector_proxy: VikingDBManagerProxy,
        query_vector: Optional[List[float]],
        sparse_query_vector: Optional[Dict[str, float]],
        context_type: Optional[str],
        target_dirs: List[str],
        scope_dsl: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Global vector search to locate initial directories."""
        results = await vector_proxy.search_global_roots_in_tenant(
            query_vector=query_vector,
            sparse_query_vector=sparse_query_vector,
            context_type=context_type,
            target_directories=target_dirs,
            extra_filter=scope_dsl,
            limit=limit,
        )
        telemetry = get_current_telemetry()
        telemetry.count("vector.searches", 1)
        telemetry.count("vector.scored", len(results))
        telemetry.count("vector.scanned", len(results))
        return results

    def _rerank_scores(
        self,
        query: str,
        documents: List[str],
        fallback_scores: List[float],
    ) -> List[float]:
        """Return rerank scores or fall back to vector scores."""
        if not self._rerank_client or not documents:
            return fallback_scores

        try:
            scores = self._rerank_client.rerank_batch(query, documents)
        except Exception as e:
            logger.warning(
                "[HierarchicalRetriever] Rerank failed, fallback to vector scores: %s", e
            )
            return fallback_scores

        if not scores or len(scores) != len(documents):
            logger.warning(
                "[HierarchicalRetriever] Invalid rerank result, fallback to vector scores"
            )
            return fallback_scores

        normalized_scores: List[float] = []
        for score, fallback in zip(scores, fallback_scores, strict=True):
            if isinstance(score, (int, float)):
                normalized_scores.append(float(score))
            else:
                normalized_scores.append(fallback)
        return normalized_scores

    def _merge_starting_points(
        self,
        query: str,
        root_uris: List[str],
        global_results: List[Dict[str, Any]],
        mode: str = "thinking",
    ) -> List[Tuple[str, float]]:
        """Merge starting points.
        Returns:
            List of (uri, parent_score) tuples
        """
        points = []
        seen = set()

        global_results = [r for r in global_results if r.get("level", 2) != 2]

        # Results from global search
        default_scores = [
            s if math.isfinite(s) else 0.0 for s in (r.get("_score", 0.0) for r in global_results)
        ]
        if self._rerank_client and mode == RetrieverMode.THINKING:
            docs = [str(r.get("abstract", "")) for r in global_results]
            query_scores = self._rerank_scores(query, docs, default_scores)
            for i, r in enumerate(global_results):
                # 只添加非 level 2 的项目到起始点
                if r.get("level", 2) != 2:
                    points.append((r["uri"], query_scores[i]))
                    seen.add(r["uri"])
        else:
            for r in global_results:
                # 只添加非 level 2 的项目到起始点
                if r.get("level", 2) != 2:
                    points.append((r["uri"], r["_score"]))
                    seen.add(r["uri"])

        # Root directories as starting points
        for uri in root_uris:
            if uri not in seen:
                points.append((uri, 0.0))
                seen.add(uri)

        return points

    def _prepare_initial_candidates(
        self,
        query: str,
        global_results: List[Dict[str, Any]],
        mode: str = RetrieverMode.THINKING,
    ) -> List[Dict[str, Any]]:
        """Preserve rerank scores for global hits added to the result pool."""
        initial_candidates = [dict(r) for r in global_results]
        if not initial_candidates:
            return []

        default_scores = [
            s if math.isfinite(s) else 0.0
            for s in (r.get("_score", 0.0) for r in initial_candidates)
        ]
        if self._rerank_client and mode == RetrieverMode.THINKING:
            docs = [str(r.get("abstract", "")) for r in initial_candidates]
            query_scores = self._rerank_scores(query, docs, default_scores)
        else:
            query_scores = default_scores

        for candidate, score in zip(initial_candidates, query_scores, strict=True):
            candidate["_score"] = score

        return initial_candidates

    async def _recursive_search(
        self,
        vector_proxy: VikingDBManagerProxy,
        query: str,
        query_vector: Optional[List[float]],
        sparse_query_vector: Optional[Dict[str, float]],
        starting_points: List[Tuple[str, float]],
        limit: int,
        mode: str,
        threshold: Optional[float] = None,
        score_gte: bool = False,
        context_type: Optional[str] = None,
        target_dirs: Optional[List[str]] = None,
        scope_dsl: Optional[Dict[str, Any]] = None,
        initial_candidates: Optional[List[Dict[str, Any]]] = None,
        level: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recursive search with directory priority return and score propagation.

        Args:
            threshold: Score threshold
            score_gte: True uses >=, False uses >
            grep_patterns: Keyword match patterns
            scope_dsl: Additional scope constraints from public find/search filter
        """
        # Use passed threshold or default threshold
        effective_threshold = threshold if threshold is not None else self.threshold

        def passes_threshold(score: float) -> bool:
            """Check if score passes threshold."""
            if score_gte:
                return score >= effective_threshold
            return score > effective_threshold

        sparse_query_vector = sparse_query_vector or None

        collected_by_uri: Dict[str, Dict[str, Any]] = {}
        dir_queue: List[tuple] = []  # Priority queue: (-score, uri)
        visited: set = set()
        prev_topk_uris: set = set()
        prev_pool_size = 0
        convergence_rounds = 0
        stagnant_rounds = 0

        # Add initial candidates that match the requested level.
        if initial_candidates:
            for r in initial_candidates:
                uri = r.get("uri", "")
                if not uri:
                    continue
                if level is None or r.get("level", 2) in level:
                    score = r.get("_score", 0.0)
                    if not passes_threshold(score):
                        logger.debug(
                            f"[RecursiveSearch] Initial candidate URI {uri} score {score:.4f} did not pass threshold {effective_threshold}"
                        )
                        continue
                    r["_final_score"] = score
                    collected_by_uri[uri] = r
                    logger.debug(
                        f"[RecursiveSearch] Added initial candidate: {uri} (score: {score:.4f})"
                    )

        alpha = self.score_propagation_alpha

        # Initialize: process starting points
        for uri, score in starting_points:
            heapq.heappush(dir_queue, (-score, uri))

        async def search_children(current_uri: str) -> List[Dict[str, Any]]:
            return await vector_proxy.search_children_in_tenant(
                parent_uri=current_uri,
                query_vector=query_vector,
                sparse_query_vector=sparse_query_vector,  # Pass sparse vector
                context_type=context_type,
                target_directories=target_dirs,
                extra_filter=scope_dsl,
                limit=max(limit * 2, 20),
            )

        parallelism = max(1, self.MAX_PARALLEL_CHILD_SEARCHES)

        while dir_queue:
            batch: List[Tuple[str, float]] = []
            while dir_queue and len(batch) < parallelism:
                temp_score, current_uri = heapq.heappop(dir_queue)
                current_score = -temp_score
                if current_uri in visited:
                    continue
                visited.add(current_uri)
                logger.info(f"[RecursiveSearch] Entering URI: {current_uri}")
                batch.append((current_uri, current_score))

            if not batch:
                continue

            batch_results = await asyncio.gather(
                *(search_children(current_uri) for current_uri, _ in batch)
            )

            telemetry = get_current_telemetry()
            for (_, current_score), results in zip(batch, batch_results, strict=True):
                telemetry.count("vector.searches", 1)
                telemetry.count("vector.scored", len(results))
                telemetry.count("vector.scanned", len(results))

                if not results:
                    continue

                query_scores = [
                    s if math.isfinite(s) else 0.0 for s in (r.get("_score", 0.0) for r in results)
                ]
                if self._rerank_client and mode == RetrieverMode.THINKING:
                    documents = [str(r.get("abstract", "")) for r in results]
                    query_scores = self._rerank_scores(query, documents, query_scores)

                for r, score in zip(results, query_scores, strict=True):
                    uri = r.get("uri", "")
                    final_score = (
                        alpha * score + (1 - alpha) * current_score if current_score else score
                    )

                    if not passes_threshold(final_score):
                        logger.debug(
                            f"[RecursiveSearch] URI {uri} score {final_score} did not pass threshold {effective_threshold}"
                        )
                        continue

                    telemetry.count("vector.passed", 1)
                    if level is None or r.get("level", 2) in level:
                        # Deduplicate by URI and keep the highest-scored candidate.
                        previous = collected_by_uri.get(uri)
                        if previous is None or final_score > previous.get("_final_score", 0):
                            r["_final_score"] = final_score
                            collected_by_uri[uri] = r
                            logger.debug(
                                "[RecursiveSearch] Updated URI: %s candidate score to %.4f",
                                uri,
                                final_score,
                            )

                    # Only recurse into directories (L0/L1). L2 files are terminal hits.
                    if uri not in visited and r.get("level", 2) != 2:
                        heapq.heappush(dir_queue, (-final_score, uri))

            # Convergence check after each parallel expansion round.
            current_topk = sorted(
                collected_by_uri.values(),
                key=lambda x: x.get("_final_score", 0),
                reverse=True,
            )[:limit]
            current_topk_uris = {c.get("uri", "") for c in current_topk}
            current_pool_size = len(collected_by_uri)

            if current_topk_uris == prev_topk_uris and len(current_topk_uris) >= limit:
                convergence_rounds += 1

                if convergence_rounds >= self.MAX_CONVERGENCE_ROUNDS:
                    break
            elif current_pool_size == prev_pool_size:
                stagnant_rounds += 1

                if stagnant_rounds >= self.MAX_CONVERGENCE_ROUNDS:
                    break
            else:
                convergence_rounds = 0
                stagnant_rounds = 0
                prev_topk_uris = current_topk_uris
                prev_pool_size = current_pool_size

        collected = sorted(
            collected_by_uri.values(),
            key=lambda x: x.get("_final_score", 0),
            reverse=True,
        )
        return collected[:limit]

    async def _graph_expand(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
        limit: int,
        target_dirs: Optional[List[str]] = None,
        level: Optional[List[int]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run graph-based expansion when link_enabled and graph_alpha > 0."""
        if not candidates:
            logger.debug("[HierarchicalRetriever] No graph seeds, skipping graph expansion")
            return candidates

        space_uris = self._get_graph_space_uris(
            ctx=ctx,
            target_dirs=target_dirs,
            candidates=candidates,
        )
        logger.info(f"[HierarchicalRetriever] Graph expand: space_uris={space_uris}")
        if not space_uris:
            logger.debug(
                "[HierarchicalRetriever] No graph memory spaces, skipping graph expansion"
            )
            return candidates

        from openviking.retrieve.graph.graph_index import get_graph_index

        index = get_graph_index(space_uris)
        if not index.is_fresh(space_uris):
            await index.build(space_uris, ctx)

        if not index.get_nodes():
            logger.debug("[HierarchicalRetriever] Graph index empty, skipping graph expansion")
            return candidates

        if self.retrieval_config.graph_strategy == "mnemis_lite":
            from openviking.retrieve.graph.mnemis_lite import MnemisLiteGraphRetriever

            graph_retriever = MnemisLiteGraphRetriever(index, self.retrieval_config)
        else:
            from openviking.retrieve.graph.graph_retriever import GraphRetriever

            graph_retriever = GraphRetriever(index, self.retrieval_config)
        return await graph_retriever.expand(
            candidates,
            ctx,
            limit,
            target_dirs=target_dirs,
            level=level,
            query_text=query_text,
        )

    def _get_graph_space_uris(
        self,
        ctx: RequestContext,
        target_dirs: Optional[List[str]],
        candidates: List[Dict[str, Any]],
    ) -> List[str]:
        """Determine memory spaces that graph retrieval should scan.

        ROOT requests cannot infer visible user/agent memory roots from ctx
        alone.  In that case, prefer the target directories already used by
        vector retrieval, then fall back to memory candidate URIs.
        """
        space_uris: List[str] = []

        for uri in target_dirs or []:
            space_uri = self._memory_space_uri_from_uri(uri)
            if space_uri:
                space_uris.append(space_uri)

        if not space_uris:
            space_uris = self._get_root_uris_for_type(ContextType.MEMORY, ctx=ctx)

        if not space_uris:
            for candidate in candidates:
                space_uri = self._memory_space_uri_from_uri(candidate.get("uri", ""))
                if space_uri:
                    space_uris.append(space_uri)

        return list(dict.fromkeys(space_uris))

    @staticmethod
    def _memory_space_uri_from_uri(uri: str) -> Optional[str]:
        """Return the owning memory-space root for a user/agent memory URI."""
        if not uri or not (
            uri.startswith("viking://user/") or uri.startswith("viking://agent/")
        ):
            return None
        marker = "/memories"
        idx = uri.find(marker)
        if idx < 0:
            return None
        return uri[: idx + len(marker)]

    async def _convert_to_matched_contexts(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
    ) -> List[MatchedContext]:
        """Convert candidate results to MatchedContext list.

        Blends semantic similarity with a hotness score derived from
        ``active_count`` and ``updated_at`` when configured. The blend weight
        is controlled by ``retrieval.hotness_alpha`` (0 disables the boost).
        """
        results = []

        for c in candidates:
            # Read related contexts and get summaries
            relations = []
            if get_viking_fs():
                related_uris = await get_viking_fs().get_relations(c.get("uri", ""), ctx=ctx)
                if related_uris:
                    related_abstracts = await get_viking_fs().read_batch(
                        related_uris[: self.MAX_RELATIONS], level="l0", ctx=ctx
                    )
                    for uri in related_uris[: self.MAX_RELATIONS]:
                        abstract = related_abstracts.get(uri, "")
                        if abstract:
                            relations.append(RelatedContext(uri=uri, abstract=abstract))

            semantic_score = c.get("_final_score", c.get("_score", 0.0))
            # Fix: clamp inf/nan scores from vector search (#inf-score)
            if not math.isfinite(semantic_score):
                semantic_score = 0.0

            alpha = self.hotness_alpha
            if alpha > 0:
                updated_at_raw = c.get("updated_at")
                if isinstance(updated_at_raw, str):
                    try:
                        updated_at_val = parse_iso_datetime(updated_at_raw)
                    except (ValueError, TypeError):
                        updated_at_val = None
                elif isinstance(updated_at_raw, datetime):
                    updated_at_val = updated_at_raw
                else:
                    updated_at_val = None

                h_score = hotness_score(
                    active_count=c.get("active_count", 0),
                    updated_at=updated_at_val,
                )
                final_score = (1 - alpha) * semantic_score + alpha * h_score
            else:
                final_score = semantic_score
            if not math.isfinite(final_score):
                final_score = 0.0
            level = c.get("level", 2)
            display_uri = self._append_level_suffix(c.get("uri", ""), level)

            # Build match_reason for graph-expanded nodes
            match_reason = ""
            if c.get("_from_graph"):
                match_reason = "Discovered via graph expansion"
                if c.get("_graph_paths"):
                    path_descs = []
                    for gp in c["_graph_paths"][:2]:
                        link_types = ", ".join(gp.get("link_types", []))
                        path_descs.append(f"Path: {gp['path']} ({link_types})")
                    match_reason = "; ".join(path_descs)

            results.append(
                MatchedContext(
                    uri=display_uri,
                    context_type=ContextType(c["context_type"])
                    if c.get("context_type")
                    else ContextType.RESOURCE,
                    level=level,
                    abstract=c.get("abstract", ""),
                    category=c.get("category", ""),
                    score=final_score,
                    match_reason=match_reason,
                    relations=relations,
                )
            )

        # Re-sort by blended score so hotness boost can change ranking
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    @classmethod
    def _append_level_suffix(cls, uri: str, level: int) -> str:
        """Return user-facing URI with L0/L1 suffix reconstructed by level."""
        suffix = cls.LEVEL_URI_SUFFIX.get(level)
        if not uri or not suffix:
            return uri
        if uri.endswith(f"/{suffix}"):
            return uri
        if uri.endswith("/.abstract.md") or uri.endswith("/.overview.md"):
            return uri
        if uri.endswith("/") and not uri.endswith("://"):
            uri = uri.rstrip("/")
        return f"{uri}/{suffix}"

    def _get_root_uris_for_type(
        self, context_type: Optional[ContextType], ctx: Optional[RequestContext] = None
    ) -> List[str]:
        """Return starting directory URI list based on context_type and user context.

        When context_type is None, returns roots for all types.
        ROOT has no space, relies on global_vector_search without URI prefix filter.
        """
        if not ctx or ctx.role == Role.ROOT:
            return []

        user_root = canonical_user_root(ctx)
        agent_root = canonical_agent_root(ctx)
        if context_type is None:
            return [
                f"{user_root}/memories",
                f"{agent_root}/memories",
                "viking://resources",
                f"{agent_root}/skills",
            ]
        elif context_type == ContextType.MEMORY:
            return [
                f"{user_root}/memories",
                f"{agent_root}/memories",
            ]
        elif context_type == ContextType.RESOURCE:
            return ["viking://resources"]
        elif context_type == ContextType.SKILL:
            return [f"{agent_root}/skills"]
        return []
