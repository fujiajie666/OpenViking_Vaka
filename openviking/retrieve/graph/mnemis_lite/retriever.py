# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Mnemis-lite dual-route graph retriever.

This strategy keeps the existing semantic route intact, then adds a bounded
graph route with rule-based query planning, typed candidate selection, and
slot-aware graph gates. It deliberately reuses the legacy GraphRetriever's
score normalization and append-only integration points so the experiment stays
close to OpenViking's current retrieval architecture.
"""

import math
from typing import Any, Dict, List

from openviking.retrieve.graph.graph_index import GraphIndex, GraphNode
from openviking.retrieve.graph.graph_retriever import GraphRetriever
from openviking.retrieve.graph.path_extractor import PathExtractor
from openviking.retrieve.graph.ppr import TypedWeightedPPR
from openviking.retrieve.graph.score_normalizer import minmax_normalize
from openviking.retrieve.graph.mnemis_lite.query_planner import (
    GraphQueryPlan,
    RuleBasedGraphQueryPlanner,
)
from openviking.retrieve.graph.mnemis_lite.scorer import MnemisLiteSlotScorer
from openviking.server.identity import RequestContext
from openviking_cli.utils.config import RetrievalConfig
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_MNEMIS_LITE_STRATEGY = "mnemis_lite_v1"
_PPR_ROUTE_SUPPORT_SCALE = 0.5
_TWO_HOP_ROUTE_SUPPORT_SCALE = 0.45
_REVERSE_ROUTE_PENALTY = 0.7


class MnemisLiteGraphRetriever(GraphRetriever):
    """Experimental Mnemis-inspired graph retriever."""

    def __init__(self, graph_index: GraphIndex, retrieval_config: RetrievalConfig):
        super().__init__(graph_index, retrieval_config)
        self._planner = RuleBasedGraphQueryPlanner()
        self._slot_scorer = MnemisLiteSlotScorer()
        self._current_plan: GraphQueryPlan | None = None

    async def expand(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
        limit: int,
        target_dirs: List[str] | None = None,
        level: List[int] | None = None,
        query_text: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Run semantic-preserving graph expansion with a second graph route."""
        self._debug_metadata = {}
        plan = self._planner.plan(query_text)
        self._current_plan = plan
        try:
            seeds = self._build_seeds(candidates)
            if not seeds:
                self._debug_metadata = self._build_retrieval_debug([], returned_uris=set())
                return candidates

            ppr_engine = TypedWeightedPPR(
                graph_index=self._graph_index,
                type_weights=self._config.graph_type_weights,
                restart=self._config.graph_ppr_restart,
                max_iter=self._config.graph_ppr_max_iter,
                tolerance=self._config.graph_ppr_tolerance,
            )
            ppr_scores = ppr_engine.run(seeds)
            direct_support = self._compute_direct_seed_support(seeds)
            if not ppr_scores and not direct_support:
                return candidates

            mnemis_support = dict(direct_support)
            self._merge_support_scores(
                mnemis_support,
                self._ppr_route_support(
                    seeds=seeds,
                    ppr_scores=ppr_scores,
                    existing_support=mnemis_support,
                    plan=plan,
                ),
            )
            self._merge_support_scores(
                mnemis_support,
                self._two_hop_route_support(
                    seeds=seeds,
                    existing_support=mnemis_support,
                    plan=plan,
                ),
            )

            logger.info(
                "[MnemisLiteGraphRetriever] strategy=%s query_type=%s seeds=%s "
                "direct=%s support_pool=%s ppr_nodes=%s",
                _MNEMIS_LITE_STRATEGY,
                plan.query_type,
                len(seeds),
                len(direct_support),
                len(mnemis_support),
                len(ppr_scores),
            )

            expanded = self._merge_mnemis_candidates(
                candidates=candidates,
                ppr_scores=ppr_scores,
                support_scores=mnemis_support,
                direct_support=direct_support,
                target_dirs=target_dirs,
                level=level,
                plan=plan,
            )
            expanded = await self._fill_abstracts_for_graph_nodes(expanded, ctx)
            expanded = self._score_graph_candidates(
                expanded,
                query_text=query_text,
                limit=limit,
            )
            scored_candidates = expanded
            expanded = self._filter_unaccepted_graph_nodes(scored_candidates)

            if self._config.graph_path_count > 0:
                try:
                    paths = PathExtractor(
                        graph_index=self._graph_index,
                        max_paths=self._config.graph_path_count,
                    ).extract(seeds, ppr_scores, top_k=limit)
                    self._attach_path_metadata(expanded, paths)
                except Exception as exc:
                    logger.warning(
                        "[MnemisLiteGraphRetriever] path extraction failed; continuing: %s",
                        exc,
                    )

            selected = self._select_expanded_candidates(expanded, limit=limit)
            self._debug_metadata = self._build_retrieval_debug(
                scored_candidates,
                returned_uris={candidate.get("uri", "") for candidate in selected},
            )
            return selected
        finally:
            self._current_plan = None

    def _merge_mnemis_candidates(
        self,
        *,
        candidates: List[Dict[str, Any]],
        ppr_scores: Dict[str, float],
        support_scores: Dict[str, float],
        direct_support: Dict[str, float],
        target_dirs: List[str] | None,
        level: List[int] | None,
        plan: GraphQueryPlan,
    ) -> List[Dict[str, Any]]:
        """Merge direct, PPR-only, and two-hop graph candidates."""
        expanded = self._merge_candidates_with_ppr(
            candidates,
            ppr_scores,
            support_scores=direct_support,
            target_dirs=target_dirs,
            level=level,
        )
        existing_by_uri = {
            candidate.get("uri", ""): candidate
            for candidate in expanded
            if candidate.get("uri", "")
        }
        for candidate in existing_by_uri.values():
            if candidate.get("_from_graph"):
                candidate["_mnemis_route"] = "direct"

        norm_support = minmax_normalize(support_scores)
        target_prefixes = self._normalize_target_dirs(target_dirs)
        allowed_levels = set(level) if level is not None else None
        pool_limit = max(self._config.graph_expansion_topk * 2, self._config.graph_expansion_topk)

        added = 0
        for uri in self._rank_mnemis_route_uris(
            ppr_scores=ppr_scores,
            support_scores=support_scores,
            direct_support=direct_support,
            limit=max(len(support_scores), pool_limit * 3),
        ):
            if uri in existing_by_uri:
                continue
            if not self._can_add_graph_uri(uri, target_prefixes, allowed_levels, plan):
                continue
            node = self._graph_index.get_node(uri)
            candidate = self._new_graph_candidate(uri, node, ppr_scores)
            candidate["_mnemis_route"] = "ppr_or_two_hop"
            self._attach_graph_signal_metadata(
                candidate,
                support_scores=support_scores,
                norm_support=norm_support,
            )
            existing_by_uri[uri] = candidate
            added += 1
            if added >= pool_limit:
                break

        if added:
            logger.info(
                "[MnemisLiteGraphRetriever] strategy=%s added %s graph-route nodes",
                _MNEMIS_LITE_STRATEGY,
                added,
            )

        return list(existing_by_uri.values())

    def _rank_mnemis_route_uris(
        self,
        *,
        ppr_scores: Dict[str, float],
        support_scores: Dict[str, float],
        direct_support: Dict[str, float],
        limit: int,
    ) -> List[str]:
        norm_ppr = minmax_normalize(ppr_scores)
        norm_support = minmax_normalize(support_scores)

        def key(uri: str) -> tuple[float, float, float, float]:
            support = norm_support.get(uri, 0.0)
            ppr = norm_ppr.get(uri, 0.0)
            direct = 1.0 if uri in direct_support else 0.0
            return (support, ppr, direct, support_scores.get(uri, 0.0))

        return sorted(support_scores, key=key, reverse=True)[:limit]

    def _score_graph_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        query_text: str | None,
        limit: int,
    ) -> List[Dict[str, Any]]:
        scored = super()._score_graph_candidates(
            candidates,
            query_text=query_text,
            limit=limit,
        )
        plan = self._current_plan or self._planner.plan(query_text)
        for candidate in scored:
            if not candidate.get("_from_graph"):
                continue
            candidate["_graph_strategy"] = _MNEMIS_LITE_STRATEGY
            candidate["_mnemis_query_type"] = plan.query_type
            candidate["_mnemis_coverage_mode"] = plan.coverage_mode
            candidate["_mnemis_uri_kind"] = self._slot_scorer.uri_kind(candidate)
            if not candidate.get("_graph_accepted"):
                candidate["_graph_debug"] = self._candidate_graph_debug(candidate)
                continue

            slot = self._slot_scorer.slot_match(candidate, plan)
            candidate["_mnemis_slot_match"] = slot.matched
            candidate["_mnemis_slot_reason"] = slot.reason
            candidate["_mnemis_coverage_group"] = self._slot_scorer.coverage_group(candidate)
            if not slot.matched:
                candidate["_graph_accepted"] = False
                candidate["_graph_boost"] = 0.0
                candidate["_graph_accept_reason"] = f"rejected:{slot.reason}"
            else:
                candidate["_graph_accept_reason"] = slot.reason
            candidate["_graph_debug"] = self._candidate_graph_debug(candidate)
        return scored

    def _select_expanded_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        semantic_candidates = [
            candidate for candidate in candidates if not candidate.get("_from_graph")
        ]
        graph_candidates = [
            candidate for candidate in candidates if candidate.get("_from_graph")
        ]
        semantic_candidates = sorted(
            semantic_candidates,
            key=lambda candidate: candidate.get("_final_score", 0.0),
            reverse=True,
        )
        graph_candidates = self._rank_graph_selection_candidates(graph_candidates)
        if not self._current_plan or not self._current_plan.coverage_mode:
            graph_limit = max(0, min(len(graph_candidates), self._config.graph_expansion_topk))
            return semantic_candidates + graph_candidates[:graph_limit]

        selected_graph: List[Dict[str, Any]] = []
        seen_groups: set[str] = set()
        graph_limit = max(0, min(len(graph_candidates), self._config.graph_expansion_topk))
        for candidate in graph_candidates:
            group = str(
                candidate.get("_mnemis_coverage_group")
                or self._slot_scorer.coverage_group(candidate)
            )
            if group in seen_groups:
                continue
            selected_graph.append(candidate)
            seen_groups.add(group)
            if len(selected_graph) >= graph_limit:
                break
        if len(selected_graph) < graph_limit:
            selected_uris = {candidate.get("uri", "") for candidate in selected_graph}
            for candidate in graph_candidates:
                if candidate.get("uri", "") in selected_uris:
                    continue
                selected_graph.append(candidate)
                if len(selected_graph) >= graph_limit:
                    break
        return semantic_candidates + selected_graph

    def _rank_graph_selection_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prefer legacy direct graph hits before broader graph expansion routes."""
        return sorted(candidates, key=self._graph_selection_key, reverse=True)

    @staticmethod
    def _graph_selection_key(
        candidate: Dict[str, Any],
    ) -> tuple[float, float, float, float, float]:
        direct_route = 1.0 if candidate.get("_mnemis_route") == "direct" else 0.0
        return (
            direct_route,
            float(candidate.get("_final_score", 0.0) or 0.0),
            float(candidate.get("_graph_evidence_own", 0.0) or 0.0),
            float(candidate.get("_graph_query_evidence", 0.0) or 0.0),
            float(candidate.get("_graph_support", 0.0) or 0.0),
        )

    def _build_retrieval_debug(
        self,
        candidates: List[Dict[str, Any]],
        *,
        returned_uris: set[str],
    ) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not candidate.get("_from_graph"):
                continue
            record = dict(candidate.get("_graph_debug") or self._candidate_graph_debug(candidate))
            record["returned"] = record.get("uri") in returned_uris
            records.append(record)
        plan = self._current_plan
        return {
            "strategy": _MNEMIS_LITE_STRATEGY,
            "query_type": plan.query_type if plan else "",
            "coverage_mode": bool(plan.coverage_mode) if plan else False,
            "candidate_count": len(records),
            "accepted_count": sum(1 for record in records if record.get("accepted")),
            "returned_count": sum(1 for record in records if record.get("returned")),
            "candidates": records,
        }

    def _candidate_graph_debug(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        record = super()._candidate_graph_debug(candidate)
        record["strategy"] = _MNEMIS_LITE_STRATEGY
        record["query_type"] = str(candidate.get("_mnemis_query_type", "") or "")
        record["coverage_mode"] = bool(candidate.get("_mnemis_coverage_mode", False))
        record["route"] = str(candidate.get("_mnemis_route", "") or "")
        record["slot_match"] = bool(candidate.get("_mnemis_slot_match", False))
        record["slot_reason"] = str(candidate.get("_mnemis_slot_reason", "") or "")
        record["coverage_group"] = str(candidate.get("_mnemis_coverage_group", "") or "")
        return record

    def _ppr_route_support(
        self,
        *,
        seeds: Dict[str, float],
        ppr_scores: Dict[str, float],
        existing_support: Dict[str, float],
        plan: GraphQueryPlan,
    ) -> Dict[str, float]:
        support: Dict[str, float] = {}
        seed_set = set(seeds)
        limit = max(self._config.graph_expansion_topk * 3, self._config.graph_expansion_topk)
        for uri, score in sorted(ppr_scores.items(), key=lambda item: item[1], reverse=True):
            if len(support) >= limit:
                break
            if uri in seed_set or uri in existing_support:
                continue
            node = self._graph_index.get_node(uri)
            if not node or node.is_summary:
                continue
            if not self._node_kind_allowed(uri, node, plan):
                continue
            support[uri] = score * _PPR_ROUTE_SUPPORT_SCALE
        return support

    def _two_hop_route_support(
        self,
        *,
        seeds: Dict[str, float],
        existing_support: Dict[str, float],
        plan: GraphQueryPlan,
    ) -> Dict[str, float]:
        support: Dict[str, float] = {}
        limit = max(self._config.graph_expansion_topk * 3, self._config.graph_expansion_topk)
        for seed_uri, seed_score in seeds.items():
            seed_specificity = self._degree_specificity(self._node_degree(seed_uri))
            for mid_uri, first_weight in self._neighbors_with_weights(seed_uri):
                for uri, second_weight in self._neighbors_with_weights(mid_uri):
                    if len(support) >= limit:
                        return support
                    if uri in seeds or uri in existing_support:
                        continue
                    node = self._graph_index.get_node(uri)
                    if not node or node.is_summary:
                        continue
                    if not self._node_kind_allowed(uri, node, plan):
                        continue
                    score = (
                        seed_score
                        * first_weight
                        * second_weight
                        * seed_specificity
                        * _TWO_HOP_ROUTE_SUPPORT_SCALE
                    )
                    support[uri] = max(score, support.get(uri, 0.0))
        return support

    def _neighbors_with_weights(self, uri: str) -> List[tuple[str, float]]:
        neighbors: List[tuple[str, float]] = []
        for edge in self._graph_index.get_forward_edges(uri):
            weight = edge.weight * self._config.graph_type_weights.get(edge.link_type, 1.0)
            neighbors.append((edge.to_uri, weight))
        for edge in self._graph_index.get_reverse_edges(uri):
            weight = (
                edge.weight
                * self._config.graph_type_weights.get(edge.link_type, 1.0)
                * _REVERSE_ROUTE_PENALTY
            )
            neighbors.append((edge.from_uri, weight))
        return neighbors

    @staticmethod
    def _merge_support_scores(target: Dict[str, float], incoming: Dict[str, float]) -> None:
        for uri, score in incoming.items():
            target[uri] = max(score, target.get(uri, 0.0))

    def _can_add_graph_uri(
        self,
        uri: str,
        target_prefixes: List[str],
        allowed_levels: set[int] | None,
        plan: GraphQueryPlan,
    ) -> bool:
        if not self._graph_index.has_node(uri):
            return False
        if allowed_levels is not None and 2 not in allowed_levels:
            return False
        if target_prefixes and not self._is_uri_under_targets(uri, target_prefixes):
            return False
        node = self._graph_index.get_node(uri)
        if not node or node.is_summary:
            return False
        return self._node_kind_allowed(uri, node, plan)

    def _node_kind_allowed(self, uri: str, node: GraphNode, plan: GraphQueryPlan) -> bool:
        kind = self._slot_scorer.uri_kind(
            {
                "uri": uri,
                "memory_type": node.memory_type or "",
                "category": node.category or "",
            }
        )
        if plan.preferred_uri_kinds and kind not in plan.preferred_uri_kinds:
            return False
        if kind in plan.risky_uri_kinds and plan.query_type not in {"relationship", "inference"}:
            return False
        return True

    @staticmethod
    def _new_graph_candidate(
        uri: str,
        node: GraphNode | None,
        ppr_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "uri": uri,
            "_score": 0.0,
            "_final_score": 0.0,
            "_ppr_score": ppr_scores.get(uri, 0.0),
            "context_type": "memory",
            "level": 2,
            "abstract": "",
            "category": node.category if node else "",
            "memory_type": node.memory_type if node else "",
            "_from_graph": True,
        }
