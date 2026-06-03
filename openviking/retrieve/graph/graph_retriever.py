# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Evidence-gated graph retrieval expansion.

The graph retriever is intentionally conservative: semantic retrieval remains
the primary result set, and graph nodes are appended only when they are both
near semantic seeds in the graph and supported by query/content overlap.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from openviking.retrieve.graph.graph_index import GraphIndex
from openviking.retrieve.graph.path_extractor import PathExtractor
from openviking.retrieve.graph.ppr import TypedWeightedPPR
from openviking.retrieve.graph.score_normalizer import minmax_normalize
from openviking.server.identity import RequestContext
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.utils.config import RetrievalConfig
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_DEGREE_PENALTY_POWER = 0.5
_REVERSE_SUPPORT_PENALTY = 0.7
_MAX_SEMANTIC_SEEDS = 12
_GRAPH_SCORING_POOL_FACTOR = 4
_MIN_GRAPH_SUPPORT = 1e-12
_MIN_QUERY_EVIDENCE = 0.16
_MIN_OWN_QUERY_EVIDENCE = 0.08
_MIN_STRONG_OWN_QUERY_EVIDENCE = 0.45
_HIGH_RISK_GRAPH_DEGREE = 24
_EDGE_EVIDENCE_WEIGHT = 0.35
_URI_EVIDENCE_WEIGHT = 0.20
_CATEGORY_EVIDENCE_WEIGHT = 0.10
_GRAPH_SCORE_CEILING_FRACTION = 0.25
_GRAPH_RETRIEVER_STRATEGY = "evidence_gated_append_v5"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CURRENT_DATE_PREFIX_RE = re.compile(
    r"^\s*current\s+date\s*:\s*\d{4}-\d{2}-\d{2}\s*\.\s*",
    re.IGNORECASE,
)
_ANSWER_DIRECTLY_PREFIX_RE = re.compile(
    r"^\s*answer\s+the\s+question\s+directly\s*:\s*",
    re.IGNORECASE,
)
_QUERY_STOPWORDS = {
    "a",
    "about",
    "according",
    "after",
    "all",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "by",
    "can",
    "current",
    "date",
    "did",
    "directly",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "question",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class GraphEvidenceSignals:
    own: float = 0.0
    edge: float = 0.0
    uri: float = 0.0
    category: float = 0.0
    combined: float = 0.0


class GraphRetriever:
    """Runs graph expansion as bounded, evidence-gated append-only recall."""

    def __init__(self, graph_index: GraphIndex, retrieval_config: RetrievalConfig):
        self._graph_index = graph_index
        self._config = retrieval_config

    async def expand(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
        limit: int,
        target_dirs: List[str] | None = None,
        level: List[int] | None = None,
        query_text: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Append graph-discovered memories that pass generic evidence gates."""
        seeds = self._build_seeds(candidates)
        if not seeds:
            return candidates

        ppr_engine = TypedWeightedPPR(
            graph_index=self._graph_index,
            type_weights=self._config.graph_type_weights,
            restart=self._config.graph_ppr_restart,
            max_iter=self._config.graph_ppr_max_iter,
            tolerance=self._config.graph_ppr_tolerance,
        )
        ppr_scores = ppr_engine.run(seeds)
        support_scores = self._compute_direct_seed_support(seeds)
        if not ppr_scores and not support_scores:
            return candidates

        logger.info(
            "[GraphRetriever] strategy=%s ppr_nodes=%s direct_support_nodes=%s seeds=%s",
            _GRAPH_RETRIEVER_STRATEGY,
            len(ppr_scores),
            len(support_scores),
            len(seeds),
        )

        expanded = self._merge_candidates_with_ppr(
            candidates,
            ppr_scores,
            support_scores=support_scores,
            target_dirs=target_dirs,
            level=level,
        )
        expanded = await self._fill_abstracts_for_graph_nodes(expanded, ctx)
        expanded = self._score_graph_candidates(
            expanded,
            query_text=query_text,
            limit=limit,
        )
        expanded = self._filter_unaccepted_graph_nodes(expanded)

        if self._config.graph_path_count > 0:
            try:
                paths = PathExtractor(
                    graph_index=self._graph_index,
                    max_paths=self._config.graph_path_count,
                ).extract(seeds, ppr_scores, top_k=limit)
                self._attach_path_metadata(expanded, paths)
            except Exception as exc:
                logger.warning(
                    "[GraphRetriever] Path extraction failed; continuing without "
                    f"graph paths: {exc}"
                )

        selected = self._select_expanded_candidates(expanded, limit=limit)
        return selected

    def _build_seeds(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Build normalized graph seeds from the strongest semantic candidates."""
        seeds: Dict[str, float] = {}
        scored_candidates = sorted(
            candidates,
            key=self._candidate_semantic_score,
            reverse=True,
        )[:_MAX_SEMANTIC_SEEDS]
        for candidate in scored_candidates:
            uri = candidate.get("uri", "")
            score = self._candidate_semantic_score(candidate)
            if uri and math.isfinite(score) and score > 0:
                seeds[uri] = score

        if not seeds:
            return seeds

        if self._config.graph_seed_include_summaries:
            mean_score = sum(seeds.values()) / len(seeds)
            for uri, node in self._graph_index.get_nodes().items():
                if node.is_summary and uri not in seeds:
                    seeds[uri] = mean_score * 0.5

        total = sum(seeds.values())
        if total > 0:
            seeds = {uri: score / total for uri, score in seeds.items()}
        return seeds

    def _compute_direct_seed_support(self, seeds: Dict[str, float]) -> Dict[str, float]:
        """Score one-hop graph support from semantic seeds."""
        support: Dict[str, float] = {}
        for seed_uri, seed_score in seeds.items():
            seed_specificity = self._degree_specificity(self._node_degree(seed_uri))
            for edge in self._graph_index.get_forward_edges(seed_uri):
                weight = edge.weight * self._config.graph_type_weights.get(edge.link_type, 1.0)
                support[edge.to_uri] = support.get(edge.to_uri, 0.0) + (
                    seed_score * weight * seed_specificity
                )
            for edge in self._graph_index.get_reverse_edges(seed_uri):
                weight = (
                    edge.weight
                    * self._config.graph_type_weights.get(edge.link_type, 1.0)
                    * _REVERSE_SUPPORT_PENALTY
                )
                support[edge.from_uri] = support.get(edge.from_uri, 0.0) + (
                    seed_score * weight * seed_specificity
                )
        return support

    def _merge_candidates_with_ppr(
        self,
        candidates: List[Dict[str, Any]],
        ppr_scores: Dict[str, float],
        support_scores: Dict[str, float] | None = None,
        target_dirs: List[str] | None = None,
        level: List[int] | None = None,
    ) -> List[Dict[str, Any]]:
        """Merge semantic candidates with a bounded graph scoring pool."""
        support_scores = support_scores or {}
        norm_support = minmax_normalize(support_scores)
        existing_by_uri: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            uri = candidate.get("uri", "")
            if uri:
                existing_by_uri[uri] = candidate

        for uri, candidate in existing_by_uri.items():
            candidate["_ppr_score"] = ppr_scores.get(uri, 0.0)
            self._attach_graph_signal_metadata(
                candidate,
                support_scores=support_scores,
                norm_support=norm_support,
            )

        allowed_levels = set(level) if level is not None else None
        target_prefixes = self._normalize_target_dirs(target_dirs)
        graph_pool_limit = max(
            self._config.graph_expansion_topk,
            self._config.graph_expansion_topk * _GRAPH_SCORING_POOL_FACTOR,
        )

        added_count = 0
        for uri in self._rank_graph_expansion_uris(
            ppr_scores=ppr_scores,
            support_scores=support_scores,
            limit=graph_pool_limit,
        ):
            if uri in existing_by_uri or not self._graph_index.has_node(uri):
                continue
            if allowed_levels is not None and 2 not in allowed_levels:
                continue
            if target_prefixes and not self._is_uri_under_targets(uri, target_prefixes):
                continue

            node = self._graph_index.get_node(uri)
            if node and node.is_summary:
                continue

            candidate = {
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
            self._attach_graph_signal_metadata(
                candidate,
                support_scores=support_scores,
                norm_support=norm_support,
            )
            existing_by_uri[uri] = candidate
            added_count += 1

        if added_count:
            logger.info(
                "[GraphRetriever] strategy=%s added %s graph nodes to scoring pool",
                _GRAPH_RETRIEVER_STRATEGY,
                added_count,
            )

        return list(existing_by_uri.values())

    @staticmethod
    def _rank_graph_expansion_uris(
        ppr_scores: Dict[str, float],
        support_scores: Dict[str, float] | None,
        limit: int,
    ) -> List[str]:
        """Rank directly supported graph nodes for pre-scoring."""
        support_scores = support_scores or {}
        if limit <= 0:
            return []

        norm_ppr = minmax_normalize(ppr_scores)
        norm_support = minmax_normalize(support_scores)

        def rank_key(uri: str) -> tuple[float, float, float]:
            support = norm_support.get(uri, 0.0)
            ppr = norm_ppr.get(uri, 0.0)
            return (support, ppr, support_scores.get(uri, 0.0))

        return sorted(support_scores, key=rank_key, reverse=True)[:limit]

    def _score_graph_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        query_text: str | None,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Gate graph-added nodes by path support, query evidence, and degree."""
        if not candidates:
            return candidates

        norm_ppr = minmax_normalize(
            {candidate["uri"]: candidate.get("_ppr_score", 0.0) for candidate in candidates}
        )
        query_evidence = self._compute_query_evidence_signals(candidates, query_text)

        semantic_scores = sorted(
            (
                self._candidate_semantic_score(candidate)
                for candidate in candidates
                if not candidate.get("_from_graph")
            ),
            reverse=True,
        )
        if semantic_scores:
            boundary_index = min(max(limit, 1), len(semantic_scores)) - 1
            semantic_floor = semantic_scores[boundary_index]
            semantic_top = semantic_scores[0]
        else:
            semantic_floor = 0.0
            semantic_top = 1.0
        semantic_span = max(semantic_top - semantic_floor, 1e-6)
        score_ceiling = semantic_floor + semantic_span * _GRAPH_SCORE_CEILING_FRACTION

        accepted_count = 0
        for candidate in candidates:
            uri = candidate.get("uri", "")
            semantic_score = self._candidate_semantic_score(candidate)
            evidence_signals = query_evidence.get(uri, GraphEvidenceSignals())
            candidate["_norm_ppr"] = norm_ppr.get(uri, 0.0)
            candidate["_graph_query_evidence"] = evidence_signals.combined
            candidate["_graph_evidence_own"] = evidence_signals.own
            candidate["_graph_evidence_edge"] = evidence_signals.edge
            candidate["_graph_evidence_uri"] = evidence_signals.uri
            candidate["_graph_evidence_category"] = evidence_signals.category
            candidate["_graph_strategy"] = _GRAPH_RETRIEVER_STRATEGY

            if not candidate.get("_from_graph"):
                candidate["_graph_accepted"] = False
                candidate["_graph_boost"] = 0.0
                candidate["_final_score"] = semantic_score
                continue

            support = candidate.get("_graph_support", 0.0)
            path_signal = (
                0.7 * candidate.get("_norm_graph_support", 0.0)
                + 0.3 * norm_ppr.get(uri, 0.0)
            ) * candidate.get("_graph_specificity", 1.0)
            evidence = evidence_signals.combined
            own_threshold = self._own_evidence_threshold(candidate)
            accepted = (
                support > _MIN_GRAPH_SUPPORT
                and path_signal > 0
                and evidence_signals.own >= own_threshold
                and evidence >= _MIN_QUERY_EVIDENCE
            )
            graph_boost = (
                self._config.graph_alpha * path_signal * evidence * semantic_span
                if accepted
                else 0.0
            )
            graph_score = min(score_ceiling, semantic_floor + graph_boost)

            candidate["_graph_path_signal"] = path_signal
            candidate["_graph_requires_strong_own_evidence"] = (
                own_threshold > _MIN_OWN_QUERY_EVIDENCE
            )
            candidate["_graph_own_evidence_threshold"] = own_threshold
            candidate["_graph_accepted"] = accepted
            candidate["_graph_boost"] = graph_score - semantic_floor if accepted else 0.0
            candidate["_final_score"] = graph_score if accepted else semantic_floor
            candidate["_graph_accept_reason"] = self._graph_accept_reason(
                support=support,
                path_signal=path_signal,
                own_evidence=evidence_signals.own,
                own_threshold=own_threshold,
                total_evidence=evidence,
                accepted=accepted,
            )
            candidate["_graph_snippet_score"] = self._graph_snippet_score(
                candidate,
                query_text,
            )
            if accepted:
                accepted_count += 1

        if accepted_count:
            logger.info(
                "[GraphRetriever] strategy=%s accepted %s graph nodes after evidence gate",
                _GRAPH_RETRIEVER_STRATEGY,
                accepted_count,
            )
        return candidates

    @staticmethod
    def _filter_unaccepted_graph_nodes(
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop graph-added nodes that did not pass the evidence gate."""
        filtered: List[Dict[str, Any]] = []
        removed = 0
        for candidate in candidates:
            if candidate.get("_from_graph") and not candidate.get("_graph_accepted"):
                removed += 1
                continue
            filtered.append(candidate)

        if removed:
            logger.info(
                "[GraphRetriever] filtered %s graph nodes rejected by evidence gate",
                removed,
            )
        return filtered

    def _select_expanded_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Preserve semantic candidates and append bounded graph additions."""
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
        graph_candidates = sorted(
            graph_candidates,
            key=lambda candidate: candidate.get("_final_score", 0.0),
            reverse=True,
        )
        graph_limit = max(0, min(len(graph_candidates), self._config.graph_expansion_topk))
        return semantic_candidates + graph_candidates[:graph_limit]

    def _attach_graph_signal_metadata(
        self,
        candidate: Dict[str, Any],
        support_scores: Dict[str, float],
        norm_support: Dict[str, float],
    ) -> None:
        uri = candidate.get("uri", "")
        degree = self._node_degree(uri)
        candidate["_graph_degree"] = degree
        candidate["_graph_specificity"] = self._degree_specificity(degree)
        candidate["_graph_support"] = support_scores.get(uri, 0.0)
        candidate["_norm_graph_support"] = norm_support.get(uri, 0.0)

    def _graph_accept_reason(
        self,
        *,
        support: float,
        path_signal: float,
        own_evidence: float,
        own_threshold: float,
        total_evidence: float,
        accepted: bool,
    ) -> str:
        if accepted:
            return "accepted"
        if support <= _MIN_GRAPH_SUPPORT:
            return "rejected:no_direct_support"
        if path_signal <= 0:
            return "rejected:no_path_signal"
        if own_evidence < own_threshold:
            return "rejected:own_evidence_below_threshold"
        if total_evidence < _MIN_QUERY_EVIDENCE:
            return "rejected:total_evidence_below_threshold"
        return "rejected:unknown"

    def _graph_snippet_score(
        self,
        candidate: Dict[str, Any],
        query_text: str | None,
    ) -> Dict[str, Any]:
        query_tokens = self._tokenize(self._normalize_graph_query(query_text))
        if not query_tokens:
            return {"overlap": 0, "density": 0.0}
        overlap, density = self._snippet_query_score(
            str(candidate.get("abstract", "") or ""),
            query_tokens,
        )
        return {"overlap": overlap, "density": self._finite_float(density)}

    def _snippet_query_score(
        self,
        snippet: str,
        query_tokens: set[str],
    ) -> tuple[int, float]:
        snippet_tokens = self._tokenize(snippet)
        overlap = len(query_tokens & snippet_tokens)
        density = overlap / max(len(snippet_tokens), 1)
        return overlap, density

    @staticmethod
    def _finite_float(value: Any) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0
        return round(float(value), 6)

    def _compute_query_evidence_signals(
        self,
        candidates: List[Dict[str, Any]],
        query_text: str | None,
    ) -> Dict[str, GraphEvidenceSignals]:
        """Return separated IDF-weighted query evidence signals."""
        query_tokens = self._tokenize(self._normalize_graph_query(query_text))
        if not query_tokens:
            return {candidate.get("uri", ""): GraphEvidenceSignals() for candidate in candidates}

        signal_tokens_by_uri: Dict[str, Dict[str, set[str]]] = {}
        combined_tokens_by_uri: Dict[str, set[str]] = {}
        for candidate in candidates:
            uri = candidate.get("uri", "")
            if not uri:
                continue
            signal_tokens = self._candidate_signal_tokens(candidate)
            signal_tokens_by_uri[uri] = signal_tokens
            combined_tokens_by_uri[uri] = set().union(*signal_tokens.values())

        if not combined_tokens_by_uri:
            return {}

        doc_count = len(combined_tokens_by_uri)
        document_frequency: Dict[str, int] = {}
        for token in query_tokens:
            document_frequency[token] = sum(
                1 for tokens in combined_tokens_by_uri.values() if token in tokens
            )
        token_weights = {
            token: math.log((doc_count + 1) / (document_frequency[token] + 1)) + 1.0
            for token in query_tokens
        }
        total_weight = sum(token_weights.values())
        if total_weight <= 0:
            return {uri: GraphEvidenceSignals() for uri in combined_tokens_by_uri}

        evidence: Dict[str, GraphEvidenceSignals] = {}
        for uri, signal_tokens in signal_tokens_by_uri.items():
            own = self._weighted_query_overlap(
                query_tokens, signal_tokens["own"], token_weights, total_weight
            )
            edge = self._weighted_query_overlap(
                query_tokens, signal_tokens["edge"], token_weights, total_weight
            )
            uri_signal = self._weighted_query_overlap(
                query_tokens, signal_tokens["uri"], token_weights, total_weight
            )
            category = self._weighted_query_overlap(
                query_tokens, signal_tokens["category"], token_weights, total_weight
            )
            combined = min(
                1.0,
                own
                + _EDGE_EVIDENCE_WEIGHT * edge
                + _URI_EVIDENCE_WEIGHT * uri_signal
                + _CATEGORY_EVIDENCE_WEIGHT * category,
            )
            evidence[uri] = GraphEvidenceSignals(
                own=own,
                edge=edge,
                uri=uri_signal,
                category=category,
                combined=combined,
            )
        return evidence

    @staticmethod
    def _weighted_query_overlap(
        query_tokens: set[str],
        doc_tokens: set[str],
        token_weights: Dict[str, float],
        total_weight: float,
    ) -> float:
        if total_weight <= 0:
            return 0.0
        overlap_weight = sum(
            token_weights[token] for token in query_tokens if token in doc_tokens
        )
        return min(1.0, overlap_weight / total_weight)

    def _candidate_signal_tokens(self, candidate: Dict[str, Any]) -> Dict[str, set[str]]:
        uri = candidate.get("uri", "")
        return {
            "own": self._tokenize(candidate.get("abstract", "")),
            "edge": self._tokenize(self._edge_evidence_text(uri)),
            "uri": self._tokenize(uri),
            "category": self._tokenize(
                " ".join(
                    str(candidate.get(key, "") or "")
                    for key in ("category", "memory_type")
                )
            ),
        }

    def _edge_evidence_text(self, uri: str) -> str:
        parts: List[str] = []
        for edge in self._graph_index.get_forward_edges(uri):
            parts.append(edge.link_type)
            parts.append(edge.description)
        for edge in self._graph_index.get_reverse_edges(uri):
            parts.append(edge.link_type)
            parts.append(edge.description)
        return " ".join(part for part in parts if part)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in _TOKEN_RE.findall(text.lower())
            if len(token) > 1 and token not in _QUERY_STOPWORDS
        }

    @staticmethod
    def _normalize_graph_query(query_text: str | None) -> str:
        text = query_text or ""
        text = _CURRENT_DATE_PREFIX_RE.sub("", text)
        text = _ANSWER_DIRECTLY_PREFIX_RE.sub("", text)
        return text.strip()

    def _node_degree(self, uri: str) -> int:
        if not self._graph_index.has_node(uri):
            return 0
        return len(self._graph_index.get_forward_edges(uri)) + len(
            self._graph_index.get_reverse_edges(uri)
        )

    @staticmethod
    def _degree_specificity(degree: int) -> float:
        return 1.0 / ((1 + max(0, degree)) ** _DEGREE_PENALTY_POWER)

    @staticmethod
    def _candidate_semantic_score(candidate: Dict[str, Any]) -> float:
        score = candidate.get("_final_score", candidate.get("_score", 0.0))
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return 0.0
        return float(score)

    def _own_evidence_threshold(self, candidate: Dict[str, Any]) -> float:
        if self._requires_strong_own_evidence(candidate):
            return _MIN_STRONG_OWN_QUERY_EVIDENCE
        return _MIN_OWN_QUERY_EVIDENCE

    def _requires_strong_own_evidence(self, candidate: Dict[str, Any]) -> bool:
        uri = str(candidate.get("uri", "") or "").lower()
        memory_type = str(candidate.get("memory_type", "") or "").lower()
        category = str(candidate.get("category", "") or "").lower()
        degree = candidate.get("_graph_degree")
        if not isinstance(degree, (int, float)) or not math.isfinite(degree):
            degree = self._node_degree(uri)

        profile_like = (
            "/entities/person/" in uri
            or uri.endswith("/profile.md")
            or memory_type in {"profile"}
            or category in {"person", "profile"}
        )
        return profile_like or degree >= _HIGH_RISK_GRAPH_DEGREE

    @staticmethod
    def _normalize_target_dirs(target_dirs: List[str] | None) -> List[str]:
        prefixes: List[str] = []
        for target_dir in target_dirs or []:
            if target_dir:
                prefixes.append(target_dir.rstrip("/"))
        return list(dict.fromkeys(prefixes))

    @staticmethod
    def _is_uri_under_targets(uri: str, target_prefixes: List[str]) -> bool:
        uri_norm = uri.rstrip("/")
        for prefix in target_prefixes:
            if uri_norm == prefix or uri_norm.startswith(prefix + "/"):
                return True
        return False

    def _attach_path_metadata(
        self,
        candidates: List[Dict[str, Any]],
        paths: List,
    ) -> None:
        """Attach graph paths to candidates for explainability."""
        candidates_by_uri = {candidate["uri"]: candidate for candidate in candidates}
        for path in paths:
            for node_uri in path.nodes:
                candidate = candidates_by_uri.get(node_uri)
                if candidate is None:
                    continue
                path_info = {
                    "path": " -> ".join(path.nodes),
                    "link_types": [edge.link_type for edge in path.edges],
                    "path_score": path.path_score,
                }
                candidate.setdefault("_graph_paths", []).append(path_info)

    async def _fill_abstracts_for_graph_nodes(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
    ) -> List[Dict[str, Any]]:
        """Read plain content for graph-discovered memory nodes."""
        graph_uris = {
            candidate["uri"]
            for candidate in candidates
            if candidate.get("_from_graph") and not candidate.get("abstract")
        }
        if not graph_uris:
            return candidates

        viking_fs = get_viking_fs()
        if not viking_fs:
            return candidates

        for candidate in candidates:
            uri = candidate.get("uri", "")
            if uri not in graph_uris:
                continue
            try:
                content = await viking_fs.read_file(uri, ctx=ctx)
                if content:
                    mf = MemoryFileUtils.read(content, uri=uri)
                    candidate["abstract"] = mf.plain_content()
            except Exception:
                logger.debug("[GraphRetriever] failed to read graph node %s", uri)

        return candidates
