# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Evidence-first graph retrieval.

Semantic retrieval remains the primary route. Graph links are used only as
bounded auxiliary evidence, selected by a pluggable edge selector.
"""

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking.models.embedder.base import embed_compat
from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex, GraphNode
from openviking.retrieve.graph.score_normalizer import minmax_normalize
from openviking.server.identity import RequestContext
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.utils.config import RetrievalConfig
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_EVIDENCE_GRAPH_STRATEGY = "evidence_graph_v2"
_EDGE_PREFILTER_LIMIT = 96
_EDGE_SELECTOR_TOPK = 24
_LLM_PREFILTER_TOPK = 24
_DIRECT_GRAPH_LIMIT = 2
_COVERAGE_GRAPH_LIMIT = 6
_DEGREE_PENALTY_POWER = 0.5
_MIN_EDGE_SUPPORT = 1e-12
_GRAPH_SCORE_CEILING_FRACTION = 0.25
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9_'-]*\b")
_CURRENT_DATE_PREFIX_RE = re.compile(
    r"^\s*current\s+date\s*:\s*\d{4}-\d{2}-\d{2}\s*\.\s*",
    re.IGNORECASE,
)
_ANSWER_DIRECTLY_PREFIX_RE = re.compile(
    r"^\s*answer\s+the\s+question\s+directly\s*:\s*",
    re.IGNORECASE,
)
_QUESTION_WORDS = {
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "who",
    "did",
    "does",
    "do",
    "has",
    "have",
    "had",
    "is",
    "are",
    "was",
    "were",
    "us",
    "u",
    "s",
}
_ANSWER_TYPE_GROUPS = (
    {"place", "city", "country", "location", "area"},
    {"book", "novel", "title"},
    {"author", "writer", "person"},
    {"media", "movie", "film", "show", "game", "music", "song"},
    {"sport", "activity", "training", "exercise", "hobby"},
    {"event", "competition", "performance", "achievement"},
    {"count", "number"},
    {"topic", "genre", "subject", "research"},
    {"item", "gift", "object", "food"},
)
_EDGE_VECTOR_CACHE: Dict[Tuple[str, str, str], List[float]] = {}


@dataclass(frozen=True)
class EvidenceQueryPlan:
    normalized_query: str
    subjects: Set[str] = field(default_factory=set)
    relation_slots: Set[str] = field(default_factory=set)
    answer_type: str = ""
    aggregation_mode: str = "single"
    confidence: float = 0.0

    @property
    def coverage_mode(self) -> bool:
        return self.aggregation_mode in {"list_all", "count", "intersection", "frequency"}


@dataclass(frozen=True)
class SelectedEdge:
    edge: GraphEdge
    score: float
    reason: str
    selector: str
    edge_id: str = ""


class EvidenceSlotRegistry:
    """Compatibility shim: exact relation-slot planning is intentionally disabled."""

    @classmethod
    def slots_for_query(cls, query: str) -> Set[str]:
        return set()


class EvidenceQueryPlanner:
    """Generic graph intent planner.

    The fallback path only predicts coarse retrieval intent. Domain-specific
    relation slots are left to edge embeddings and evidence fields.
    """

    async def plan(self, query_text: Optional[str], intent_source: str = "fallback") -> EvidenceQueryPlan:
        query = self._normalize_query(query_text)
        if intent_source == "llm":
            llm_plan = await self._llm_plan(query)
            if llm_plan is not None:
                return llm_plan
        return self.fallback_plan(query)

    def fallback_plan(self, query_text: Optional[str]) -> EvidenceQueryPlan:
        query = self._normalize_query(query_text)
        answer_type = self._answer_type(query)
        aggregation_mode = self._aggregation_mode(query)
        subjects = self._subjects(query)
        confidence = 0.0
        if subjects:
            confidence += 0.30
        if answer_type:
            confidence += 0.40
        if aggregation_mode != "single":
            confidence += 0.30
        return EvidenceQueryPlan(
            normalized_query=query,
            subjects=subjects,
            relation_slots=set(),
            answer_type=answer_type,
            aggregation_mode=aggregation_mode,
            confidence=min(1.0, confidence),
        )

    async def _llm_plan(self, query: str) -> Optional[EvidenceQueryPlan]:
        if not query:
            return None
        prompt = (
            "Analyze this retrieval query for graph evidence selection. "
            "Return JSON only with keys: subjects (array of names/entities), "
            "answer_kind (one of place, person, book, author, media, activity, "
            "item, event, topic, date, number, unknown), aggregation_mode "
            "(one of single, list_all, count, intersection, frequency).\n"
            f"Query: {query}"
        )
        try:
            from openviking_cli.utils.config import get_openviking_config
            from openviking_cli.utils.llm import parse_json_from_response

            response = await get_openviking_config().vlm.get_completion_async(prompt)
            data = parse_json_from_response(response)
            if not isinstance(data, dict):
                return None
            subjects = {
                _normalize_label(str(item))
                for item in data.get("subjects", [])
                if _normalize_label(str(item))
            }
            answer_type = self._normalize_answer_type(data.get("answer_kind", ""))
            aggregation_mode = self._normalize_aggregation_mode(data.get("aggregation_mode", ""))
            confidence = 0.0
            if subjects:
                confidence += 0.30
            if answer_type:
                confidence += 0.40
            if aggregation_mode != "single":
                confidence += 0.30
            return EvidenceQueryPlan(
                normalized_query=query,
                subjects=subjects,
                relation_slots=set(),
                answer_type=answer_type,
                aggregation_mode=aggregation_mode,
                confidence=min(1.0, confidence),
            )
        except Exception as exc:
            logger.debug("[EvidenceQueryPlanner] LLM intent planning failed: %s", exc)
            return None

    @staticmethod
    def _normalize_query(query_text: Optional[str]) -> str:
        text = query_text or ""
        text = _CURRENT_DATE_PREFIX_RE.sub("", text)
        text = _ANSWER_DIRECTLY_PREFIX_RE.sub("", text)
        return text.strip()

    @staticmethod
    def _subjects(query: str) -> Set[str]:
        subjects: Set[str] = set()
        for token in _CAPITALIZED_RE.findall(query):
            normalized = token.lower().strip("'")
            if len(normalized) <= 1 or normalized in _QUESTION_WORDS:
                continue
            subjects.add(normalized)
        return subjects

    @staticmethod
    def _answer_type(query: str) -> str:
        q = query.lower()
        if re.search(r"\bauthors?\b|\bwriters?\b", q):
            return "author"
        if re.search(r"\bbooks?\b|\bnovels?\b", q):
            return "book"
        if re.search(r"\bwhere\b|\bplaces?\b|\blocations?\b|\bcit(?:y|ies)\b|\bareas?\b|\bcountr(?:y|ies)\b", q):
            return "place"
        if re.search(r"\bmovies?\b|\bfilms?\b|\bshows?\b|\bseries\b|\bgames?\b|\bmusic\b|\bsongs?\b", q):
            return "media"
        if re.search(r"\bsports?\b|\bactivities\b|\bhobbies\b|\btraining\b|\bexercises?\b|\bdestress\b|\brelax\b", q):
            return "activity"
        if re.search(r"\bitems?\b|\bgifts?\b|\bdesserts?\b|\bfood\b|\bbought\b|\breceived\b", q):
            return "item"
        if re.search(r"\bwho\b|\bpeople\b|\bchildren\b|\bkids\b|\bnames?\b", q):
            return "person"
        if re.search(r"\bevents?\b|\btournaments?\b|\bperformances?\b|\bachievements?\b|\bwon\b", q):
            return "event"
        if re.search(r"\bresearch(?:ed)?\b|\bsubject\b|\binspired\b|\bwritings?\b", q):
            return "topic"
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q):
            return "count"
        return ""

    @staticmethod
    def _aggregation_mode(query: str) -> str:
        q = query.lower()
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q):
            return "count"
        if re.search(r"\bboth\b|\bshare\b|\bcommon\b", q):
            return "intersection"
        if re.search(r"\bmost frequently\b|\bmost often\b", q):
            return "frequency"
        if re.search(r"\bwhich\b|\bwhat (?:are|books|authors|items|sports|activities|cities|locations|performances|movies|games|gifts|events|places|people|names)\b", q):
            return "list_all"
        return "single"

    @staticmethod
    def _normalize_answer_type(value: Any) -> str:
        normalized = _normalize_label(str(value)).replace(" ", "_")
        aliases = {
            "answer": "",
            "unknown": "",
            "location": "place",
            "city": "place",
            "country": "place",
            "area": "place",
            "title": "book",
            "movie": "media",
            "film": "media",
            "game": "media",
            "music": "media",
            "song": "media",
            "sport": "activity",
            "training": "activity",
            "exercise": "activity",
            "hobby": "activity",
            "gift": "item",
            "food": "item",
            "object": "item",
            "achievement": "event",
            "performance": "event",
            "research": "topic",
            "subject": "topic",
            "number": "count",
            "date": "date",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {"place", "person", "book", "author", "media", "activity", "item", "event", "topic", "date", "count"}
        return normalized if normalized in allowed else ""

    @staticmethod
    def _normalize_aggregation_mode(value: Any) -> str:
        normalized = _normalize_label(str(value)).replace(" ", "_")
        aliases = {
            "list": "list_all",
            "list_or_set": "list_all",
            "many": "list_all",
            "number": "count",
            "both": "intersection",
            "common": "intersection",
            "shared": "intersection",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"single", "list_all", "count", "intersection", "frequency"} else "single"


class EdgeSelector:
    """Base selector with shared cheap gates and edge document helpers."""

    selector_name = "base"

    def __init__(self, graph_index: GraphIndex, retrieval_config: RetrievalConfig, embedder: Any = None):
        self._graph_index = graph_index
        self._config = retrieval_config
        self._embedder = embedder

    async def select(
        self,
        query: str,
        query_vector: Optional[List[float]],
        plan: EvidenceQueryPlan,
        edges: List[GraphEdge],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[SelectedEdge]:
        raise NotImplementedError

    def _prefilter_edges(
        self,
        query: str,
        plan: EvidenceQueryPlan,
        edges: List[GraphEdge],
        limit: int = _EDGE_PREFILTER_LIMIT,
    ) -> List[tuple[GraphEdge, float, str]]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        scored: List[tuple[GraphEdge, float, str]] = []
        for edge in edges:
            allowed, reason = self._cheap_gate(edge, plan)
            if not allowed:
                continue
            cheap_score = self._cheap_score(edge, plan, query_tokens)
            if cheap_score <= 0:
                continue
            scored.append((edge, cheap_score, reason))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _cheap_gate(self, edge: GraphEdge, plan: EvidenceQueryPlan) -> tuple[bool, str]:
        if edge.link_type != "evidence_for":
            return False, "rejected:non_evidence_link_type"
        if _subject_mismatch(edge, plan):
            return False, "rejected:subject_mismatch"
        if not edge.answer_value and not edge.source_span:
            return False, "rejected:no_answer_value_or_source_span"
        edge_type = _edge_answer_type(self._graph_index, edge)
        if plan.answer_type and edge_type and not _equivalent_type(edge_type, plan.answer_type):
            return False, "rejected:answer_kind_mismatch"
        return True, "accepted:cheap_gate"

    def _cheap_score(self, edge: GraphEdge, plan: EvidenceQueryPlan, query_tokens: Set[str]) -> float:
        edge_doc = self.edge_document(edge)
        edge_tokens = set(_TOKEN_RE.findall(edge_doc.lower()))
        overlap = len(query_tokens & edge_tokens) / max(len(query_tokens), 1)
        edge_type = _edge_answer_type(self._graph_index, edge)
        type_score = 1.0 if plan.answer_type and _equivalent_type(edge_type, plan.answer_type) else 0.0
        slot_score = 0.0
        completeness = 1.0 if edge.answer_value and edge.source_span else 0.6
        coverage = 1.0 if plan.coverage_mode else 0.5
        return (
            0.35 * overlap
            + 0.35 * type_score
            + 0.00 * slot_score
            + 0.15 * max(0.0, min(float(edge.weight or 0.0), 1.0))
            + 0.05 * completeness
            + 0.05 * coverage
        )

    def edge_document(self, edge: GraphEdge) -> str:
        node = self._graph_index.get_node(edge.to_uri) or self._graph_index.get_node(edge.from_uri)
        parts = [
            f"subject: {edge.subject}",
            f"relation: {edge.relation_slot}",
            f"answer: {', '.join(edge.answer_value or [])}",
            f"evidence: {edge.source_span}",
            f"description: {edge.description}",
            f"target: {_uri_title(edge.to_uri)}",
        ]
        if node:
            parts.append(f"target_type: {node.memory_type or node.category}")
        return " | ".join(part for part in parts if part and not part.endswith(": "))


class EmbeddingEdgeSelector(EdgeSelector):
    """Default low-latency selector: cheap gates plus edge/query embedding similarity."""

    selector_name = "embedding"

    async def select(
        self,
        query: str,
        query_vector: Optional[List[float]],
        plan: EvidenceQueryPlan,
        edges: List[GraphEdge],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[SelectedEdge]:
        prefiltered = self._prefilter_edges(query, plan, edges)
        if not prefiltered:
            return []
        if not self._embedder or not query_vector:
            return [
                SelectedEdge(edge=edge, score=score, reason=f"{reason}:lexical_fallback", selector=self.selector_name)
                for edge, score, reason in prefiltered[: self._topk(plan)]
            ]

        scored: List[SelectedEdge] = []
        sem = asyncio.Semaphore(8)

        async def _score(edge: GraphEdge, cheap_score: float, reason: str) -> Optional[SelectedEdge]:
            async with sem:
                vector = await self._edge_vector(edge)
            if not vector:
                return SelectedEdge(
                    edge=edge,
                    score=cheap_score,
                    reason=f"{reason}:edge_embedding_missing",
                    selector=self.selector_name,
                )
            similarity = max(0.0, _cosine_similarity(query_vector, vector))
            score = 0.75 * similarity + 0.25 * cheap_score
            return SelectedEdge(
                edge=edge,
                score=score,
                reason=f"{reason}:embedding_similarity={similarity:.3f}",
                selector=self.selector_name,
            )

        results = await asyncio.gather(
            *(_score(edge, cheap_score, reason) for edge, cheap_score, reason in prefiltered)
        )
        scored = [item for item in results if item is not None and item.score > 0]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self._topk(plan)]

    async def _edge_vector(self, edge: GraphEdge) -> List[float]:
        key = (
            _embedder_identity(self._embedder),
            getattr(self._graph_index, "_space_key", "") or "",
            _edge_fingerprint(edge),
        )
        cached = _EDGE_VECTOR_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            result = await embed_compat(self._embedder, self.edge_document(edge), is_query=False)
            vector = result.dense_vector or []
        except Exception as exc:
            logger.debug("[EmbeddingEdgeSelector] edge embedding failed: %s", exc)
            vector = []
        _EDGE_VECTOR_CACHE[key] = vector
        return vector

    def _topk(self, plan: EvidenceQueryPlan) -> int:
        base = _EDGE_SELECTOR_TOPK if plan.coverage_mode else min(_EDGE_SELECTOR_TOPK, 12)
        return max(1, min(base, self._config.graph_expansion_topk))


class LLMEdgeSelector(EdgeSelector):
    """Optional high-quality selector with embedding prefilter and safe fallback."""

    selector_name = "llm"

    def __init__(self, graph_index: GraphIndex, retrieval_config: RetrievalConfig, embedder: Any = None):
        super().__init__(graph_index, retrieval_config, embedder)
        self._fallback = EmbeddingEdgeSelector(graph_index, retrieval_config, embedder)

    async def select(
        self,
        query: str,
        query_vector: Optional[List[float]],
        plan: EvidenceQueryPlan,
        edges: List[GraphEdge],
        semantic_candidates: List[Dict[str, Any]],
    ) -> List[SelectedEdge]:
        candidates = await self._fallback.select(query, query_vector, plan, edges, semantic_candidates)
        candidates = candidates[:_LLM_PREFILTER_TOPK]
        if not candidates:
            return []
        try:
            prompt = self._prompt(query, plan, candidates)
            from openviking_cli.utils.config import get_openviking_config

            response = await get_openviking_config().vlm.get_completion_async(prompt)
            selected = self._parse_response(response)
            if not selected:
                return [
                    SelectedEdge(item.edge, item.score, f"{item.reason}:fallback_llm_empty", item.selector, item.edge_id)
                    for item in candidates
                ]
            by_id = {item.edge_id: item for item in candidates}
            reranked: List[SelectedEdge] = []
            for edge_id, confidence, reason in selected:
                item = by_id.get(edge_id)
                if item is None:
                    continue
                reranked.append(
                    SelectedEdge(
                        edge=item.edge,
                        score=max(item.score, confidence),
                        reason=f"llm_selected:{reason or item.reason}",
                        selector=self.selector_name,
                        edge_id=edge_id,
                    )
                )
            return reranked or [
                SelectedEdge(item.edge, item.score, f"{item.reason}:fallback_llm_no_valid_ids", item.selector, item.edge_id)
                for item in candidates
            ]
        except Exception as exc:
            logger.debug("[LLMEdgeSelector] failed, falling back to embedding: %s", exc)
            return [
                SelectedEdge(item.edge, item.score, f"{item.reason}:fallback_llm_error", item.selector, item.edge_id)
                for item in candidates
            ]

    def _prompt(self, query: str, plan: EvidenceQueryPlan, candidates: List[SelectedEdge]) -> str:
        lines = [
            "Select evidence edges useful for answering the query. Return JSON only.",
            f"Query: {query}",
            f"Need: answer_kind={plan.answer_type or 'unknown'}, aggregation={plan.aggregation_mode}",
            "Edges:",
        ]
        for idx, item in enumerate(candidates):
            edge_id = f"E{idx}"
            object.__setattr__(item, "edge_id", edge_id)
            edge = item.edge
            lines.append(
                f"{edge_id} | subject={edge.subject} | slot={edge.relation_slot} | "
                f"answer={', '.join(edge.answer_value or [])} | evidence={edge.source_span[:220]}"
            )
        lines.append(
            '{"selected_edges":[{"id":"E0","confidence":0.9,"reason":"short reason"}]}'
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_response(response: Any) -> List[tuple[str, float, str]]:
        text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        data = json.loads(text)
        raw_items = data.get("selected_edges") or data.get("edges") or data.get("edge_ids") or []
        parsed: List[tuple[str, float, str]] = []
        for item in raw_items:
            if isinstance(item, str):
                parsed.append((item, 1.0, ""))
            elif isinstance(item, dict):
                edge_id = str(item.get("id") or item.get("edge_id") or "")
                if not edge_id:
                    continue
                confidence = float(item.get("confidence", item.get("score", 1.0)) or 1.0)
                parsed.append((edge_id, max(0.0, min(1.0, confidence)), str(item.get("reason", ""))))
        return parsed


class EvidenceGraphRetriever:
    """Semantic-preserving graph expansion using selected evidence links."""

    def __init__(self, graph_index: GraphIndex, retrieval_config: RetrievalConfig):
        self._graph_index = graph_index
        self._config = retrieval_config
        self._planner = EvidenceQueryPlanner()

    async def expand(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
        limit: int,
        target_dirs: List[str] | None = None,
        level: List[int] | None = None,
        query_text: str | None = None,
        query_vector: Optional[List[float]] = None,
        embedder: Any = None,
    ) -> List[Dict[str, Any]]:
        plan = await self._planner.plan(query_text, self._config.graph_intent_source)
        logger.info(
            "[EvidenceGraphRetriever] intent_plan source=%s query=%r plan=%s",
            self._config.graph_intent_source,
            query_text,
            self._debug_plan(plan),
        )
        if not candidates or not plan.normalized_query or plan.confidence < 0.40:
            return candidates

        selector = self._make_selector(embedder)
        selected_edges = await selector.select(
            query=plan.normalized_query,
            query_vector=query_vector,
            plan=plan,
            edges=self._graph_index.get_all_edges(),
            semantic_candidates=candidates,
        )
        support_scores, support_edges, support_selected = self._support_from_selected(selected_edges, plan)
        if not support_scores:
            return candidates

        logger.info(
            "[EvidenceGraphRetriever] strategy=%s selector=%s plan=%s selected=%s support=%s",
            _EVIDENCE_GRAPH_STRATEGY,
            getattr(selector, "selector_name", ""),
            self._debug_plan(plan),
            len(selected_edges),
            len(support_scores),
        )

        expanded = self._merge_candidates(
            candidates=candidates,
            support_scores=support_scores,
            support_edges=support_edges,
            support_selected=support_selected,
            target_dirs=target_dirs,
            level=level,
            plan=plan,
        )
        expanded = await self._fill_abstracts_for_graph_nodes(expanded, ctx)
        expanded = self._score_candidates(expanded, support_scores, support_edges, support_selected, plan, limit)
        expanded = self._aggregation_complete(
            expanded,
            support_edges,
            support_selected,
            support_scores,
            plan,
            target_dirs,
            level,
        )
        expanded = await self._fill_abstracts_for_graph_nodes(expanded, ctx)
        return self._select_candidates(expanded, plan)

    def _make_selector(self, embedder: Any) -> EdgeSelector:
        if self._config.graph_edge_selector == "llm":
            return LLMEdgeSelector(self._graph_index, self._config, embedder)
        return EmbeddingEdgeSelector(self._graph_index, self._config, embedder)

    def _support_from_selected(
        self,
        selected_edges: List[SelectedEdge],
        plan: EvidenceQueryPlan,
    ) -> tuple[Dict[str, float], Dict[str, GraphEdge], Dict[str, SelectedEdge]]:
        support_scores: Dict[str, float] = {}
        support_edges: Dict[str, GraphEdge] = {}
        support_selected: Dict[str, SelectedEdge] = {}
        for selected in sorted(selected_edges, key=lambda item: item.score, reverse=True):
            if not self._edge_can_include(selected.edge, plan):
                continue
            for uri in self._candidate_uris_from_edge(selected.edge):
                score = selected.score * self._degree_specificity(self._node_degree(uri))
                if score <= _MIN_EDGE_SUPPORT or score <= support_scores.get(uri, 0.0):
                    continue
                support_scores[uri] = score
                support_edges[uri] = selected.edge
                support_selected[uri] = selected
        return support_scores, support_edges, support_selected

    def _merge_candidates(
        self,
        *,
        candidates: List[Dict[str, Any]],
        support_scores: Dict[str, float],
        support_edges: Dict[str, GraphEdge],
        support_selected: Dict[str, SelectedEdge],
        target_dirs: List[str] | None,
        level: List[int] | None,
        plan: EvidenceQueryPlan,
    ) -> List[Dict[str, Any]]:
        existing: Dict[str, Dict[str, Any]] = {
            candidate.get("uri", ""): candidate
            for candidate in candidates
            if candidate.get("uri", "")
        }
        norm_support = minmax_normalize(support_scores)
        target_prefixes = self._normalize_target_dirs(target_dirs)
        allowed_levels = set(level) if level is not None else None
        pool_limit = max(self._config.graph_expansion_topk * 2, self._config.graph_expansion_topk)
        added = 0
        for uri in sorted(support_scores, key=support_scores.get, reverse=True):
            if uri in existing:
                continue
            if not self._can_add_uri(uri, target_prefixes, allowed_levels):
                continue
            edge = support_edges.get(uri)
            if edge and not self._edge_can_include(edge, plan):
                continue
            node = self._graph_index.get_node(uri)
            candidate = self._new_graph_candidate(uri, node)
            self._attach_graph_metadata(candidate, support_scores, norm_support, support_edges, support_selected, plan)
            existing[uri] = candidate
            added += 1
            if added >= pool_limit:
                break
        if added:
            logger.info("[EvidenceGraphRetriever] added %s selected evidence graph candidates", added)
        return list(existing.values())

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        support_scores: Dict[str, float],
        support_edges: Dict[str, GraphEdge],
        support_selected: Dict[str, SelectedEdge],
        plan: EvidenceQueryPlan,
        limit: int,
    ) -> List[Dict[str, Any]]:
        semantic_scores = sorted(
            (
                self._candidate_semantic_score(candidate)
                for candidate in candidates
                if not candidate.get("_from_graph")
            ),
            reverse=True,
        )
        semantic_floor = semantic_scores[min(max(limit, 1), len(semantic_scores)) - 1] if semantic_scores else 0.0
        semantic_top = semantic_scores[0] if semantic_scores else 1.0
        semantic_span = max(semantic_top - semantic_floor, 1e-6)
        score_ceiling = semantic_floor + semantic_span * _GRAPH_SCORE_CEILING_FRACTION
        norm_support = minmax_normalize(support_scores)

        for candidate in candidates:
            uri = candidate.get("uri", "")
            semantic_score = self._candidate_semantic_score(candidate)
            candidate["_graph_strategy"] = _EVIDENCE_GRAPH_STRATEGY
            candidate["_evidence_query_plan"] = self._debug_plan(plan)
            candidate["_mnemis_coverage_mode"] = plan.coverage_mode
            if not candidate.get("_from_graph"):
                candidate["_final_score"] = semantic_score
                candidate["_graph_accepted"] = False
                continue

            edge = support_edges.get(uri)
            selected = support_selected.get(uri)
            accepted, reason = self._candidate_acceptance(candidate, edge, plan)
            path_signal = norm_support.get(uri, 0.0) * candidate.get("_graph_specificity", 1.0)
            edge_score = selected.score if selected else 0.0
            boost = self._config.graph_alpha * path_signal * max(edge_score, 0.0) * semantic_span if accepted else 0.0
            candidate["_graph_accepted"] = accepted
            candidate["_graph_accept_reason"] = reason
            candidate["_graph_path_signal"] = path_signal
            candidate["_graph_boost"] = boost
            candidate["_final_score"] = min(score_ceiling, semantic_floor + boost) if accepted else semantic_floor
        return candidates

    def _candidate_acceptance(
        self,
        candidate: Dict[str, Any],
        edge: Optional[GraphEdge],
        plan: EvidenceQueryPlan,
    ) -> tuple[bool, str]:
        if edge is None:
            return False, "rejected:no_supporting_evidence_edge"
        if not self._edge_can_include(edge, plan):
            return False, "rejected:evidence_gate"
        if self._node_kind(candidate.get("uri", ""), candidate) == "person/profile":
            return False, "rejected:profile_hub"
        return True, "accepted:selected_evidence_edge"

    def _edge_can_include(self, edge: GraphEdge, plan: EvidenceQueryPlan) -> bool:
        if edge.link_type != "evidence_for":
            return False
        if _subject_mismatch(edge, plan):
            return False
        if not edge.answer_value and not edge.source_span:
            return False
        edge_type = _edge_answer_type(self._graph_index, edge)
        if plan.answer_type and edge_type and not _equivalent_type(edge_type, plan.answer_type):
            return False
        return True

    def _aggregation_complete(
        self,
        candidates: List[Dict[str, Any]],
        support_edges: Dict[str, GraphEdge],
        support_selected: Dict[str, SelectedEdge],
        support_scores: Dict[str, float],
        plan: EvidenceQueryPlan,
        target_dirs: List[str] | None,
        level: List[int] | None,
    ) -> List[Dict[str, Any]]:
        if not plan.coverage_mode:
            return candidates
        accepted_keys = {
            self._aggregation_key(edge, plan)
            for edge in support_edges.values()
            if self._aggregation_key(edge, plan) and self._edge_can_include(edge, plan)
        }
        if not accepted_keys:
            return candidates

        existing = {candidate.get("uri", ""): candidate for candidate in candidates if candidate.get("uri", "")}
        target_prefixes = self._normalize_target_dirs(target_dirs)
        allowed_levels = set(level) if level is not None else None
        added = 0
        for edge in self._graph_index.get_all_edges():
            if self._aggregation_key(edge, plan) not in accepted_keys:
                continue
            if not self._edge_can_include(edge, plan):
                continue
            selected = SelectedEdge(
                edge=edge,
                score=max(0.05, min(1.0, float(edge.weight or 0.0))),
                reason="accepted:aggregation_answer_kind_completion",
                selector="aggregation",
            )
            for uri in self._candidate_uris_from_edge(edge):
                if uri in existing:
                    continue
                if not self._can_add_uri(uri, target_prefixes, allowed_levels):
                    continue
                node = self._graph_index.get_node(uri)
                candidate = self._new_graph_candidate(uri, node)
                support_scores[uri] = selected.score
                support_edges[uri] = edge
                support_selected[uri] = selected
                norm_support = minmax_normalize(support_scores)
                self._attach_graph_metadata(candidate, support_scores, norm_support, support_edges, support_selected, plan)
                candidate["_evidence_aggregation_completed"] = True
                candidate["_graph_accepted"] = True
                candidate["_graph_accept_reason"] = selected.reason
                candidate["_final_score"] = selected.score
                existing[uri] = candidate
                added += 1
                if added >= self._config.graph_expansion_topk:
                    break
        if added:
            logger.info("[EvidenceGraphRetriever] aggregation completed with %s members", added)
        return list(existing.values())

    def _candidate_uris_from_edge(self, edge: GraphEdge) -> List[str]:
        uris = []
        for uri in (edge.to_uri, edge.from_uri):
            node = self._graph_index.get_node(uri)
            if not node or node.is_summary:
                continue
            kind = self._node_kind(uri, {"memory_type": node.memory_type or "", "category": node.category or ""})
            if kind == "person/profile":
                continue
            uris.append(uri)
        return list(dict.fromkeys(uris))

    def _attach_graph_metadata(
        self,
        candidate: Dict[str, Any],
        support_scores: Dict[str, float],
        norm_support: Dict[str, float],
        support_edges: Dict[str, GraphEdge],
        support_selected: Dict[str, SelectedEdge],
        plan: EvidenceQueryPlan,
    ) -> None:
        uri = candidate.get("uri", "")
        edge = support_edges.get(uri)
        selected = support_selected.get(uri)
        degree = self._node_degree(uri)
        candidate["_graph_degree"] = degree
        candidate["_graph_specificity"] = self._degree_specificity(degree)
        candidate["_graph_support"] = support_scores.get(uri, 0.0)
        candidate["_norm_graph_support"] = norm_support.get(uri, 0.0)
        candidate["_evidence_subject"] = edge.subject if edge else ""
        candidate["_evidence_relation_slot"] = edge.relation_slot if edge else ""
        candidate["_evidence_answer_type"] = _edge_answer_type(self._graph_index, edge) if edge else ""
        candidate["_evidence_role"] = edge.evidence_role if edge else ""
        candidate["_evidence_answer_value"] = edge.answer_value if edge else []
        candidate["_evidence_source_span"] = edge.source_span if edge else ""
        candidate["_evidence_aggregation_key"] = self._aggregation_key(edge, plan) if edge else ""
        candidate["_evidence_edge_score"] = selected.score if selected else 0.0
        candidate["_evidence_selector"] = selected.selector if selected else ""
        candidate["_evidence_gate_reason"] = selected.reason if selected else ""

    def _select_candidates(self, candidates: List[Dict[str, Any]], plan: EvidenceQueryPlan) -> List[Dict[str, Any]]:
        semantic = [candidate for candidate in candidates if not candidate.get("_from_graph")]
        graph = [
            candidate
            for candidate in candidates
            if candidate.get("_from_graph") and candidate.get("_graph_accepted")
        ]
        semantic = sorted(semantic, key=lambda c: c.get("_final_score", 0.0), reverse=True)
        graph = sorted(
            graph,
            key=lambda c: (
                bool(c.get("_evidence_aggregation_completed")),
                float(c.get("_evidence_edge_score", 0.0) or 0.0),
                float(c.get("_graph_support", 0.0) or 0.0),
            ),
            reverse=True,
        )
        graph_limit = _COVERAGE_GRAPH_LIMIT if plan.coverage_mode else _DIRECT_GRAPH_LIMIT
        graph_limit = max(0, min(self._config.graph_expansion_topk, graph_limit, len(graph)))
        return semantic + graph[:graph_limit]

    @classmethod
    def _aggregation_key(cls, edge: GraphEdge, plan: EvidenceQueryPlan) -> str:
        if not edge or not edge.subject:
            return ""
        subject = _normalize_label(edge.subject).replace(" ", "_")
        answer_type = _edge_answer_type_static(edge) or plan.answer_type
        if not answer_type:
            return ""
        return f"{subject}.{answer_type}"

    def _can_add_uri(
        self,
        uri: str,
        target_prefixes: List[str],
        allowed_levels: Optional[Set[int]],
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
        return True

    @staticmethod
    def _new_graph_candidate(uri: str, node: Optional[GraphNode]) -> Dict[str, Any]:
        return {
            "uri": uri,
            "_score": 0.0,
            "_final_score": 0.0,
            "_ppr_score": 0.0,
            "context_type": "memory",
            "level": 2,
            "abstract": "",
            "category": node.category if node else "",
            "memory_type": node.memory_type if node else "",
            "_from_graph": True,
        }

    async def _fill_abstracts_for_graph_nodes(
        self,
        candidates: List[Dict[str, Any]],
        ctx: RequestContext,
    ) -> List[Dict[str, Any]]:
        graph_uris = {
            candidate["uri"]
            for candidate in candidates
            if candidate.get("_from_graph") and not candidate.get("abstract")
        }
        if not graph_uris:
            return candidates
        try:
            viking_fs = get_viking_fs()
        except RuntimeError:
            return candidates
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
                logger.debug("[EvidenceGraphRetriever] failed to read graph node %s", uri)
        return candidates

    @staticmethod
    def _candidate_semantic_score(candidate: Dict[str, Any]) -> float:
        score = candidate.get("_final_score", candidate.get("_score", 0.0))
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return 0.0
        return float(score)

    def _node_degree(self, uri: str) -> int:
        if not self._graph_index.has_node(uri):
            return 0
        return len(self._graph_index.get_forward_edges(uri)) + len(self._graph_index.get_reverse_edges(uri))

    @staticmethod
    def _degree_specificity(degree: int) -> float:
        return 1.0 / ((1 + max(0, degree)) ** _DEGREE_PENALTY_POWER)

    @staticmethod
    def _normalize_target_dirs(target_dirs: List[str] | None) -> List[str]:
        return list(dict.fromkeys(target_dir.rstrip("/") for target_dir in target_dirs or [] if target_dir))

    @staticmethod
    def _is_uri_under_targets(uri: str, target_prefixes: List[str]) -> bool:
        uri_norm = uri.rstrip("/")
        return any(uri_norm == prefix or uri_norm.startswith(prefix + "/") for prefix in target_prefixes)

    @staticmethod
    def _node_kind(uri: str, candidate: Dict[str, Any]) -> str:
        uri_lower = str(uri or "").lower()
        memory_type = str(candidate.get("memory_type", "") or "").lower()
        category = str(candidate.get("category", "") or "").lower()
        if "/events/" in uri_lower or "/entities/event/" in uri_lower or memory_type in {"event", "events"}:
            return "event"
        if (
            "/entities/person/" in uri_lower
            or uri_lower.endswith("/profile.md")
            or memory_type in {"person", "profile"}
            or category in {"person", "profile"}
        ):
            return "person/profile"
        if "/preferences/" in uri_lower or memory_type in {"preference", "preferences"}:
            return "preference"
        if "/entities/" in uri_lower or memory_type in {"entity", "entities"}:
            return "entity"
        return "other"

    @staticmethod
    def _debug_plan(plan: EvidenceQueryPlan) -> Dict[str, Any]:
        return {
            "subjects": sorted(plan.subjects),
            "relation_slots": sorted(plan.relation_slots),
            "answer_type": plan.answer_type,
            "aggregation_mode": plan.aggregation_mode,
            "confidence": round(plan.confidence, 3),
        }


def _subject_mismatch(edge: GraphEdge, plan: EvidenceQueryPlan) -> bool:
    if not plan.subjects:
        return False
    if not edge.subject:
        return True
    return _normalize_label(edge.subject) not in plan.subjects


def _equivalent_type(left: str, right: str) -> bool:
    left = (left or "").strip().lower()
    right = (right or "").strip().lower()
    if not left or not right:
        return False
    if left == right:
        return True
    for group in _ANSWER_TYPE_GROUPS:
        if left in group and right in group:
            return True
    if right == "count" and left in {"event", "game", "competition", "person", "item"}:
        return True
    if right == "activity" and left in {"sport", "training", "exercise", "hobby"}:
        return True
    if right == "author" and left in {"book", "title"}:
        return True
    if right == "book" and left in {"author", "writer"}:
        return True
    if right == "item" and left in {"gift", "food", "object"}:
        return True
    if right == "media" and left in {"book", "title"}:
        return True
    return False


def _normalize_label(value: str) -> str:
    tokens = _TOKEN_RE.findall(str(value or "").lower())
    return " ".join(tokens)


def _slot_answer_type(relation_slot: str) -> str:
    slot = (relation_slot or "").strip().lower()
    explicit = {
        "visited_place": "place",
        "planned_trip_to": "place",
        "read_book": "book",
        "recommended_book": "book",
        "read_author": "author",
        "likes_sport": "activity",
        "does_activity": "activity",
        "does_training": "activity",
        "won_event": "event",
        "career_high_performance": "event",
        "writes_genre": "topic",
        "painted_subject": "topic",
        "received_gift": "item",
        "bought_item": "item",
        "destresses_by": "activity",
        "researched_topic": "topic",
        "has_child": "person",
    }
    if slot in explicit:
        return explicit[slot]
    slot_words = slot.replace("_", " ")
    if any(word in slot_words for word in ("place", "location", "city", "country", "destination", "travel", "trip")):
        return "place"
    if any(word in slot_words for word in ("book", "novel")):
        return "book"
    if any(word in slot_words for word in ("author", "writer")):
        return "author"
    if any(word in slot_words for word in ("movie", "film", "media", "franchise", "game", "music", "song")):
        return "media"
    if any(word in slot_words for word in ("sport", "activity", "training", "exercise", "hobby", "destress", "volunteer")):
        return "activity"
    if any(word in slot_words for word in ("gift", "item", "food", "dessert", "object", "bought", "received", "owns")):
        return "item"
    if any(word in slot_words for word in ("child", "children", "person", "people", "fan")):
        return "person"
    if any(word in slot_words for word in ("event", "tournament", "performance", "competition", "achievement", "meeting")):
        return "event"
    if any(word in slot_words for word in ("research", "topic", "subject", "genre", "inspired", "writing")):
        return "topic"
    return ""


def _edge_answer_type(graph_index: GraphIndex, edge: Optional[GraphEdge]) -> str:
    if edge is None:
        return ""
    slot_type = _slot_answer_type(edge.relation_slot)
    if slot_type:
        return slot_type
    node = graph_index.get_node(edge.to_uri) or graph_index.get_node(edge.from_uri)
    text = " ".join(
        [
            edge.to_uri,
            edge.from_uri,
            edge.description,
            edge.source_span,
            " ".join(edge.answer_value or []),
            node.category if node else "",
            node.memory_type if node else "",
        ]
    ).lower()
    if any(token in text for token in ("location", "place", "city", "country", "mountain", "california", "london")):
        return "place"
    if any(token in text for token in ("book", "novel")):
        return "book"
    if any(token in text for token in ("author", "writer")):
        return "author"
    if any(token in text for token in ("movie", "film", "game", "music", "song", "franchise")):
        return "media"
    if any(token in text for token in ("sport", "training", "exercise", "activity", "hobby")):
        return "activity"
    if any(token in text for token in ("gift", "item", "food", "dessert", "object")):
        return "item"
    if any(token in text for token in ("person", "people", "child", "children")):
        return "person"
    if any(token in text for token in ("event", "tournament", "performance", "competition")):
        return "event"
    return ""


def _edge_answer_type_static(edge: Optional[GraphEdge]) -> str:
    return _slot_answer_type(edge.relation_slot) if edge else ""


def _uri_title(uri: str) -> str:
    name = str(uri or "").rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".md"):
        name = name[:-3]
    return name.replace("_", " ").replace("-", " ")


def _edge_fingerprint(edge: GraphEdge) -> str:
    raw = "|".join(
        [
            edge.from_uri,
            edge.to_uri,
            edge.link_type,
            edge.subject,
            edge.relation_slot,
            ",".join(edge.answer_value or []),
            edge.source_span,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _embedder_identity(embedder: Any) -> str:
    if embedder is None:
        return "none"
    return ":".join(
        str(part)
        for part in (
            type(embedder).__module__,
            type(embedder).__qualname__,
            getattr(embedder, "model", ""),
            getattr(embedder, "model_name", ""),
        )
    )


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)
