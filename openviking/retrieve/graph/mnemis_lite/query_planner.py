# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Rule-based query planning for Mnemis-lite graph retrieval."""

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][a-z0-9]+(?:\s+(?:and\s+)?[A-Z][a-z0-9]+)*\b"
)
_CURRENT_DATE_PREFIX_RE = re.compile(
    r"^\s*current\s+date\s*:\s*\d{4}-\d{2}-\d{2}\s*\.\s*",
    re.IGNORECASE,
)
_ANSWER_DIRECTLY_PREFIX_RE = re.compile(
    r"^\s*answer\s+the\s+question\s+directly\s*:\s*",
    re.IGNORECASE,
)

_STOPWORDS = {
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
class GraphQueryPlan:
    """Lightweight retrieval-only plan for graph candidate selection."""

    query_type: str
    anchors: set[str] = field(default_factory=set)
    required_slots: set[str] = field(default_factory=set)
    constraints: set[str] = field(default_factory=set)
    preferred_uri_kinds: set[str] = field(default_factory=set)
    risky_uri_kinds: set[str] = field(default_factory=set)
    coverage_mode: bool = False
    normalized_query: str = ""


class RuleBasedGraphQueryPlanner:
    """Classify graph-retrieval needs without adding LLM calls."""

    def plan(self, query_text: str | None) -> GraphQueryPlan:
        query = self._normalize_query(query_text)
        query_type = self._query_type(query)
        required_slots = self._required_slots(query_type)
        preferred_kinds = self._preferred_uri_kinds(query_type)
        risky_kinds = self._risky_uri_kinds(query_type)
        return GraphQueryPlan(
            query_type=query_type,
            anchors=self._anchors(query),
            required_slots=required_slots,
            constraints=self._constraints(query),
            preferred_uri_kinds=preferred_kinds,
            risky_uri_kinds=risky_kinds,
            coverage_mode=query_type in {"count", "list_or_set", "multi_hop"},
            normalized_query=query,
        )

    @staticmethod
    def _normalize_query(query_text: str | None) -> str:
        text = query_text or ""
        text = _CURRENT_DATE_PREFIX_RE.sub("", text)
        text = _ANSWER_DIRECTLY_PREFIX_RE.sub("", text)
        return text.strip()

    @staticmethod
    def _query_type(query: str) -> str:
        q = query.lower()
        if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q):
            return "count"
        if re.search(
            r"what (?:recommendations?|advice|pointers?|tips|suggestions?)|"
            r"what activities|what games|what books|what .* (?:has|have) "
            r".* (?:played|done|participated|recommended|suggested|visited|attended|"
            r"written|watched)|list|kinds? of|types? of",
            q,
        ):
            return "list_or_set"
        if re.search(
            r"\bwhen\b|what (?:date|day|month|year)|which (?:date|day|month|year)|"
            r"\bhow long\b",
            q,
        ):
            return "time"
        if re.search(
            r"\bwhere\b|which cit(?:y|ies)|which (?:country|city|place|school)|"
            r"what (?:country|city|place|location|school)|location|moved from",
            q,
        ):
            return "place"
        if re.search(r"\bwhy\b|reason|motivat|decided|because", q):
            return "reason"
        if re.search(r"\bwhich\b", q):
            return "list_or_set"
        if re.search(
            r"likely|would|might|potentially|considering|advice|improve|support|"
            r"help|learn from|growth",
            q,
        ):
            return "inference"
        if re.search(r"relationship|status|partner|married|single|friend", q):
            return "relationship"
        return "attribute"

    @staticmethod
    def _constraints(query: str) -> set[str]:
        q = query.lower()
        constraints: set[str] = set()
        if re.search(r"\bbefore\b|\bafter\b|\bin\s+20\d{2}\b|\bon\s+20\d{2}", q):
            constraints.add("temporal")
        if re.search(
            r"\bcountry\b|\bcity\b|\bfestival\b|\bvenue\b|\bschool\b|\bplace\b|"
            r"\blocation\b",
            q,
        ):
            constraints.add("place")
        return constraints

    @staticmethod
    def _required_slots(query_type: str) -> set[str]:
        return {
            "time": {"date"},
            "place": {"place"},
            "count": {"item"},
            "list_or_set": {"item"},
            "reason": {"causal"},
            "inference": {"supporting_event"},
            "relationship": {"profile_or_relation"},
        }.get(query_type, {"attribute"})

    @staticmethod
    def _preferred_uri_kinds(query_type: str) -> set[str]:
        return {
            "time": {"event"},
            "place": {"event", "entity"},
            "count": {"event", "preference", "entity"},
            "list_or_set": {"event", "preference", "entity"},
            "reason": {"event"},
            "inference": {"event", "preference", "person/profile"},
            "relationship": {"person/profile", "entity"},
        }.get(query_type, {"event", "preference", "entity", "person/profile"})

    @staticmethod
    def _risky_uri_kinds(query_type: str) -> set[str]:
        if query_type in {"time", "count", "list_or_set", "reason", "place"}:
            return {"person/profile"}
        if query_type in {"attribute"}:
            return {"person/profile", "entity"}
        return set()

    @staticmethod
    def _anchors(query: str) -> set[str]:
        anchors = {
            token
            for token in _TOKEN_RE.findall(query.lower())
            if len(token) > 1 and token not in _STOPWORDS
        }
        for phrase in _CAPITALIZED_PHRASE_RE.findall(query):
            normalized = " ".join(
                token
                for token in _TOKEN_RE.findall(phrase.lower())
                if len(token) > 1 and token not in _STOPWORDS
            )
            if normalized:
                anchors.add(normalized)
                anchors.update(normalized.split())
        return anchors
