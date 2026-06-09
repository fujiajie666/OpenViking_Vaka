# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Slot-aware graph candidate scoring for Mnemis-lite."""

import re
from dataclasses import dataclass
from typing import Any

from openviking.retrieve.graph.mnemis_lite.query_planner import GraphQueryPlan

_DATE_RE = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|20\d{2}|"
    r"\d{1,2}\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|"
    r"jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)|"
    r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
    r"aug|august|sep|september|oct|october|nov|november|dec|december)\s+\d{1,2}|"
    r"yesterday|today|tomorrow|last\s+(?:week|month|year|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|next\s+(?:week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_PLACE_RE = re.compile(
    r"\b(?:city|country|school|festival|venue|park|museum|shelter|center|centre|"
    r"studio|restaurant|cafe|gym|church|library|beach|mountain|trail|office|"
    r"boston|tokyo|sweden|rio|california|hawaii|japan|united states|usa)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|reason|decided|motivated|due to|so that|therefore|after losing|"
    r"lost (?:his|her|their)?\s*job|wanted to|in order to|led to|caused|inspired)\b",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(
    r"\b(?:played|visited|attended|recommended|suggested|wrote|read|joined|"
    r"participated|donated|volunteered|bought|made|make|making|went|met|"
    r"started|received|shared|liked|likes|disliked|dislikes)\b",
    re.IGNORECASE,
)
_PLACE_EVENT_RE = re.compile(
    r"\b(?:visited|traveled|travelled|went|moved|attended|stayed|lived|met)\b",
    re.IGNORECASE,
)
_ENTITY_SUBTYPE_RE = re.compile(r"/entities/([^/]+)/", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TARGET_EQUIVALENCE_GROUPS = (
    {
        "company",
        "companies",
        "organization",
        "organizations",
        "organisation",
        "organisations",
    },
    {"meal", "meals", "food", "foods", "recipe", "recipes"},
    {"book", "books", "novel", "novels"},
    {"game", "games"},
    {"city", "cities", "country", "countries", "place", "places", "location", "locations"},
    {"school", "schools", "venue", "venues"},
)


@dataclass(frozen=True)
class SlotMatch:
    matched: bool
    reason: str


class MnemisLiteSlotScorer:
    """Check whether a graph candidate can answer the query's required slot."""

    def slot_match(self, candidate: dict[str, Any], plan: GraphQueryPlan) -> SlotMatch:
        content_text = self._candidate_content_text(candidate)
        content_match_text = self._candidate_match_text(candidate, include_uri=False)
        all_match_text = self._candidate_match_text(candidate, include_uri=True)
        uri_kind = str(candidate.get("_graph_uri_kind") or candidate.get("_uri_kind") or "")
        if not uri_kind:
            uri_kind = self.uri_kind(candidate)

        if plan.query_type == "time":
            if uri_kind == "event" and _DATE_RE.search(content_text):
                return SlotMatch(True, "slot:time:date_present")
            return SlotMatch(False, "slot_mismatch:time_requires_date")

        if plan.query_type == "place":
            if _PLACE_RE.search(content_match_text):
                return SlotMatch(True, "slot:place:place_signal")
            if uri_kind == "entity" and self._entity_has_place_signal(
                candidate,
                all_match_text,
            ):
                return SlotMatch(True, "slot:place:entity_anchor")
            if (
                uri_kind == "event"
                and _PLACE_EVENT_RE.search(content_match_text)
                and self._has_anchor_overlap(content_match_text, plan)
            ):
                return SlotMatch(True, "slot:place:event_context")
            return SlotMatch(False, "slot_mismatch:place_requires_place_signal")

        if plan.query_type == "reason":
            if _CAUSAL_RE.search(content_text):
                return SlotMatch(True, "slot:reason:causal_signal")
            return SlotMatch(False, "slot_mismatch:reason_requires_causal_signal")

        if plan.query_type in {"count", "list_or_set"}:
            anchor_overlap = self._has_anchor_overlap(content_match_text, plan)
            item_signal = bool(_ITEM_RE.search(content_match_text))
            if uri_kind == "preference" and anchor_overlap:
                return SlotMatch(True, f"slot:{plan.query_type}:coverage_item")
            if uri_kind == "event" and anchor_overlap and item_signal:
                return SlotMatch(True, f"slot:{plan.query_type}:coverage_item")
            if uri_kind == "entity" and (
                self._entity_matches_query_target(candidate, plan)
                or (anchor_overlap and item_signal)
            ):
                return SlotMatch(True, f"slot:{plan.query_type}:coverage_item")
            return SlotMatch(False, f"slot_mismatch:{plan.query_type}_requires_item")

        if plan.query_type == "relationship":
            if uri_kind in {"person/profile", "entity"}:
                return SlotMatch(True, "slot:relationship:profile_or_entity")
            return SlotMatch(False, "slot_mismatch:relationship_requires_profile")

        if plan.query_type == "inference":
            if uri_kind in {"event", "preference", "person/profile", "entity"}:
                return SlotMatch(True, "slot:inference:supporting_context")
            return SlotMatch(False, "slot_mismatch:inference_requires_context")

        if uri_kind in plan.risky_uri_kinds and not self._has_anchor_overlap(
            content_text,
            plan,
        ):
            return SlotMatch(False, "slot_mismatch:attribute_risky_without_anchor")
        return SlotMatch(True, "slot:attribute:accepted")

    @staticmethod
    def uri_kind(candidate: dict[str, Any]) -> str:
        uri = str(candidate.get("uri", "") or "").lower()
        memory_type = str(candidate.get("memory_type", "") or "").lower()
        category = str(candidate.get("category", "") or "").lower()
        if "/events/" in uri or "/entities/event/" in uri or memory_type in {"event", "events"}:
            return "event"
        if (
            "/entities/person/" in uri
            or uri.endswith("/profile.md")
            or memory_type in {"person", "profile"}
            or category in {"person", "profile"}
        ):
            return "person/profile"
        if "/preferences/" in uri or memory_type in {"preference", "preferences"}:
            return "preference"
        if "/entities/" in uri or memory_type in {"entity", "entities"}:
            return "entity"
        return "other"

    @staticmethod
    def coverage_group(candidate: dict[str, Any]) -> str:
        uri = str(candidate.get("uri", "") or "")
        match = re.search(r"/events/(\d{4})/(\d{2})/(\d{2})/([^/.]+)", uri)
        if match:
            return f"event:{match.group(1)}-{match.group(2)}-{match.group(3)}:{match.group(4)}"
        parts = [part for part in uri.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return uri

    @staticmethod
    def _candidate_content_text(candidate: dict[str, Any]) -> str:
        return " ".join(
            str(candidate.get(key, "") or "")
            for key in ("abstract", "category", "memory_type")
        ).lower()

    @classmethod
    def _candidate_match_text(
        cls,
        candidate: dict[str, Any],
        *,
        include_uri: bool,
    ) -> str:
        keys = (
            ("uri", "abstract", "category", "memory_type")
            if include_uri
            else ("abstract", "category", "memory_type")
        )
        return cls._normalize_match_text(
            " ".join(str(candidate.get(key, "") or "") for key in keys)
        )

    @classmethod
    def _has_anchor_overlap(cls, text: str, plan: GraphQueryPlan) -> bool:
        normalized = cls._normalize_match_text(text)
        text_tokens = set(_TOKEN_RE.findall(normalized))
        padded_text = f" {normalized} "
        for anchor in plan.anchors:
            anchor_text = cls._normalize_match_text(anchor)
            if not anchor_text:
                continue
            if " " in anchor_text:
                if f" {anchor_text} " in padded_text:
                    return True
            elif anchor_text in text_tokens:
                return True
        return False

    @classmethod
    def _entity_has_place_signal(cls, candidate: dict[str, Any], all_match_text: str) -> bool:
        labels = cls._entity_label_tokens(candidate)
        return (
            _PLACE_RE.search(all_match_text) is not None
            or "location" in labels
            or "place" in labels
        )

    @classmethod
    def _entity_matches_query_target(
        cls,
        candidate: dict[str, Any],
        plan: GraphQueryPlan,
    ) -> bool:
        entity_terms = cls._expand_equivalent_terms(cls._entity_label_tokens(candidate))
        query_terms = cls._expand_equivalent_terms(
            set().union(*(_TOKEN_RE.findall(anchor.lower()) for anchor in plan.anchors))
        )
        return bool(entity_terms & query_terms)

    @classmethod
    def _entity_label_tokens(cls, candidate: dict[str, Any]) -> set[str]:
        uri = str(candidate.get("uri", "") or "").lower()
        labels = [
            str(candidate.get("category", "") or ""),
            str(candidate.get("memory_type", "") or ""),
        ]
        match = _ENTITY_SUBTYPE_RE.search(uri)
        if match:
            labels.append(match.group(1))
        return set(_TOKEN_RE.findall(cls._normalize_match_text(" ".join(labels))))

    @staticmethod
    def _expand_equivalent_terms(terms: set[str]) -> set[str]:
        expanded = set(terms)
        for group in _TARGET_EQUIVALENCE_GROUPS:
            if expanded & group:
                expanded.update(group)
        return expanded

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
