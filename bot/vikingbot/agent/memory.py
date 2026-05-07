"""Memory system for persistent agent memory."""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger

from vikingbot.config.loader import load_config
from vikingbot.openviking_mount.ov_server import VikingClient
from vikingbot.utils.helpers import ensure_dir


@dataclass
class MemoryQueryPlan:
    original_query: str
    queries: list[str]
    exact_terms: list[str]
    grep_terms: list[str]
    facets: list[str]


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    _QUOTED_TEXT_PATTERN = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]")
    _FILE_LIKE_PATTERN = re.compile(
        r"\b[\w .()@+-]{2,80}\.(?:md|txt|csv|xlsx|xls|docx|doc|pptx|pdf|png|jpg|jpeg)\b",
        re.IGNORECASE,
    )
    _EN_TERM_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_./+-]{2,}\b")
    _EN_PHRASE_PATTERN = re.compile(
        r"\b[A-Za-z][A-Za-z0-9_./+-]{2,}(?:\s+[A-Za-z][A-Za-z0-9_./+-]{2,})+\b"
    )
    _CJK_TEXT_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
    _MEMORY_FIELDS_PATTERN = re.compile(r"<!--\s*MEMORY_FIELDS\s*(.*?)\s*-->", re.DOTALL)
    _DATE_LIKE_TERM_PATTERN = re.compile(
        r"^(?:20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|"
        r"20\d{2}年\d{1,2}月\d{1,2}日?|"
        r"\d{1,2}[-/.]\d{1,2}|"
        r"\d{1,2}月\d{1,2}日?)$"
    )

    _EN_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "what", "why", "how",
        "which", "should", "could", "would", "please", "user", "this", "that",
        "with", "from", "into", "about", "when", "where",
    }

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _normalize_date_text(text: str) -> list[str]:
        dates: list[str] = []

        for year, month, day in re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text):
            dates.append(f"{year}-{int(month):02d}-{int(day):02d}")

        for year, month, day in re.findall(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text):
            dates.append(f"{year}-{int(month):02d}-{int(day):02d}")

        dates.extend(re.findall(r"20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}日?", text))
        dates.extend(re.findall(r"\d{1,2}[-/.月]\d{1,2}日?", text))

        return MemoryStore._dedupe_keep_order(dates)

    @classmethod
    def _extract_exact_terms(cls, query: str) -> list[str]:
        terms: list[str] = []

        terms.extend(cls._QUOTED_TEXT_PATTERN.findall(query))
        terms.extend(cls._FILE_LIKE_PATTERN.findall(query))
        terms.extend(cls._normalize_date_text(query))

        for term in cls._EN_TERM_PATTERN.findall(query):
            lower = term.lower()
            if lower in cls._EN_STOPWORDS:
                continue
            if len(term) < 3 and not re.search(r"\d", term):
                continue
            terms.append(term)

        return cls._dedupe_keep_order(terms)[:20]

    @classmethod
    def _extract_grep_terms(cls, query: str) -> list[str]:
        terms: list[str] = []

        terms.extend(cls._QUOTED_TEXT_PATTERN.findall(query))
        terms.extend(cls._FILE_LIKE_PATTERN.findall(query))

        for phrase in cls._EN_PHRASE_PATTERN.findall(query):
            words = [
                word for word in re.findall(r"[A-Za-z][A-Za-z0-9_./+-]{2,}", phrase)
                if word.lower() not in cls._EN_STOPWORDS
            ]
            if len(words) >= 2:
                terms.append(" ".join(words))

        high_precision_terms = []
        for term in cls._dedupe_keep_order(terms):
            if cls._is_date_like_term(term):
                continue
            if len(term) < 4:
                continue
            high_precision_terms.append(term)

        return high_precision_terms[:8]

    @classmethod
    def _extract_facets(cls, query: str) -> list[str]:
        facets: list[str] = []

        for part in re.split(r"[,，、/]|(?:\s+and\s+)|(?:\s+or\s+)", query, flags=re.IGNORECASE):
            words = [
                word for word in cls._EN_TERM_PATTERN.findall(part)
                if word.lower() not in cls._EN_STOPWORDS
            ]
            if words:
                facets.append(" ".join(words[:3]))

        facets.extend(cls._QUOTED_TEXT_PATTERN.findall(query))
        facets.extend(cls._FILE_LIKE_PATTERN.findall(query))

        return [
            facet for facet in cls._dedupe_keep_order(facets)
            if len(facet) >= 3 and not cls._is_date_like_term(facet)
        ][:8]

    @classmethod
    def _is_date_like_term(cls, term: str) -> bool:
        return bool(cls._DATE_LIKE_TERM_PATTERN.fullmatch(term.strip()))

    @classmethod
    def _build_anchor_query(cls, exact_terms: list[str]) -> str:
        non_date_terms = [
            term for term in exact_terms
            if not cls._is_date_like_term(term)
        ]
        if not non_date_terms:
            return ""

        date_terms = [
            term for term in exact_terms
            if cls._is_date_like_term(term)
        ]
        return " ".join(cls._dedupe_keep_order(non_date_terms + date_terms)[:16])

    @classmethod
    def _build_memory_query_plan(cls, current_message: str) -> MemoryQueryPlan:
        original = current_message.strip()
        exact_terms = cls._extract_exact_terms(original)

        queries: list[str] = [original]
        anchor_query = cls._build_anchor_query(exact_terms)
        if anchor_query:
            queries.append(anchor_query)

        return MemoryQueryPlan(
            original_query=original,
            queries=cls._dedupe_keep_order(queries),
            exact_terms=exact_terms,
            grep_terms=cls._extract_grep_terms(original),
            facets=cls._extract_facets(original),
        )

    @staticmethod
    def _memory_to_dict(memory: Any) -> dict[str, Any]:
        if isinstance(memory, dict):
            return dict(memory)
        return {
            "uri": getattr(memory, "uri", ""),
            "score": getattr(memory, "score", 0.0),
            "abstract": getattr(memory, "abstract", ""),
            "category": getattr(memory, "category", ""),
            "level": getattr(memory, "level", None),
        }

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    @staticmethod
    def _extract_grep_matches(result: Any) -> list[Any]:
        if isinstance(result, dict):
            matches = result.get("matches", [])
            return matches if isinstance(matches, list) else []
        matches = getattr(result, "matches", [])
        return matches if isinstance(matches, list) else []

    @staticmethod
    def _grep_match_uri(match: Any) -> str:
        if isinstance(match, dict):
            return str(match.get("uri") or "")
        return str(getattr(match, "uri", "") or "")

    @staticmethod
    def _grep_match_content(match: Any) -> str:
        if isinstance(match, dict):
            return str(match.get("content") or "")
        return str(getattr(match, "content", "") or "")

    @staticmethod
    def _add_lexical_match(memory: dict[str, Any], pattern: str, content: str = "") -> None:
        patterns = memory.setdefault("lexical_patterns", [])
        if pattern not in patterns:
            patterns.append(pattern)
            memory["lexical_score"] = float(memory.get("lexical_score") or 0.0) + 0.08
        memory["lexical_match_count"] = int(memory.get("lexical_match_count") or 0) + 1
        if content and not memory.get("abstract"):
            memory["abstract"] = content[:300]

    async def _merge_grep_matches(
        self,
        client: Any,
        merged: dict[str, dict[str, dict[str, Any]]],
        *,
        plan: MemoryQueryPlan,
        user_ids: list[str],
        agent_user_id: str,
        node_limit: int = 8,
    ) -> None:
        if not plan.grep_terms:
            return

        targets: list[tuple[str, str]] = [
            (f"viking://user/{user_id}/memories/", "user_memory")
            for user_id in user_ids
        ]
        try:
            agent_space_name = client.get_agent_space_name(agent_user_id)
            targets.append((f"viking://agent/{agent_space_name}/memories/", "agent_memory"))
        except Exception as exc:
            logger.warning(f"[READ_USER_MEMORY]: build agent grep target failed: {exc}")

        for term in plan.grep_terms:
            pattern = re.escape(term)
            for target_uri, bucket in targets:
                try:
                    result = await client.grep(
                        target_uri,
                        pattern,
                        case_insensitive=True,
                        node_limit=node_limit,
                    )
                except Exception as exc:
                    logger.warning(f"[READ_USER_MEMORY]: grep failed term={term}: {exc}")
                    continue

                for match in self._extract_grep_matches(result):
                    uri = self._grep_match_uri(match)
                    if not uri or "/memories/" not in uri:
                        continue
                    memory = merged[bucket].get(uri)
                    if memory is None:
                        memory = {
                            "uri": uri,
                            "score": 0.0,
                            "raw_score": 0.0,
                            "rrf_score": 0.0,
                            "matched_query_count": 0,
                        }
                        merged[bucket][uri] = memory
                    self._add_lexical_match(memory, term, self._grep_match_content(match))

    async def _search_memory_with_query_plan(
        self,
        client: Any,
        plan: MemoryQueryPlan,
        user_ids: list[str],
        agent_user_id: str,
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        merged: dict[str, dict[str, dict[str, Any]]] = {
            "user_memory": {},
            "agent_memory": {},
        }

        weighted_queries: list[tuple[str, float]] = []
        seen_queries: set[str] = set()
        for query, weight in (
            [(query, 1.0) for query in plan.queries]
            + [(facet, 0.35) for facet in plan.facets]
        ):
            normalized = query.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            weighted_queries.append((normalized, weight))

        for query, query_weight in weighted_queries:
            try:
                result = await client.search_memory(
                    query=query,
                    user_ids=user_ids,
                    agent_user_id=agent_user_id,
                    limit=limit,
                )
            except Exception as exc:
                logger.warning(f"[READ_USER_MEMORY]: query variant failed: {exc}")
                continue

            for bucket in ("user_memory", "agent_memory"):
                for rank, raw_memory in enumerate(result.get(bucket, []), start=1):
                    memory = self._memory_to_dict(raw_memory)
                    uri = str(memory.get("uri") or "")
                    if not uri:
                        continue

                    previous = merged[bucket].get(uri)
                    rrf = self._rrf_score(rank) * query_weight

                    if previous is None:
                        memory["raw_score"] = float(memory.get("score") or 0.0)
                        memory["rrf_score"] = rrf
                        memory["matched_query_count"] = 1
                        merged[bucket][uri] = memory
                    else:
                        previous["rrf_score"] = float(previous.get("rrf_score") or 0.0) + rrf
                        previous["matched_query_count"] = int(previous.get("matched_query_count") or 1) + 1
                        previous["raw_score"] = max(
                            float(previous.get("raw_score") or 0.0),
                            float(memory.get("score") or 0.0),
                        )

        await self._merge_grep_matches(
            client,
            merged,
            plan=plan,
            user_ids=user_ids,
            agent_user_id=agent_user_id,
        )

        for bucket in ("user_memory", "agent_memory"):
            for memory in merged[bucket].values():
                memory["score"] = (
                    float(memory.get("lexical_score") or 0.0)
                    + float(memory.get("rrf_score") or 0.0)
                )

        return {
            bucket: sorted(
                items.values(),
                key=lambda item: (
                    float(item.get("lexical_score") or 0.0),
                    float(item.get("rrf_score") or 0.0),
                    int(item.get("matched_query_count") or 0),
                    float(item.get("raw_score") or 0.0),
                ),
                reverse=True,
            )
            for bucket, items in merged.items()
        }

    @staticmethod
    def _memory_bucket(uri: str) -> str:
        if "/events/" in uri:
            return "events"
        if "/preferences/" in uri:
            return "preferences"
        if "/entities/" in uri:
            return "entities"
        if "/memories/cases/" in uri:
            return "cases"
        if "/memories/patterns/" in uri:
            return "patterns"
        if uri.endswith("profile.md"):
            return "profile"
        return "other"

    @staticmethod
    def _is_low_value_memory_uri(uri: str) -> bool:
        normalized = uri.rstrip("/")
        return (
            normalized.endswith("/.overview.md")
            or normalized.endswith(".overview.md")
            or normalized.endswith("/.abstract.md")
            or normalized.endswith(".abstract.md")
        )

    @staticmethod
    def _uri_display_name(uri: str) -> str:
        path = uri.rstrip("/").rsplit("/", 1)[-1]
        if path.endswith(".md"):
            path = path[:-3]
        return unquote(path).replace("_", " ").replace("-", " ")

    @classmethod
    def _memory_metadata_text(cls, content: str) -> str:
        match = cls._MEMORY_FIELDS_PATTERN.search(content)
        if not match:
            return ""
        try:
            fields = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError, TypeError):
            return ""
        if not isinstance(fields, dict):
            return ""
        values: list[str] = []
        for key in ("topic", "event_name", "goal", "user", "tool_name", "skill_name"):
            value = fields.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        return " ".join(values)

    @classmethod
    def _memory_title_text(cls, uri: str, content: str, abstract: str) -> str:
        parts = [cls._uri_display_name(uri), abstract]
        for line in content.splitlines()[:8]:
            stripped = line.strip()
            if stripped.startswith("#"):
                parts.append(stripped.lstrip("#").strip())
                break
        metadata = cls._memory_metadata_text(content)
        if metadata:
            parts.append(metadata)
        return " ".join(part for part in parts if part)

    @classmethod
    def _title_overlap_score(cls, query: str, title_text: str) -> float:
        query_lower = query.lower()
        title_lower = title_text.lower()
        score = 0.0

        for term in cls._EN_TERM_PATTERN.findall(query):
            lower = term.lower()
            if lower in cls._EN_STOPWORDS or cls._is_date_like_term(term):
                continue
            if lower in title_lower:
                score += 0.018

        for chunk in cls._CJK_TEXT_PATTERN.findall(title_text):
            if len(chunk) <= 2:
                continue
            max_hit = 0
            max_size = min(8, len(chunk))
            for size in range(max_size, 1, -1):
                if max_hit:
                    break
                for start in range(0, len(chunk) - size + 1):
                    if chunk[start:start + size] in query_lower:
                        max_hit = size
                        break
            if max_hit:
                score += min(0.03, 0.006 * max_hit)

        return min(score, 0.12)

    @classmethod
    def _matched_facets(
        cls,
        plan: MemoryQueryPlan | None,
        uri: str,
        content: str,
        abstract: str,
    ) -> list[str]:
        if not plan or not plan.facets:
            return []

        haystack = " ".join(
            [
                uri,
                abstract,
                cls._memory_title_text(uri, content, abstract),
                content,
            ]
        ).lower()
        matched: list[str] = []
        for facet in plan.facets:
            words = [
                word.lower() for word in cls._EN_TERM_PATTERN.findall(facet)
                if word.lower() not in cls._EN_STOPWORDS
            ]
            if words and all(word in haystack for word in words):
                matched.append(facet)
            elif facet.lower() in haystack:
                matched.append(facet)
        return cls._dedupe_keep_order(matched)

    @classmethod
    def _extract_focused_lines(
        cls,
        content: str,
        terms: list[str],
        max_chars: int = 1400,
    ) -> str:
        if not terms:
            return cls._clip_memory_content(content, exact_terms=[], max_chars=max_chars)

        lines = [line.rstrip() for line in content.splitlines()]
        selected_indices: set[int] = set()
        lowered_terms = [
            term.lower() for term in terms
            if term.strip() and not cls._is_date_like_term(term.strip())
        ]

        for idx, line in enumerate(lines):
            lowered = line.lower()
            if any(term in lowered for term in lowered_terms):
                for nearby in range(max(0, idx - 2), min(len(lines), idx + 3)):
                    selected_indices.add(nearby)

        if not selected_indices:
            return cls._clip_memory_content(content, exact_terms=terms, max_chars=max_chars)

        chunks: list[str] = []
        total = 0
        last_idx = -2
        for idx in sorted(selected_indices):
            line = lines[idx].strip()
            if not line:
                continue
            if idx > last_idx + 1 and chunks:
                chunks.append("...")
                total += 3
            if total + len(line) > max_chars:
                break
            chunks.append(line)
            total += len(line)
            last_idx = idx

        return "\n".join(chunks) if chunks else cls._clip_memory_content(
            content,
            exact_terms=terms,
            max_chars=max_chars,
        )

    @classmethod
    def _clip_memory_content(
        cls,
        content: str,
        exact_terms: list[str],
        max_chars: int = 1400,
    ) -> str:
        content = content.strip()
        if len(content) <= max_chars:
            return content

        lowered = content.lower()
        best_pos = -1
        for term in exact_terms:
            pos = lowered.find(term.lower())
            if pos >= 0:
                best_pos = pos
                break

        if best_pos < 0:
            return content[:max_chars].rstrip() + "\n...[truncated]"

        start = max(0, best_pos - max_chars // 3)
        end = min(len(content), start + max_chars)
        snippet = content[start:end].strip()
        if start > 0:
            snippet = "[truncated]...\n" + snippet
        if end < len(content):
            snippet += "\n...[truncated]"
        return snippet

    @staticmethod
    def _content_fingerprint(content: str) -> str:
        normalized = re.sub(r"\s+", " ", content.strip())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""

    @classmethod
    def _content_relevance_score(
        cls,
        memory: dict[str, Any],
        content: str,
        abstract: str,
        exact_terms: list[str],
        query: str,
    ) -> float:
        text = f"{content}\n{abstract}".lower()
        score = float(memory.get("score") or 0.0)

        matched_terms = 0
        for term in exact_terms:
            normalized = term.strip()
            if not normalized:
                continue
            if normalized.lower() not in text:
                continue

            if cls._is_date_like_term(normalized):
                continue

            matched_terms += 1
            if len(normalized) >= 8 or re.search(r"\s", normalized):
                score += 0.035
            else:
                score += 0.012

        score += min(matched_terms, 6) * 0.006
        score += float(memory.get("lexical_score") or 0.0) * 1.5
        score += cls._title_overlap_score(
            query,
            cls._memory_title_text(str(memory.get("uri") or ""), content, abstract),
        )

        uri = str(memory.get("uri") or "")
        if uri.endswith("profile.md"):
            score -= 0.015
        return score

    async def _parse_viking_memory(
        self,
        result: Any,
        client: Any,
        *,
        plan: MemoryQueryPlan | None = None,
        min_score: float = 0.0,
        max_chars: int = 5000,
        max_count: int = 12,
        max_summary_count: int = 4,
    ) -> str:
        if not result:
            return ""

        exact_terms = plan.exact_terms if plan else []
        focus_terms = self._dedupe_keep_order(exact_terms + (plan.facets if plan else []))
        memories = [self._memory_to_dict(memory) for memory in result]

        def get_score(memory: dict[str, Any]) -> float:
            return float(memory.get("score") or 0.0)

        def get_uri(memory: dict[str, Any]) -> str:
            return str(memory.get("uri") or "")

        raw_candidates = [
            memory for memory in memories
            if get_uri(memory) and get_score(memory) >= min_score
        ]
        candidates = [
            memory for memory in raw_candidates
            if not self._is_low_value_memory_uri(get_uri(memory))
        ] or raw_candidates
        candidates.sort(key=get_score, reverse=True)

        enriched_candidates: list[dict[str, Any]] = []
        for memory in candidates:
            uri = get_uri(memory)
            abstract = str(memory.get("abstract") or "").strip()

            content = ""
            try:
                content = await client.read_content(uri, level="read")
            except Exception as exc:
                logger.warning(f"Failed to read content from {uri}: {exc}")

            bucket = self._memory_bucket(uri)
            if content and bucket == "preferences" and len(content.strip()) <= 2200:
                display_content = content.strip()
            elif content:
                display_content = self._extract_focused_lines(content, terms=focus_terms)
            else:
                display_content = ""
            enriched = dict(memory)
            enriched["_content"] = content
            enriched["_display_content"] = display_content
            enriched["_matched_facets"] = self._matched_facets(plan, uri, content, abstract)
            enriched["_ranking_score"] = self._content_relevance_score(
                enriched,
                content,
                abstract,
                exact_terms,
                plan.original_query if plan else "",
            )
            enriched_candidates.append(enriched)

        enriched_candidates.sort(
            key=lambda memory: (
                float(memory.get("_ranking_score") or 0.0),
                float(memory.get("lexical_score") or 0.0),
                get_score(memory),
            ),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        seen_uris: set[str] = set()
        bucket_counts: dict[str, int] = defaultdict(int)

        max_per_bucket = max(3, max_count // 2)

        def add_selected(memory: dict[str, Any], *, enforce_bucket: bool = True) -> bool:
            if len(selected) >= max_count:
                return False
            uri = get_uri(memory)
            bucket = self._memory_bucket(uri)
            if uri in seen_uris:
                return False
            if enforce_bucket and bucket_counts[bucket] >= max_per_bucket:
                return False
            seen_uris.add(uri)
            bucket_counts[bucket] += 1
            selected.append(memory)
            return True

        for facet in (plan.facets if plan else []):
            for memory in enriched_candidates:
                if facet in memory.get("_matched_facets", []):
                    if add_selected(memory):
                        break

        for memory in enriched_candidates:
            if len(selected) >= max_count:
                break
            add_selected(memory)

        for memory in enriched_candidates:
            if len(selected) >= max_count:
                break
            add_selected(memory, enforce_bucket=False)

        parts: list[str] = []
        total_chars = 0
        summary_count = 0
        seen_fingerprints: set[str] = set()

        for idx, memory in enumerate(selected, start=1):
            uri = get_uri(memory)
            score = get_score(memory)
            abstract = str(memory.get("abstract") or "").strip()
            bucket = self._memory_bucket(uri)

            content = str(memory.get("_display_content") or "")

            fingerprint = self._content_fingerprint(content or abstract)
            if fingerprint:
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

            if content:
                memory_text = (
                    f'<memory index="{idx}" type="full" bucket="{bucket}">\n'
                    f"  <uri>{uri}</uri>\n"
                    f"  <score>{score}</score>\n"
                    f"  <content>{content}</content>\n"
                    f"</memory>"
                )
            else:
                if summary_count >= max_summary_count:
                    continue
                summary_count += 1
                memory_text = (
                    f'<memory index="{idx}" type="summary" bucket="{bucket}">\n'
                    f"  <uri>{uri}</uri>\n"
                    f"  <score>{score}</score>\n"
                    f"  <abstract>{abstract[:600]}</abstract>\n"
                    f"</memory>"
                )

            extra_chars = len(memory_text) + (1 if parts else 0)
            if total_chars + extra_chars > max_chars:
                if summary_count >= max_summary_count:
                    continue
                summary_count += 1
                fallback_text = (
                    f'<memory index="{idx}" type="summary" bucket="{bucket}">\n'
                    f"  <uri>{uri}</uri>\n"
                    f"  <score>{score}</score>\n"
                    f"  <abstract>{abstract[:600]}</abstract>\n"
                    f"</memory>"
                )
                fallback_chars = len(fallback_text) + (1 if parts else 0)
                if total_chars + fallback_chars > max_chars:
                    continue
                memory_text = fallback_text
                extra_chars = fallback_chars

            parts.append(memory_text)
            total_chars += extra_chars

        return "\n".join(parts)

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def get_viking_memory_context(
        self, current_message: str, workspace_id: str, sender_id: str, user_ids: list[str] | None = None
    ) -> str:
        try:
            config = load_config().ov_server
            admin_user_id = config.admin_user_id
            # Use provided user_ids or fall back to sender_id
            search_user_ids = user_ids if user_ids else [sender_id]
            logger.info(f'workspace_id={workspace_id}')
            logger.info(f'user_ids={search_user_ids}')
            logger.info(f'admin_user_id={admin_user_id}')
            client = await VikingClient.create(agent_id=workspace_id)
            plan = self._build_memory_query_plan(current_message)
            logger.info(
                f"[READ_USER_MEMORY_PLAN]: queries={plan.queries}, "
                f"exact_terms={plan.exact_terms}, grep_terms={plan.grep_terms}, "
                f"facets={plan.facets}"
            )
            result = await self._search_memory_with_query_plan(
                client=client,
                plan=plan,
                user_ids=search_user_ids,
                agent_user_id=admin_user_id,
                limit=30,
            )
            if not result:
                return ""

            # Log raw search results for debugging
            memory_list = []
            memory_list.append(f"user_memory[{len(result['user_memory'])}]:")

            for i, mem in enumerate(result['user_memory']):
                uri = mem.get('uri', '') if isinstance(mem, dict) else getattr(mem, 'uri', '')
                score = mem.get('score', 0) if isinstance(mem, dict) else getattr(mem, 'score', 0)
                memory_list.append(f"{i},{uri},{score}")
            memory_list.append(f"agent_memory[{len(result['agent_memory'])}]:")
            for i, mem in enumerate(result['agent_memory']):
                uri = mem.get('uri', '') if isinstance(mem, dict) else getattr(mem, 'uri', '')
                score = mem.get('score', 0) if isinstance(mem, dict) else getattr(mem, 'score', 0)
                memory_list.append(f"{i},{uri},{score}")
            raw_memories_log = "\n".join(memory_list)
            logger.info(f"[RAW_MEMORIES]\n{raw_memories_log}")
            user_memory = await self._parse_viking_memory(
                result["user_memory"],
                client,
                plan=plan,
                min_score=0.0,
                max_chars=5000,
                max_count=12,
                max_summary_count=4,
            )
            agent_memory = await self._parse_viking_memory(
                result["agent_memory"],
                client,
                plan=plan,
                min_score=0.0,
                max_chars=1500,
                max_count=4,
                max_summary_count=2,
            )
            return f"### user memories:\n{user_memory}\n### agent memories:\n{agent_memory}"
        except Exception as e:
            logger.error(f"[READ_USER_MEMORY]: search error. {e}")
            return ""

    async def get_viking_user_profile(self, workspace_id: str, user_id: str) -> str:
        client = await VikingClient.create(agent_id=workspace_id)
        result = await client.read_user_profile(user_id)
        if not result:
            return ""
        return result

    async def get_viking_user_profiles(self, workspace_id: str, user_ids: list[str]) -> str:
        """Get multiple user profiles concurrently.

        Args:
            workspace_id: Workspace ID
            user_ids: List of user IDs to get profiles for

        Returns:
            Formatted string with all user profiles
        """
        if not user_ids:
            return ""

        client = await VikingClient.create(agent_id=workspace_id)

        async def fetch_profile(user_id: str) -> tuple[str, str]:
            """Fetch a single user profile."""
            try:
                start_time = time.time()
                profile = await client.read_user_profile(user_id)
                cost = round(time.time() - start_time, 2)
                logger.info(
                    f"[READ_USER_PROFILE]: user_id={user_id}, cost {cost}s, "
                    f"profile={profile[:50] if profile else 'None'}"
                )
                return (user_id, profile or "")
            except Exception as e:
                logger.error(f"[READ_USER_PROFILE]: user_id={user_id}, error. {e}")
                return (user_id, "")

        # Fetch all profiles concurrently
        tasks = [fetch_profile(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build the result string
        parts = []
        for result in results:
            if isinstance(result, Exception):
                continue
            user_id, profile = result
            if profile:
                parts.append(f"## User profile for {user_id}: \n{profile}")

        return "\n\n".join(parts) if parts else ""
