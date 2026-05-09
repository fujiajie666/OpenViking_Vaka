"""Memory system for persistent agent memory."""

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from vikingbot.config.loader import load_config
from vikingbot.openviking_mount.ov_server import VikingClient
from vikingbot.utils.helpers import ensure_dir


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

    def _pick_relevant_excerpt(self, text: str, query: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 0:
            return ""

        terms = re.findall(
            r"[A-Za-z][A-Za-z0-9_./-]{2,}|\d+(?:[./-]\d+)*|[\u4e00-\u9fff]{2,18}",
            query,
        )

        lowered = text.lower()
        window = max_chars
        candidate_starts = {0}
        seen_terms = []
        for term in (t.strip() for t in terms):
            if len(term) < 2 or term in seen_terms:
                continue
            seen_terms.append(term)
            needle = term.lower()
            start = lowered.find(needle)
            while start != -1 and len(candidate_starts) < 80:
                candidate_starts.add(max(0, start - window // 3))
                start = lowered.find(needle, start + max(1, len(needle)))

        best_start = 0
        best_score = -1
        for start in candidate_starts:
            chunk = lowered[start:start + window]
            score = sum(1 for term in seen_terms[:32] if term.lower() in chunk)
            if score > best_score:
                best_start = start
                best_score = score

        excerpt = text[best_start:best_start + window].strip()
        if best_start > 0:
            excerpt = "..." + excerpt
        if best_start + window < len(text):
            excerpt += "..."
        return excerpt

    async def _parse_viking_memory(
        self,
        result: Any,
        client: Any,
        min_score: float = 0.3,
        max_chars: int = 6000,
        max_count: int = 12,
        current_message: str = "",
    ) -> str:
        """Parse viking memory with score filtering and character limit.
        Automatically reads full content for memories above threshold.

        Args:
            result: Memory search results
            client: VikingClient instance to read content
            min_score: Minimum score threshold (default: 0.4)
            max_chars: Maximum character limit for output (default: 4000)
            max_count: Maximum number of memories to return (default: 12)
            current_message: User message used to select relevant excerpts from long memories

        Returns:
            Formatted memory string within character limit
        """
        if not result or len(result) == 0:
            return ""

        # Filter by min_score and sort by score descending
        def get_score(m):
            return m.get("score", 0) if isinstance(m, dict) else getattr(m, "score", 0.0)

        def get_uri(m):
            return m.get("uri", "") if isinstance(m, dict) else getattr(m, "uri", "")

        def get_abstract(m):
            return m.get("abstract", "") if isinstance(m, dict) else getattr(m, "abstract", "")

        filtered_memories = [
            memory for memory in result if get_score(memory) >= min_score
        ]
        filtered_memories.sort(key=get_score, reverse=True)

        user_memories = []
        total_chars = 0
        seen_content_hashes = set()
        link_candidates: list[tuple[str, Any]] = []
        max_link_candidates = 5
        max_full_content_chars = 4000 if max_chars <= 4000 else 5000
        compact_content_chars = 2000 if max_chars <= 4000 else 3000

        def build_memory_str(index: int, memory_type: str, body_tag: str = "", body: str = "") -> str:
            lines = [
                f'<memory index="{index}" type="{memory_type}">',
                f"  <uri>{uri}</uri>",
                f"  <score>{score}</score>",
            ]
            if body_tag:
                lines.append(f"  <{body_tag}>{body}</{body_tag}>")
            lines.append("</memory>")
            return "\n".join(lines)

        def append_memory(memory_str: str) -> bool:
            nonlocal total_chars
            memory_chars = len(memory_str) + (1 if user_memories else 0)
            if total_chars + memory_chars > max_chars:
                return False
            user_memories.append(memory_str)
            total_chars += memory_chars
            return True

        def append_compact(index: int, body_tag: str, body: str) -> bool:
            empty_compact = build_memory_str(index, "compact", body_tag, "")
            available_body_chars = max_chars - total_chars - (1 if user_memories else 0) - len(empty_compact)
            if available_body_chars <= 80:
                return False
            excerpt = self._pick_relevant_excerpt(
                body, current_message, min(compact_content_chars, available_body_chars)
            )
            return append_memory(build_memory_str(index, "compact", body_tag, excerpt))

        for memory in filtered_memories:
            if len(user_memories) >= max_count:
                break

            uri = get_uri(memory)
            abstract = get_abstract(memory)
            score = get_score(memory)

            # First, try to build full memory with content
            content = ""
            try:
                content = await client.read_content(uri, level="read")
            except Exception as e:
                logger.warning(f"Failed to read content from {uri}: {e}")

            # Deduplicate by content hash (use content or abstract as key)
            content_to_hash = content or abstract
            content_hash = hash(content_to_hash)
            if content_to_hash and content_hash in seen_content_hashes:
                continue
            if content_to_hash:
                seen_content_hashes.add(content_hash)

            memory_index = len(user_memories) + 1
            if content:
                # Try full version first for short memories (no abstract when content is present)
                if len(content) <= max_full_content_chars:
                    full_memory_str = build_memory_str(memory_index, "full", "content", content)
                    if append_memory(full_memory_str):
                        continue

                if append_compact(memory_index, "abstract" if abstract else "content", abstract or content):
                    continue
            elif abstract:
                if append_compact(memory_index, "abstract", abstract):
                    continue
            else:
                # No content available, use link-only version only within the same caps
                # Keep unread links out of the main evidence block; expose only a few as read candidates.
                logger.info(f"Collecting link-only candidate for {uri} (read failed or empty)")
                if len(link_candidates) < max_link_candidates:
                    link_candidates.append((uri, score))
                continue
            break

        parts = ["\n".join(user_memories)] if user_memories else []
        if link_candidates:
            lines = [
                "### candidate_link_uris",
                "These URIs were retrieved but not loaded. They are not evidence.",
                "If one looks necessary for the user question, call openviking_multi_read before using it.",
            ]
            for uri, score in link_candidates:
                lines.append(f"- {uri} (score={score})")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def _create_client(self, workspace_id: str) -> Optional[VikingClient]:
        """Helper method to create a VikingClient with proper error handling."""
        try:
            return await VikingClient.create(agent_id=workspace_id)
        except Exception as e:
            logger.error(f"Failed to create VikingClient for workspace {workspace_id}: {e}")
            return None

    async def get_viking_memory_context(
        self,
        current_message: str,
        workspace_id: str,
        sender_id: str,
        user_ids: list[str] | None = None,
    ) -> str:
        client = None
        try:
            config = load_config().ov_server
            admin_user_id = config.admin_user_id
            # Use provided user_ids or fall back to sender_id
            search_user_ids = user_ids if user_ids else [sender_id]
            logger.info(f"workspace_id={workspace_id}")
            logger.info(f"user_ids={search_user_ids}")
            logger.info(f"admin_user_id={admin_user_id}")

            client = await self._create_client(workspace_id)
            if not client:
                return ""
            result = await client.search_memory(
                query=current_message, user_ids=search_user_ids, agent_user_id=admin_user_id, limit=30
            )
            if not result:
                return ""

            # Log raw search results for debugging
            memory_list = []
            memory_list.append(f"user_memory[{len(result['user_memory'])}]:")

            for i, mem in enumerate(result["user_memory"]):
                uri = mem.get("uri", "") if isinstance(mem, dict) else getattr(mem, "uri", "")
                score = mem.get("score", 0) if isinstance(mem, dict) else getattr(mem, "score", 0)
                memory_list.append(f"{i},{uri},{score}")
            memory_list.append(f"agent_memory[{len(result['agent_memory'])}]:")
            for i, mem in enumerate(result["agent_memory"]):
                uri = mem.get("uri", "") if isinstance(mem, dict) else getattr(mem, "uri", "")
                score = mem.get("score", 0) if isinstance(mem, dict) else getattr(mem, "score", 0)
                memory_list.append(f"{i},{uri},{score}")
            raw_memories_log = "\n".join(memory_list)
            logger.info(f"[RAW_MEMORIES]\n{raw_memories_log}")
            user_memory = await self._parse_viking_memory(
                result["user_memory"], client, min_score=0.1, max_chars=4000, current_message=current_message
            )
            agent_memory = await self._parse_viking_memory(
                result["agent_memory"], client, min_score=0.1, max_chars=2000, current_message=current_message
            )
            return f"### user memories:\n{user_memory}\n### agent memories:\n{agent_memory}"
        except Exception as e:
            logger.error(f"[READ_USER_MEMORY]: search error. {e}")
            return ""
        finally:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing VikingClient: {e}")

    async def get_viking_user_profile(self, workspace_id: str, user_id: str) -> str:
        client = None
        try:
            client = await self._create_client(workspace_id)
            if not client:
                return ""
            result = await client.read_user_profile(user_id)
            return result or ""
        except Exception as e:
            logger.warning(f"[READ_USER_PROFILE]: user_id={user_id}, error. {e}")
            return ""
        finally:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing VikingClient: {e}")

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

        client = None
        try:
            client = await self._create_client(workspace_id)
            if not client:
                return ""

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
        except Exception as e:
            logger.error(f"[READ_USER_PROFILES]: error. {e}")
            return ""
        finally:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing VikingClient: {e}")