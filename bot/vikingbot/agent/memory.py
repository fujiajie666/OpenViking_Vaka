"""Memory system for persistent agent memory."""

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from vikingbot.config.loader import load_config
from vikingbot.openviking_mount.ov_server import VikingClient
from vikingbot.utils.helpers import ensure_dir


COVERAGE_QUERY_TYPES = {"count", "list_or_set", "multi_hop"}
DEFAULT_USER_MEMORY_CHARS = 4000
COVERAGE_USER_MEMORY_CHARS = 6000




class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.latest_graph_retrieval_debug: list[dict[str, Any]] | None = None

    @staticmethod
    def _get_score(memory: Any) -> float:
        raw_score = (
            memory.get("score", 0) if isinstance(memory, dict) else getattr(memory, "score", 0.0)
        )
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _limit_memory_groups(
        cls,
        result: dict[str, list[Any]],
        limit: int,
    ) -> dict[str, list[Any]]:
        user_memories = result.get("user_memory", [])
        agent_memories = result.get("agent_memory", [])
        ranked: list[tuple[float, str, int, Any]] = []

        for group, memories in (
            ("user_memory", user_memories),
            ("agent_memory", agent_memories),
        ):
            for index, memory in enumerate(memories):
                ranked.append((cls._get_score(memory), group, index, memory))

        selected = {
            (group, index)
            for _, group, index, _ in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]
        }
        return {
            "user_memory": [
                memory
                for index, memory in enumerate(user_memories)
                if ("user_memory", index) in selected
            ],
            "agent_memory": [
                memory
                for index, memory in enumerate(agent_memories)
                if ("agent_memory", index) in selected
            ],
        }

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    async def _parse_viking_memory(
        self,
        result: Any,
        client: Any,
        min_score: float = 0.3,
        max_chars: int = 4000,
        query_text: str | None = None,
    ) -> str:
        """Parse viking memory with score filtering and character limit.
        Automatically reads full content for memories above threshold.

        Args:
            result: Memory search results
            client: VikingClient instance to read content
            min_score: Minimum score threshold (default: 0.4)
            max_chars: Maximum character limit for output (default: 4000)

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

        def get_match_reason(m):
            return (
                m.get("match_reason", "")
                if isinstance(m, dict)
                else getattr(m, "match_reason", "")
            )

        def get_debug_metadata(m):
            return (
                m.get("debug_metadata", {})
                if isinstance(m, dict)
                else getattr(m, "debug_metadata", {})
            ) or {}

        def safe_float(value: Any) -> float:
            return float(value) if isinstance(value, (int, float)) else 0.0

        normalized_query = (query_text or "").lower()
        recommendation_query = re.search(
            r"\b(recommendations?|advice|pointers?|tips|suggestions?)\b",
            normalized_query,
        )
        giver_match = re.search(r"\bfrom\s+([a-z][a-z0-9_-]*)", normalized_query)
        recipient_match = re.search(
            r"\b(?:has|have)\s+([a-z][a-z0-9_-]*)\s+received\b",
            normalized_query,
        )
        recommendation_terms = r"recommend\w*|advi[cs]\w*|pointer\w*|tip\w*|suggest\w*|shared?|provided|told"
        giver = giver_match.group(1) if giver_match else ""
        recipient = recipient_match.group(1) if recipient_match else ""

        def recommendation_direction_signal(m) -> float:
            if not recommendation_query:
                return 0.0
            text = f"{get_uri(m)} {get_abstract(m)}".lower()
            intent_signal = 1.0 if re.search(recommendation_terms, text) else 0.0
            if not giver or not recipient:
                return intent_signal
            giver_action = re.search(rf"\b{re.escape(giver)}\b.*\b(?:{recommendation_terms})\b", text)
            recipient_action = re.search(
                rf"\b{re.escape(recipient)}\b.*\b(?:{recommendation_terms})\b",
                text,
            )
            recipient_present = re.search(rf"\b{re.escape(recipient)}\b", text)
            giver_present = re.search(rf"\b{re.escape(giver)}\b", text)
            if giver_action and recipient_present:
                return 3.0
            if recipient_action and giver_present:
                return -1.0
            return intent_signal

        def graph_snippet_priority(m) -> tuple[float, float, float, float, float]:
            metadata = get_debug_metadata(m)
            snippet_score = metadata.get("snippet_score", {})
            if not isinstance(snippet_score, dict):
                snippet_score = {}
            return (
                recommendation_direction_signal(m),
                safe_float(snippet_score.get("overlap")),
                safe_float(metadata.get("own_evidence")),
                safe_float(metadata.get("total_evidence")),
                safe_float(snippet_score.get("density")),
                safe_float(get_score(m)),
            )

        filtered_memories = [memory for memory in result if get_score(memory) >= min_score]
        filtered_memories.sort(key=get_score, reverse=True)
        graph_snippet_memories = [memory for memory in filtered_memories if get_match_reason(memory)]
        filtered_memories = [memory for memory in filtered_memories if not get_match_reason(memory)]
        graph_snippet_memories.sort(key=graph_snippet_priority, reverse=True)

        user_memories = []
        link_only_memories: list[tuple[str, float]] = []
        total_chars = 0
        seen_content_hashes = set()
        next_index = 1

        for memory in filtered_memories:
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

            if content:
                # Try full version first (no abstract when content is present)
                full_memory_str = (
                    f'<memory index="{next_index}" type="full">\n'
                    f"  <uri>{uri}</uri>\n"
                    f"  <score>{score}</score>\n"
                    f"  <content>{content}</content>\n"
                    f"</memory>"
                )
                full_chars = len(full_memory_str)
                if user_memories:
                    full_chars += 1

                if total_chars + full_chars <= max_chars:
                    user_memories.append(full_memory_str)
                    total_chars += full_chars
                    next_index += 1
                else:
                    link_only_memories.append((uri, score))
            else:
                # No content available, use link-only version (always add)
                logger.info(f"Using link-only for {uri} (read failed or empty)")
                link_only_memories.append((uri, score))


        graph_snippet_chars = 0
        graph_snippet_budget = 3200
        graph_snippet_max_chars = 600
        graph_snippet_limit = 8
        graph_snippet_count = 0
        for memory in graph_snippet_memories:
            if graph_snippet_count >= graph_snippet_limit:
                break
            uri = get_uri(memory)
            abstract = " ".join(get_abstract(memory).split())
            if not uri or not abstract:
                continue
            if len(abstract) > graph_snippet_max_chars:
                abstract = abstract[: graph_snippet_max_chars - 3].rstrip() + "..."
            score = get_score(memory)
            memory_str = (
                f'<memory index="{next_index}" type="graph_snippet">\n'
                f"  <uri>{uri}</uri>\n"
                f"  <score>{score}</score>\n"
                f"  <content>{abstract}</content>\n"
                f"</memory>"
            )
            if graph_snippet_chars + len(memory_str) > graph_snippet_budget:
                break
            user_memories.append(memory_str)
            graph_snippet_chars += len(memory_str)
            graph_snippet_count += 1
            next_index += 1

        for uri, score in link_only_memories:
            memory_str = (
                f'<memory index="{next_index}" type="link">\n'
                f"  <uri>{uri}</uri>\n"
                f"  <score>{score}</score>\n"
                f"</memory>"
            )
            user_memories.append(memory_str)
            next_index += 1

        return "\n".join(user_memories)

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def get_viking_memory_context(
        self,
        current_message: str,
        workspace_id: str,
        sender_id: str,
        user_ids: list[str] | None = None,
    ) -> str:
        client = None
        self.latest_graph_retrieval_debug = None
        try:
            config = load_config().ov_server
            admin_user_id = config.admin_user_id
            # Use provided user_ids or fall back to sender_id
            search_user_ids = user_ids if user_ids else [sender_id]
            logger.info(f"workspace_id={workspace_id}")
            logger.info(f"user_ids={search_user_ids}")
            logger.info(f"admin_user_id={admin_user_id}")

            client = await VikingClient.create(agent_id=workspace_id)
            result = await client.search_memory(
                query=current_message,
                user_ids=search_user_ids,
                agent_user_id=admin_user_id,
                limit=10,
            )
            if not result:
                return ""

            result = self._limit_memory_groups(result, limit=10)


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
                result["user_memory"],
                client,
                min_score=0.1,
                query_text=current_message,
            )
            agent_memory = await self._parse_viking_memory(
                result["agent_memory"],
                client,
                min_score=0.1,
                max_chars=2000,
                query_text=current_message,
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

    async def get_viking_experience_context(self, query: str, workspace_id: str) -> str:
        """用当前任务 query 检索 experience 记忆，注入到 system prompt。"""
        client = None
        try:
            ov_cfg = load_config().ov_server
            client = await VikingClient.create(agent_id=workspace_id)
            experiences = await client.search_experiences(query, limit=ov_cfg.exp_recall_limit)
            logger.info(
                f"[READ_EXPERIENCE_MEMORY]: found {len(experiences)} experiences, query={query[:50]}"
            )
            for i, exp in enumerate(experiences):
                uri = exp.get("uri", "") if isinstance(exp, dict) else getattr(exp, "uri", "")
                score = exp.get("score", 0) if isinstance(exp, dict) else getattr(exp, "score", 0)
                logger.info(f"  {i},{uri},{score}")
            if not experiences:
                return ""
            return await self._parse_viking_memory(
                experiences, client, min_score=0.3, max_chars=ov_cfg.exp_recall_max_chars
            )
        except Exception as e:
            logger.error(f"[READ_EXPERIENCE_MEMORY]: error. {e}")
            return ""
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    pass

    async def get_viking_user_profile(self, workspace_id: str, user_id: str) -> str:
        client = None
        try:
            client = await VikingClient.create(agent_id=workspace_id)
            result = await client.read_user_profile(user_id)
            return result or ""
        except Exception as e:
            logger.error(f"[READ_USER_PROFILE]: user_id={user_id}, error. {e}")
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
        except Exception as e:
            logger.error(f"[READ_USER_PROFILES]: error. {e}")
            return ""
        finally:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing VikingClient: {e}")
