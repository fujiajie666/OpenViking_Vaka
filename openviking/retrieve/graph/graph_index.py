# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""In-memory graph index built from MEMORY_FIELDS links across memory spaces."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openviking.server.identity import RequestContext
from openviking.session.memory.utils.memory_file_utils import MemoryFileUtils
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_TTL_SECONDS = 300.0  # 5 minutes
_READ_SEMAPHORE = 16
_GRAPH_INDEX_CACHE_STRATEGY = "per_space_index_v17"


@dataclass
class GraphNode:
    uri: str
    memory_type: Optional[str] = None
    category: str = ""
    is_summary: bool = False


@dataclass
class GraphEdge:
    from_uri: str
    to_uri: str
    link_type: str
    weight: float
    description: str = ""
    match_text: str = ""
    subject: str = ""
    relation_slot: str = ""
    answer_value: List[str] = field(default_factory=list)
    evidence_role: str = "context"
    source_span: str = ""

    def is_evidence_edge(self) -> bool:
        return self.link_type == "evidence_for" or self.evidence_role in {
            "direct",
            "list_member",
            "count_member",
            "derived_intersection",
        }

    def is_navigation_or_context(self) -> bool:
        return self.link_type in {"context_for", "belongs_to", "related_to"} or self.evidence_role in {
            "context",
            "navigation",
        }


class GraphIndex:
    """In-memory graph index built from MEMORY_FIELDS links."""

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._forward_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._reverse_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._space_key: Optional[str] = None
        self._built_at: float = 0.0
        self._lock = asyncio.Lock()

    async def build(self, space_uris: List[str], ctx: RequestContext) -> None:
        """Build graph index from filesystem."""
        async with self._lock:
            # Double-check after acquiring lock
            key = "+".join(sorted(space_uris))
            if key == self._space_key and (time.monotonic() - self._built_at) <= _TTL_SECONDS:
                return

            viking_fs = get_viking_fs()
            if not viking_fs:
                logger.warning("[GraphIndex] VikingFS not available, skipping build")
                return

            nodes: Dict[str, GraphNode] = {}
            forward_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
            reverse_edges: Dict[str, List[GraphEdge]] = defaultdict(list)

            for space_uri in space_uris:
                try:
                    await self._build_space(
                        viking_fs, space_uri, ctx, nodes, forward_edges, reverse_edges
                    )
                except Exception as e:
                    logger.error(
                        f"[GraphIndex] Failed to build space {space_uri}: {type(e).__name__}: {e}",
                        exc_info=True,
                    )

            # Atomic assignment
            self._nodes = nodes
            self._forward_edges = forward_edges
            self._reverse_edges = reverse_edges
            self._space_key = key
            self._built_at = time.monotonic()

            logger.info(
                f"[GraphIndex] Built graph: {len(nodes)} nodes, "
                f"{sum(len(e) for e in forward_edges.values())} edges"
            )

    async def _build_space(
        self,
        viking_fs: Any,
        space_uri: str,
        ctx: RequestContext,
        nodes: Dict[str, GraphNode],
        forward_edges: Dict[str, List[GraphEdge]],
        reverse_edges: Dict[str, List[GraphEdge]],
    ) -> None:
        entries = await viking_fs.tree(
            space_uri, node_limit=1000000, level_limit=None,
            show_all_hidden=True, ctx=ctx
        )
        logger.info(f"[GraphIndex] tree({space_uri}) returned {len(entries)} entries")
        md_uris = []
        summary_uris = []
        for entry in entries:
            if entry.get("isDir"):
                continue
            rel_path = entry.get("rel_path", "")
            if not rel_path.endswith(".md"):
                continue
            uri = entry["uri"]
            if rel_path.endswith("/.abstract.md") or rel_path.endswith("/.overview.md"):
                summary_uris.append(uri)
            else:
                md_uris.append(uri)

        logger.info(
            f"[GraphIndex] Found {len(md_uris)} memory files, "
            f"{len(summary_uris)} summary files under {space_uri}"
        )

        # Read and parse all files in parallel
        sem = asyncio.Semaphore(_READ_SEMAPHORE)

        async def _read_and_parse(uri: str, is_summary: bool) -> Optional[tuple]:
            async with sem:
                try:
                    content = await viking_fs.read_file(uri, ctx=ctx)
                    if not content:
                        return None
                    mf = MemoryFileUtils.read(content, uri=uri)
                    return uri, mf, is_summary
                except Exception as e:
                    logger.debug(f"[GraphIndex] Failed to read {uri}: {e}")
                    return None

        tasks = [_read_and_parse(u, False) for u in md_uris]
        tasks += [_read_and_parse(u, True) for u in summary_uris]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is None:
                continue
            uri, mf, is_summary = result
            memory_type = mf.memory_type or ""
            category = mf.extra_fields.get("category", "")

            nodes[uri] = GraphNode(
                uri=uri,
                memory_type=memory_type,
                category=category,
                is_summary=is_summary,
            )

            # Forward links only (backlinks are the same edges stored on the
            # target file, so processing both would double-count every edge).
            seen_edges: set = set()
            for link_data in mf.links:
                if not isinstance(link_data, dict):
                    continue
                to_uri = link_data.get("to_uri", "")
                if not to_uri:
                    continue
                link_type = link_data.get("link_type", "related_to")
                edge_key = (
                    uri,
                    to_uri,
                    link_type,
                    str(link_data.get("relation_slot", "") or ""),
                    tuple(self._string_list(link_data.get("answer_value"))),
                )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge = GraphEdge(
                    from_uri=uri,
                    to_uri=to_uri,
                    link_type=link_type,
                    weight=float(link_data.get("weight", 1.0)),
                    description=link_data.get("description", ""),
                    match_text=str(link_data.get("match_text", "") or ""),
                    subject=str(link_data.get("subject", "") or ""),
                    relation_slot=str(link_data.get("relation_slot", "") or ""),
                    answer_value=self._string_list(link_data.get("answer_value")),
                    evidence_role=str(link_data.get("evidence_role", "context") or "context"),
                    source_span=str(link_data.get("source_span", "") or ""),
                )
                forward_edges[edge.from_uri].append(edge)
                reverse_edges[edge.to_uri].append(edge)

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    def is_fresh(self, space_uris: List[str]) -> bool:
        """Check if cached index is still valid for the given space URIs."""
        key = "+".join(sorted(space_uris))
        if key != self._space_key:
            return False
        if not self._nodes:
            return False
        if (time.monotonic() - self._built_at) > _TTL_SECONDS:
            return False
        return True

    def get_nodes(self) -> Dict[str, GraphNode]:
        return self._nodes

    def get_forward_edges(self, uri: str) -> List[GraphEdge]:
        return self._forward_edges.get(uri, [])

    def get_reverse_edges(self, uri: str) -> List[GraphEdge]:
        return self._reverse_edges.get(uri, [])

    def get_all_edges(self) -> List[GraphEdge]:
        edges: List[GraphEdge] = []
        for edge_list in self._forward_edges.values():
            edges.extend(edge_list)
        return edges

    def has_node(self, uri: str) -> bool:
        return uri in self._nodes

    def get_node(self, uri: str) -> Optional[GraphNode]:
        return self._nodes.get(uri)


# Module-level singleton
_instance: Optional[GraphIndex] = None
_instances_by_space_key: Dict[str, GraphIndex] = {}


def get_graph_index(space_uris: Optional[List[str]] = None) -> GraphIndex:
    """Return a graph index scoped to the requested memory spaces.

    Retrieval requests for different memory spaces can run concurrently. A
    single mutable process-wide index lets one request rebuild the graph while
    another request is still scoring candidates from a previous graph. Keep the
    legacy no-argument singleton for callers that do not have a space key, but
    prefer a per-space index when the caller can provide one.
    """
    global _instance
    if space_uris is not None:
        key = "+".join(sorted(space_uris))
        if not key:
            key = "__empty__"
        index = _instances_by_space_key.get(key)
        if index is None:
            index = GraphIndex()
            _instances_by_space_key[key] = index
            logger.info(
                "[GraphIndex] strategy=%s created scoped graph index for space_key=%s",
                _GRAPH_INDEX_CACHE_STRATEGY,
                key,
            )
        return index

    if _instance is None:
        _instance = GraphIndex()
    return _instance
