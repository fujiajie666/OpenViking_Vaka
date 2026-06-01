# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lightweight path extraction for graph retrieval explainability."""

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from openviking.retrieve.graph.graph_index import GraphEdge, GraphIndex


@dataclass
class GraphPath:
    nodes: List[str]
    edges: List["GraphEdge"]
    path_score: float


class PathExtractor:
    """Extract key paths from seeds to high-scoring PPR nodes."""

    def __init__(self, graph_index: "GraphIndex", max_paths: int = 3, max_path_length: int = 4):
        self._graph = graph_index
        self._max_paths = max_paths
        self._max_path_length = max_path_length

    def extract(
        self,
        seeds: Dict[str, float],
        ppr_scores: Dict[str, float],
        top_k: int = 5,
    ) -> List[GraphPath]:
        """Extract paths from seeds to top PPR nodes via weighted BFS.

        For each target (top-k PPR node not in seeds), find the shortest
        weighted path to any seed node by traversing reverse edges.

        Returns up to max_paths paths, ranked by path_score * target_ppr_score.
        """
        if not seeds or not ppr_scores or self._max_paths <= 0:
            return []

        seed_set = set(seeds.keys())

        # Get top-k PPR nodes that are NOT seeds
        candidates = sorted(
            ((uri, score) for uri, score in ppr_scores.items() if uri not in seed_set),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        if not candidates:
            return []

        all_paths: List[GraphPath] = []

        for target_uri, target_ppr in candidates:
            path = self._find_path_to_seed(target_uri, seed_set)
            if path is not None:
                all_paths.append(path)

        # Rank by path_score * target_ppr_score and return top-N
        all_paths.sort(
            key=lambda p: p.path_score * ppr_scores.get(p.nodes[0], 0.0),
            reverse=True,
        )
        return all_paths[: self._max_paths]

    def _find_path_to_seed(
        self, target_uri: str, seed_set: Set[str]
    ) -> "GraphPath | None":
        """Find the highest-weight path from target to any seed via reverse BFS.

        Uses Dijkstra-like approach: explore neighbors in order of accumulated weight
        (higher is better), stopping when a seed node is reached.
        """
        # Priority queue: (-accumulated_weight, counter, uri, path_nodes, path_edges)
        # Negative because heapq is a min-heap and we want max-weight first
        counter = count()
        frontier: List[tuple] = [(-1.0, next(counter), target_uri, [target_uri], [])]
        visited: Set[str] = set()

        while frontier:
            neg_acc, _order, current_uri, path_nodes, path_edges = heapq.heappop(frontier)
            acc_weight = -neg_acc

            if current_uri in visited:
                continue
            visited.add(current_uri)

            if current_uri in seed_set and len(path_nodes) > 1:
                # Reached a seed — construct path
                return GraphPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    path_score=acc_weight,
                )

            if len(path_nodes) >= self._max_path_length:
                continue

            # Explore reverse edges only (walking backwards from target toward seeds).
            # Reverse edges at current_uri have to_uri == current_uri, so following
            # from_uri takes us toward nodes that point TO current_uri — i.e. closer
            # to seeds along the forward direction of the random walk.
            for edge in self._graph.get_reverse_edges(current_uri):
                if edge.from_uri in visited:
                    continue
                # Weight is the product of edge weights along the path
                new_acc = acc_weight * edge.weight
                heapq.heappush(
                    frontier,
                    (
                        -new_acc,
                        next(counter),
                        edge.from_uri,
                        path_nodes + [edge.from_uri],
                        path_edges + [edge],
                    ),
                )

        return None
