# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Typed Weighted Personalized PageRank for graph-based retrieval."""

from typing import TYPE_CHECKING, Dict

from openviking_cli.utils.logger import get_logger

if TYPE_CHECKING:
    from openviking.retrieve.graph.graph_index import GraphIndex

logger = get_logger(__name__)

# Reverse-direction edges get a penalty factor since backlinks are weaker signals.
_REVERSE_DIRECTION_PENALTY = 0.7


class TypedWeightedPPR:
    """Personalized PageRank with typed edge weights and restart probability."""

    def __init__(
        self,
        graph_index: "GraphIndex",
        type_weights: Dict[str, float],
        restart: float = 0.15,
        max_iter: int = 50,
        tolerance: float = 1e-4,
    ):
        self._graph = graph_index
        self._type_weights = type_weights
        self._restart = restart
        self._max_iter = max_iter
        self._tolerance = tolerance

    def run(self, seed_uris: Dict[str, float]) -> Dict[str, float]:
        """Run PPR from seed nodes.

        Args:
            seed_uris: Map of seed URI -> initial score. Normalized to sum to 1 internally.

        Returns:
            Map of URI -> PPR score for all reachable nodes.
        """
        if not seed_uris:
            return {}

        # Normalize seeds to sum to 1
        total = sum(seed_uris.values())
        if total <= 0:
            return {}
        personalization = {k: v / total for k, v in seed_uris.items()}

        # Build transition weights: for each node, compute normalized outgoing probabilities
        # considering both forward and reverse edges with type weighting.
        transition = self._compute_transition_weights(personalization)

        # Initialize PPR vector with personalization
        nodes = self._graph.get_nodes()
        p = dict(personalization)

        for iteration in range(self._max_iter):
            p_new: Dict[str, float] = {}

            # Propagate: p_new[v] = sum over u of (1-restart) * transition[u][v] * p[u]
            for u, p_u in p.items():
                if p_u == 0:
                    continue
                out_edges = transition.get(u)
                if not out_edges:
                    # Dangling node: redistribute to seeds (standard PPR fix)
                    for seed_uri, seed_prob in personalization.items():
                        p_new[seed_uri] = p_new.get(seed_uri, 0.0) + (1 - self._restart) * p_u * seed_prob
                    continue
                for v, trans_prob in out_edges.items():
                    p_new[v] = p_new.get(v, 0.0) + (1 - self._restart) * p_u * trans_prob

            # Add restart (teleport) to personalization vector
            for seed_uri, seed_prob in personalization.items():
                p_new[seed_uri] = p_new.get(seed_uri, 0.0) + self._restart * seed_prob

            # Convergence check: L1-norm
            diff = 0.0
            all_keys = set(p.keys()) | set(p_new.keys())
            for k in all_keys:
                diff += abs(p_new.get(k, 0.0) - p.get(k, 0.0))

            p = p_new

            if diff < self._tolerance:
                logger.debug(f"[PPR] Converged at iteration {iteration + 1}, L1-diff={diff:.6f}")
                break

        # Filter to nodes that exist in the graph
        return {k: v for k, v in p.items() if k in nodes and v > 0}

    def _compute_transition_weights(
        self, personalization: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Build normalized transition matrix with type-weighted edges.

        For each node u, the transition probability to neighbor v is:
            w(u,v) * type_weight(link_type) / sum_over_all_neighbors(w(u,v') * type_weight)
        Edges are considered in both forward and reverse directions.
        """
        result: Dict[str, Dict[str, float]] = {}
        seed_uris = set(personalization.keys())

        for uri in self._graph.get_nodes():
            outgoing: Dict[str, float] = {}

            # Forward edges (this node is the source)
            for edge in self._graph.get_forward_edges(uri):
                tw = self._type_weights.get(edge.link_type, 1.0)
                weight = edge.weight * tw
                outgoing[edge.to_uri] = outgoing.get(edge.to_uri, 0.0) + weight

            # Reverse edges (this node is the target) with direction penalty
            for edge in self._graph.get_reverse_edges(uri):
                tw = self._type_weights.get(edge.link_type, 1.0)
                weight = edge.weight * tw * _REVERSE_DIRECTION_PENALTY
                outgoing[edge.from_uri] = outgoing.get(edge.from_uri, 0.0) + weight

            if not outgoing:
                # Dangling node: will be handled in run() by redistributing to seeds
                continue

            # Normalize so probabilities sum to 1
            total_weight = sum(outgoing.values())
            if total_weight > 0:
                result[uri] = {v: w / total_weight for v, w in outgoing.items()}

        return result
