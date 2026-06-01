# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from typing import Dict

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Configuration for retrieval ranking behavior."""

    hotness_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for blending hotness into final retrieval scores. "
            "0 disables hotness boost; 1 uses only hotness."
        ),
    )
    score_propagation_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for each child result's own score when blending with its parent score "
            "during hierarchical retrieval. 0 uses only the parent score; "
            "1 uses only the child score."
        ),
    )

    # --- Graph retrieval fields (active only when memory.link_enabled is True) ---

    graph_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for bounded graph (PPR) boosts in the final hybrid ranking. "
            "0 disables graph retrieval entirely; higher values give graph-supported "
            "candidates more room to move toward the strongest semantic score. "
            "Only effective when memory.link_enabled is True."
        ),
    )
    graph_ppr_restart: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Restart probability for Personalized PageRank. "
        "Higher values keep the walk closer to seed nodes.",
    )
    graph_ppr_max_iter: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum PPR iterations before convergence check.",
    )
    graph_ppr_tolerance: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="L1-norm convergence threshold for PPR.",
    )
    graph_expansion_topk: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of top PPR-scoring nodes to add to the candidate pool.",
    )
    graph_path_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of key paths to extract for explainability. 0 disables path extraction.",
    )
    graph_type_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "belongs_to": 1.5,
            "caused_by": 1.3,
            "derived_from": 1.2,
            "evolved_from": 1.1,
            "related_to": 1.0,
            "contradicts": 0.8,
        },
        description="Multiplier applied to edge weight by link_type during PPR transition.",
    )
    graph_seed_include_summaries: bool = Field(
        default=False,
        description=(
            "When True, .abstract.md and .overview.md nodes are included as seed "
            "candidates. Disabled by default to keep graph walks anchored to "
            "retrieved memories."
        ),
    )

    model_config = {"extra": "forbid"}
