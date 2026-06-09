# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Link merge and dedup logic for MEMORY_FIELDS links field.

Dedup key: from_uri + to_uri + match_text + relation_slot + answer_value
Merge rules:
- Weight conflict: take max
- link_type and description: latest write wins
"""

import json
from typing import Any, Dict, List


def _dedup_key(link: Dict[str, Any]) -> str:
    """Compute dedup key for a link."""
    answer_value = link.get("answer_value") or []
    if not isinstance(answer_value, list):
        answer_value = [answer_value]
    answer_key = json.dumps([str(item) for item in answer_value], ensure_ascii=False, sort_keys=True)
    return (
        f"{link.get('from_uri', '')}|{link.get('to_uri', '')}|"
        f"{link.get('match_text', '')}|{link.get('relation_slot', '')}|{answer_key}"
    )


def merge_links(existing_links: List[Dict], new_links: List[Dict]) -> List[Dict]:
    """
    Merge link lists with dedup and conflict resolution.

    Dedup key: from_uri + to_uri + match_text + relation_slot + answer_value
    Weight conflict: take max
    link_type and description: latest write wins
    """
    link_map: Dict[str, Dict[str, Any]] = {}

    # Process existing links first
    for link in existing_links:
        key = _dedup_key(link)
        link_map[key] = dict(link)

    # Process new links (override existing on conflict)
    for link in new_links:
        key = _dedup_key(link)
        if key in link_map:
            existing = link_map[key]
            # Weight: take max
            existing["weight"] = max(existing.get("weight", 1.0), link.get("weight", 1.0))
            # link_type, description, and evidence metadata: latest non-empty write wins
            if "link_type" in link:
                existing["link_type"] = link["link_type"]
            if "description" in link:
                existing["description"] = link["description"]
            for field in (
                "subject",
                "relation_slot",
                "evidence_role",
                "source_span",
            ):
                if link.get(field):
                    existing[field] = link[field]
            if link.get("answer_value"):
                existing["answer_value"] = link["answer_value"]
            # created_at: keep the original
        else:
            link_map[key] = dict(link)

    return list(link_map.values())
