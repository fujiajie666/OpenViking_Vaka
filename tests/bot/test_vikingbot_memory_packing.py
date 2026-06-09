from pathlib import Path

import pytest

from vikingbot.agent.memory import MemoryStore


class FakeClient:
    def __init__(self, contents: dict[str, str]):
        self.contents = contents

    async def read_content(self, uri: str, level: str = "read", raw: bool = False) -> str:
        return self.contents.get(uri, "")


def test_limit_memory_groups_preserves_graph_candidates():
    result = {
        "user_memory": [
            {"uri": f"viking://user/u/memories/{i}.md", "score": 1.0 - i * 0.01}
            for i in range(12)
        ]
        + [
            {
                "uri": "viking://user/u/memories/graph.md",
                "score": 0.1,
                "match_reason": "Discovered via graph expansion",
            }
        ],
        "agent_memory": [],
    }

    limited = MemoryStore._limit_memory_groups(result, limit=10, graph_limit=2)

    uris = [memory["uri"] for memory in limited["user_memory"]]
    assert len(uris) == 11
    assert "viking://user/u/memories/10.md" not in uris
    assert "viking://user/u/memories/graph.md" in uris


def test_pick_evidence_span_prefers_matching_structured_evidence():
    content = """- Long profile

<!-- MEMORY_FIELDS
{
  "links": [
    {
      "subject": "tim",
      "relation_slot": "recommended_book",
      "answer_value": ["The Name of the Wind"],
      "evidence_role": "direct",
      "source_span": "Tim recommended The Name of the Wind.",
      "weight": 0.9
    },
    {
      "subject": "tim",
      "relation_slot": "visited_location",
      "answer_value": ["Harry Potter related location in London"],
      "evidence_role": "direct",
      "source_span": "I went to a place in London a few years ago.",
      "weight": 0.9
    }
  ]
}
-->
"""

    spans = MemoryStore._pick_evidence_spans(
        content,
        "Which geographical locations has Tim been to?",
    )

    assert spans == ["I went to a place in London a few years ago."]


@pytest.mark.asyncio
async def test_long_memory_falls_back_to_evidence_snippet_instead_of_link_only(tmp_path: Path):
    uri = "viking://user/conv-43/memories/entities/person/tim.md"
    content = "\n".join(
        [
            "- Enjoys reading fantasy novels",
            "- Visited a Harry Potter related location in London a few years prior to 2023",
            "- Owns serenity memory foam shoes",
            "x" * 4500,
            """<!-- MEMORY_FIELDS
{"links":[{"subject":"tim","relation_slot":"visited_location","answer_value":["Harry Potter related location in London"],"source_span":"I went to a place in London a few years ago."}]}
-->""",
        ]
    )
    store = MemoryStore(tmp_path)

    packed = await store._parse_viking_memory(
        [{"uri": uri, "score": 0.9, "abstract": ""}],
        FakeClient({uri: content}),
        min_score=0.1,
        max_chars=4000,
        query_text="Which geographical locations has Tim been to?",
    )

    assert 'type="evidence_snippet"' in packed
    assert "London" in packed
    assert 'type="link"' not in packed
