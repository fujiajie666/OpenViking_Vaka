"""Compile conflict counts use complete visible files and exact metadata states."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.service.compile_targets import CompileLifecycle


def comment(*items):
    return (
        b"<!-- CONFLICTS_CONTENT\n"
        + b",\n".join(json.dumps(item).encode() for item in items)
        + b"\n-->"
    )


@pytest.mark.parametrize("count", [0, 1, 31, 1001])
async def test_counts_all_records_and_blocks_without_matching_body_text(count):
    body = (
        b'body: {"state":"unfixed"}\n<!-- ordinary {"state":"unfixed"} -->\n'
        + comment(*([{"state": "unfixed"}] * count))
        + comment(
            {"state": "fixed"},
            {"state": "UNFIXED"},
            {"nested": {"state": "unfixed"}},
            {"match_text": '{"state":"unfixed"}'},
        )
    )
    fs = SimpleNamespace(
        tree=AsyncMock(
            return_value=[
                {"uri": "to/nested", "isDir": True},
                {"uri": "to/nested/page.md", "rel_path": "nested/page.md", "isDir": False},
                {"uri": "to/.hidden.md", "rel_path": ".hidden.md", "isDir": False},
                {"uri": "to/.hidden/page.md", "rel_path": ".hidden/page.md", "isDir": False},
            ]
        ),
        read_file_bytes=AsyncMock(return_value=body),
    )
    assert await CompileLifecycle(None, fs)._count_conflicts("to", None) == count
    fs.tree.assert_awaited_once_with(
        "to", show_all_hidden=False, node_limit=None, level_limit=None, ctx=None
    )
    fs.read_file_bytes.assert_awaited_once_with("to/nested/page.md", ctx=None)


async def test_empty_tree_and_unannotated_binary_count_zero():
    fs = SimpleNamespace(tree=AsyncMock(return_value=[]), read_file_bytes=AsyncMock())
    lifecycle = CompileLifecycle(None, fs)
    assert await lifecycle._count_conflicts("to", None) == 0
    fs.read_file_bytes.assert_not_awaited()
    fs.tree.return_value = [{"uri": "to/image.png", "rel_path": "image.png", "isDir": False}]
    fs.read_file_bytes.return_value = b"\x89PNG\xff\x00"
    assert await lifecycle._count_conflicts("to", None) == 0


@pytest.mark.parametrize(
    "body",
    [
        b'<!-- CONFLICTS_CONTENT {"state":"unfixed"}',
        b'<!-- CONFLICTS_CONTENT {"state":broken} -->',
        b'<!-- CONFLICTS_CONTENT "unfixed" -->',
    ],
)
async def test_malformed_file_is_skipped_without_losing_other_files(body):
    fs = SimpleNamespace(
        tree=AsyncMock(
            return_value=[
                {"uri": "to/good.md", "rel_path": "good.md", "isDir": False},
                {"uri": "to/bad.md", "rel_path": "bad.md", "isDir": False},
            ]
        ),
        read_file_bytes=AsyncMock(
            side_effect=[comment({"state": "unfixed"}), comment({"state": "unfixed"}) + body]
        ),
    )
    assert await CompileLifecycle(None, fs)._count_conflicts("to", None) == 1


async def test_unreadable_file_is_skipped_and_reads_stay_bounded():
    active = peak = finished = 0

    async def read(uri, **kwargs):
        nonlocal active, peak, finished
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            if uri == "to/0":
                raise OSError("unreadable output")
            return comment({"state": "unfixed"})
        finally:
            active -= 1
            finished += 1

    fs = SimpleNamespace(
        tree=AsyncMock(
            return_value=[{"uri": f"to/{i}", "rel_path": str(i), "isDir": False} for i in range(20)]
        ),
        read_file_bytes=read,
    )
    assert await CompileLifecycle(None, fs)._count_conflicts("to", None) == 19
    assert active == 0 and peak == 8 and finished == 20


async def test_cancellation_is_not_treated_as_a_file_error():
    fs = SimpleNamespace(
        tree=AsyncMock(return_value=[{"uri": "to/page.md", "rel_path": "page.md", "isDir": False}]),
        read_file_bytes=AsyncMock(side_effect=asyncio.CancelledError),
    )
    with pytest.raises(asyncio.CancelledError):
        await CompileLifecycle(None, fs)._count_conflicts("to", None)
