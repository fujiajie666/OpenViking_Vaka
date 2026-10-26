#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for VikingFS.mkdir() — verifies the target directory is actually created
and that backend errors are propagated instead of silently swallowed."""

import contextvars
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.pyagfs.exceptions import (
    AGFSAlreadyExistsError,
    AGFSClientError,
    AGFSIoError,
    AGFSNotADirectoryError,
    AGFSNotFoundError,
    AGFSPermissionDeniedError,
    AGFSResourceExhaustedError,
)
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make_viking_fs():
    """Create a VikingFS instance with mocked AGFS backend."""
    from openviking.storage.viking_fs import VikingFS

    fs = VikingFS.__new__(VikingFS)
    fs.agfs = MagicMock()
    fs._async_agfs = MagicMock()
    fs._async_agfs.mkdir = AsyncMock(return_value=None)
    fs._async_agfs.ensure_parent_dirs = AsyncMock(return_value=None)
    fs._async_agfs.stat = AsyncMock(return_value={"isDir": True})
    fs._async_agfs.write = AsyncMock(return_value=None)
    fs.query_embedder = None
    fs.vector_store = None
    fs.acl_manager = None
    fs._uri_prefix = "viking://"
    fs._bound_ctx = contextvars.ContextVar("vikingfs_bound_ctx", default=None)
    return fs


class TestMkdir:
    """Test that mkdir() actually creates the target directory."""

    @pytest.mark.asyncio
    async def test_mkdir_calls_agfs_mkdir(self):
        """mkdir() must call agfs.mkdir with the target path."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()

        await fs.mkdir("viking://resources/new_dir")

        fs._async_agfs.mkdir.assert_called_once()
        call_path = fs._async_agfs.mkdir.call_args[0][0]
        assert call_path.endswith("resources/new_dir")

    @pytest.mark.asyncio
    async def test_mkdir_exist_ok_true_existing(self):
        """mkdir(exist_ok=True) should tolerate an already-exists backend error."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()
        fs._async_agfs.mkdir = AsyncMock(side_effect=Exception("directory already exists"))

        await fs.mkdir("viking://resources/existing_dir", exist_ok=True)

        fs._async_agfs.mkdir.assert_called_once()

    @pytest.mark.asyncio
    async def test_mkdir_exist_ok_true_not_existing(self):
        """mkdir(exist_ok=True) should create dir if it does not exist."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()

        await fs.mkdir("viking://resources/new_dir", exist_ok=True)

        fs._async_agfs.mkdir.assert_called_once()
        call_path = fs._async_agfs.mkdir.call_args[0][0]
        assert call_path.endswith("resources/new_dir")

    @pytest.mark.asyncio
    async def test_mkdir_exist_ok_false_default(self):
        """mkdir(exist_ok=False) should always attempt to create."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()

        await fs.mkdir("viking://resources/another_dir")

        fs._async_agfs.mkdir.assert_called_once()

    @pytest.mark.asyncio
    async def test_mkdir_ensures_parents_first(self):
        """mkdir() must call _ensure_parent_dirs before creating target."""
        fs = _make_viking_fs()
        call_order = []
        fs._ensure_parent_dirs = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append("parents")
        )
        fs._async_agfs.mkdir = AsyncMock(side_effect=lambda *a, **kw: call_order.append("mkdir"))

        await fs.mkdir("viking://a/b/c")

        assert call_order == ["parents", "mkdir"]

    @pytest.mark.asyncio
    async def test_mkdir_propagates_backend_error(self):
        """mkdir() must not swallow backend errors (permissions, quota, I/O)."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()
        fs._async_agfs.mkdir = AsyncMock(side_effect=Exception("permission denied"))

        with pytest.raises(Exception, match="permission denied"):
            await fs.mkdir("viking://resources/protected_dir")

    @pytest.mark.asyncio
    async def test_mkdir_exist_ok_true_propagates_non_exists_error(self):
        """mkdir(exist_ok=True) only tolerates already-exists errors, not others."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()
        fs._async_agfs.mkdir = AsyncMock(
            side_effect=Exception("resource exhausted: quota exceeded")
        )

        with pytest.raises(Exception, match="resource exhausted"):
            await fs.mkdir("viking://resources/full_dir", exist_ok=True)

    @pytest.mark.asyncio
    async def test_mkdir_exist_ok_false_raises_on_existing(self):
        """mkdir(exist_ok=False) surfaces the already-exists error to the caller."""
        fs = _make_viking_fs()
        fs._ensure_parent_dirs = AsyncMock()
        fs._async_agfs.mkdir = AsyncMock(side_effect=Exception("directory already exists"))

        with pytest.raises(Exception, match="already exists"):
            await fs.mkdir("viking://resources/existing_dir")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            AGFSAlreadyExistsError("target already exists"),
            FileExistsError("target already exists"),
            AGFSClientError("file exists"),
            RuntimeError("directory already exists"),
        ],
    )
    async def test_existing_directory_keeps_tenant_and_lease(self, error):
        fs = _make_viking_fs()
        fs._ensure_access = AsyncMock()
        fs._async_agfs.mkdir.side_effect = error
        ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ROOT)
        lease = {"lease_ref": "lease-1"}

        await fs.mkdir("viking://resources/existing", exist_ok=True, ctx=ctx, lease_ref=lease)

        path = "/local/acme/resources/existing"
        fs_ctx = {"account_id": "acme", "lease_ref": "lease-1"}
        fs._async_agfs.ensure_parent_dirs.assert_awaited_once_with(path, fs_ctx=fs_ctx)
        fs._async_agfs.mkdir.assert_awaited_once_with(path, fs_ctx=fs_ctx)
        fs._async_agfs.stat.assert_awaited_once_with(path, fs_ctx=fs_ctx)

    @pytest.mark.asyncio
    async def test_existing_regular_file_is_not_accepted_as_directory(self):
        fs = _make_viking_fs()
        error = AGFSAlreadyExistsError("target exists")
        fs._async_agfs.mkdir.side_effect = error
        fs._async_agfs.stat.return_value = {"isDir": False}

        with pytest.raises(AGFSAlreadyExistsError) as raised:
            await fs.mkdir("viking://resources/existing", exist_ok=True)

        assert raised.value is error

    @pytest.mark.asyncio
    async def test_existing_directory_stat_failure_propagates(self):
        fs = _make_viking_fs()
        fs._async_agfs.mkdir.side_effect = AGFSAlreadyExistsError("target exists")
        error = AGFSPermissionDeniedError("cannot inspect target")
        fs._async_agfs.stat.side_effect = error

        with pytest.raises(AGFSPermissionDeniedError) as raised:
            await fs.mkdir("viking://resources/existing", exist_ok=True)

        assert raised.value is error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            Exception("parent does not exist"),
            RuntimeError("request already cancelled"),
            AGFSNotFoundError("parent does not exist"),
            AGFSPermissionDeniedError("permission denied; target already exists"),
            AGFSResourceExhaustedError("quota exceeded"),
            AGFSIoError("storage unavailable"),
        ],
    )
    async def test_exist_ok_does_not_suppress_unrelated_errors(self, error):
        fs = _make_viking_fs()
        fs._async_agfs.mkdir.side_effect = error

        with pytest.raises(type(error)) as raised:
            await fs.mkdir("viking://resources/output", exist_ok=True)

        assert raised.value is error
        fs._async_agfs.stat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        AGFSPermissionDeniedError("permission denied"),
        AGFSResourceExhaustedError("quota exceeded"),
        AGFSIoError("write unavailable"),
        AGFSNotADirectoryError("parent is a file"),
        AGFSNotFoundError("parent does not exist"),
        TimeoutError("parent creation timed out"),
    ],
)
async def test_parent_creation_failure_stops_write_and_preserves_error(error):
    fs = _make_viking_fs()
    fs._ensure_access = AsyncMock()
    fs._async_agfs.ensure_parent_dirs.side_effect = error
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ROOT)
    lease = {"lease_ref": "lease-1"}

    with pytest.raises(type(error)) as raised:
        await fs.write_file(
            "viking://resources/notes/output.md", "content", ctx=ctx, lease_ref=lease
        )

    assert raised.value is error
    fs._async_agfs.ensure_parent_dirs.assert_awaited_once_with(
        "/local/acme/resources/notes/output.md",
        fs_ctx={"account_id": "acme", "lease_ref": "lease-1"},
    )
    fs._async_agfs.write.assert_not_awaited()
    fs._async_agfs.mkdir.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_creation_success_preserves_write_context():
    fs = _make_viking_fs()
    fs._ensure_access = AsyncMock()
    ctx = RequestContext(user=UserIdentifier("acme", "alice"), role=Role.ROOT)
    lease = {"lease_ref": "lease-1"}

    await fs.write_file("viking://resources/notes/output.md", "content", ctx=ctx, lease_ref=lease)

    path = "/local/acme/resources/notes/output.md"
    fs_ctx = {"account_id": "acme", "lease_ref": "lease-1"}
    fs._async_agfs.ensure_parent_dirs.assert_awaited_once_with(path, fs_ctx=fs_ctx)
    fs._async_agfs.write.assert_awaited_once_with(
        path, b"content", fs_ctx=fs_ctx, auto_pathlock=True
    )


@pytest.mark.asyncio
async def test_local_backend_creates_parents_and_preserves_internal_spaces(tmp_path):
    """The shared test backend implements parent creation without renaming valid paths."""
    from openviking.storage.viking_fs import VikingFS
    from tests.utils.mock_agfs import MockLocalAGFS

    backend = MockLocalAGFS(root_path=tmp_path)
    fs = VikingFS(agfs=backend)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
    uri = "viking://resources/业务 域/章节 一.md"

    await fs.write_file(uri, "完整正文", ctx=ctx)

    path = f"/local/{ctx.account_id}/resources/业务 域/章节 一.md"
    assert backend.read_file(path) == "完整正文".encode("utf-8")
