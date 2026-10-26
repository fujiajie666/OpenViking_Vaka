"""Focused pipeline children using the existing AgentLoop and isolated file tools."""

from __future__ import annotations

import inspect
import json
import uuid
from collections import Counter
from copy import copy

import json_repair

from vikingbot.agent.tools.base import Tool
from vikingbot.agent.tools.compile import CompileChildTool
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.compile import file_ops
from vikingbot.compile.hashing import content_hash
from vikingbot.compile.models import COMPILE_DRAFT_ROOT
from vikingbot.compile.pipeline_io import ROOT, ModelCallError, TaskFiles, retry_allowed
from vikingbot.compile.renderer import validate_relative_file_path
from vikingbot.compile.results import FileResponse, RecordResponse
from vikingbot.compile.schemas import result_schema
from vikingbot.compile.skill_resources import EvidenceReader
from vikingbot.providers.base import LLMProvider


class ChildProvider(LLMProvider):
    """Route each child request through the task's bounded, metered provider calls."""

    def __init__(self, model, schema, stage="agent", *, submit=None, context=None):
        super().__init__()
        self.model = model
        self.stage = stage
        self.submit = submit
        self.context = context
        self.failures = Counter()

    def get_default_model(self):
        return self.model.model

    async def chat(self, messages, tools=None, **kwargs):
        """Retry truncated output without executing incomplete tool calls."""
        while True:
            if self.context is not None:
                await self.context.bound_messages(messages, tools)
            response = await self.model.call(
                self.stage, messages, tools, max_tokens=self.model.max_tokens
            )
            if response.finish_reason not in {"length", "max_tokens"}:
                return response
            error = "Agent output was truncated; use smaller writes or content_ref and complete tool calls"
            if not retry_allowed(self.failures, error):
                raise ValueError(error)
            self.model.metrics["repairs"] += 1
            messages = [*messages, {"role": "user", "content": error}]


async def complete_tool_result(result, session_key):
    """Keep scoped tool evidence complete for subsequent model requests.

    Child file tools cannot read shared preview spill files, so head/tail previews
    are not a lossless delivery mechanism for these isolated assignments.
    """
    return result


class ScopedReadTool(Tool):
    """Page private file reads by Unicode characters, including files with very long lines.

    Line selection remains owned by the bound read tool. Character offsets address
    that selected text; a non-null next_offset_chars permits lossless continuation.
    """

    def __init__(self, tool, max_chars):
        self.tool, self.max_chars = tool, max_chars

    @property
    def name(self):
        return self.tool.name

    @property
    def description(self):
        return self.tool.description + " Returns a bounded text page; follow next_offset_chars."

    @property
    def parameters(self):
        parameters = copy(self.tool.parameters)
        parameters["properties"] = dict(parameters["properties"])
        parameters["properties"].update(
            offset_chars={
                "type": "integer",
                "minimum": 0,
                "description": "Zero-based character offset within the selected lines.",
            },
            limit_chars={
                "type": "integer",
                "minimum": 1,
                "maximum": self.max_chars,
                "description": "Maximum characters in this page; defaults to the page budget.",
            },
        )
        return parameters

    async def execute(self, tool_context, *, offset_chars=0, limit_chars=None, **kwargs):
        """Read a bounded page without changing the wrapped tool's path permissions."""
        if offset_chars < 0 or (limit_chars is not None and limit_chars < 1):
            raise ValueError("Character offset must be nonnegative and limit must be positive")
        text = await self.tool.execute(tool_context, **kwargs)
        end = min(len(text), offset_chars + min(limit_chars or self.max_chars, self.max_chars))
        return json.dumps(
            {
                "text": text[offset_chars:end],
                "offset_chars": offset_chars,
                "next_offset_chars": end if end < len(text) else None,
                "total_chars": len(text),
            },
            ensure_ascii=False,
        )


class FileContext:
    """Keep collision-agent requests bounded with reversible private-file history.

    Complete tool exchanges are archived together to preserve call/response pairing.
    The original task and fixed instructions remain in context; oversized fixed
    content fails locally. Archive paths remain discoverable in context-history.json.
    """

    def __init__(self, model, root):
        self.model, self.root = model, root
        self.sandbox = model.files.sandbox
        self.history = []
        self.page_chars = max(1, (model.budget - model.reserve - 1200) // 8)

    async def store(self, content, suffix):
        """Save complete text, update the archive index and return the private relative path."""
        path = f"history-{len(self.history)}.{suffix}"
        await self.sandbox.write_file(f"{self.root}/{path}", content)
        self.history.append(path)
        await self.sandbox.write_file(
            f"{self.root}/context-history.json", json.dumps(self.history, ensure_ascii=False)
        )
        return path

    async def preview_result(self, result, session_key):
        """Replace oversized tool text with a handle to its complete, pageable content."""
        text = result if isinstance(result, str) else str(result)
        # JSON page envelopes and escaped text must remain readable without being re-spilled.
        if len(text) <= self.page_chars * 2 + 512:
            return result
        path = await self.store(text, "txt")
        return json.dumps({"content_file": path, "total_chars": len(text)}, ensure_ascii=False)

    async def bound_messages(self, messages, tools):
        """Archive oldest complete exchanges in place before any provider call or retry."""
        overhead = len(json.dumps(tools, ensure_ascii=False)) + self.model.reserve + 1200
        while len(json.dumps(messages, ensure_ascii=False)) + overhead > self.model.budget:
            for start, message in enumerate(messages):
                if message.get("role") != "assistant" or not message.get("tool_calls"):
                    continue
                end = start + 1
                while end < len(messages) and messages[end].get("role") == "tool":
                    end += 1
                expected = {call["id"] for call in message["tool_calls"]}
                delivered = {item.get("tool_call_id") for item in messages[start + 1 : end]}
                if expected == delivered:
                    break
            else:
                raise ModelCallError("Collision agent fixed context exceeds the request budget")
            await self.store(json.dumps(messages[start:end], ensure_ascii=False, indent=2), "json")
            notice = {
                "role": "user",
                "content": '{"archived_tool_exchanges":"context-history.json"}',
            }
            messages[start:end] = [] if notice in messages else [notice]


async def prepare_assignment(data, sandbox, root):
    """Store an oversized assignment in private scratch, replacing source bodies with read handles.

    Return only a manifest handle; the caller retains full data for lineage validation.
    Candidate bodies use text files so line-based reads do not expand an entire JSON string.
    """
    assignment = {k: v for k, v in data.items() if k != "original_evidence"}
    assignment["inputs"] = []
    for index, item in enumerate(data.get("inputs", [])):
        payload = item.get("payload", {})
        path = f"input-{index}.json"
        await sandbox.write_file(
            f"{root}/{path}", json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if "text" in payload and "uri" in payload:
            payload = {"uri": payload["uri"], "source_ranges": [item["id"]]}
        else:
            payload = {k: payload[k] for k in ("source_ranges", "evidence_spans") if k in payload}
        view = {**item, "payload": payload, "payload_file": path}
        if item.get("ready_file", {}).get("content") is not None:
            ready_path = f"ready-{index}.txt"
            await sandbox.write_file(f"{root}/{ready_path}", item["ready_file"]["content"])
            view["ready_file"] = {
                **{k: v for k, v in item["ready_file"].items() if k != "content"},
                "content_file": ready_path,
            }
        assignment["inputs"].append(view)
    if "candidates" in data:
        assignment["candidates"] = []
        for index, candidate in enumerate(data["candidates"]):
            if candidate.get("binary"):
                assignment["candidates"].append(candidate)
                continue
            path = f"candidate-{index}.txt"
            await sandbox.write_file(f"{root}/{path}", candidate["content"])
            assignment["candidates"].append(
                {**{k: v for k, v in candidate.items() if k != "content"}, "content_file": path}
            )
    await sandbox.write_file(
        f"{root}/assignment.json", json.dumps(assignment, ensure_ascii=False, indent=2)
    )
    return {
        "assignment_file": "assignment.json",
        "instructions": "Read assignment.json in ranges with read_file. Read payload_file and content_file "
        "paths for complete input records and candidate bodies. Use read_evidence for assigned original sources. "
        "Process all assigned inputs before submitting the complete result.",
    }


class EmitResult(Tool):
    """Validate child results with independent retry allowances for distinct errors."""

    name = "emit"
    description = "Submit the complete typed transformation result."

    def __init__(
        self,
        schema,
        validate,
        sandbox=None,
        root="",
        data=None,
        metrics=None,
    ):
        self.schema, self.validate = schema, validate
        self.sandbox, self.root = sandbox, root
        self.result = None
        self.failures = Counter()
        self.can_retry = True
        self.last_error = ""
        self.data = data or {}
        self.metrics = metrics

    async def read_content(self, relative: str) -> bytes:
        """Read private scratch or an assigned ready snapshot; foreign snapshots are not accessible."""
        supplied = {
            item.get("ready_file", {}).get("content_ref") for item in self.data.get("inputs", [])
        }
        supplied.update(item.get("content_ref") for item in self.data.get("candidates", []))
        if relative in supplied:
            saved = await TaskFiles(self.sandbox).get(relative)
            raw = file_ops.file_bytes(saved)
            actual = content_hash(raw)
            if saved.get("sha256", actual) != actual:
                raise ValueError("Stored file hash mismatch")
            return raw
        return await self.sandbox.read_file_bytes(f"{self.root}/{relative}")

    @property
    def parameters(self):
        parameters = result_schema(self.schema, self.data)
        if self.sandbox is None:
            return parameters
        parameters["properties"]["result_ref"] = {
            "type": "string",
            "description": "Relative path used with write_file for a JSON file containing the "
            "complete emit arguments, without result_ref. Submit result_ref alone instead of "
            "inline fields. Use for large results.",
        }
        parameters["required"] = []
        return parameters

    async def execute(self, tool_context, **kwargs):
        """Return bounded validation feedback; only valid output terminates the child."""
        try:
            if (
                set(kwargs) == {"raw"}
                and isinstance(kwargs["raw"], str)
                and "raw" not in self.schema.model_fields
            ):
                kwargs = json_repair.loads(kwargs["raw"], stream_stable=True)
            if isinstance(kwargs, dict) and "result_ref" in kwargs:
                if self.sandbox is None:
                    raise ValueError(
                        "result_ref is unavailable; submit inline emit arguments, without a result file"
                    )
                if len(kwargs) != 1:
                    raise ValueError("result_ref is an alternative to all inline result fields")
                relative = validate_relative_file_path(kwargs["result_ref"])
                raw = await self.sandbox.read_file_bytes(f"{self.root}/{relative}")
                kwargs = json_repair.loads(raw.decode("utf-8"), stream_stable=True)
            result = self.schema.model_validate(kwargs)
            if isinstance(result, RecordResponse):
                for draft in result.records:
                    if draft.ready_content_ref is not None:
                        if (
                            self.sandbox is None
                            or draft.ready_content is not None
                            or draft.ready_content_base64 is not None
                        ):
                            raise ValueError(
                                "Ready file reference requires scratch access without inline text"
                            )
                        relative = validate_relative_file_path(draft.ready_content_ref)
                        raw = await self.read_content(relative)
                        body = file_ops.encode_content(raw)
                        draft.ready_content = body["content"]
                        draft.ready_content_base64 = body["content_base64"]
                        draft.ready_content_ref = None
            if isinstance(result, FileResponse):
                for draft in result.files:
                    if draft.content_ref is not None:
                        if (
                            self.sandbox is None
                            or draft.content is not None
                            or draft.content_base64 is not None
                            or draft.patches
                        ):
                            raise ValueError(
                                "File reference requires scratch access without content/patches"
                            )
                        relative = validate_relative_file_path(draft.content_ref)
                        raw = await self.read_content(relative)
                        actual = content_hash(raw)
                        if draft.content_sha256 and draft.content_sha256 != actual:
                            raise ValueError("Scratch artifact hash mismatch")
                        body = file_ops.encode_content(raw)
                        draft.content, draft.content_base64 = (
                            body["content"],
                            body["content_base64"],
                        )
                        draft.content_sha256 = actual
                        draft.content_ref = None
            if self.validate:
                checked = self.validate(result)
                if inspect.isawaitable(checked):
                    await checked
            self.result = result
            return "Result accepted."
        except (ValueError, TypeError, OSError) as exc:
            self.last_error = str(exc)[:1600]
            self.can_retry = retry_allowed(self.failures, self.last_error)
            if self.metrics is not None:
                self.metrics["validation_failures"] += 1
                if self.can_retry:
                    self.metrics["repairs"] += 1
            feedback = "Error: " + self.last_error
            if (
                self.can_retry
                and self.sandbox is not None
                and isinstance(kwargs, dict)
                and "result_ref" not in kwargs
            ):
                # Keep complete rejected data private; edits still pass every emit validator.
                candidate = f"rejected-{uuid.uuid4().hex}.json"
                try:
                    content = (
                        kwargs["raw"]
                        if set(kwargs) == {"raw"} and isinstance(kwargs["raw"], str)
                        else json.dumps(kwargs, ensure_ascii=False)
                    )
                    await self.sandbox.write_file(f"{self.root}/{candidate}", content)
                    feedback += (
                        f"\nRepair {candidate} with edit_file, then emit only "
                        f'{{"result_ref":"{candidate}"}}.'
                    )
                except (OSError, ValueError):
                    pass
            return feedback


def agent_runner(loop, session_key, connection, limits):
    """Bind an isolated child adapter; queueing/cancellation remain owned by Pipeline.

    Each call has fresh history, a private scratch root and its stage's iteration
    limit. Large files are submitted by reference, then hashed and checked.
    Skill reads use the authenticated task client. Final review can browse task files,
    with writes confined to its staged outputs and no shell access.
    """

    async def run(system, data, schema, validate, model, *, stage="agent"):
        child_id = uuid.uuid4().hex
        final_review = stage == "final_review_agent"
        root = ROOT if final_review else f"{COMPILE_DRAFT_ROOT}/{child_id}"
        context = FileContext(model, root) if stage == "reduce_merge_agent" else None
        evidence = EvidenceReader(model.files, data)
        submit = EmitResult(
            schema,
            validate,
            None if final_review else model.files.sandbox,
            root,
            data,
            model.metrics,
        )
        assignment = data
        if context is not None or not model.fits(system, data, schema):
            assignment = await prepare_assignment(data, model.files.sandbox, root)
            evidence.delivered.clear()
        registry = ToolRegistry(config=loop.config)
        registry.register(submit)
        if final_review:
            tool = loop.tools.get("openviking_glob")
            if tool is not None:
                registry.register(tool)
        if model.resources:
            registry.register(model.resources)
        if evidence.allowed - evidence.delivered:
            registry.register(evidence)
        for name in (
            "read_file",
            "write_file",
            "edit_file",
            *(["list_dir"] if final_review else []),
        ):
            tool = loop.tools.get(name)
            if tool is not None:
                bound = CompileChildTool(tool, root, merge_only=name != "list_dir")
                if final_review and name in {"write_file", "edit_file"}:

                    async def write_artifact(tool_context, _execute=bound.execute, **arguments):
                        """Accept task-relative write/edit paths only inside the staged artifact directory."""
                        path = validate_relative_file_path(arguments["path"])
                        if not path.startswith(data["artifact_directory"] + "/"):
                            raise ValueError(
                                f"Write only inside {data['artifact_directory']}/; use the same path as read_file"
                            )
                        arguments["path"] = path
                        return await _execute(tool_context, **arguments)

                    bound.execute = write_artifact
                registry.register(
                    ScopedReadTool(bound, context.page_chars)
                    if context is not None and name == "read_file"
                    else bound
                )
        tool = None if final_review else loop.tools.get("exec")
        if (
            loop.config.sandbox.backend == "direct"
            and not loop.config.sandbox.backends.direct.allow_compile_exec
        ):
            tool = None
        if tool is not None:
            await model.files.sandbox.write_file(f"{root}/.keep", "")
            registry.register(CompileChildTool(tool, root))
            system += (
                "\nUse exec with pipelines, Python or shell commands when they improve efficiency."
            )
        child = copy(loop)
        child.provider = ChildProvider(model, schema, stage, submit=submit, context=context)
        child._preview_tool_result = context.preview_result if context else complete_tool_result
        instructions = (
            "\nUse emit to submit. Supplied Skill attachments are complete; "
            "they fulfill the Skill's reading requirements without a tool call. Do not reread them. "
            "read_skill_resource is available for additional references. "
            "Use assigned evidence, Skill attachments and your scratch files for this job. "
            "Prefer one inline emit for small finished results; scratch writing and rereading "
            "are optional, not mandatory verification steps. "
            "Write large/multiple files using write_file, then emit content_ref paths "
            "relative to your scratch root, without repeating their content. Runtime hashes "
            "the files and validates paths, revisions and source coverage before acceptance. "
            "For a large record collection, build a JSON result file incrementally with file "
            "tools, then emit only result_ref pointing to that file. Finished independent Map "
            "files may use ready_path plus ready_content_ref and concise routing payloads. "
            "read_skill_resource provides local paths for Skill scripts and their dependencies. "
            "Use edit_file to repair existing JSON."
        )
        if stage == "map_agent" or final_review:
            instructions = ""  # These stages define their own file access and submission rules.
        if context is not None:
            instructions += (
                "\nFile reads return text pages; follow next_offset_chars with offset_chars until null. "
                "Large tool results use content_file handles. Earlier tool exchanges remain in "
                "context-history.json and its listed files. Save accumulated outputs and remaining "
                "work in scratch files so they can be resumed without retaining all read text."
            )
        await child._run_agent_loop(
            messages=[
                {
                    "role": "system",
                    "content": system + instructions,
                },
                {"role": "user", "content": json.dumps(assignment, ensure_ascii=False)},
            ],
            session_key=session_key,
            publish_events=False,
            tool_registry=registry,
            stop_tool_names=["emit"],
            openviking_tool_names=set(registry.tool_names),
            openviking_connection=connection,
            allow_final_fallback=False,
            inject_write_experience=False,
            agent_id=child_id,
            context_compact_budget=None,
            max_iterations=(
                limits.final_review_iterations if final_review else limits.subagent_iterations
            ),
            should_stop=lambda: not submit.can_retry,
        )
        if submit.result is None:
            raise ValueError(
                "Required agent transform ended without validated output: "
                + (submit.last_error or "No emit result was submitted")
            )
        return submit.result

    return run
