# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Core domain data classes for memory system.
"""

import json
from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, WithJsonSchema, model_validator

from openviking.session.memory.merge_op.base import (
    FieldType,
    MergeOp,
)

T = TypeVar("T")


# ============================================================================
# Link Type and Link Models
# ============================================================================


LINK_TYPE_DEFAULT = "related_to"


class LinkType(str, Enum):
    """Legacy predefined link labels kept for compatibility in tests/call sites."""

    RELATED_TO = LINK_TYPE_DEFAULT
    EVIDENCE_FOR = "evidence_for"
    CONTEXT_FOR = "context_for"
    BELONGS_TO = "belongs_to"
    CAUSED_BY = "caused_by"
    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    EVOLVED_FROM = "evolved_from"


_CANONICAL_LINK_TYPES = {item.value for item in LinkType}
_LINK_TYPE_ALIASES = {
    "evidence": LinkType.EVIDENCE_FOR.value,
    "answer_evidence": LinkType.EVIDENCE_FOR.value,
    "direct_evidence": LinkType.EVIDENCE_FOR.value,
    "context": LinkType.CONTEXT_FOR.value,
    "background": LinkType.CONTEXT_FOR.value,
    "context_with": LinkType.CONTEXT_FOR.value,
    "contextual": LinkType.CONTEXT_FOR.value,
    "belong_to": LinkType.BELONGS_TO.value,
    "part_of": LinkType.BELONGS_TO.value,
    "member_of": LinkType.BELONGS_TO.value,
    "relates_to": LinkType.RELATED_TO.value,
    "topic": LinkType.RELATED_TO.value,
    "topic_related": LinkType.RELATED_TO.value,
    "associated_with": LinkType.RELATED_TO.value,
    "derived": LinkType.DERIVED_FROM.value,
    "extracted_from": LinkType.DERIVED_FROM.value,
    "caused": LinkType.CAUSED_BY.value,
    "causes": LinkType.CAUSED_BY.value,
    "contradict": LinkType.CONTRADICTS.value,
    "conflicts_with": LinkType.CONTRADICTS.value,
    "evolved": LinkType.EVOLVED_FROM.value,
    "changed_from": LinkType.EVOLVED_FROM.value,
}
_EVIDENCE_ROLE_ALIASES = {
    "list": "list_member",
    "count": "count_member",
    "intersection": "derived_intersection",
    "shared": "derived_intersection",
    "background": "context",
    "profile": "navigation",
}


def _snake_case(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_link_payload(data: Any) -> Any:
    """Normalize LLM-produced link fields while safely degrading unknown labels."""
    if not isinstance(data, dict):
        return data

    raw_weight = data.get("weight")
    if raw_weight is not None:
        try:
            data["weight"] = min(1.0, max(0.0, float(raw_weight)))
        except (TypeError, ValueError):
            data["weight"] = 0.5

    raw_link_type = _snake_case(data.get("link_type") or LINK_TYPE_DEFAULT)
    link_type = _LINK_TYPE_ALIASES.get(raw_link_type, raw_link_type)
    if link_type not in _CANONICAL_LINK_TYPES:
        link_type = LINK_TYPE_DEFAULT
        data["weight"] = min(float(data.get("weight", 0.5)), 0.45)
        data["evidence_role"] = "context"
    data["link_type"] = link_type

    raw_relation_slot = data.get("relation_slot")
    if raw_relation_slot is not None:
        data["relation_slot"] = _snake_case(raw_relation_slot)

    raw_evidence_role = data.get("evidence_role")
    if raw_evidence_role is not None:
        role = _snake_case(raw_evidence_role)
        data["evidence_role"] = _EVIDENCE_ROLE_ALIASES.get(role, role)

    raw_answer_value = data.get("answer_value")
    if raw_answer_value is not None:
        if isinstance(raw_answer_value, str):
            data["answer_value"] = [raw_answer_value]
        elif isinstance(raw_answer_value, list):
            data["answer_value"] = [str(item) for item in raw_answer_value if str(item).strip()]
        else:
            data["answer_value"] = [str(raw_answer_value)]

    return data


class WikiLink(BaseModel):
    """Link output by LLM during extraction, using temporary page_ids.

    f and t use WithJsonSchema to appear as required int in the JSON schema
    sent to the LLM, but parse as Optional[int] to tolerate null values.
    Invalid links (null f/t) are filtered in _resolve_links.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_link_fields(cls, data: Any) -> Any:
        return _normalize_link_payload(data)

    f: Annotated[Optional[int], WithJsonSchema({"type": "integer"})] = Field(
        ..., description="From page_id. Use the page_id from the item's 'page_id' field."
    )
    t: Annotated[Optional[int], WithJsonSchema({"type": "integer"})] = Field(
        ..., description="To page_id. Use the page_id from the item's 'page_id' field."
    )
    link_type: str = Field(
        LINK_TYPE_DEFAULT,
        description=(
            "Retrieval safety label, not the fact predicate. Use evidence_for only when the "
            "target states an answerable fact; use context_for for background, belongs_to for "
            "profile/navigation membership, and related_to only as a weak topic fallback."
        ),
    )
    weight: float = Field(
        0.5,
        description=(
            "Evidence strength from 0 to 1. Use high weight only for evidence_for links with "
            "subject, relation_slot, and answer_value or source_span. Use <=0.45 for broad "
            "topic, profile, navigation, or related_to links."
        ),
    )
    match_text: Annotated[
        Optional[str],
        WithJsonSchema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
    ] = Field(
        ...,
        description=(
            "A single WORD from the original conversation to be linkified. "
            "This field must always be included in link output. "
            "Use a single exact word from the original conversation whenever possible; "
            "only use null when no valid single-word anchor exists. "
            "Rules: (1) must be a single word only (NOT a phrase or multi-word text); "
            "(2) must exist verbatim in the original conversation messages; "
            "(3) pick the most specific/identifying word"
        ),
    )
    description: str = Field("", description="Brief explanation of the relationship")
    subject: Optional[str] = Field(
        None,
        description="Person or entity the fact is about; use the speaker/owner, not every mentioned name.",
    )
    relation_slot: Optional[str] = Field(
        None,
        description=(
            "Reusable snake_case fact slot, not an event-specific phrase. Prefer slots such as "
            "visited_place, read_book, read_author, watched_movie, likes_sport, does_training, "
            "won_event, writes_genre, destresses_by, researched_topic, bought_item, received_gift, "
            "has_child, or career_high_performance."
        ),
    )
    answer_value: List[str] = Field(
        default_factory=list,
        description=(
            "Smallest answer text to return for this fact: place, person, title, author, activity, "
            "item, topic, event, date, or number. Leave empty for context links."
        ),
    )
    evidence_role: str = Field(
        "context",
        description=(
            "Optional role hint. Use direct for answer evidence, context for background, and "
            "navigation for profile/path links. Use list_member/count_member/intersection only "
            "when the memory explicitly lists, counts, or compares facts."
        ),
    )
    source_span: Optional[str] = Field(
        None,
        description="Shortest phrase from the memory text or image metadata proving answer_value.",
    )


class StoredLink(BaseModel):
    """Persisted link in MEMORY_FIELDS, with URIs instead of page_ids."""

    @model_validator(mode="before")
    @classmethod
    def normalize_link_fields(cls, data: Any) -> Any:
        return _normalize_link_payload(data)

    from_uri: str
    to_uri: str
    link_type: str = LINK_TYPE_DEFAULT
    weight: float = 0.5
    match_text: Optional[str] = None  # single word, must exist verbatim in conversation
    description: str = ""
    created_at: str = ""
    subject: Optional[str] = None
    relation_slot: Optional[str] = None
    answer_value: List[str] = Field(default_factory=list)
    evidence_role: str = "context"
    source_span: Optional[str] = None


# ============================================================================
# Memory Field and Schema Definitions
# ============================================================================


class MemoryField(BaseModel):
    """Memory field definition."""

    name: str = Field(..., description="Field name")
    field_type: FieldType = Field(..., description="Field type")
    description: str = Field("", description="Field description")
    merge_op: MergeOp = Field(MergeOp.PATCH, description="Merge strategy")
    init_value: Optional[str] = Field(None, description="Initial value for this field")


class MemoryTypeSchema(BaseModel):
    """Memory type schema definition."""

    memory_type: str = Field(..., description="Memory type name")
    description: str = Field("", description="Type description")
    fields: List[MemoryField] = Field(default_factory=list, description="Field definitions")
    filename_template: str = Field("", description="Filename template")
    content_template: Optional[str] = Field(
        None, description="Content template (for template mode)"
    )
    embedding_template: Optional[str] = Field(None, description="Embedding text template")
    directory: str = Field("", description="Directory path")
    enabled: bool = Field(True, description="Whether this memory type is enabled")
    operation_mode: str = Field(
        "upsert", description="Operation mode: 'upsert' (default), 'add_only', or 'update_only'"
    )
    agent_only: bool = Field(
        False, description="If true, only used by agent memory extraction, not user memory"
    )
    overview_template: Optional[str] = Field(
        None, description="Overview template for auto-generating .overview.md files"
    )

    def filename_has_variables(self):
        return "{{" in self.filename_template and "}}" in self.filename_template


class MemoryData(BaseModel):
    """Dynamic memory data."""

    memory_type: str = Field(..., description="Memory type name")
    uri: Optional[str] = Field(None, description="Memory URI (for updates)")
    fields: dict[str, Any] = Field(default_factory=dict, description="Dynamic field data")
    abstract: Optional[str] = Field(None, description="L0 abstract")
    overview: Optional[str] = Field(None, description="L1 overview")
    content: Optional[str] = Field(None, description="L2 content")
    name: Optional[str] = Field(None, description="Memory name")
    tags: List[str] = Field(default_factory=list, description="Tags")
    created_at: Optional[datetime] = Field(None, description="Created time")
    updated_at: Optional[datetime] = Field(None, description="Updated time")

    def get_field(self, field_name: str) -> Any:
        """Get field value."""
        return self.fields.get(field_name)

    def set_field(self, field_name: str, value: Any) -> None:
        """Set field value."""
        self.fields[field_name] = value


class MemoryFile(BaseModel):
    """Typed representation of a memory file's parsed content."""

    uri: Optional[str] = None
    content: str = ""
    links: List[Dict[str, Any]] = []
    backlinks: List[Dict[str, Any]] = []
    memory_type: Optional[str] = None
    extra_fields: Dict[str, Any] = {}

    def plain_content(self) -> str:
        from openviking.session.memory.utils.link_renderer import LinkRenderer

        return LinkRenderer.strip_links(self.content)

    @classmethod
    def from_parsed(cls, uri: Optional[str] = None, parsed: Dict[str, Any] = None) -> "MemoryFile":
        """Build from parse_memory_file_with_fields result."""
        if parsed is None:
            parsed = {}
        content = parsed.pop("content", "")
        links = parsed.pop("links", []) or []
        backlinks = parsed.pop("backlinks", []) or []
        memory_type = parsed.pop("memory_type", None)
        # Remaining keys are dynamic schema fields + system fields
        return cls(
            uri=uri,
            content=content,
            links=links,
            backlinks=backlinks,
            memory_type=memory_type,
            extra_fields=parsed,
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for serialize_with_metadata."""
        metadata = dict(self.extra_fields)
        metadata["content"] = self.content
        if self.links:
            metadata["links"] = self.links
        if self.backlinks:
            metadata["backlinks"] = self.backlinks
        if self.memory_type:
            metadata["memory_type"] = self.memory_type
        return metadata


class ResolvedOperation(BaseModel):
    old_memory_file_content: Optional[MemoryFile] = None
    memory_fields: Dict
    memory_type: str  # The memory type (e.g., 'tools', 'skills', 'events')
    uris: List[str]
    page_id: Optional[int] = None  # Temporary page_id for link resolution (not persisted)

    def is_edit(self):
        return self.old_memory_file_content is not None


class ResolvedOperations(BaseModel):
    upsert_operations: List[ResolvedOperation]
    delete_file_contents: List[MemoryFile]
    errors: List[str]
    resolved_links: List[StoredLink] = Field(default_factory=list)

    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ============================================================================
# Fault Tolerant Base Model (参考 vikingdb BaseModelCompat)
# ============================================================================


class FaultTolerantBaseModel(BaseModel):
    """
    支持验证前自动容错的 BaseModel，类似 vikingdb 的 BaseModelCompat。

    在 model_validator(mode='before') 中对所有字段做类型容错处理，
    使得模型可以接受 LLM 输出的不标准格式数据。
    """

    @model_validator(mode="before")
    @classmethod
    def values_fault_tolerance(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """在验证前对所有字段做容错处理"""
        if isinstance(data, dict):
            field_types = get_type_hints(cls)
            for field_name, value in data.items():
                if field_name in field_types:
                    data[field_name] = cls.value_fault_tolerance(field_types[field_name], value)
            return data
        return {}

    @classmethod
    def get_origin_type(cls, annotation) -> type:
        """从 Optional 或 Union 类型中提取基础类型"""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            if len(args) == 2 and args[1] is type(None):
                return cls.get_origin_type(args[0])
        elif origin is list:
            return list
        return annotation

    @classmethod
    def get_arg_type(cls, annotation) -> type:
        """从 List annotation 中提取元素类型"""
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            if len(args) == 2 and args[1] is type(None):
                return cls.get_arg_type(args[0])
        elif origin is list:
            args = get_args(annotation)
            if args:
                return args[0]
        return None

    @classmethod
    def any_to_str(cls, value) -> str:
        """将任意值转换为字符串"""
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(map(str, value))
        elif isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (int, bool, float)):
            return f"{value}"
        return str(value)

    @classmethod
    def value_fault_tolerance(cls, field_type, value):
        """
        字段级别的容错处理：
        - 'None' -> None (非 str 类型)
        - list/dict/number -> str (目标是 str)
        - str -> int/float (目标是数字)
        - str/dict -> list (目标是 list)
        - list 元素类型容错
        - 非法 LinkType -> related_to
        """
        origin_type = cls.get_origin_type(field_type)

        # json_repair 会把 None 转换成 'None'
        if value == "None" and origin_type is not str:
            return None

        if isinstance(origin_type, type) and issubclass(origin_type, Enum):
            if origin_type is LinkType:
                try:
                    return origin_type(value)
                except (ValueError, TypeError):
                    return LinkType.RELATED_TO

        if origin_type is str:
            return cls.any_to_str(value)
        elif origin_type is int:
            if isinstance(value, str):
                if value is None or value == "None":
                    return 0
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
        elif origin_type is float:
            if isinstance(value, str):
                if value is None or value == "None":
                    return 0.0
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
        elif origin_type is list:
            if isinstance(value, str):
                return [value]
            elif isinstance(value, dict):
                return [value]
            elif isinstance(value, list):
                arg_type = cls.get_arg_type(field_type)
                if arg_type is str:
                    return [cls.any_to_str(v) for v in value]
        return value


# ============================================================================
# Memory Operations
# ============================================================================


class MemoryOperationsProtocol(Protocol):
    """Protocol for memory operations (for type checking)."""

    reasoning: str
    write_uris: List[Any]
    edit_uris: List[Any]
    delete_uris: List[str]

    def is_empty(self) -> bool: ...


class StructuredMemoryOperations(FaultTolerantBaseModel):
    """
    Fallback memory operations model with fault tolerance.

    Use SchemaModelGenerator.create_structured_operations_model() to get
    the actual type-safe implementation with per-memory_type fields.
    """

    reasoning: str = Field(
        "",
        description="reasoning",
    )
    write_uris: List[Any] = Field(
        default_factory=list,
        description="Write operations with flat data format",
    )
    edit_uris: List[Any] = Field(
        default_factory=list,
        description="Edit operations with flat data format",
    )
    delete_uris: List[str] = Field(
        default_factory=list,
        description="Delete operations as URI strings",
    )

    def is_empty(self) -> bool:
        """Check if there are any operations."""
        return len(self.write_uris) == 0 and len(self.edit_uris) == 0 and len(self.delete_uris) == 0

    def to_legacy_operations(self) -> Dict[str, Any]:
        """Convert to legacy format (identity for fallback)."""
        return {
            "write_uris": self.write_uris,
            "edit_uris": self.edit_uris,
            "delete_uris": self.delete_uris,
        }

    model_config = {"extra": "ignore"}


# Backward compatibility alias
MemoryOperations = StructuredMemoryOperations
