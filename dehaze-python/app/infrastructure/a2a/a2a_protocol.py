"""A2A 协议核心对象模型

对齐后端实现文档 §5.4.2：
- Part 多态单元：TextPart / FilePart / DataPart
- Message：role（user/agent）+ parts
- Artifact：任务产出物，可含多个 Part
- Task：工作单元，完整生命周期状态机
- JSON-RPC 2.0 请求/响应信封

任务状态机：submitted → working → (input_required / auth_required) → completed /
failed / canceled / rejected
"""

from __future__ import annotations

import base64
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskStatus = Literal[
    "submitted",
    "working",
    "input_required",
    "auth_required",
    "completed",
    "failed",
    "canceled",
    "rejected",
]


# ── Part（多态单元）───────────────────────────────────────────────


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    file: dict[str, Any] = Field(
        default_factory=dict, description="文件对象，含 bytes（base64）或 url 二者其一"
    )


class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict, description="结构化数据")


Part = TextPart | FilePart | DataPart


def parse_part(raw: dict) -> Part:
    """按 type 将原始 dict 解析为对应 Part。

    未知/缺省 type 兜底为 text：剥离 type 键再按 TextPart 解析，
    避免 {"type": "weird"} 这类未知值触发 Literal 校验失败。
    """
    part_type = raw.get("type", "text")
    if part_type == "file":
        return FilePart.model_validate(raw)
    if part_type == "data":
        return DataPart.model_validate(raw)
    fallback = {k: v for k, v in raw.items() if k != "type"}
    return TextPart.model_validate(fallback)


def part_to_dict(part: Part) -> dict:
    """序列化 Part 为 dict（丢弃空默认字段）。"""
    return part.model_dump(exclude_none=True)


def part_to_text(part: Part) -> str:
    """提取 Part 的文本内容（文本直接返回，文件/data 返回摘要）。"""
    if isinstance(part, TextPart):
        return part.text
    if isinstance(part, FilePart):
        return part.file.get("url") or f"<file bytes:{len(part.file.get('bytes', ''))}>"
    return str(part.data)


# ── Message / Artifact ───────────────────────────────────────────


class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)

    def to_text(self) -> str:
        return "\n".join(part_to_text(p) for p in self.parts)


class Artifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    artifact_id: str = Field(..., alias="artifactId")
    name: str | None = Field(default=None, alias="name")
    parts: list[Part] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata")

    def to_text(self) -> str:
        return "\n".join(part_to_text(p) for p in self.parts)


# ── Task ─────────────────────────────────────────────────────────


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    context_id: str | None = Field(default=None, alias="contextId")
    status: TaskStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdateEvent(BaseModel):
    """status-update SSE 事件载荷"""

    id: str
    status: TaskStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskArtifactUpdateEvent(BaseModel):
    """artifact-update SSE 事件载荷"""

    id: str
    artifact: Artifact


# ── JSON-RPC 2.0 信封 ────────────────────────────────────────────


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = Field(default=None)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = Field(default=None)
    result: Any | None = Field(default=None)
    error: JsonRpcError | None = Field(default=None)


def encode_bytes(data: bytes) -> str:
    """二进制内容 base64 编码（FilePart.file.bytes）。"""
    return base64.b64encode(data).decode()


def decode_bytes(b64: str) -> bytes:
    """FilePart.file.bytes base64 解码。"""
    return base64.b64decode(b64)
