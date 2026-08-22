"""A2A Task/Message/Artifact ↔ dehaze 会话/消息/产物 双向映射

A2A 侧对象见 a2a_protocol.py；dehaze 侧为：
- Message → dehaze 消息列表（[{role, content}]，供推理链路组装）
- Task → dehaze 临时会话的推理结果（final_response + artifacts）
- Artifact → dehaze 产物（SysAiArtifact），图像/指标以引用挂载
"""

from __future__ import annotations

from typing import Any

from app.infrastructure.llm.a2a_protocol import (
    Artifact,
    DataPart,
    FilePart,
    Message,
    Part,
    Task,
    TextPart,
    decode_bytes,
)

# dehaze 产物类型 → A2A FilePart 的 mime 映射（供引用挂载展示）
_ARTIFACT_MIME = {
    "image": "image/png",
    "metric": "application/json",
    "report": "text/markdown",
}


class A2ATaskMapper:
    """A2A 与 dehaze 模型双向映射（单例）"""

    @staticmethod
    def messages_to_dehaze(messages: list[Message], system_prompt: str | None = None) -> list[dict]:
        """将 A2A Message 列表转为 dehaze 消息列表。

        仅保留 user/agent 角色的文本内容；system_prompt 单独承载系统指令。
        """
        converted = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})
        for msg in messages:
            if msg.role not in ("user", "agent"):
                continue
            converted.append({"role": msg.role, "content": msg.to_text()})
        return converted

    @staticmethod
    def extract_files(messages: list[Message]) -> list[dict]:
        """从 A2A Message 中提取 FilePart 引用（url 或 bytes），供多模态上下文使用。"""
        files = []
        for msg in messages:
            for part in msg.parts:
                if getattr(part, "type", None) == "file":
                    files.append(
                        {
                            "name": part.file.get("name"),
                            "url": part.file.get("url"),
                            "mime_type": part.file.get("mime_type"),
                        }
                    )
        return files

    @staticmethod
    def task_to_message(task: Task) -> str:
        """从 Task 中提取最末 agent 消息文本作为任务输入上下文。"""
        for msg in reversed(task.history or []):
            if msg.role == "user":
                return msg.to_text()
        return ""

    # ── dehaze → A2A ────────────────────────────────────────────

    @staticmethod
    def build_task(
        task_id: str,
        status: str,
        final_response: str = "",
        artifacts: list[dict] | None = None,
        context_id: str | None = None,
        history: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """将 dehaze 推理结果映射为 A2A Task。

        artifacts 为 dehaze 产物列表（含 type/summary/ref_type/ref_id），
        图像/指标以引用形式挂载到 Artifact 的 FilePart/DataPart。
        """
        result_artifacts = [A2ATaskMapper.dehaze_artifact_to_artifact(a) for a in (artifacts or [])]
        # 最终文本回复作为一条 TextPart Artifact 挂载（A2A 标准产物形态）
        if final_response and not result_artifacts:
            result_artifacts.append(
                Artifact(
                    artifactId=f"{task_id}:output",
                    name="response",
                    parts=[TextPart(text=final_response)],
                )
            )
        return Task(
            id=task_id,
            context_id=context_id,
            status=status,  # type: ignore[arg-type]
            artifacts=result_artifacts,
            history=history or [],
            metadata=metadata or {},
        )

    @staticmethod
    def dehaze_artifact_to_artifact(item: dict) -> Artifact:
        """将 dehaze 产物记录（SysAiArtifact）映射为 A2A Artifact。

        产物元数据（ref_type/ref_id/summary）以 DataPart 挂载，
        图像类产物以 FilePart 引用（url 由调用方在 summary 中提供）。
        """
        artifact_type = item.get("type", "data")
        parts: list[Part] = []
        summary = item.get("summary") or {}
        url = summary.get("url")
        if url:
            parts.append(
                FilePart(
                    file={
                        "name": summary.get("name", artifact_type),
                        "url": url,
                        "mime_type": _ARTIFACT_MIME.get(artifact_type, "application/octet-stream"),
                    }
                )
            )
        # 引用元数据（ref_type/ref_id）作为 DataPart，避免 URL 之外的敏感数据入上下文
        ref = {
            "ref_type": item.get("ref_type"),
            "ref_id": item.get("ref_id"),
            "artifact_type": artifact_type,
            **{k: v for k, v in summary.items() if k != "url"},
        }
        parts.append(DataPart(data=ref))
        return Artifact(
            artifact_id=str(item.get("id") or artifact_type),
            name=item.get("name") or artifact_type,
            parts=parts,
        )

    @staticmethod
    def artifact_to_context(artifact: Artifact) -> dict:
        """将 A2A Artifact 反解为 dehaze 上下文片段（供后续消息携带引用）。"""
        texts = []
        files = []
        for part in artifact.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
            elif isinstance(part, FilePart):
                if part.file.get("url"):
                    files.append({"url": part.file["url"]})
                elif part.file.get("bytes"):
                    files.append(
                        {"bytes": decode_bytes(part.file["bytes"]), "name": part.file.get("name")}
                    )
            elif isinstance(part, DataPart):
                texts.append(str(part.data))
        return {"artifact_id": artifact.artifact_id, "text": "\n".join(texts), "files": files}


a2a_task_mapper = A2ATaskMapper()
