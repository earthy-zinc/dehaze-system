import pytest
from pydantic import ValidationError

from app.infrastructure.llm.a2a_protocol import (
    Artifact,
    DataPart,
    FilePart,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    Task,
    TextPart,
    decode_bytes,
    encode_bytes,
    parse_part,
    part_to_text,
)
from app.service.ai.a2a_task_mapper import a2a_task_mapper


class TestPart:
    def test_parse_text_part(self):
        part = parse_part({"type": "text", "text": "你好"})
        assert isinstance(part, TextPart)
        assert part.text == "你好"

    def test_parse_file_part(self):
        part = parse_part({"type": "file", "file": {"url": "https://x/y.png"}})
        assert isinstance(part, FilePart)
        assert part.file["url"] == "https://x/y.png"

    def test_parse_data_part(self):
        part = parse_part({"type": "data", "data": {"k": 1}})
        assert isinstance(part, DataPart)
        assert part.data == {"k": 1}

    def test_default_text_part(self):
        part = parse_part({})
        assert isinstance(part, TextPart)
        assert part.text == ""

    def test_part_to_text_variants(self):
        assert part_to_text(TextPart(text="hi")) == "hi"
        assert part_to_text(FilePart(file={"url": "u"})) == "u"
        assert part_to_text(DataPart(data={"a": 1})) == "{'a': 1}"

    def test_unknown_part_type_falls_back_to_text(self):
        part = parse_part({"type": "weird", "text": "x"})
        assert isinstance(part, TextPart)
        assert part.text == "x"


class TestBase64:
    def test_roundtrip(self):
        raw = b"\x00\x01\x89\xff"
        assert decode_bytes(encode_bytes(raw)) == raw


class TestJsonRpc:
    def test_request_envelope(self):
        req = JsonRpcRequest(id=1, method="tasks/send", params={"id": "t1"})
        d = req.model_dump(by_alias=True)
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["method"] == "tasks/send"

    def test_error_response(self):
        resp = JsonRpcResponse(
            id=2,
            result=None,
            error=JsonRpcError(code=-32601, message="Method not found"),
        )
        d = resp.model_dump(exclude_none=True)
        assert d["error"]["code"] == -32601
        assert "result" not in d

    def test_success_response(self):
        resp = JsonRpcResponse(id=3, result={"status": "completed"}, error=None)
        d = resp.model_dump(exclude_none=True)
        assert d["result"] == {"status": "completed"}
        assert "error" not in d

    def test_invalid_jsonrpc_version_rejected(self):
        with pytest.raises(ValidationError):
            JsonRpcRequest(jsonrpc="1.0", method="x")


class TestTaskModel:
    def test_task_serialization_with_alias(self):
        task = Task(id="t1", contextId="c1", status="working")
        d = task.model_dump(by_alias=True)
        assert d["contextId"] == "c1"
        assert d["status"] == "working"

    def test_task_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            Task(id="t1", status="not-a-status")

    def test_task_message_to_text(self):
        task = Task(
            id="t1",
            status="completed",
            history=[Message(role="user", parts=[TextPart(text="hello")])],
        )
        assert task.history[0].to_text() == "hello"


class TestTaskMapper:
    def test_messages_to_dehaze(self):
        msgs = [
            Message(role="user", parts=[TextPart(text="你好")]),
            Message(role="agent", parts=[TextPart(text="你好！")]),
        ]
        out = a2a_task_mapper.messages_to_dehaze(msgs, system_prompt="sys")
        assert out[0] == {"role": "system", "content": "sys"}
        assert out[1] == {"role": "user", "content": "你好"}
        assert out[2] == {"role": "agent", "content": "你好！"}

    def test_extract_files(self):
        msgs = [
            Message(
                role="user",
                parts=[
                    TextPart(text="hi"),
                    FilePart(file={"url": "https://x/a.png", "mime_type": "image/png"}),
                ],
            )
        ]
        files = a2a_task_mapper.extract_files(msgs)
        assert files == [{"name": None, "url": "https://x/a.png", "mime_type": "image/png"}]

    def test_task_to_message_returns_latest_user(self):
        task = Task(
            id="t1",
            status="completed",
            history=[
                Message(role="user", parts=[TextPart(text="old")]),
                Message(role="agent", parts=[TextPart(text="mid")]),
                Message(role="user", parts=[TextPart(text="new")]),
            ],
        )
        assert a2a_task_mapper.task_to_message(task) == "new"

    def test_build_task_with_final_response(self):
        task = a2a_task_mapper.build_task("t1", "completed", final_response="done")
        assert task.id == "t1"
        assert task.status == "completed"
        assert any(
            isinstance(a.parts[0], TextPart) and a.parts[0].text == "done" for a in task.artifacts
        )

    def test_dehaze_artifact_to_artifact_with_url(self):
        art = a2a_task_mapper.dehaze_artifact_to_artifact(
            {
                "id": 1,
                "type": "image",
                "ref_type": "SYS",
                "ref_id": "r1",
                "summary": {"url": "https://cdn/x.png", "name": "out.png"},
            }
        )
        assert art.artifact_id == "1"
        types = [type(p).__name__ for p in art.parts]
        assert "FilePart" in types and "DataPart" in types

    def test_artifact_to_context_with_bytes(self):
        art = Artifact(
            artifactId="a1",
            parts=[
                FilePart(file={"bytes": encode_bytes(b"\x01\x02"), "name": "b.bin"}),
            ],
        )
        ctx = a2a_task_mapper.artifact_to_context(art)
        assert ctx["files"][0]["bytes"] == b"\x01\x02"


class TestTaskStateMachine:
    def test_task_ctx_index_ttl_constants(self):
        from app.service.ai.a2a_server import _CTX_INDEX_PREFIX, _TASK_TTL

        assert _TASK_TTL == 86400
        assert _CTX_INDEX_PREFIX == "a2a:task:ctx:"

    def test_serialize_task_shape(self):
        import json

        from app.service.ai.a2a_server import A2AServer

        task = Task(id="t1", contextId="c1", status="submitted", history=[])
        payload = A2AServer._serialize_task(task, "submitted")
        assert payload["id"] == "t1"
        assert payload["contextId"] == "c1"
        assert payload["status"] == "submitted"
        json.dumps(payload)
