import json
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai import compatible_api_service as m
from tests.stubs import FakeInternalResponse


def _internal_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text_delta(text: str, index: int = 0) -> str:
    return _internal_sse(
        "content_block.delta", {"index": index, "delta": {"type": "text_delta", "text": text}}
    )


def _message_end(usage: dict | None = None, stop_reason: str = "stop") -> str:
    return _internal_sse(
        "message.end",
        {
            "stopReason": stop_reason,
            "usage": usage or {"inputTokens": 10, "outputTokens": 5, "credits": 2},
        },
    )


def _thought_event() -> str:
    return _internal_sse("thought", {"thought": "正在思考..."})


async def _collect(agen):
    return [c async for c in agen]


def _sse_payload(line: str) -> dict:
    for ln in line.splitlines():
        if ln.startswith("data:"):
            return json.loads(ln[5:].strip())
    raise AssertionError(f"SSE 行缺少 data: {line!r}")


def _api_key(**kw):
    defaults = dict(
        id=1, model_whitelist=None, daily_quota=None, monthly_quota=None, rpm_limit=None
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _make_request(body):
    state = SimpleNamespace(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"}, request_id="req-1")

    async def _json():
        return body

    return SimpleNamespace(state=state, client=SimpleNamespace(host="1.2.3.4"), json=_json)


async def _pass(*a, **k):
    return None


async def test_openai_stream_text_delta_and_done():
    resp = FakeInternalResponse(_text_delta("你好"), _text_delta("世界"), _message_end())
    lines = await _collect(m._openai_stream(resp, "gpt-4", "chatcmpl-1", 1700000000, None))
    first = json.loads(lines[0].removeprefix("data: ").strip())
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["delta"]["content"] == ""
    text = json.loads(lines[1].removeprefix("data: ").strip())
    assert text["object"] == "chat.completion.chunk"
    assert text["choices"][0]["delta"]["content"] == "你好"
    text2 = json.loads(lines[2].removeprefix("data: ").strip())
    assert text2["choices"][0]["delta"]["content"] == "世界"
    end = json.loads(lines[3].removeprefix("data: ").strip())
    assert end["choices"][0]["finish_reason"] == "stop"
    assert end["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert lines[-1] == "data: [DONE]\n\n"


async def test_openai_stream_error_yields_done():
    resp = FakeInternalResponse(_internal_sse("error", {"message": "推理失败"}))
    lines = await _collect(m._openai_stream(resp, "gpt-4", "chatcmpl-1", 1700000000, None))
    error = json.loads(lines[1].removeprefix("data: ").strip())
    assert error["choices"][0]["finish_reason"] == "stop"
    assert lines[-1] == "data: [DONE]\n\n"


async def test_claude_stream_event_sequence_and_done():
    resp = FakeInternalResponse(
        _text_delta("你好"), _message_end(usage={"inputTokens": 10, "outputTokens": 5})
    )
    lines = await _collect(m._claude_stream(resp, "claude-3-5", "msg_1", None))
    start = _sse_payload(lines[0])
    assert lines[0].startswith("event: message_start")
    assert start["type"] == "message_start"
    assert start["message"]["id"] == "msg_1"
    assert start["message"]["model"] == "claude-3-5"
    assert lines[1].startswith("event: content_block_delta")
    delta = _sse_payload(lines[1])
    assert delta["delta"] == {"type": "text_delta", "text": "你好"}
    assert lines[2].startswith("event: message_delta")
    mdelta = _sse_payload(lines[2])
    assert mdelta["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert lines[3].startswith("event: message_stop")
    assert _sse_payload(lines[3])["type"] == "message_stop"
    assert len(lines) == 4


async def test_openai_non_stream_structure():
    resp = FakeInternalResponse(_text_delta("你好"), _text_delta("世界"), _message_end())
    out = await m._openai_non_stream(resp, "gpt-4", "chatcmpl-1", 1700000000, None)
    assert out["id"] == "chatcmpl-1"
    assert out["object"] == "chat.completion"
    assert out["model"] == "gpt-4"
    assert out["created"] == 1700000000
    assert out["choices"][0]["message"]["content"] == "你好世界"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def test_stream_does_not_expose_thought():
    openai_lines = await _collect(
        m._openai_stream(
            FakeInternalResponse(_thought_event(), _text_delta("正常回复"), _message_end()),
            "gpt-4",
            "chatcmpl-1",
            1700000000,
            None,
        )
    )
    for line in openai_lines:
        assert "thought" not in line
    claude_lines = await _collect(
        m._claude_stream(
            FakeInternalResponse(_thought_event(), _text_delta("正常回复"), _message_end()),
            "claude-3-5",
            "msg_1",
            None,
        )
    )
    for line in claude_lines:
        assert "thought" not in line


def test_extract_system_prompt_from_messages():
    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "hi"},
    ]
    assert m._extract_system_prompt(messages) == "你是助手"
    assert m._extract_system_prompt([{"role": "user", "content": "hi"}]) is None
    assert m._extract_system_prompt([{"role": "system", "content": ["非文本"]}]) is None


async def test_resolve_conversation_creates_new(monkeypatch):
    created = {}

    async def _create(db, user_id, form):
        created["systemPrompt"] = form.systemPrompt
        created["model"] = form.model
        created["title"] = form.title
        return SimpleNamespace(id=100)

    monkeypatch.setattr(m.AiConversationService, "create_conversation", _create)
    first_user_msg = "请分析华东区与华南区季度销售数据，并结合库存周转率给出下季度备货建议。" * 2
    conv_id = await m.CompatibleApiService._resolve_conversation(
        None, 7, None, "claude-3-5", "全局系统提示", first_user_msg
    )
    assert conv_id == 100
    assert created["systemPrompt"] == "全局系统提示"
    assert created["model"] == "claude-3-5"
    assert created["title"] == first_user_msg[:50]


async def test_resolve_conversation_reuses_existing(monkeypatch):
    fetched = []
    created = []

    async def _get(db, conv_id, user_id):
        fetched.append((conv_id, user_id))
        return SimpleNamespace(id=42)

    monkeypatch.setattr(m.ai_conversation_repository, "get_by_id_and_user", _get)
    monkeypatch.setattr(
        m.AiConversationService, "create_conversation", lambda *a, **k: created.append(a)
    )
    conv_id = await m.CompatibleApiService._resolve_conversation(None, 7, "42", "gpt-4", None, "hi")
    assert conv_id == 42
    assert fetched == [(42, 7)]
    assert created == []


async def test_enforce_model_whitelist(monkeypatch):
    api_key = _api_key(model_whitelist=["gpt-4"])
    await m.CompatibleApiService._enforce_model_whitelist(None, 1, api_key, "claude-3", None)

    async def _get_by_ids(db, user_id, conv_ids):
        return [SimpleNamespace(model="claude-3")]

    monkeypatch.setattr(m.ai_conversation_repository, "get_by_ids", _get_by_ids)
    with pytest.raises(m.GovernanceError):
        await m.CompatibleApiService._enforce_model_whitelist(None, 1, api_key, None, 42)


def _db(monkeypatch, api_key):
    async def _get_by_id(db, key_id):
        return api_key if api_key is not None and api_key.id == key_id else None

    monkeypatch.setattr(m, "api_key_repository", SimpleNamespace(get_by_id=_get_by_id))
    return None


async def test_model_not_available_maps_403_openai(monkeypatch):
    monkeypatch.setattr(m.CompatibleGovernanceService, "precheck", _pass)

    async def _handler(body, audit, api_key):
        raise BusinessException(ResultCode.AI_MODEL_NOT_AVAILABLE, "模型不可用")

    req = _make_request({"messages": [{"role": "user", "content": "hi"}], "model": "bad-model"})
    resp = await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key()),
        SimpleNamespace(id=7),
        protocol="openai",
        endpoint="chat/completions",
        handler=_handler,
    )
    assert resp.status_code == 403
    assert '"type":"permission_error"' in resp.body.decode()


async def test_list_models_openai_whitelist_chain(monkeypatch):
    model_a = SimpleNamespace(model_id="gpt-4", create_time=None, provider_id=1)
    api_key = SimpleNamespace(id=1, model_whitelist=["gpt-4"])
    vip_user_id = {"uid": None}

    async def _vip_list(db, redis, user_id):
        vip_user_id["uid"] = user_id
        return [model_a, SimpleNamespace(model_id="claude-3", create_time=None, provider_id=2)]

    monkeypatch.setattr(m.AiModelService, "list_enabled_models", _vip_list)
    result = await m.CompatibleApiService.list_models_openai(object(), object(), 9, api_key)
    assert [d["id"] for d in result["data"]] == ["gpt-4"]
    assert vip_user_id["uid"] == 9
