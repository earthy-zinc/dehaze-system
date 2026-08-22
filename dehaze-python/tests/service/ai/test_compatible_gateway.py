from types import SimpleNamespace

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai import compatible_api_service as m
from app.service.ai.compatible_governance import GovernanceError
from tests.stubs import FakeInternalResponse


def _api_key(**kw):
    defaults = dict(
        id=1,
        model_whitelist=None,
        daily_quota=None,
        monthly_quota=None,
        rpm_limit=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _db(monkeypatch, api_key=None):
    async def _get_by_id(db, key_id):
        if api_key is not None and api_key.id == key_id:
            return api_key
        return None

    monkeypatch.setattr(m, "api_key_repository", SimpleNamespace(get_by_id=_get_by_id))
    return None


class _User:
    id = 7


def _request(*, api_key_info=None, body=None):
    state = SimpleNamespace(api_key_info=api_key_info, request_id="req-1")
    req_body = body or {"messages": [{"role": "user", "content": "hi"}]}
    req = SimpleNamespace(
        state=state,
        client=SimpleNamespace(host="1.2.3.4"),
    )

    async def _json():
        return req_body

    req.json = _json
    return req


class _GovStub:
    calls = []

    @staticmethod
    async def precheck(redis, api_key, model, endpoint):
        _GovStub.calls.append(("precheck", model, endpoint))


def _patch_gov(monkeypatch):
    _GovStub.calls = []
    monkeypatch.setattr(m, "compatible_governance_service", _GovStub)


def _patch_record(monkeypatch):
    recorded = {}

    def _record(**kw):
        recorded.update(kw)

    monkeypatch.setattr(m, "record_call", _record)
    return recorded


async def _ok_handler(body, audit, api_key):
    return {"ok": True, "audit": audit}


async def test_governance_429_openai_format(monkeypatch):
    _patch_gov(monkeypatch)

    async def _block(*a, **k):
        raise GovernanceError(429, "rate_limit_error", "超限")

    monkeypatch.setattr(_GovStub, "precheck", _block)

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key()),
        _User(),
        protocol="openai",
        endpoint="chat/completions",
        handler=_ok_handler,
    )
    assert resp.status_code == 429
    body = resp.body.decode()
    assert '"type":"rate_limit_error"' in body
    assert '"message":"超限"' in body
    assert '"code"' in body


async def test_governance_429_claude_format(monkeypatch):
    _patch_gov(monkeypatch)

    async def _block(*a, **k):
        raise GovernanceError(429, "rate_limit_error", "超限")

    monkeypatch.setattr(_GovStub, "precheck", _block)

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_compatible_call(
        req, _db(monkeypatch, _api_key()), _User(),
        protocol="claude", endpoint="messages", handler=_ok_handler,
    )
    assert resp.status_code == 429
    body = resp.body.decode()
    assert '"type":"error"' in body
    assert '"code"' not in body
    assert '"type":"rate_limit_error"' in body


async def test_governance_403_permission_error(monkeypatch):
    _patch_gov(monkeypatch)

    async def _block(*a, **k):
        raise GovernanceError(403, "permission_error", "模型不在白名单")

    monkeypatch.setattr(_GovStub, "precheck", _block)

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key()),
        _User(),
        protocol="openai",
        endpoint="chat/completions",
        handler=_ok_handler,
    )
    assert resp.status_code == 403
    assert '"type":"permission_error"' in resp.body.decode()


async def test_business_quota_insufficient_maps_402_openai(monkeypatch):
    _patch_gov(monkeypatch)

    async def _handler(body, audit, api_key):
        raise BusinessException(ResultCode.QUOTA_INSUFFICIENT)

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key()),
        _User(),
        protocol="openai",
        endpoint="chat/completions",
        handler=_handler,
    )
    assert resp.status_code == 402
    assert '"type":"insufficient_quota"' in resp.body.decode()


async def test_business_quota_insufficient_maps_402_claude(monkeypatch):
    _patch_gov(monkeypatch)

    async def _handler(body, audit, api_key):
        raise BusinessException(ResultCode.QUOTA_EXCEEDED)

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_compatible_call(
        req, _db(monkeypatch, _api_key()), _User(),
        protocol="claude", endpoint="messages", handler=_handler,
    )
    assert resp.status_code == 402
    assert '"type":"insufficient_quota"' in resp.body.decode()


def test_error_status_code_mapping():
    assert m._error_status_code(ResultCode.RATE_LIMIT) == 429
    assert m._error_status_code(ResultCode.QUOTA_INSUFFICIENT) == 402
    assert m._error_status_code(ResultCode.QUOTA_EXCEEDED) == 402
    assert m._error_status_code(ResultCode.AI_MODEL_NOT_AVAILABLE) == 403
    assert m._error_status_code(ResultCode.PARAM_ERROR) == 400


async def test_governance_error_records_audit(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _block(*a, **k):
        raise GovernanceError(429, "rate_limit_error", "超限")

    monkeypatch.setattr(_GovStub, "precheck", _block)

    req = _request(api_key_info={"key_id": 5, "key_prefix": "dhak_xy"})
    await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key(id=5)),
        _User(),
        protocol="openai",
        endpoint="chat/completions",
        handler=_ok_handler,
    )
    assert recorded["status_code"] == 429
    assert recorded["key_id"] == 5
    assert recorded["key_prefix"] == "dhak_xy"
    assert recorded["user_id"] == 7
    assert recorded["endpoint"] == "chat/completions"
    assert recorded["protocol"] == "openai"
    assert recorded["client_ip"] == "1.2.3.4"
    assert recorded["error_msg"] == "超限"


async def test_success_non_stream_audit_fields(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _handler(body, audit, api_key):
        internal = FakeInternalResponse(
            'event: content_block.delta\ndata: {"type":"text_delta","text":"你好，世界"}\n\n',
            'event: message.end\n'
            'data: {"stopReason":"stop","usage":{"inputTokens":10,"outputTokens":5,'
            '"credits":2}}\n\n',
        )
        return await m._openai_non_stream(internal, "gpt-4", "chatcmpl-x", 1234567, audit)

    req = _request(
        api_key_info={"key_id": 1, "key_prefix": "dhak_ab"},
        body={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "stream": True,
            "conversation_id": 42,
        },
    )
    await m._run_compatible_call(
        req,
        _db(monkeypatch, _api_key()),
        _User(),
        protocol="openai",
        endpoint="chat/completions",
        handler=_handler,
    )
    assert recorded["endpoint"] == "chat/completions"
    assert recorded["protocol"] == "openai"
    assert recorded["is_stream"] is True
    assert recorded["conversation_id"] == 42
    assert recorded["model"] == "gpt-4"
    assert recorded["input_tokens"] == 10
    assert recorded["output_tokens"] == 5
    assert recorded["credits"] == 2
    assert recorded["status_code"] == 200
    assert recorded["duration_ms"] >= 0


async def test_session_auth_skips_governance(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _handler(body, audit, api_key):
        m._record_audit(audit, status_code=200)
        return {"ok": True}

    req = _request()
    await m._run_compatible_call(
        req, _db(monkeypatch), _User(),
        protocol="openai", endpoint="chat/completions", handler=_handler,
    )
    assert _GovStub.calls == []
    assert recorded["key_id"] is None


async def test_list_models_openai_filters_whitelist(monkeypatch):
    model_a = SimpleNamespace(model_id="gpt-4", create_time=None, provider_id=1)
    model_b = SimpleNamespace(model_id="claude-3", create_time=None, provider_id=2)
    api_key = SimpleNamespace(id=1, model_whitelist=["gpt-4"])
    called = {}

    async def _vip_list(db, redis, user_id):
        called["user_id"] = user_id
        return [model_a, model_b]

    monkeypatch.setattr(m.ai_model_service, "list_enabled_models", _vip_list)
    result = await m.CompatibleApiService.list_models_openai(_db(monkeypatch), object(), 7, api_key)
    ids = [d["id"] for d in result["data"]]
    assert ids == ["gpt-4"]
    assert called["user_id"] == 7


async def test_list_models_openai_no_whitelist_passthrough(monkeypatch):
    model_a = SimpleNamespace(model_id="gpt-4", create_time=None, provider_id=1)
    model_b = SimpleNamespace(model_id="claude-3", create_time=None, provider_id=2)
    api_key = SimpleNamespace(id=1, model_whitelist=None)

    async def _vip_list(db, redis, user_id):
        return [model_a, model_b]

    monkeypatch.setattr(m.ai_model_service, "list_enabled_models", _vip_list)
    result = await m.CompatibleApiService.list_models_openai(_db(monkeypatch), object(), 7, api_key)
    assert len(result["data"]) == 2


async def test_list_models_claude_filters_whitelist(monkeypatch):
    model_a = SimpleNamespace(model_id="gpt-4", create_time=None, provider_id=1, display_name="A")
    model_b = SimpleNamespace(
        model_id="claude-3", create_time=None, provider_id=2, display_name="B"
    )
    api_key = SimpleNamespace(id=1, model_whitelist=["claude-3"])

    async def _vip_list(db, redis, user_id):
        return [model_a, model_b]

    monkeypatch.setattr(m.ai_model_service, "list_enabled_models", _vip_list)
    result = await m.CompatibleApiService.list_models_claude(_db(monkeypatch), object(), 7, api_key)
    assert [d["id"] for d in result["data"]] == ["claude-3"]


async def test_models_endpoint_runs_precheck_and_audits(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _handler(db, redis, user_id, api_key):
        return {"data": []}

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_models_call(
        req, _db(monkeypatch, _api_key()), _User(), protocol="openai", handler=_handler
    )
    assert resp.status_code == 200
    assert ("precheck", None, "models") in _GovStub.calls
    assert recorded["endpoint"] == "models"
    assert recorded["status_code"] == 200
    assert recorded["key_id"] == 1


async def test_models_endpoint_governance_429_audited(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _block(*a, **k):
        raise GovernanceError(429, "rate_limit_error", "Key 日配额已用尽")

    monkeypatch.setattr(_GovStub, "precheck", _block)

    async def _handler(db, redis, user_id, api_key):
        raise AssertionError("治理拒绝后不应执行列表查询")

    req = _request(api_key_info={"key_id": 1, "key_prefix": "dhak_ab"})
    resp = await m._run_models_call(
        req, _db(monkeypatch, _api_key()), _User(), protocol="openai", handler=_handler
    )
    assert resp.status_code == 429
    assert recorded["status_code"] == 429
    assert recorded["endpoint"] == "models"


async def test_models_endpoint_session_auth_skips_governance(monkeypatch):
    _patch_gov(monkeypatch)
    recorded = _patch_record(monkeypatch)

    async def _handler(db, redis, user_id, api_key):
        assert api_key is None
        return {"data": []}

    req = _request()
    resp = await m._run_models_call(
        req, _db(monkeypatch), _User(), protocol="openai", handler=_handler
    )
    assert resp.status_code == 200
    assert _GovStub.calls == []
    assert recorded["key_id"] is None
