from types import SimpleNamespace

from app.service.ai.service.suggestion_service import suggestion_service
from tests.stubs.fakes import RecorderEmitter


def _msg(input_tokens=100, output_tokens=50, cached_input_tokens=0, credits=10, model="gpt-4o"):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        credits=credits,
        model=model,
    )


def _conv(suggest=True):
    return SimpleNamespace(suggestions_enabled=int(suggest), model="gpt-4o")


def _patch_service(monkeypatch, *, conv, msg, gen_result):
    service = suggestion_service
    emitter = RecorderEmitter()

    class _ConvRepo:
        async def get_by_id(self, db, cid):
            return conv

    class _MsgRepo:
        async def get_by_id(self, db, mid):
            return msg

    async def _settle(
        db, redis, user_id, conversation_id, message_id, model_id, actual_model_id, usage, **kwargs
    ):
        return {}

    async def _calc(db, model_id, it, ot, ct):
        return it + ot

    monkeypatch.setattr(service, "_generate_questions", gen_result)
    monkeypatch.setattr("app.service.ai.service.suggestion_service.ai_conversation_repository", _ConvRepo())
    monkeypatch.setattr("app.service.ai.service.suggestion_service.ai_message_repository", _MsgRepo())
    monkeypatch.setattr(
        "app.service.ai.service.suggestion_service.billing_service",
        type("B", (), {"settle": staticmethod(_settle)})(),
    )
    monkeypatch.setattr("app.service.ai.service.suggestion_service.calculate_credits", _calc)
    monkeypatch.setattr("app.service.ai.service.suggestion_service.sse_emitter_manager", emitter)
    return service, emitter


async def test_switch_off_skips(monkeypatch):
    service, emitter = _patch_service(
        monkeypatch,
        conv=_conv(suggest=False),
        msg=_msg(),
        gen_result=None,
    )
    result = await service.generate(1, 2, "回答", 7, "s1")
    assert result is None
    assert emitter.events == []


async def test_generate_success_counts_token_and_pushes(monkeypatch):
    async def _gen_success(db, model_id, reply, **kwargs):
        return ["追问一", "追问二"], {"input_tokens": 10, "output_tokens": 20}

    msg = _msg()
    service, emitter = _patch_service(
        monkeypatch, conv=_conv(), msg=msg, gen_result=_gen_success
    )

    result = await service.generate(1, 2, "回答", 7, "s1")

    assert result == ["追问一", "追问二"]
    assert msg.input_tokens == 110
    assert msg.output_tokens == 70
    assert emitter.events[-1][0] == "suggestions"
    assert emitter.events[-1][1] == {"questions": [{"question": "追问一"}, {"question": "追问二"}]}


async def test_generate_failure_returns_none(monkeypatch):
    async def _gen_fail(db, model_id, reply, **kwargs):
        return None

    service, emitter = _patch_service(
        monkeypatch, conv=_conv(), msg=_msg(), gen_result=_gen_fail
    )
    result = await service.generate(1, 2, "回答", 7, "s1")
    assert result is None
    assert emitter.events == []


async def test_empty_reply_skips(monkeypatch):
    service, emitter = _patch_service(monkeypatch, conv=_conv(), msg=_msg(), gen_result=None)
    result = await service.generate(1, 2, "", 7, "s1")
    assert result is None
    assert emitter.events == []


async def test_parse_questions_json_array():
    svc = suggestion_service
    assert await svc._parse_questions('["追问一", "追问二"]') == ["追问一", "追问二"]
    assert await svc._parse_questions('说明如下\n["追问"]\n结束') == ["追问"]
    assert await svc._parse_questions("不是数组") is None
    assert await svc._parse_questions("") is None
