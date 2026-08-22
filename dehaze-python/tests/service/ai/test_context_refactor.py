from types import SimpleNamespace

from app.service.ai.context_manager import ContextManager
from app.service.ai.prompt_composer import STABLE_SYSTEM_PROMPT, compose_system_prompt
from app.service.ai.scene_templates import SCENE_VALUES, get_scene_prompt
from app.service.ai.reasoning_service import reasoning_service
from app.service.ai.summary_service import summary_service, _PRIOR_SUMMARY_MAX_LEN
from tests.stubs import NullDBSession, make_conv, repo_returns


class _AgentSnapshot:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt


class _Conv:
    def __init__(self, system_prompt: str | None):
        self.system_prompt = system_prompt


class _Msg:
    def __init__(self, mid, role, content):
        self.id = mid
        self.role = role
        self.content = content


def _finalize_msg(**overrides):
    fields = {
        "content": None,
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "credits": None,
        "status": None,
        "used_memory_ids": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _install_summary_mocks(monkeypatch, prior_summary, load_messages, recompress=None):
    class _Model:
        max_context_tokens = 10_000

    class _Conv:
        def __init__(self):
            self.summary = prior_summary
            self.summary_upto_message_id = None

    conv = _Conv()

    async def _get_model(db, model_id):
        return _Model()

    async def _build_context(self, db, conv, model_id):
        return [{"role": "user", "content": "x" * 100}], "system", []

    async def _estimate(messages, system_prompt):
        return 9_000

    async def _gen_summary(db, model_id, msgs):
        return "新摘要"

    async def _no_memory_extract(db, conv, msgs):
        return None

    monkeypatch.setattr(
        "app.service.ai.summary_service.ai_model_repository.get_by_model_id", _get_model
    )
    monkeypatch.setattr(ContextManager, "build_context", _build_context)
    monkeypatch.setattr("app.service.ai.summary_service.estimate_context_tokens", _estimate)
    monkeypatch.setattr(summary_service, "_load_messages_to_summarize", staticmethod(load_messages))
    monkeypatch.setattr(summary_service, "_generate_summary", staticmethod(_gen_summary))
    monkeypatch.setattr(summary_service, "_extract_episodic_memory", staticmethod(_no_memory_extract))
    if recompress is not None:
        monkeypatch.setattr(summary_service, "_recompress_prior_summary", staticmethod(recompress))
    return conv


def _install_build_context_mocks(monkeypatch, chain_msgs, memory, artifact_refs=None):
    async def _get_chain_by_id(db, conv_id, start_id, limit=None):
        return chain_msgs

    async def _no_snapshot(db, conv):
        return None

    monkeypatch.setattr(ContextManager, "_load_agent_snapshot", staticmethod(_no_snapshot))
    monkeypatch.setattr(
        "app.service.ai.context_manager.ai_message_repository.get_chain_by_id",
        _get_chain_by_id,
    )
    monkeypatch.setattr("app.service.ai.context_manager.inject_memories", memory)
    if artifact_refs is not None:
        monkeypatch.setattr(
            "app.service.ai.context_manager.ai_artifact_service.get_message_artifact_refs",
            staticmethod(artifact_refs),
        )


def _install_finalize_mocks(monkeypatch, msg, credits):
    monkeypatch.setattr(
        "app.service.ai.reasoning_service.ai_message_repository", repo_returns(msg)
    )
    monkeypatch.setattr("app.service.ai.reasoning_service.get_db_session", NullDBSession)

    async def _calc(db, model_id, it, ot, cit):
        return credits

    monkeypatch.setattr("app.service.ai.reasoning_service.calculate_credits", _calc)
    return msg


def test_compose_system_prompt_three_layers_stable_first():
    snapshot = _AgentSnapshot("你是图像处理专家。")
    conv = _Conv("请用 RIDCP 算法处理图像。")
    result = compose_system_prompt(snapshot, conv)
    assert result.startswith(STABLE_SYSTEM_PROMPT)
    parts = result.split("\n\n")
    assert parts[0] == STABLE_SYSTEM_PROMPT
    assert result.index(STABLE_SYSTEM_PROMPT) < result.index("你是图像处理专家。")
    assert result.index("你是图像处理专家。") < result.index("请用 RIDCP 算法处理图像。")


def test_compose_system_prompt_skips_empty_layers():
    assert compose_system_prompt(None, None) == STABLE_SYSTEM_PROMPT
    assert compose_system_prompt(None, _Conv("会话提示")) == (STABLE_SYSTEM_PROMPT + "\n\n会话提示")
    assert compose_system_prompt(_AgentSnapshot("agent"), _Conv("")) == (
        STABLE_SYSTEM_PROMPT + "\n\nagent"
    )


def test_get_scene_prompt_known_scenes():
    assert set(SCENE_VALUES) == {
        "general",
        "image_dispatch",
        "multi_step",
        "algorithm_recommend",
        "scheduled_task",
    }
    for scene in SCENE_VALUES:
        prompt = get_scene_prompt(scene)
        assert "【角色】" in prompt and "【任务】" in prompt and "【格式】" in prompt


def test_get_scene_prompt_fallback_to_general():
    assert get_scene_prompt(None) == get_scene_prompt("general")
    assert get_scene_prompt("unknown_scene") == get_scene_prompt("general")


async def test_builder_system_prompt_excludes_conversation(monkeypatch):
    import app.service.ai.deep_agent_builder as builder_mod
    from app.service.ai.deep_agent_builder import DeepAgentBuilder

    captured = {}

    def _fake_create_deep_agent(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return object()

    monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)

    snapshot = {
        "system_prompt": "Agent 人设内容",
        "config": {
            "guardrails": {},
            "mcp_namespaces": [],
            "token_budget": 100,
            "max_steps": 1,
            "max_steps_react": 5,
        },
        "model_id": "m1",
        "name": "a1",
        "subagents": [],
    }
    await DeepAgentBuilder.build_from_snapshot(None, None, snapshot)
    built = captured["system_prompt"]
    assert built.startswith(STABLE_SYSTEM_PROMPT)
    assert "Agent 人设内容" in built
    assert "会话场景" not in built


async def test_summary_watermark_selects_only_after_watermark(monkeypatch):
    all_rows = [
        SimpleNamespace(id=i, role="user" if i % 2 else "assistant", content=f"c{i}")
        for i in range(1, 41)
    ]
    conv = SimpleNamespace(id=1, summary_upto_message_id=10)

    async def _list_for_summary(db, conv_id, watermark):
        return [r for r in all_rows if r.id > watermark][::-1]

    monkeypatch.setattr(
        "app.service.ai.summary_service.ai_message_repository.list_for_summary",
        _list_for_summary,
    )
    selected = await summary_service._load_messages_to_summarize(None, conv)
    assert selected
    assert selected[0]["id"] == 11
    assert selected[-1]["id"] == 20
    assert all(10 < m["id"] <= 20 for m in selected)


async def test_maybe_compress_appends_summary_and_advances_watermark(monkeypatch):
    async def _load_messages(db, conv):
        return [{"id": 15, "role": "user", "content": "待压缩的近期消息"}]

    conv = _install_summary_mocks(monkeypatch, "旧摘要", _load_messages)
    await summary_service.maybe_compress(NullDBSession(), conv, "gpt")
    assert conv.summary == "前序摘要：旧摘要\n近期摘要：新摘要"
    assert conv.summary_upto_message_id == 15


async def test_maybe_compress_recompresses_oversized_prior_summary(monkeypatch):
    async def _load_messages(db, conv):
        return [{"id": 3, "role": "user", "content": "待压缩"}]

    async def _recompress(db, model_id, old_summary):
        return "再压缩后的前序摘要"

    conv = _install_summary_mocks(
        monkeypatch, "旧" * (_PRIOR_SUMMARY_MAX_LEN + 1), _load_messages, _recompress
    )
    await summary_service.maybe_compress(NullDBSession(), conv, "gpt")
    assert conv.summary == "前序摘要：再压缩后的前序摘要\n近期摘要：新摘要"


def test_artifact_ref_line_format():
    refs = [{"id": 7, "type": "image_result", "summary": {"algorithm": "RIDCP"}}]
    lines = ContextManager._build_artifact_ref_lines(refs)
    assert lines == ["[[产物 #7] image_result：{'algorithm': 'RIDCP'}]"]


def test_artifact_ref_summary_truncated_200():
    refs = [{"id": 1, "type": "metric_report", "summary": {"detail": "x" * 250}}]
    line = ContextManager._build_artifact_ref_lines(refs)[0]
    summary_part = line.split("：")[1].rstrip("]")
    assert summary_part.endswith("…")
    assert len(summary_part) == 200 + 1


def test_artifact_ref_empty_and_missing_summary():
    assert ContextManager._build_artifact_ref_lines(None) == []
    assert ContextManager._build_artifact_ref_lines([]) == []
    line = ContextManager._build_artifact_ref_lines([{"id": 3, "type": "file_ref"}])[0]
    assert line == "[[产物 #3] file_ref：]"


async def test_build_context_attaches_artifact_refs(monkeypatch):
    msgs = {
        1: _Msg(1, "user", "帮我处理这张图"),
        2: _Msg(2, "assistant", "已处理完成"),
    }

    async def _no_memories(*a, **k):
        return None, []

    async def _artifact_refs(db, message_ids):
        return {2: [{"id": 7, "type": "image_result", "summary": {"algorithm": "RIDCP"}}]}

    _install_build_context_mocks(monkeypatch, [msgs[1], msgs[2]], _no_memories, _artifact_refs)

    messages, system_prompt, _injected = await ContextManager().build_context(
        object(), make_conv(current_branch_message_id=2, system_prompt="会话提示"), "gpt"
    )
    assert messages[1]["role"] == "assistant"
    assert "[[产物 #7] image_result：{'algorithm': 'RIDCP'}]" in messages[1]["content"]
    assert system_prompt.startswith(STABLE_SYSTEM_PROMPT)
    assert "会话提示" in system_prompt


async def test_build_context_injects_memory_system_block(monkeypatch):
    async def _with_memory(*a, **k):
        return "【用户画像】用户偏好 RIDCP", [
            {"memory_id": 1, "memory_type": "semantic", "content": "偏好 RIDCP", "source": "preference"}
        ]

    _install_build_context_mocks(monkeypatch, [_Msg(1, "user", "处理雾图")], _with_memory)

    messages, _system_prompt, injected = await ContextManager().build_context(
        object(), make_conv(current_branch_message_id=1, system_prompt="会话提示"), "gpt"
    )
    assert messages[0] == {"role": "system", "content": "【用户画像】用户偏好 RIDCP"}
    assert messages[1]["role"] == "user"
    assert injected == [
        {"memory_id": 1, "memory_type": "semantic", "content": "偏好 RIDCP", "source": "preference"}
    ]


async def test_finalize_message_writes_used_memory_ids(monkeypatch):
    msg = _install_finalize_mocks(monkeypatch, _finalize_msg(), 9)

    result = {
        "final_response": "ok",
        "stop_reason": "stop",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2},
    }
    await reasoning_service._finalize_message(1, result, "gpt", used_memory_ids=[7, 8, 9])
    assert msg.used_memory_ids == [7, 8, 9]
    assert msg.status == 2
    assert msg.credits == 9


async def test_finalize_message_skips_empty_used_memory_ids(monkeypatch):
    msg = _install_finalize_mocks(monkeypatch, _finalize_msg(used_memory_ids="keep"), 0)

    result = {"final_response": "ok", "stop_reason": "stop", "usage": {}}
    await reasoning_service._finalize_message(1, result, "gpt", used_memory_ids=[])
    assert msg.used_memory_ids == "keep"
