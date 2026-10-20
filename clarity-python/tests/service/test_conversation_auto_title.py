"""会话标题自动生成：LLM 输出清洗、超 20 字与调用失败降级（T-CV-010 / T-CV-011）。"""

from types import SimpleNamespace

import pytest

from app.service import ai_conversation_service as m
from app.service.ai_conversation_service import (
    AiConversationService,
    _clean_generated_title,
)
from tests.stubs.fakes import LLMChunk, NullDBSession


class _FakeLlm:
    """流式桩：按序产出 LLMChunk，或迭代时抛异常模拟调用失败。"""

    def __init__(self, chunks=(), exc=None):
        self._chunks = chunks
        self._exc = exc
        self.calls: list[dict] = []

    async def stream_chat(self, db, model_id, messages, **kwargs):
        self.calls.append({"model_id": model_id, "messages": messages, **kwargs})
        if self._exc:
            raise self._exc
        for chunk in self._chunks:
            yield chunk


class _FakeConvRepo:
    def __init__(self, conv):
        self._conv = conv
        self.updated = []

    async def get_by_id(self, db, cid):
        return self._conv

    async def update_title(self, db, cid, title, title_source="auto"):
        self.updated.append((cid, title, title_source))


@pytest.fixture(autouse=True)
def _null_session(monkeypatch):
    monkeypatch.setattr(m, "get_db_session", lambda: NullDBSession())


def _conv():
    return SimpleNamespace(id=1, model="m1", user_id=7, title="新对话", title_source="auto")


def _service(llm, repo):
    return AiConversationService(ai_conversation_repository=repo, llm_client=llm)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  雾天图像去雾总结  ", "雾天图像去雾总结"),
        ("“雾天图像去雾总结”", "雾天图像去雾总结"),
        ("标题：雾天图像去雾总结", "雾天图像去雾总结"),
        ("标题：\n雾天图像去雾总结", ""),  # 首行仅前缀，清洗后为空
        ("雾天去雾方案\n以下是我的解释", "雾天去雾方案"),
        ("一二三四五六七八九十一二三四五六七八九十", "一二三四五六七八九十一二三四五六七八九十"),
        ("", ""),
        ("    ", ""),
        ("这是一个超过二十个字的非常冗长的标题它不应该被采纳使用", ""),
        ("好的，用户让我为对话“1+1=?”生成一个不超过20字的标题。首先，我需要理解。", ""),
        # 思考链残留（Qwen3 未关思考时随正文回传）：闭合/未闭合两种形态都必须剥掉
        ("<think>用户在问 1+1</think>1+1 等于 2", "1+1 等于 2"),
        ("嗯，用户让我为「1+1=?」生成标题。首先", ""),
        # 短到能过长度校验的推理自述也不是标题（此前会漏过）
        ("嗯，用户让我为对话起标题", ""),
        ("首先分析一下用户的问题", ""),
        # 正文里出现「用户」不构成推理自述，正常标题要保留
        ("用户增长与留存策略分析", "用户增长与留存策略分析"),
    ],
)
def test_clean_generated_title(raw, expected):
    assert _clean_generated_title(raw) == expected


async def test_reasoning_blob_falls_back_to_user_content():
    blob = "好的，用户让我为对话“1+1=?”生成一个不超过20字的标题。首先，我需要理解这个对话的内容。"
    repo = _FakeConvRepo(_conv())
    await _service(_FakeLlm([LLMChunk("text_delta", blob)]), repo)._auto_generate_title(
        1, "1+1=? 只回答数字"
    )
    assert repo.updated == [(1, "1+1=? 只回答数字", "auto")]


async def test_clean_title_kept_after_normalize():
    llm = _FakeLlm(
        [
            LLMChunk("text_delta", "标题："),
            LLMChunk("text_delta", "“雾天图像去雾方案”"),
            LLMChunk("thinking_delta", "这段是思考流，不应计入标题"),
        ]
    )
    repo = _FakeConvRepo(_conv())
    await _service(llm, repo)._auto_generate_title(1, "帮我看看这张雾图")
    assert repo.updated == [(1, "雾天图像去雾方案", "auto")]


async def test_llm_failure_falls_back():
    repo = _FakeConvRepo(_conv())
    await _service(_FakeLlm(exc=RuntimeError("boom")), repo)._auto_generate_title(
        1, "分析这张图片的雾霾浓度"
    )
    assert repo.updated == [(1, "分析这张图片的雾霾浓度", "auto")]


async def test_long_first_message_truncated_to_20():
    repo = _FakeConvRepo(_conv())
    await _service(_FakeLlm(), repo)._auto_generate_title(1, "雾" * 50)
    assert repo.updated == [(1, "雾" * 20, "auto")]


async def test_title_call_passes_system_prompt_to_disable_thinking():
    """标题调用必须带 system 提示：本地 provider 据此在 system 尾部注入 /no_think 关思考，
    否则 Qwen3 的推理过程会混进标题（设计评审已发现「嗯，用户让我为…」标题）。"""
    llm = _FakeLlm([LLMChunk("text_delta", "图像去雾算法总结")])
    repo = _FakeConvRepo(_conv())
    await _service(llm, repo)._auto_generate_title(1, "介绍下去雾算法")
    assert llm.calls, "未调用 LLM"
    system_prompt = llm.calls[0].get("system_prompt") or ""
    assert "标题" in system_prompt
    assert repo.updated == [(1, "图像去雾算法总结", "auto")]
