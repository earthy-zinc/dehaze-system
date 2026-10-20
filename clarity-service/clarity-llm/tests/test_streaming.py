"""流式对话契约：usage 块顺序、断连即停、推理锁排队与取消。"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from fastapi.responses import StreamingResponse

from app import inference
from app.routers import openai_compat


def _body(response: Any) -> AsyncGenerator[str, None]:
    """取流式分支的 SSE 生成器：非流式分支返回 dict，测试断言只走流式分支。"""
    assert isinstance(response, StreamingResponse)
    return cast("AsyncGenerator[str, None]", response.body_iterator)


def _request(stream: bool = True) -> openai_compat.ChatCompletionRequest:
    return openai_compat.ChatCompletionRequest(
        messages=[openai_compat.ChatMessage(role="user", content="你好")], stream=stream
    )


async def wait_unlocked(lock, timeout: float = 2.0) -> None:
    """等待推理锁被工作线程释放（生成线程退出有约 0.5s 的投递复查窗口）。"""
    for _ in range(int(timeout / 0.05)):
        if not lock.locked():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("推理锁未在规定时间内释放")


async def _payloads(response: Any) -> list[str]:
    """驱动 StreamingResponse 的 SSE 生成器，返回 data 载荷列表。"""
    payloads = []
    async for chunk in _body(response):
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        payloads.extend(
            line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")
        )
    return payloads


async def test_stream_finish_usage_done_order(chat_llm):
    chat_llm.tokens = ("你", "好")
    response = await openai_compat.chat_completions(_request())
    payloads = await _payloads(response)

    assert payloads[-1] == "[DONE]"
    usage_chunk = json.loads(payloads[-2])
    assert usage_chunk["choices"] == []
    usage = usage_chunk["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == usage["prompt_tokens"] + 2
    # 尾块（finish_reason=stop）必须在 usage 块之前
    tail = json.loads(payloads[-3])
    assert tail["choices"][0]["finish_reason"] == "stop"
    deltas = [json.loads(p)["choices"][0]["delta"]["content"] for p in payloads[:-3]]
    assert deltas == ["你", "好"]
    first = json.loads(payloads[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["model"] == "qwen3-0.6b"
    assert first["choices"][0]["finish_reason"] is None

    await wait_unlocked(inference._infer_lock)


async def test_stream_strips_template_residue(chat_llm):
    chat_llm.tokens = ("正文", "<|im_end|>残留")
    response = await openai_compat.chat_completions(_request())
    payloads = await _payloads(response)
    deltas = [json.loads(p)["choices"][0]["delta"]["content"] for p in payloads[:-3]]
    assert deltas == ["正文"]
    await wait_unlocked(inference._infer_lock)


async def test_disconnect_stops_generation(chat_llm):
    chat_llm.tokens = tuple(f"t{i}" for i in range(30))
    chat_llm.stream_delay = 0.01
    response = await openai_compat.chat_completions(_request())
    agen = _body(response)
    assert "data: " in await anext(agen)

    await agen.aclose()  # 客户端断连：连接被取消，生成器收 GeneratorExit
    await wait_unlocked(inference._infer_lock)
    # 断连后不再空转生成剩余 token
    assert 0 < chat_llm.generated < 30


async def test_concurrent_requests_serialized_by_shared_lock(chat_llm):
    chat_llm.tokens = ("a", "b", "c")
    chat_llm.stream_delay = 0.02
    first = await openai_compat.chat_completions(_request())
    second = await openai_compat.chat_completions(_request())
    await asyncio.gather(_payloads(first), _payloads(second))
    assert chat_llm.max_concurrent == 1  # 两个请求未同时进入 llama.cpp
    await wait_unlocked(inference._infer_lock)


async def test_request_queues_while_lock_held(chat_llm):
    await inference._infer_lock.acquire()
    try:
        response = await openai_compat.chat_completions(_request())
        agen = _body(response)
        pending = asyncio.create_task(anext(agen))
        await asyncio.sleep(0.1)
        assert not pending.done()  # 排在锁等待处，未开始生成
        assert chat_llm.generated == 0
        assert chat_llm.active == 0
    finally:
        inference._infer_lock.release()

    assert (await asyncio.wait_for(pending, timeout=5)).startswith("data: ")
    assert chat_llm.generated >= 1
    await agen.aclose()
    await wait_unlocked(inference._infer_lock)


async def test_queued_request_cancelled_before_generation(chat_llm):
    await inference._infer_lock.acquire()
    try:
        response = await openai_compat.chat_completions(_request())
        agen = _body(response)
        pending = asyncio.create_task(anext(agen))
        await asyncio.sleep(0.1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        # 取消发生在锁等待处：不派发生成工作线程，也不影响当前持锁者
        assert inference._infer_lock.locked()
        assert chat_llm.generated == 0
        assert chat_llm.active == 0
    finally:
        inference._infer_lock.release()


async def test_generation_error_propagates_and_releases_lock(chat_llm, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("llama 崩溃")

    monkeypatch.setattr(chat_llm, "create_completion", _boom)
    response = await openai_compat.chat_completions(_request())
    with pytest.raises(RuntimeError, match="llama 崩溃"):
        await _payloads(response)
    # 生成异常必须释放推理锁，否则后续所有推理请求被饿死
    await wait_unlocked(inference._infer_lock)
