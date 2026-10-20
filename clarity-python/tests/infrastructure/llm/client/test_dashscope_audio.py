"""阿里云百炼（DashScope）语音协议适配器：协议结构与事件归一化纯函数单测。

覆盖：run-task / finish-task / continue-task 指令帧结构、ASR 下行事件归一化
（heartbeat 跳过、sentence_end 真假、非结果事件跳过、二进制帧跳过）、task-failed
错误码映射、WS 地址派生。均为不依赖 DB/Redis 的纯逻辑断言。
"""

import json
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosed

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.llm.client import dashscope_audio_client as client


def _result_event(text: str, *, sentence_end: bool = False, heartbeat: bool = False) -> dict:
    """构造下行 result-generated 事件（sentence 字段对齐百炼协议）"""
    return {
        "header": {"event": "result-generated", "task_id": "t-1"},
        "payload": {
            "output": {
                "sentence": {
                    "text": text,
                    "sentence_end": sentence_end,
                    "heartbeat": heartbeat,
                }
            }
        },
    }


class _FakeWs:
    """假 WS：记录 send、按序出帧（recv 单取 / __aiter__ 迭代），close 置位"""

    def __init__(self, frames: list) -> None:
        self.frames = list(frames)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        return self.frames.pop(0)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        while self.frames:
            yield self.frames.pop(0)


# --- 指令帧结构 ----------------------------------------------------------------


def test_run_task_frame_structure():
    payload = client._asr_payload("paraformer-realtime-v2")
    task = client._run_task("tid-1", payload)

    assert task["header"] == {"action": "run-task", "task_id": "tid-1", "streaming": "duplex"}
    assert task["payload"] == {
        "task_group": "audio",
        "task": "asr",
        "function": "recognition",
        "model": "paraformer-realtime-v2",
        "parameters": {"format": "pcm", "sample_rate": 16000, "disfluency_removal_enabled": False},
        "input": {},
    }


def test_finish_task_frame_structure():
    task = client._finish_task("tid-1")

    assert task["header"] == {"action": "finish-task", "task_id": "tid-1", "streaming": "duplex"}
    assert task["payload"] == {"input": {}}


def test_continue_task_frame_structure():
    task = client._continue_task("tid-1", "你好")

    assert task["header"] == {"action": "continue-task", "task_id": "tid-1"}
    assert task["payload"] == {"input": {"text": "你好"}}


def test_tts_payload_structure():
    payload = client._tts_payload("cosyvoice-v1", "longxiaochun", 1.0, "mp3", 22050)

    assert payload == {
        "task_group": "audio",
        "task": "tts",
        "function": "SpeechSynthesizer",
        "model": "cosyvoice-v1",
        "parameters": {
            "text_type": "PlainText",
            "voice": "longxiaochun",
            "format": "mp3",
            "sample_rate": 22050,
            "rate": 1.0,
        },
        "input": {},
    }


# --- 下行事件归一化 ------------------------------------------------------------


def test_normalize_result_not_final():
    out = client._normalize_asr_event(_result_event("你好", sentence_end=False))

    assert out is not None
    assert json.loads(out) == {"text": "你好", "is_final": False}


def test_normalize_result_final():
    out = client._normalize_asr_event(_result_event("你好。", sentence_end=True))

    assert out is not None
    assert json.loads(out) == {"text": "你好。", "is_final": True}


def test_normalize_heartbeat_skipped():
    assert client._normalize_asr_event(_result_event("", heartbeat=True)) is None


def test_normalize_non_result_event_skipped():
    assert client._normalize_asr_event({"header": {"event": "task-started"}}) is None
    assert client._normalize_asr_event({"header": {"event": "task-finished"}}) is None


# --- task-failed 错误映射 ------------------------------------------------------


def test_task_failed_maps_known_code():
    event = {
        "header": {
            "event": "task-failed",
            "error_code": "InvalidParameter",
            "error_message": "model 参数无效",
        }
    }

    with pytest.raises(BusinessException) as exc:
        client._normalize_asr_event(event)

    assert exc.value.code == ResultCode.PARAM_ERROR
    assert exc.value.message == "语音引擎服务错误：model 参数无效"


def test_task_failed_unknown_code_falls_back_business_error():
    event = {"header": {"event": "task-failed", "error_code": "Whatever", "error_message": "boom"}}

    with pytest.raises(BusinessException) as exc:
        client._normalize_asr_event(event)

    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert exc.value.message == "语音引擎服务错误：boom"


def test_task_failed_without_message_falls_back_to_code():
    event = {"header": {"event": "task-failed", "error_code": "Throttling"}}

    with pytest.raises(BusinessException) as exc:
        client._normalize_asr_event(event)

    assert exc.value.code == ResultCode.BUSINESS_ERROR
    assert exc.value.message == "语音引擎服务错误：Throttling"


# --- WS 地址派生 ---------------------------------------------------------------


def test_ws_url_derives_from_bare_host():
    provider = SimpleNamespace(api_base_url="https://dashscope.aliyuncs.com", provider_code="ali")

    assert client._ws_url(provider) == "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


def test_ws_url_keeps_existing_path_and_downgrades_scheme():
    provider = SimpleNamespace(
        api_base_url="http://dashscope.aliyuncs.com/api-ws/v1/inference/", provider_code="ali"
    )

    assert client._ws_url(provider) == "ws://dashscope.aliyuncs.com/api-ws/v1/inference"


# --- 会话：run-task/start/finish-task 与下行归一路径 ---------------------------


async def test_start_then_eos_reuse_same_task_id():
    ws = _FakeWs([json.dumps({"header": {"event": "task-started"}})])
    session = client.AsrStreamSession(ws, "paraformer-realtime-v2")

    await session.start()
    await session.send_eos()

    run = json.loads(ws.sent[0])
    fin = json.loads(ws.sent[-1])
    assert run["header"]["action"] == "run-task"
    assert fin["header"]["action"] == "finish-task"
    assert fin["header"]["task_id"] == run["header"]["task_id"]


async def test_session_recv_normalizes_and_stops_at_task_finished():
    frames = [
        json.dumps(_result_event("", heartbeat=True)),
        json.dumps(_result_event("你好", sentence_end=False)),
        b"\x00\x01",  # ASR 下行不应出现的二进制帧，须被跳过
        json.dumps(_result_event("你好世界。", sentence_end=True)),
        json.dumps({"header": {"event": "task-finished"}}),
        json.dumps(_result_event("tail", sentence_end=False)),  # task-finished 之后不再产出
    ]
    session = client.AsrStreamSession(_FakeWs(frames), "paraformer-realtime-v2")

    out = [json.loads(m) async for m in session.recv_messages()]

    assert out == [
        {"text": "你好", "is_final": False},
        {"text": "你好世界。", "is_final": True},
    ]


async def test_send_audio_raises_engine_stream_closed_on_disconnect():
    class _ClosedWs:
        async def send(self, data) -> None:
            raise ConnectionClosed(None, None)

    session = client.AsrStreamSession(_ClosedWs(), "paraformer-realtime-v2")

    with pytest.raises(client.EngineStreamClosed):
        await session.send_audio(b"pcm")
