"""火山引擎语音协议适配器纯函数单测（不依赖 DB/Redis/网络）。

覆盖：二进制帧编解码（头 4 字节位域、seq/size 大端、带/不带 sequence、末帧负 seq、
错误帧）、ASR 结果归一化（text/definite/错误码）、请求 JSON 结构与鉴权头组装
（Bearer; 分号、default_headers 读取）。
"""

import gzip
import json

import pytest

from app.core.exceptions import BusinessException
from app.infrastructure.llm.client import volc_audio_client as volc

# ===== 测试用假对象 =====


class _Provider:
    """最小供应商桩（经典口径）：仅含本模块读取的属性"""

    provider_code = "volc"
    api_base_url = "https://openspeech.bytedance.com"
    request_timeout_seconds = None
    default_headers = {
        "X-Api-App-Key": "appid-123",
        "asr_resource_id": "volc.seedasr.sauc.duration",
    }


class _SingleKeyProvider(_Provider):
    """单密钥口径（新版控制台 auth_header）：ASR/TTS 共用行、resource-id 按能力分列。"""

    default_headers = {
        "X-Api-App-Key": "appid-123",
        "asr_resource_id": "volc.seedasr.sauc.duration",
        "tts_resource_id": "seed-tts-2.0",
        "auth_header": "X-Api-Key",
    }


class _FakeWs:
    """记录 send 的 WS 桩"""

    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    async def send(self, data) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


class _FakeRecvWs:
    """按序产出帧的 WS 桩（支持 async for）"""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for frame in self._frames:
            yield frame

    async def close(self) -> None:
        pass


# ===== 帧编码 =====


def test_encode_frame_header_and_big_endian_size():
    """头 4 字节位域正确，payload_size 为大端 uint32，payload 为 gzip"""
    payload = b"hello"
    frame = volc.encode_frame(
        volc.MSG_FULL_CLIENT_REQUEST,
        volc.FLAG_NO_SEQUENCE,
        payload,
        serialization=volc.SERIALIZATION_JSON,
        compression=volc.COMPRESSION_GZIP,
    )
    assert frame[0] == 0x11  # 版本 1 | 头长度 1
    assert frame[1] == (0b0001 << 4) | 0b0000  # full client request | 无 seq
    assert frame[2] == (0b0001 << 4) | 0b0001  # JSON | gzip
    assert frame[3] == 0x00
    gz = gzip.compress(payload)
    assert frame[4:8] == len(gz).to_bytes(4, "big")  # 无 seq：payload_size 紧跟头
    assert frame[8:] == gz


def test_encode_frame_with_sequence_layout():
    """带正 sequence：帧体为 seq(4B 有符号大端) + payload_size(4B 大端) + payload"""
    frame = volc.encode_frame(
        volc.MSG_AUDIO_ONLY_REQUEST,
        volc.FLAG_POSITIVE_SEQUENCE,
        b"ab",
        sequence=5,
        compression=volc.COMPRESSION_NONE,
    )
    assert frame[0] == 0x11
    assert frame[1] == (0b0010 << 4) | 0b0001  # audio only request | 正 seq
    assert frame[2] == 0x00  # 无序列化 | 不压缩
    assert frame[4:8] == (5).to_bytes(4, "big", signed=True)
    assert frame[8:12] == (2).to_bytes(4, "big", signed=False)
    assert frame[12:] == b"ab"


# ===== 帧解码 =====


def test_decode_frame_roundtrip_with_sequence():
    raw = volc.encode_frame(
        volc.MSG_AUDIO_ONLY_REQUEST,
        volc.FLAG_POSITIVE_SEQUENCE,
        b"chunk",
        sequence=7,
        compression=volc.COMPRESSION_GZIP,
    )
    frame = volc.decode_frame(raw)
    assert frame.message_type == volc.MSG_AUDIO_ONLY_REQUEST
    assert frame.flags == volc.FLAG_POSITIVE_SEQUENCE
    assert frame.serialization == volc.SERIALIZATION_NONE
    assert frame.compression == volc.COMPRESSION_GZIP
    assert frame.sequence == 7
    assert frame.payload == b"chunk"
    assert frame.header_size == 4


def test_decode_frame_without_sequence():
    raw = volc.encode_frame(
        volc.MSG_FULL_CLIENT_REQUEST,
        volc.FLAG_NO_SEQUENCE,
        b"{}",
        serialization=volc.SERIALIZATION_JSON,
    )
    frame = volc.decode_frame(raw)
    assert frame.sequence is None
    assert frame.payload == b"{}"


def test_decode_frame_negative_sequence_marks_last():
    raw = volc.encode_frame(
        volc.MSG_AUDIO_ONLY_REQUEST,
        volc.FLAG_NEGATIVE_SEQUENCE,
        b"",
        sequence=-3,
        compression=volc.COMPRESSION_NONE,
    )
    frame = volc.decode_frame(raw)
    assert frame.flags == volc.FLAG_NEGATIVE_SEQUENCE
    assert frame.sequence == -3


def test_decode_frame_last_no_sequence_has_no_seq_field():
    """flags=0b0010（最后一包无 seq）：帧体不含 sequence 字段"""
    header = bytes([0x11, (volc.MSG_AUDIO_ONLY_REQUEST << 4) | 0b0010, 0x00, 0x00])
    raw = header + (2).to_bytes(4, "big") + b"hi"
    frame = volc.decode_frame(raw)
    assert frame.sequence is None
    assert frame.payload == b"hi"


def test_decode_error_frame_and_parse_payload():
    """错误帧 payload：int32 大端错误码 + uint32 大端消息长度 + UTF-8 消息"""
    message = b"invalid access key"
    body = (4001).to_bytes(4, "big", signed=True) + len(message).to_bytes(4, "big") + message
    raw = volc.encode_frame(
        volc.MSG_ERROR, volc.FLAG_NO_SEQUENCE, body, compression=volc.COMPRESSION_NONE
    )
    frame = volc.decode_frame(raw)
    assert frame.message_type == volc.MSG_ERROR
    assert volc.parse_error_payload(frame.payload) == (4001, "invalid access key")


def test_decode_tolerates_missing_gzip_magic():
    """压缩位标了 gzip 但实为裸音频：按原样返回，不应抛 BadGzipFile"""
    header = bytes([0x11, (volc.MSG_AUDIO_ONLY_SERVER_RESPONSE << 4) | 0b0000, 0x01, 0x00])
    raw_audio = b"\x01\x02\x03\x04"
    raw = header + len(raw_audio).to_bytes(4, "big") + raw_audio
    frame = volc.decode_frame(raw)
    assert frame.payload == raw_audio


def test_decode_frame_too_short_raises():
    with pytest.raises(ValueError, match="不足 4 字节"):
        volc.decode_frame(b"\x11\x10")


# ===== ASR 结果归一化 =====


def test_normalize_asr_prefers_result_text():
    payload = json.dumps(
        {
            "code": 1000,
            "result": {
                "text": "你好世界",
                "utterances": [
                    {"text": "你好", "definite": True},
                    {"text": "世界", "definite": True},
                ],
            },
        }
    ).encode()
    out = json.loads(volc.normalize_asr_payload(payload))
    assert out == {"text": "你好世界", "is_final": True}


def test_normalize_asr_joins_utterances_when_text_empty():
    payload = json.dumps(
        {
            "code": 1000,
            "result": {
                "text": "",
                "utterances": [
                    {"text": "你好", "definite": False},
                    {"text": "世界", "definite": True},
                ],
            },
        }
    ).encode()
    out = json.loads(volc.normalize_asr_payload(payload))
    assert out == {"text": "你好世界", "is_final": True}


def test_normalize_asr_is_final_false_when_last_not_definite():
    payload = json.dumps(
        {
            "code": 1000,
            "result": {"text": "你好", "utterances": [{"text": "你好", "definite": False}]},
        }
    ).encode()
    out = json.loads(volc.normalize_asr_payload(payload))
    assert out == {"text": "你好", "is_final": False}


def test_normalize_asr_error_code_raises(caplog):
    payload = json.dumps({"code": 1002, "message": "engine busy"}).encode()
    with pytest.raises(BusinessException) as exc:
        volc.normalize_asr_payload(payload)
    assert exc.value.message == "语音引擎服务错误：engine busy"
    assert "engine busy" in caplog.text  # 错误原文只进日志


# ===== 鉴权头与请求体 =====


def test_asr_headers_classic_mode():
    """经典口径：APP ID 必填 + Key 写 X-Api-Access-Key + resource-id 取自 asr_resource_id"""
    assert volc._asr_headers(_Provider(), "plain-key") == {
        "X-Api-App-Key": "appid-123",
        "X-Api-Access-Key": "plain-key",
        "X-Api-Resource-Id": "volc.seedasr.sauc.duration",
    }


def test_asr_headers_missing_resource_id_raises():
    class _Bad(_Provider):
        default_headers = {}

    with pytest.raises(BusinessException) as exc:
        volc._asr_headers(_Bad(), "k")
    assert "default_headers.asr_resource_id" in exc.value.message


def test_asr_headers_classic_mode_requires_app_id():
    class _NoAppId(_Provider):
        default_headers = {"asr_resource_id": "volc.seedasr.sauc.duration"}

    with pytest.raises(BusinessException) as exc:
        volc._asr_headers(_NoAppId(), "k")
    assert "default_headers.X-Api-App-Key" in exc.value.message


def test_asr_headers_single_key_app_id_optional():
    """单密钥口径：APP ID 变可选，Key 写 auth_header 指定头名"""

    class _NoAppId(_SingleKeyProvider):
        default_headers = {
            "asr_resource_id": "volc.seedasr.sauc.duration",
            "auth_header": "X-Api-Key",
        }

    assert volc._asr_headers(_NoAppId(), "secret") == {
        "X-Api-Key": "secret",
        "X-Api-Resource-Id": "volc.seedasr.sauc.duration",
    }
    assert volc._asr_headers(_SingleKeyProvider(), "secret") == {
        "X-Api-Key": "secret",
        "X-Api-App-Key": "appid-123",
        "X-Api-Resource-Id": "volc.seedasr.sauc.duration",
    }


def test_tts_headers_classic_uses_semicolon():
    assert volc._tts_headers(_Provider(), "secret") == {"Authorization": "Bearer; secret"}


def test_tts_headers_single_key_uses_v3_headers():
    headers = volc._tts_headers(_SingleKeyProvider(), "secret")
    assert headers["X-Api-Key"] == "secret"
    assert headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert headers["X-Api-Request-Id"]
    assert "Authorization" not in headers


def test_tts_v1_request_json_structure():
    frame = volc._tts_v1_request(_Provider(), "tok", "你好", "BV700_streaming", 1.0, "mp3", 24000)
    decoded = volc.decode_frame(frame)
    assert decoded.message_type == volc.MSG_FULL_CLIENT_REQUEST
    req = json.loads(decoded.payload)
    assert req["app"] == {"appid": "appid-123", "token": "tok", "cluster": "volcano_tts"}
    assert req["audio"] == {
        "voice_type": "BV700_streaming",
        "encoding": "mp3",
        "speed_ratio": 1.0,
        "sample_rate": 24000,
    }
    assert req["request"]["text"] == "你好"
    assert req["request"]["operation"] == "query"
    assert req["request"]["text_type"] == "plain"
    assert req["request"]["reqid"]


def test_tts_v1_request_cluster_override():
    class _WithCluster(_Provider):
        default_headers = {"X-Api-App-Key": "appid-123", "cluster": "my_cluster"}

    frame = volc._tts_v1_request(_WithCluster(), "tok", "hi", "v", 1.0, "mp3", 16000)
    assert json.loads(volc.decode_frame(frame).payload)["app"]["cluster"] == "my_cluster"


def test_tts_v3_request_header_and_body():
    """v3 请求帧：头 [0x11,0x10,0x10,0x00]（无 event/seq、JSON 不压缩），正文只含 req_params"""
    frame = volc._tts_v3_request("你好", "zh_female_vv_uranus_bigtts", 1.0, "mp3", 24000)
    assert frame[:4] == bytes([0x11, 0x10, 0x10, 0x00])
    decoded = volc.decode_frame(frame)
    assert decoded.message_type == volc.MSG_FULL_CLIENT_REQUEST
    assert decoded.compression == volc.COMPRESSION_NONE
    body = json.loads(decoded.payload)
    assert "app" not in body  # 经典口径的 app{appid,token,cluster} 已弃用
    assert body["req_params"] == {
        "text": "你好",
        "speaker": "zh_female_vv_uranus_bigtts",
        "audio_params": {"format": "mp3", "sample_rate": 24000, "speech_rate": 0},
    }


@pytest.mark.parametrize(
    ("speed", "expected"),
    [(0.5, -50), (1.0, 0), (1.2, 20), (2.0, 100), (3.0, 100), (0.1, -50)],
)
def test_speech_rate_maps_openai_speed(speed, expected):
    assert volc._speech_rate(speed) == expected


def test_asr_first_frame_json_structure():
    frame = volc._asr_first_frame("bigmodel", "uid-1")
    decoded = volc.decode_frame(frame)
    assert decoded.message_type == volc.MSG_FULL_CLIENT_REQUEST
    assert decoded.flags == volc.FLAG_NO_SEQUENCE
    req = json.loads(decoded.payload)
    assert req["user"] == {"uid": "uid-1"}
    assert req["audio"] == {
        "format": "pcm",
        "codec": "raw",
        "rate": 16000,
        "bits": 16,
        "channel": 1,
    }
    assert req["request"]["model_name"] == "bigmodel"
    assert req["request"]["enable_itn"] is True
    assert req["request"]["enable_punc"] is True
    assert req["request"]["show_utterances"] is True


# ===== URL 组装 =====


def test_ws_url_uses_scheme_and_host_only():
    class _WithPath:
        api_base_url = "https://openspeech.bytedance.com/api/v3/sauc/bigmodel"

    assert volc._ws_url(_WithPath(), volc._ASR_PATH) == (
        "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
    )

    class _Http:
        api_base_url = "http://host:8080/prefix"

    assert volc._ws_url(_Http(), "/p") == "ws://host:8080/p"


# ===== 会话帧翻译（send/recv） =====


async def test_session_send_audio_and_eos_frames():
    ws = _FakeWs()
    session = volc.AsrStreamSession(ws)
    await session.send_audio(b"chunk1")
    await session.send_audio(b"chunk2")
    await session.send_eos()

    assert len(ws.sent) == 3
    f0 = volc.decode_frame(ws.sent[0])
    assert f0.message_type == volc.MSG_AUDIO_ONLY_REQUEST
    assert f0.sequence == 1
    assert f0.payload == b"chunk1"
    f1 = volc.decode_frame(ws.sent[1])
    assert f1.sequence == 2
    assert f1.payload == b"chunk2"
    eos = volc.decode_frame(ws.sent[2])
    assert eos.flags == volc.FLAG_NEGATIVE_SEQUENCE
    assert eos.sequence == -2  # 末帧序号取负


async def test_session_recv_messages_normalizes_and_stops_on_negative_seq():
    first = volc.encode_frame(
        volc.MSG_FULL_SERVER_RESPONSE,
        volc.FLAG_POSITIVE_SEQUENCE,
        json.dumps(
            {
                "code": 1000,
                "result": {"text": "你好", "utterances": [{"text": "你好", "definite": False}]},
            }
        ).encode(),
        sequence=1,
        serialization=volc.SERIALIZATION_JSON,
        compression=volc.COMPRESSION_GZIP,
    )
    last = volc.encode_frame(
        volc.MSG_FULL_SERVER_RESPONSE,
        volc.FLAG_NEGATIVE_SEQUENCE,
        json.dumps(
            {
                "code": 1000,
                "result": {
                    "text": "你好世界",
                    "utterances": [{"text": "你好世界", "definite": True}],
                },
            }
        ).encode(),
        sequence=-2,
        serialization=volc.SERIALIZATION_JSON,
        compression=volc.COMPRESSION_GZIP,
    )
    tail = volc.encode_frame(
        volc.MSG_FULL_SERVER_RESPONSE,
        volc.FLAG_POSITIVE_SEQUENCE,
        json.dumps({"code": 1000, "result": {"text": "不应读到"}}).encode(),
        sequence=3,
        serialization=volc.SERIALIZATION_JSON,
    )
    session = volc.AsrStreamSession(_FakeRecvWs([first, last, tail]))

    messages = [json.loads(raw) async for raw in session.recv_messages()]
    assert messages[0] == {"text": "你好", "is_final": False}
    assert messages[-1] == {"text": "你好世界", "is_final": True}
    assert len(messages) == 2  # 负 seq 后停止，tail 不再产出


async def test_session_recv_messages_error_frame_raises():
    message = b"bad key"
    body = (4001).to_bytes(4, "big", signed=True) + len(message).to_bytes(4, "big") + message
    error = volc.encode_frame(
        volc.MSG_ERROR, volc.FLAG_NO_SEQUENCE, body, compression=volc.COMPRESSION_NONE
    )
    session = volc.AsrStreamSession(_FakeRecvWs([error]))

    with pytest.raises(BusinessException) as exc:
        async for _ in session.recv_messages():
            pass
    assert exc.value.message == "语音引擎服务错误：bad key"


async def test_collect_tts_audio_aggregates_and_ignores_meta():
    frames = [
        volc.encode_frame(
            volc.MSG_AUDIO_ONLY_SERVER_RESPONSE,
            volc.FLAG_POSITIVE_SEQUENCE,
            b"AAA",
            sequence=1,
            compression=volc.COMPRESSION_NONE,
        ),
        volc.encode_frame(  # 0b1001 元信息帧应被忽略
            volc.MSG_FULL_SERVER_RESPONSE,
            volc.FLAG_POSITIVE_SEQUENCE,
            json.dumps({"code": 1000}).encode(),
            sequence=2,
            serialization=volc.SERIALIZATION_JSON,
        ),
        volc.encode_frame(
            volc.MSG_AUDIO_ONLY_SERVER_RESPONSE,
            volc.FLAG_POSITIVE_SEQUENCE,
            b"BBB",
            sequence=3,
            compression=volc.COMPRESSION_NONE,
        ),
        volc.encode_frame(  # 末帧：负 seq
            volc.MSG_AUDIO_ONLY_SERVER_RESPONSE,
            volc.FLAG_NEGATIVE_SEQUENCE,
            b"CCC",
            sequence=-4,
            compression=volc.COMPRESSION_NONE,
        ),
        volc.encode_frame(  # 负 seq 后不应再读取
            volc.MSG_AUDIO_ONLY_SERVER_RESPONSE,
            volc.FLAG_POSITIVE_SEQUENCE,
            b"DDD",
            sequence=5,
            compression=volc.COMPRESSION_NONE,
        ),
    ]
    audio = await volc._collect_tts_audio(_FakeRecvWs(frames))
    assert audio == b"AAABBBCCC"


# ===== v3 单向流式帧（单密钥口径 TTS）=====


def _v3_frame(message_type: int, event: int, payload: bytes, *, session_id: str = "sid") -> bytes:
    """构造 v3 下行帧：4 字节头(WithEvent) + event(int32) + session_id(长度前缀) + payload"""
    header = bytes(
        [
            0x11,
            (message_type << 4) | volc.FLAG_WITH_EVENT,
            (volc.SERIALIZATION_JSON << 4) | volc.COMPRESSION_NONE,
            0x00,
        ]
    )
    sid = session_id.encode()
    return (
        header
        + event.to_bytes(4, "big", signed=True)
        + len(sid).to_bytes(4, "big")
        + sid
        + len(payload).to_bytes(4, "big")
        + payload
    )


def test_decode_v3_frame_event_and_session_id():
    parsed = volc.decode_v3_frame(
        _v3_frame(volc.MSG_AUDIO_ONLY_SERVER_RESPONSE, volc.EVENT_TTS_RESPONSE, b"AUDIO")
    )
    assert parsed.message_type == volc.MSG_AUDIO_ONLY_SERVER_RESPONSE
    assert parsed.event == volc.EVENT_TTS_RESPONSE
    assert parsed.payload == b"AUDIO"


def test_decode_v3_frame_without_event_field():
    """无 event 的帧（如错误帧）：头后直接接 payload_size + payload"""
    body = (4001).to_bytes(4, "big", signed=True) + len(b"bad").to_bytes(4, "big") + b"bad"
    raw = bytes([0x11, (volc.MSG_ERROR << 4) | volc.FLAG_NO_SEQUENCE, 0x00, 0x00])
    raw += len(body).to_bytes(4, "big") + body
    parsed = volc.decode_v3_frame(raw)
    assert parsed.message_type == volc.MSG_ERROR
    assert volc.parse_error_payload(parsed.payload) == (4001, "bad")


async def test_collect_v3_tts_audio_aggregates_until_session_finished():
    frames = [
        _v3_frame(volc.MSG_FULL_SERVER_RESPONSE, 350, json.dumps({"text": ""}).encode()),
        _v3_frame(volc.MSG_AUDIO_ONLY_SERVER_RESPONSE, volc.EVENT_TTS_RESPONSE, b"AAA"),
        _v3_frame(volc.MSG_AUDIO_ONLY_SERVER_RESPONSE, volc.EVENT_TTS_RESPONSE, b"BBB"),
        _v3_frame(
            volc.MSG_FULL_SERVER_RESPONSE,
            volc.EVENT_SESSION_FINISHED,
            json.dumps({"usage": {"text_words": 2}}).encode(),
        ),
        _v3_frame(volc.MSG_AUDIO_ONLY_SERVER_RESPONSE, volc.EVENT_TTS_RESPONSE, b"CCC"),
    ]
    assert await volc._collect_v3_tts_audio(_FakeRecvWs(frames)) == b"AAABBB"


async def test_collect_v3_tts_audio_session_failed_raises(caplog):
    frames = [
        _v3_frame(volc.MSG_FULL_SERVER_RESPONSE, volc.EVENT_SESSION_FAILED, b'{"message":"quota"}')
    ]
    with pytest.raises(BusinessException):
        await volc._collect_v3_tts_audio(_FakeRecvWs(frames))
    assert "quota" in caplog.text  # 错误原文只进日志


async def test_collect_v3_tts_audio_error_frame_raises():
    message = b"bad key"
    body = (4001).to_bytes(4, "big", signed=True) + len(message).to_bytes(4, "big") + message
    raw = bytes([0x11, (volc.MSG_ERROR << 4) | volc.FLAG_NO_SEQUENCE, 0x00, 0x00])
    raw += len(body).to_bytes(4, "big") + body
    with pytest.raises(BusinessException) as exc:
        await volc._collect_v3_tts_audio(_FakeRecvWs([raw]))
    assert exc.value.message == "语音引擎服务错误：bad key"


# ===== 状态 =====


async def test_status_contract_fields():
    result = await volc.status(object())
    assert result == {
        "engine_status": "online",
        "stream_model_loaded": False,
        "offline_model_loaded": False,
        "voice_model_loaded": False,
    }
