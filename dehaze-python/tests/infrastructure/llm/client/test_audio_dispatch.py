"""语音协议分派：值域内每个协议族都必须分派到自己的适配器。

值域声明处为 `sys_ai_provider.protocol_type` 列注释 / 需求规格「协议类型」/ 管理端协议下拉。
新增协议族须同步 `VOICE_PROTOCOL_TYPES` 与 `get_audio_client`，否则调用仍走内置引擎。
"""

from app.infrastructure.llm.client import (
    dashscope_audio_client,
    openai_audio_client,
    volc_audio_client,
)
from app.infrastructure.llm.client.audio_dispatch import (
    REALTIME_VOICE_PROTOCOLS,
    VOICE_PROTOCOL_TYPES,
    get_audio_client,
)


def test_voice_protocols_dispatch_to_own_client():
    expected = {
        "openai_audio": openai_audio_client,
        "dashscope_realtime": dashscope_audio_client,
        "volc_realtime": volc_audio_client,
    }
    assert set(expected) == VOICE_PROTOCOL_TYPES
    # 实时协议族（无 OpenAI 兼容 HTTP 端点，模型可用性测试据此不发 HTTP 探测）
    assert {"dashscope_realtime", "volc_realtime"} == REALTIME_VOICE_PROTOCOLS

    for protocol_type, client in expected.items():
        assert get_audio_client(protocol_type) is client


def test_unknown_protocol_falls_back_to_builtin_audio():
    assert get_audio_client("not_a_voice_protocol") is openai_audio_client
