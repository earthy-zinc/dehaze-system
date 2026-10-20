"""探针语义：/health 恒 200 且纯只读，/ready 只在预热完成后 200。"""

from app import inference


def _forbid_load(slot, monkeypatch, message: str) -> None:
    def _boom() -> str:
        raise AssertionError(message)

    monkeypatch.setattr(slot, "_ensure", _boom)


async def test_health_is_read_only(client, monkeypatch):
    monkeypatch.setattr(inference._chat, "_llm", None)
    _forbid_load(inference._chat, monkeypatch, "/health 不得触发模型加载")
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "qwen3-0.6b", "loaded": False}


async def test_ready_503_before_prewarm_and_read_only(client, monkeypatch):
    for slot in (inference._chat, inference._embed):
        monkeypatch.setattr(slot, "_llm", None)
        _forbid_load(slot, monkeypatch, "/ready 不得触发下载或加载")
    resp = await client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["components"] == {"chat": "DOWN", "embedding": "DOWN", "rerank": "DOWN"}


async def test_ready_200_when_chat_and_embedding_loaded(client, monkeypatch):
    monkeypatch.setattr(inference._chat, "_llm", object())
    monkeypatch.setattr(inference._embed, "_llm", object())
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    # 重排是可选能力：不参与就绪判定
    assert resp.json()["components"]["rerank"] == "DOWN"


async def test_prewarm_retries_until_loaded(monkeypatch):
    attempts = {"chat": 0}

    class _Slot:
        def __init__(self, name: str) -> None:
            self.name = name
            self.loaded = False

        def load(self) -> None:
            if self.name == "chat":
                attempts["chat"] += 1
                if attempts["chat"] == 1:
                    raise RuntimeError("下载失败")
            self.loaded = True

    monkeypatch.setattr(inference, "_PREWARM_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(inference, "_chat", _Slot("chat"))
    monkeypatch.setattr(inference, "_embed", _Slot("embedding"))

    await inference.prewarm()

    assert attempts["chat"] == 2  # 首次失败后重试成功
