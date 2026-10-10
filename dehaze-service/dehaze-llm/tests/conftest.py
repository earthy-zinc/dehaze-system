"""dehaze-llm 单测共享夹具：以桩替换 llama_cpp，不依赖真实 GGUF 模型与 GPU。"""

import asyncio
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app import inference  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.routers import openai_compat  # noqa: E402

INTERNAL_KEY = "test-internal-key"
YES_ID, NO_ID, N_VOCAB = 1, 2, 8


@pytest.fixture(autouse=True)
def _auth_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", INTERNAL_KEY)


@pytest.fixture(autouse=True)
def _fresh_infer_lock(monkeypatch):
    """推理锁是模块级单例，而 pytest 每个用例一个事件循环，需按用例重建。

    生产只有 uvicorn 单事件循环，无此问题。
    """
    monkeypatch.setattr(inference, "_infer_lock", asyncio.Lock())


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {INTERNAL_KEY}"}


class FakeLlama:
    """llama_cpp.Llama 桩：支持流式/非流式对话、向量化、重排低层 eval 接口。"""

    def __init__(
        self,
        tokens: tuple[str, ...] = ("你好", "世界"),
        stream_delay: float = 0.0,
        rerank_table: dict[str, tuple[float, float]] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        self.tokens = list(tokens)
        self.stream_delay = stream_delay
        self.rerank_table = rerank_table or {}
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.prompts: list[str] = []
        self.generated = 0
        self.active = 0
        self.max_concurrent = 0
        self.logits = (0.0, 0.0)
        self.ctx = object()

    # ---- 分词（对话用量计数与重排打分共用入口）----
    def tokenize(self, text, add_bos: bool = True, special: bool = False):
        if text == b"yes":
            return [YES_ID]
        if text == b"no":
            return [NO_ID]
        prompt = text.decode("utf-8", "ignore")
        self.prompts.append(prompt)
        self.logits = next(
            ((yes, no) for key, (yes, no) in self.rerank_table.items() if key in prompt),
            (0.0, 0.0),
        )
        return list(text)  # 逐字节近似 token 计数（超长截断判定用）

    # ---- 对话 ----
    def create_completion(self, prompt, max_tokens=1024, temperature=0.7, stop=None, stream=False):
        if stream:
            return self._stream()
        with self._busy():
            completion = len(self.tokens)
            return {
                "choices": [{"text": "".join(self.tokens), "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": completion,
                    "total_tokens": len(prompt) + completion,
                },
            }

    def _stream(self):
        with self._busy():
            for token in self.tokens:
                if self.stream_delay:
                    time.sleep(self.stream_delay)
                self.generated += 1
                yield {"choices": [{"text": token, "finish_reason": None}]}

    # ---- 向量 ----
    def create_embedding(self, text):
        return {"data": [{"embedding": self.embedding}], "usage": {"prompt_tokens": len(text)}}

    # ---- 重排低层接口 ----
    def n_vocab(self) -> int:
        return N_VOCAB

    def reset(self) -> None:
        pass

    def eval(self, tokens) -> None:
        pass

    @contextmanager
    def _busy(self):
        self.active += 1
        self.max_concurrent = max(self.max_concurrent, self.active)
        try:
            yield
        finally:
            self.active -= 1


def _fake_last_token_logits(llm: FakeLlama) -> list[float]:
    """替代 llama_cpp 低层 logits 读取：把桩记录的 (yes, no) logit 填入词表数组。"""
    logits = [0.0] * N_VOCAB
    logits[YES_ID], logits[NO_ID] = llm.logits
    return logits


@pytest.fixture
def chat_llm(monkeypatch) -> FakeLlama:
    llm = FakeLlama()
    monkeypatch.setattr(inference._chat, "_llm", llm)
    return llm


@pytest.fixture
def embed_llm(monkeypatch) -> FakeLlama:
    llm = FakeLlama(embedding=[0.5, 0.25])
    monkeypatch.setattr(inference._embed, "_llm", llm)
    return llm


@pytest.fixture
def rerank_llm(monkeypatch) -> FakeLlama:
    llm = FakeLlama(
        rerank_table={"高相关": (12.0, -6.0), "弱相关": (-2.0, -1.5), "无关": (-4.0, 2.0)}
    )
    monkeypatch.setattr(inference._rerank, "_llm", llm)
    monkeypatch.setattr(inference, "_last_token_logits", _fake_last_token_logits)
    monkeypatch.setattr(openai_compat, "is_rerank_downloaded", lambda: True)
    return llm


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app, auth_headers):
    """带正确 Bearer 的客户端（验证业务契约用）。"""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=auth_headers
    ) as c:
        yield c


@pytest.fixture
async def raw_client(app):
    """不带任何鉴权头的客户端（验证 401 语义用）。"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
