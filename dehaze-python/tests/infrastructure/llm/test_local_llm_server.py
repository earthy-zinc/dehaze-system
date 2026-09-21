"""本地 /v1/rerank 端点与打分逻辑测试（mock llama_cpp，不依赖真模型）。

Qwen3-Reranker 官方打分方式：空思考块后末位 logits 的 yes/no 二选一 softmax，
此处锁定 prompt 模板格式、logit 换算边界、端点响应契约（降序 + top_n）。
"""

import math

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.infrastructure.llm.local import local_llm_server
from app.infrastructure.llm.local.local_llm_server import (
    RerankRequest,
    _build_rerank_prompt,
    _score_pair,
    _yes_no_score,
)

YES_ID, NO_ID, N_VOCAB = 1, 2, 8


class _FakeLlm:
    """llama 桩：按 prompt 命中关键词在末位 logits 里填 yes/no logit 值。"""

    def __init__(self, table: dict[str, tuple[float, float]]):
        self.table = table
        self.prompts: list[str] = []
        self._logits = (0.0, 0.0)

    def tokenize(self, text, add_bos: bool = True, special: bool = False):
        if text == b"yes":
            return [YES_ID]
        if text == b"no":
            return [NO_ID]
        s = text.decode("utf-8", "ignore")
        self.prompts.append(s)
        self._logits = next(((y, n) for key, (y, n) in self.table.items() if key in s), (0.0, 0.0))
        return list(text)  # 逐字节近似 token 计数（超长截断测试用）

    def reset(self) -> None:
        pass

    def eval(self, tokens) -> None:
        pass

    def n_vocab(self) -> int:
        return N_VOCAB


@pytest.fixture
def fake_logits(monkeypatch):
    """把 _last_token_logits 指向桩记录的 (logit_yes, logit_no)。"""

    def _last(llm):
        arr = np.zeros(N_VOCAB, dtype=np.float32)
        arr[YES_ID], arr[NO_ID] = llm._logits
        return arr

    monkeypatch.setattr(local_llm_server, "_last_token_logits", _last)


# ===== prompt 模板 =====


def test_build_rerank_prompt_matches_official_format():
    prompt = _build_rerank_prompt("机器学习", "监督学习简介")
    # Qwen3-Reranker 官方 system/user 模板原文
    assert (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n"
    ) in prompt
    assert (
        "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n"
        "<Query>: 机器学习\n\n<Document>: 监督学习简介"
    ) in prompt
    # 官方 assistant 后缀必须含空思考块（占住 <think> 生成位，末位直接输出 yes/no）
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "/no_think" not in prompt


# ===== logit 换算 =====


def test_yes_no_score_softmax_pairwise():
    assert _yes_no_score(0.0, 0.0) == pytest.approx(0.5)
    # logit 差 1 → sigmoid(1)
    assert _yes_no_score(-1.0, -2.0) == pytest.approx(1 / (1 + math.exp(-1)))
    # 数值稳定：大 logit 不上溢
    assert _yes_no_score(-10000.0, -10001.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert _yes_no_score(19.0, -11.0) == pytest.approx(1.0)


def test_yes_no_score_missing_logprob_boundaries():
    # 缺 yes → 0 分；缺 no → 满分；双缺（打分不可得）→ 最不相关 0 分
    assert _yes_no_score(-math.inf, -0.5) == 0.0
    assert _yes_no_score(-0.5, -math.inf) == 1.0
    assert _yes_no_score(-math.inf, -math.inf) == 0.0


# ===== 单对打分（mock llama）=====


def test_score_pair_uses_last_token_logits(fake_logits):
    llm = _FakeLlm({"机器学习": (9.2, -3.0)})
    score = _score_pair(llm, "机器学习", "监督学习详解")
    assert score == pytest.approx(1 / (1 + math.exp(-(9.2 - -3.0))))
    # 官方格式：prompt 含空思考块后缀
    assert llm.prompts[0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_score_pair_truncates_overlong_doc(fake_logits):
    llm = _FakeLlm({})
    doc = "相关" * 20000  # 远超 LOCAL_LLM_CTX_SIZE，触发截断
    _score_pair(llm, "q", doc)
    # prompts[-1] 为截断后重新分词的 prompt（prompts[0] 是截断前的首次分词）
    sent = llm.prompts[-1]
    doc_part = sent.split("<Document>: ")[1].split("<|im_end|>")[0]
    assert len(doc_part) < len(doc)
    assert doc_part.startswith("相关")  # 保留头部


# ===== /v1/rerank 端点契约 =====


def _patch_ready(monkeypatch, fake_llm, downloaded: bool = True):
    monkeypatch.setattr(local_llm_server, "is_rerank_downloaded", lambda: downloaded)
    monkeypatch.setattr(local_llm_server, "_rerank_llm", fake_llm)
    monkeypatch.setattr(local_llm_server, "_trigger_rerank_download", lambda: None)


async def test_rerank_endpoint_sorted_desc_and_top_n(monkeypatch, fake_logits):
    llm = _FakeLlm(
        {
            "高相关": (12.0, -6.0),
            "弱相关": (-2.0, -1.5),
            "无关": (-4.0, 2.0),
        }
    )
    _patch_ready(monkeypatch, llm)
    docs = ["弱相关内容", "高相关内容", "无关内容"]
    async with AsyncClient(
        transport=ASGITransport(app=local_llm_server.app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/rerank",
            json={"model": "qwen3-reranker-0.6b", "query": "机器学习", "documents": docs},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "qwen3-reranker-0.6b"
    results = body["results"]
    assert [r["index"] for r in results] == [1, 0, 2]  # 按分数降序
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1 / (1 + math.exp(-18.0)))
    assert scores[1] == pytest.approx(1 / (1 + math.exp(0.5)))
    assert scores[2] == pytest.approx(1 / (1 + math.exp(6.0)))
    # top_n 截取
    async with AsyncClient(
        transport=ASGITransport(app=local_llm_server.app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/rerank",
            json={
                "model": "qwen3-reranker-0.6b",
                "query": "机器学习",
                "documents": docs,
                "top_n": 2,
            },
        )
    assert resp.status_code == 200
    assert [r["index"] for r in resp.json()["results"]] == [1, 0]


async def test_rerank_endpoint_empty_documents(monkeypatch):
    _patch_ready(monkeypatch, _FakeLlm({}))
    async with AsyncClient(
        transport=ASGITransport(app=local_llm_server.app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/rerank", json={"model": "m", "query": "q", "documents": []})
    assert resp.status_code == 200
    assert resp.json() == {"model": "m", "results": []}


async def test_rerank_endpoint_503_when_model_missing(monkeypatch):
    _patch_ready(monkeypatch, _FakeLlm({}), downloaded=False)
    triggered = []
    monkeypatch.setattr(local_llm_server, "_trigger_rerank_download", lambda: triggered.append(1))
    async with AsyncClient(
        transport=ASGITransport(app=local_llm_server.app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": ["d"]},
        )
    assert resp.status_code == 503
    assert "下载" in resp.json()["detail"]
    assert triggered == [1]  # 503 的同时已触发后台下载


async def test_rerank_request_schema_defaults():
    req = RerankRequest(query="q", documents=["d"])
    assert req.top_n is None
