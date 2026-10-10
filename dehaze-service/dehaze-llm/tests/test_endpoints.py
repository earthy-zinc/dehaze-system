"""能力端点契约：对话（非流式）、向量、重排、能力发现、错误模型。"""

from app import inference
from app.config import settings
from app.model_store import (
    CHAT_MODEL_FILE,
    CHAT_MODEL_SIZE,
    EMBEDDING_MODEL_FILE,
    EMBEDDING_MODEL_SIZE,
    RERANK_MODEL_FILE,
    RERANK_MODEL_SIZE,
)
from app.routers import openai_compat


def _write_sparse(path, size: int) -> None:
    """生成指定大小的稀疏文件（只校验字节数，不实际占盘）。"""
    with path.open("wb") as f:
        f.truncate(size)


# ===== 对话 =====


def test_build_prompt_injects_no_think_and_chatml():
    prompt = inference.build_prompt([("system", "你是助手"), ("user", "你好")])
    assert prompt.startswith("<|im_start|>system\n你是助手 /no_think<|im_end|>\n")
    assert "<|im_start|>user\n你好<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_clean_content_strips_template_residue():
    assert inference._clean_content("正文<|im_start|>残留") == "正文"
    assert inference._clean_content("  正文<|im_end|>") == "正文"


async def test_chat_non_stream_shape(client, chat_llm):
    chat_llm.tokens = ("你", "好")
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "qwen3-0.6b"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好"},
            "finish_reason": "stop",
        }
    ]
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}


async def test_chat_ignores_openai_tool_fields(client, chat_llm):
    """本地小模型不解析工具协议，但必须保证接口形状兼容不报错。"""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "x"}}],
            "tool_choice": "auto",
        },
    )
    assert resp.status_code == 200


# ===== 向量 =====


async def test_embeddings_single(client, embed_llm):
    resp = await client.post(
        "/v1/embeddings", json={"model": "qwen3-embedding-0.6b", "input": "文本"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "qwen3-embedding-0.6b"
    assert body["data"] == [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}]
    assert body["usage"] == {"prompt_tokens": 2, "total_tokens": 2}


async def test_embeddings_batch_order_and_token_sum(client, embed_llm):
    resp = await client.post("/v1/embeddings", json={"model": "m", "input": ["a", "bbb"]})
    assert resp.status_code == 200
    body = resp.json()
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["usage"]["prompt_tokens"] == 4  # 1 + 3


# ===== 重排 =====


async def test_rerank_sorted_desc_and_top_n(client, rerank_llm):
    docs = ["弱相关内容", "高相关内容", "无关内容"]
    resp = await client.post(
        "/v1/rerank",
        json={"model": "qwen3-reranker-0.6b", "query": "机器学习", "documents": docs},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "qwen3-reranker-0.6b"
    assert [r["index"] for r in body["results"]] == [1, 0, 2]
    scores = [r["relevance_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)

    resp = await client.post(
        "/v1/rerank",
        json={"model": "m", "query": "机器学习", "documents": docs, "top_n": 2},
    )
    assert [r["index"] for r in resp.json()["results"]] == [1, 0]


async def test_rerank_empty_documents(client, rerank_llm):
    resp = await client.post("/v1/rerank", json={"model": "m", "query": "q", "documents": []})
    assert resp.status_code == 200
    assert resp.json() == {"model": "m", "results": []}


async def test_rerank_503_when_model_missing(client, monkeypatch):
    triggered = []
    monkeypatch.setattr(openai_compat, "is_rerank_downloaded", lambda: False)
    monkeypatch.setattr(inference, "trigger_rerank_download", lambda: triggered.append(1))
    resp = await client.post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["d"]})
    assert resp.status_code == 503
    assert resp.json()["code"] == "MODEL_NOT_READY"
    assert triggered == [1]  # 503 的同时已触发后台下载


def test_yes_no_score_boundaries():
    import math

    assert inference._yes_no_score(0.0, 0.0) == 0.5
    assert inference._yes_no_score(-1.0, -2.0) == 1 / (1 + math.exp(-1))
    assert inference._yes_no_score(-math.inf, -0.5) == 0.0
    assert inference._yes_no_score(-0.5, -math.inf) == 1.0
    assert inference._yes_no_score(-math.inf, -math.inf) == 0.0


def test_rerank_prompt_uses_official_format():
    prompt = inference._build_rerank_prompt("机器学习", "监督学习简介")
    assert (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    ) in prompt
    assert (
        "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n\n"
        "<Query>: 机器学习\n\n<Document>: 监督学习简介"
    ) in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "/no_think" not in prompt


# ===== 能力发现 =====


async def test_models_declares_three_types(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path))
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [m["id"] for m in data] == [
        "qwen3-0.6b",
        "qwen3-embedding-0.6b",
        "qwen3-reranker-0.6b",
    ]
    assert [m["type"] for m in data] == ["chat", "embedding", "rerank"]
    for card in data:
        assert set(card) == {
            "id",
            "type",
            "available",
            "unavailable_reason",
            "capabilities",
            "limits",
        }
        assert card["available"] is False
        assert card["unavailable_reason"]
    assert data[0]["capabilities"] == {"streaming": True}
    assert data[1]["capabilities"] == {"dimensions": 1024}
    assert data[2]["capabilities"] == {"top_n": True}
    assert data[0]["limits"]["context_tokens"] == settings.CTX_SIZE
    assert all(card["limits"]["max_concurrency"] == 1 for card in data)


async def test_models_available_when_weights_present(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "MODEL_CACHE_DIR", str(tmp_path))
    _write_sparse(tmp_path / CHAT_MODEL_FILE, CHAT_MODEL_SIZE)
    _write_sparse(tmp_path / EMBEDDING_MODEL_FILE, EMBEDDING_MODEL_SIZE)
    _write_sparse(tmp_path / RERANK_MODEL_FILE, RERANK_MODEL_SIZE)
    data = (await client.get("/v1/models")).json()["data"]
    assert all(card["available"] is True for card in data)
    assert all(card["unavailable_reason"] is None for card in data)


async def test_model_detail_and_404(client):
    resp = await client.get("/v1/models/qwen3-reranker-0.6b")
    assert resp.status_code == 200
    assert resp.json()["type"] == "rerank"
    resp = await client.get("/v1/models/no-such-model")
    assert resp.status_code == 404
    assert resp.json()["code"] == "MODEL_NOT_FOUND"


# ===== 错误模型 =====


async def test_invalid_request_returns_400(client):
    resp = await client.post("/v1/chat/completions", json={"stream": False})
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_REQUEST"


async def test_metrics_exposed_with_auth(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200  # 不得重定向（Prometheus 抓取直取 200）
    assert "# HELP" in resp.text


async def test_inference_metrics_recorded(client, embed_llm):
    """推理打点：向量请求后 /metrics 可见 embedding 维度的计数与耗时。"""
    await client.post("/v1/embeddings", json={"model": "m", "input": "文本"})
    text = (await client.get("/metrics")).text
    assert 'dehaze_inference_requests_total{status="success",type="embedding"}' in text
    assert 'dehaze_inference_duration_seconds_count{type="embedding"}' in text


async def test_unknown_route_returns_error_envelope(client):
    """FastAPI 自身抛出的 HTTPException 同样走 {code,message} 信封。"""
    resp = await client.get("/v1/not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"
    assert resp.json()["message"]


async def test_model_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(inference._chat, "_llm", None)

    def _fail() -> str:
        raise RuntimeError("模型文件不存在: /models/x.gguf")

    monkeypatch.setattr(inference._chat, "_ensure", _fail)
    resp = await client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 503
    assert resp.json()["code"] == "MODEL_NOT_READY"
