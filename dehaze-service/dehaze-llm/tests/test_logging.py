"""日志契约：NDJSON 字段（service/exc_info）与 ACCESS 日志的噪声路径跳过。"""

import json
import logging

from app.logging import JsonFormatter


def test_json_formatter_emits_service_and_exc_info():
    record = logging.LogRecord("app.x", logging.ERROR, __file__, 1, "boom", (), None)
    record.exc_info = (RuntimeError, RuntimeError("炸了"), None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["service"] == "llm"
    assert payload["level"] == "ERROR"
    assert payload["logger"] == "app.x"
    assert payload["message"] == "boom"
    assert "炸了" in payload["exc_info"]


async def test_trace_id_propagated_and_echoed(client, caplog):
    """编排侧透传的 X-Trace-Id 进入日志上下文并回写响应头；缺失时自动生成。"""
    with caplog.at_level(logging.INFO, logger="app.request"):
        resp = await client.get("/v1/models", headers={"X-Trace-Id": "trace-abc"})

    assert resp.headers["X-Trace-Id"] == "trace-abc"
    payload = json.loads(JsonFormatter().format(caplog.records[-1]))
    assert payload["trace_id"] == "trace-abc"

    assert (await client.get("/v1/models")).headers["X-Trace-Id"]


async def test_access_log_skips_probe_paths(client, caplog):
    """能力端点落 ACCESS（含请求上下文），探针与 /metrics 不落日志。"""
    with caplog.at_level(logging.INFO, logger="app.request"):
        await client.get("/health")
        await client.get("/v1/models")

    records = [r for r in caplog.records if r.name == "app.request"]
    assert [r.getMessage() for r in records] == ["ACCESS"]
    payload = json.loads(JsonFormatter().format(records[0]))
    assert payload["method"] == "GET"
    assert payload["path"] == "/v1/models"
    assert payload["status"] == 200
    assert payload["duration_ms"] >= 0
