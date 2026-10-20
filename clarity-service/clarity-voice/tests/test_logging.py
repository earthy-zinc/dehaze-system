"""文件日志：NDJSON 字段契约、按日期分目录与按大小分片。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from app.logging import DateDirectoryFileHandler, JsonFormatter, set_request_context


def _logger(handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger("test.voice.logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def test_handler_writes_ndjson_with_contract_fields(tmp_path: Path) -> None:
    handler = DateDirectoryFileHandler(str(tmp_path), "info.log", retention_days=0, max_bytes=0)
    handler.setFormatter(JsonFormatter())
    logger = _logger(handler)
    try:
        set_request_context(trace_id="trace-1", method="POST", path="/v1/audio/speech")
        logger.info(
            "TTS 合成完成",
            extra={"voice_id": "huayan", "duration_ms": 12.3, "audio_bytes": 1024},
        )
        handler.close()

        log_file = tmp_path / datetime.now().strftime("%Y-%m-%d") / "info.log"
        entry = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert entry["service"] == "voice"
        assert entry["level"] == "INFO"
        assert entry["message"] == "TTS 合成完成"
        assert entry["trace_id"] == "trace-1"
        assert entry["method"] == "POST"
        assert entry["path"] == "/v1/audio/speech"
        assert entry["voice_id"] == "huayan"
        assert entry["duration_ms"] == 12.3
        assert entry["logger"] == "test.voice.logging"
        assert "timestamp" in entry
    finally:
        set_request_context()


def test_error_log_carries_exc_info(tmp_path: Path) -> None:
    handler = DateDirectoryFileHandler(str(tmp_path), "error.log", retention_days=0, max_bytes=0)
    handler.setFormatter(JsonFormatter())
    logger = _logger(handler)
    try:
        try:
            raise RuntimeError("推理线程崩溃")
        except RuntimeError:
            logger.exception("未处理异常")
        handler.close()

        log_file = tmp_path / datetime.now().strftime("%Y-%m-%d") / "error.log"
        entry = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert entry["exc_info"].startswith("Traceback")
        assert "RuntimeError: 推理线程崩溃" in entry["exc_info"]
    finally:
        set_request_context()


def test_handler_archives_on_size_limit(tmp_path: Path) -> None:
    handler = DateDirectoryFileHandler(str(tmp_path), "info.log", retention_days=0, max_bytes=1)
    handler.setFormatter(JsonFormatter())
    logger = _logger(handler)
    date_dir = tmp_path / datetime.now().strftime("%Y-%m-%d")
    try:
        logger.info("第一条")
        assert (date_dir / "info.1.log").exists()
        assert (date_dir / "info.log").exists()
        logger.info("第二条")
        assert (date_dir / "info.2.log").exists()
    finally:
        handler.close()
