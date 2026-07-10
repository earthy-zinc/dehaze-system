"""
自定义 JSON 响应类

递归移除 null 值，匹配 Java Jackson NON_NULL 序列化行为。
"""
import json
from typing import Any

from fastapi.responses import JSONResponse


def _remove_none(obj: Any) -> Any:
    """递归移除 dict/list 中的 None 值"""
    if isinstance(obj, dict):
        return {k: _remove_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_remove_none(item) for item in obj]
    return obj


class NonNullJSONResponse(JSONResponse):
    """JSON 响应类，序列化时排除 null 值（匹配 Java Jackson NON_NULL）"""

    def render(self, content: Any) -> bytes:
        cleaned = _remove_none(content)
        return json.dumps(
            cleaned,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
