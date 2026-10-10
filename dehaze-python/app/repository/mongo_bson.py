"""MongoDB 共享集合的 BSON 数值类型口径（跨端互认）。

`audit_log` / `login_log` / `ai_api_call_log` 由 java / go / python **三端共写共读**，
而同一个 Python 整数在 pymongo 里按值域编码：绝对值 < 2^31 落 **int32**，否则 int64。
另两端查这些归属字段时是**单类型**（java `Long` → int64、go `int64` → int64），
BSON 数值类型不相等 ⇒ 读不到 python 写入的记录，**且不报错**（表现为静默空列表）。

故两端口径必须成对：
- 写侧：被跨端查询的整数字段统一用 `bson.int64.Int64` 编码（本模块 `as_int64`）；
- 读侧：同一字段用 int32/int64 双类型 `$in`（本模块 `int64_or_int32_in`），
  这样既能命中历史 int32 记录（python 旧数据、java `Integer`），也能命中 int64 记录。
"""

from collections.abc import Iterable
from typing import Any

from bson.int64 import Int64

__all__ = ["as_int64", "int64_candidates", "int64_or_int32_in"]


def as_int64(value: Any) -> Any:
    """把整数统一编码为 BSON int64；非整数原样返回。

    - `int`/`Int64` → `Int64`（含 bool 除外：bool 是 int 子类，必须保持布尔语义）
    - `list`/`tuple` → 逐元素转换（审计 `target_id` 批量写入传的是 id 列表）
    - `str`/`None`/`dict` 等 → 原样（审计 target_id 也可能是订单号字符串，不能强转）
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return Int64(value)
    if isinstance(value, list):
        return [as_int64(item) for item in value]
    if isinstance(value, tuple):
        return tuple(as_int64(item) for item in value)
    return value


def int64_candidates(values: Iterable[int]) -> list[Any]:
    """把待匹配的整数展开为 int32/int64 两种编码候选（供 `$in` 使用）。"""
    return [candidate for value in values for candidate in (value, Int64(value))]


def int64_or_int32_in(value: int) -> dict[str, list[Any]]:
    """int32/int64 双类型匹配条件（等价 `{"$in": [value, Int64(value)]}`）。"""
    return {"$in": int64_candidates([value])}
