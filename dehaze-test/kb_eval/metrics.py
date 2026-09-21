"""分块质量指标计算（纯函数）。

输入原文与 chunk 列表（只需具有 .content / .token_count 属性），输出标量指标
或聚合 dict。不 import 分块引擎，避免评估器与被测实现相互纠缠。
"""

from __future__ import annotations

import math
import re
from collections import Counter

# 句末标点集合，与分块引擎 _SENTENCE_END 保持一致
SENTENCE_END = "。！？!?；;"
# 碎块阈值：与引擎 _MIN_TOKENS 对齐（低于此值的块入库前会被合并）
FRAGMENT_TOKENS = 50


def replacement_char_count(chunks) -> int:
    """U+FFFD 计数：任何策略下都应为 0，非 0 说明 token 边界切断多字节字符产生乱码。"""
    return sum(c.content.count("\ufffd") for c in chunks)


def sentence_complete_rate(chunks) -> float | None:
    """非末块中块尾（rstrip 后）落在句末标点上的比例；块数 < 2 时无意义返回 None。"""
    if len(chunks) < 2:
        return None
    tails = [c.content.rstrip() for c in chunks[:-1]]
    complete = sum(1 for t in tails if t and t[-1] in SENTENCE_END)
    return complete / len(tails)


def code_fence_broken(chunks) -> int:
    """``` 围栏被块边界切断的块数：块内围栏标记出现奇数次即说明跨块断开。"""
    return sum(1 for c in chunks if c.content.count("```") % 2 == 1)


def table_row_broken(text: str, chunks) -> int:
    """表格行连续性破坏计数，覆盖两类破坏：

    1. 拆断：原文中物理相邻的表格行被拆进相邻两块（跨块边界行对在原文相邻对集合中）；
    2. 篡改行：块内出现原文中不存在的表格行——行被粘连（如 semantic 切超长段时
       丢弃换行把多行拼成一行）或被拦腰截断（token 硬切的残行）。

    引擎的空白清理会压缩行内空格，因此两侧行都先做同样的规范化再按内容匹配。
    按内容匹配在"同一行内容出现在文档多处表格"时有极小误判概率，可忽略。
    """
    def norm(line: str) -> str:
        return re.sub(r"[ \t]+", " ", line.strip())

    lines = [norm(line) for line in text.split("\n")]
    adjacent = {
        (a, b) for a, b in zip(lines, lines[1:]) if a.startswith("|") and b.startswith("|")
    }
    if not adjacent:
        return 0
    rows_per_chunk = [
        [norm(line) for line in c.content.split("\n") if norm(line).startswith("|")]
        for c in chunks
    ]
    broken = 0
    for prev, nxt in zip(rows_per_chunk, rows_per_chunk[1:]):
        if prev and nxt and (prev[-1], nxt[0]) in adjacent:
            broken += 1
    known_lines = set(lines)
    for rows in rows_per_chunk:
        broken += sum(1 for r in rows if r not in known_lines)
    return broken


def token_percentile(chunks, q: float) -> float:
    """块 token 数的 q 分位数（最近秩法，不引入 numpy）。"""
    if not chunks:
        return 0.0
    xs = sorted(c.token_count for c in chunks)
    return float(xs[max(0, math.ceil(q * len(xs)) - 1)])


def fragment_rate(chunks) -> float:
    """token < 50 的块占比。"""
    if not chunks:
        return 0.0
    return sum(1 for c in chunks if c.token_count < FRAGMENT_TOKENS) / len(chunks)


def oversized_rate(chunks, chunk_size: int) -> float:
    """token > chunk_size×1.2 的块占比（软预算余量；qa/table 为保留结构允许超预算）。"""
    if not chunks:
        return 0.0
    budget = chunk_size * 1.2
    return sum(1 for c in chunks if c.token_count > budget) / len(chunks)


def coverage(text: str, chunks) -> float:
    """块内容（去空白）对原文（去空白）的字符多重集覆盖率，理想值 ≈ 1.0。

    overlap 导致的块间重复字符无碍（取 min 计入）；丢段、丢行、字符被乱码
    替换都会拉低覆盖率，因此它同时兜住"内容丢失"与"字符损坏"两类缺陷。
    """
    def counter(s: str) -> Counter:
        return Counter(ch for ch in s if not ch.isspace())

    orig = counter(text)
    if not orig:
        return 1.0
    got = counter("".join(c.content for c in chunks))
    matched = sum(min(n, got.get(ch, 0)) for ch, n in orig.items())
    return matched / sum(orig.values())


def compute_metrics(text: str, chunks, chunk_size: int, sentence_applicable: bool = True) -> dict:
    """一篇文档 × 一个策略的全部指标。sentence_applicable=False 时句子完整性标 N/A（None）。"""
    return {
        "chunk_count": len(chunks),
        "replacement_char_count": replacement_char_count(chunks),
        "sentence_complete_rate": sentence_complete_rate(chunks) if sentence_applicable else None,
        "code_fence_broken": code_fence_broken(chunks),
        "table_row_broken": table_row_broken(text, chunks),
        "token_p50": token_percentile(chunks, 0.50),
        "token_p95": token_percentile(chunks, 0.95),
        "fragment_rate": fragment_rate(chunks),
        "oversized_rate": oversized_rate(chunks, chunk_size),
        "coverage": coverage(text, chunks),
    }
