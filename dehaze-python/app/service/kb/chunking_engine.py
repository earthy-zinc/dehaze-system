"""AI 知识库分块引擎：按多种策略将文本切分为 Chunk。

纯算法组件，不依赖数据库。策略与算法细节对齐《后端实现-文档管理.md》§4。
token 计数使用 tiktoken（cl100k_base），延迟导入避免加重启动开销。
"""

import re
from dataclasses import dataclass, field

# 句末标点（用于切分时避免切断句子）
_SENTENCE_END = "。！？!?；;"

_encoder = None


def _get_encoder():
    """获取 tiktoken 编码器（延迟导入）。"""
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


@dataclass
class Chunk:
    content: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, strategy: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """统一分块入口：按 strategy 切分文本并执行后处理。

    参数：
        text: 待分块文本
        strategy: fixed / semantic / recursive / qa / table
        chunk_size: 单个分块目标 token 数（限制在 50-2000）
        chunk_overlap: 相邻分块重叠 token 数（限制在 0 ~ chunk_size-1）
    返回：
        list[Chunk]，每条已附加 chunk_index 元数据
    """
    strategy = (strategy or "fixed").lower()
    chunk_size = max(50, min(int(chunk_size or 800), 2000))
    chunk_overlap = max(0, min(int(chunk_overlap or 80), chunk_size - 1))
    raw = _dispatch(strategy, text, chunk_size, chunk_overlap)
    # qa/table 结构语义必须保留，不做小片段合并与强制切分
    return _postprocess(raw, preserve_structure=strategy in ("qa", "table"))


def _dispatch(strategy: str, text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if strategy == "qa":
        return _chunk_qa(text, chunk_size, chunk_overlap)
    if strategy == "table":
        return _chunk_table(text, chunk_size, chunk_overlap)
    if strategy == "semantic":
        return _chunk_semantic(text, chunk_size, chunk_overlap)
    if strategy == "recursive":
        return _chunk_recursive(text, chunk_size, chunk_overlap)
    return _chunk_fixed(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# 固定长度分块
# ---------------------------------------------------------------------------
def _chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    enc = _get_encoder()
    tokens = enc.encode(text)
    step = max(chunk_size - chunk_overlap, 1)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        content = _cut_at_sentence(enc.decode(tokens[start:end]))
        chunks.append(Chunk(content=content, token_count=_count_tokens(content)))
        start += step
    return chunks


def _cut_at_sentence(content: str, min_ratio: float = 0.6) -> str:
    """在内容末尾附近寻找换行/句末边界，避免切断句子。

    从尾部向前找第一个边界；仅当边界之后无非空白内容（尾部为空白）时才截断，
    绝不丢弃边界后的非空白尾部内容，避免固定分块静默丢失数据。
    """
    text = content.rstrip()
    if not text:
        return content
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] == "\n" or text[idx] in _SENTENCE_END:
            if idx >= len(text) * min_ratio and not text[idx + 1 :].strip():
                return text[: idx + 1]
            break
    return text


# ---------------------------------------------------------------------------
# 语义分块：先按双换行分段，超长段在句子边界切分，小段合并
# ---------------------------------------------------------------------------
def _chunk_semantic(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged_paras = _merge_small_paragraphs(paragraphs, min_tokens=50)
    chunks = []
    for para in merged_paras:
        if _count_tokens(para) <= chunk_size:
            chunks.append(Chunk(content=para, token_count=_count_tokens(para)))
        else:
            chunks.extend(_split_paragraph(para, chunk_size, chunk_overlap))
    return chunks


def _merge_small_paragraphs(paragraphs: list[str], min_tokens: int) -> list[str]:
    """token 数小于 min_tokens 的段落与下一段合并。"""
    merged = []
    buffer = ""
    for para in paragraphs:
        if buffer and _count_tokens(buffer + "\n\n" + para) < min_tokens:
            buffer = f"{buffer}\n\n{para}"
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)
    return merged


def _split_paragraph(para: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """超长段落在句子边界切分，相邻块在边界处添加 overlap。"""
    pieces = _split_at_sentences(para, chunk_size)
    if len(pieces) <= 1 or chunk_overlap <= 0:
        return [Chunk(content=p, token_count=_count_tokens(p)) for p in pieces]
    result = []
    for i, piece in enumerate(pieces):
        prefix = pieces[i - 1][-chunk_overlap:] if i > 0 else ""
        content = prefix + piece
        result.append(Chunk(content=content, token_count=_count_tokens(content)))
    return result


def _split_at_sentences(text: str, chunk_size: int) -> list[str]:
    """按句末标点/换行切成小段后贪心重组为不超过 chunk_size 的文本块。"""
    pieces = [p for p in re.split(r"(?<=[。！？!?.])\s*|\n+", text) if p.strip()]
    chunks = []
    current = ""
    for piece in pieces:
        if current and _count_tokens(current + piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# 递归分块：按分隔符优先级层层细化
# ---------------------------------------------------------------------------
_SEPARATORS = ["\n\n", "\n", "。", "，", " "]


def _chunk_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks = []
    _recursive_split(text, _SEPARATORS, chunk_size, chunk_overlap, chunks)
    return chunks


def _recursive_split(
    text: str, separators: list[str], chunk_size: int, chunk_overlap: int, out: list[Chunk]
):
    if not text.strip() or _count_tokens(text) <= chunk_size:
        out.append(Chunk(content=text, token_count=_count_tokens(text)))
        return
    if not separators:
        _hard_split(text, chunk_size, chunk_overlap, out)
        return
    sep = separators[0]
    parts = [p for p in text.split(sep) if p.strip()]
    # 当前分隔符切出的所有片都足够小 → 贪心合并
    if all(_count_tokens(p) <= chunk_size for p in parts):
        _greedy_group(parts, sep, chunk_size, out)
        return
    # 仍有超大片 → 用下一级分隔符继续细化
    for p in parts:
        _recursive_split(p, separators[1:], chunk_size, chunk_overlap, out)


def _greedy_group(parts: list[str], sep: str, chunk_size: int, out: list[Chunk]):
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if _count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                out.append(Chunk(content=current, token_count=_count_tokens(current)))
            current = part
    if current:
        out.append(Chunk(content=current, token_count=_count_tokens(current)))


def _hard_split(text: str, chunk_size: int, chunk_overlap: int, out: list[Chunk]):
    """无分隔符可用时按 token 硬切。"""
    enc = _get_encoder()
    tokens = enc.encode(text)
    step = max(chunk_size - chunk_overlap, 1)
    for i in range(0, len(tokens), step):
        content = enc.decode(tokens[i : i + chunk_size])
        out.append(Chunk(content=content, token_count=_count_tokens(content)))


# ---------------------------------------------------------------------------
# 问答对分块
# ---------------------------------------------------------------------------
def _chunk_qa(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """识别 Q:/A: 标记（忽略大小写），Q 与 A 分别独立 chunk（chunk_index 相邻）。"""
    if not re.search(r"(?im)^\s*[QA][:：]", text):
        # 无 Q/A 标记 → 退化为固定切分，避免内容丢失
        return _chunk_fixed(text, chunk_size, chunk_overlap)
    segments = re.split(r"(?ims)^\s*(Q[:：]|A[:：])\s*", text)
    chunks = []
    pending_q = None
    i = 1  # segments[0] 为首个标记前的文本，忽略
    while i < len(segments) - 1:
        marker = segments[i].strip()
        content = segments[i + 1].strip()
        i += 2
        if marker.upper().startswith("Q"):
            if pending_q:
                chunks.append(_qa_chunk(pending_q, "question"))
            pending_q = content
        else:  # A
            if pending_q:
                chunks.append(_qa_chunk(pending_q, "question"))
                pending_q = None
            chunks.append(_qa_chunk(content, "answer"))
    if pending_q:
        chunks.append(_qa_chunk(pending_q, "question"))
    return chunks


def _qa_chunk(content: str, qa_type: str) -> Chunk:
    return Chunk(content=content, token_count=_count_tokens(content), metadata={"type": qa_type})


# ---------------------------------------------------------------------------
# 表格感知分块：表格整体保留不分块
# ---------------------------------------------------------------------------
def _chunk_table(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    lines = text.split("\n")
    chunks = []
    buffer = []
    i = 0
    in_code_fence = False
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        # 跟踪 Markdown 代码围栏开关，围栏内的 |行| 不作表格识别
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence
            buffer.append(raw)
            i += 1
            continue
        is_table = not in_code_fence and (
            stripped.startswith("|") or stripped.lower().startswith("<table")
        )
        if is_table:
            if buffer:
                chunks.extend(_fixed_text_chunks("\n".join(buffer), chunk_size, chunk_overlap))
                buffer = []
            table_text, i = _collect_table(lines, i)
            if table_text:
                rows, cols = _table_dims(table_text)
                chunks.append(
                    Chunk(
                        content=table_text,
                        token_count=_count_tokens(table_text),
                        metadata={"type": "table", "rows": rows, "cols": cols},
                    )
                )
        else:
            buffer.append(raw)
            i += 1
    if buffer:
        chunks.extend(_fixed_text_chunks("\n".join(buffer), chunk_size, chunk_overlap))
    return chunks


def _collect_table(lines: list[str], start: int) -> tuple[str, int]:
    """收集一个连续的表格块，返回 (表格文本, 下一行索引)。"""
    first = lines[start].strip()
    if first.lower().startswith("<table"):
        body = [lines[start]]
        i = start + 1
        while i < len(lines) and "</table>" not in lines[i].lower():
            body.append(lines[i])
            i += 1
        if i < len(lines):
            body.append(lines[i])
            i += 1
        return "\n".join(body), i
    body = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        body.append(lines[i])
        i += 1
    return "\n".join(body), i


def _table_dims(table_text: str) -> tuple[int, int]:
    """估算表格行列数（支持 Markdown 与 HTML 表格），rows 只计数据行（不含表头）。"""
    lines = [ln for ln in table_text.split("\n") if ln.strip()]
    if table_text.strip().lower().startswith("<table"):
        tr_count = len(re.findall(r"<tr[ >]", table_text, re.I))
        cols = len(re.findall(r"<(?:td|th)[ >]", table_text.split("</tr>")[0], re.I))
        # tr 首行通常为表头，数据行数 = tr 数 - 1
        return max(tr_count - 1, 0), cols
    # Markdown 表格：去掉 |---| 分隔行后，首行为表头，其余为数据行
    data_rows = [ln for ln in lines if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", ln)]
    cols = len([c for c in data_rows[0].strip().strip("|").split("|")]) if data_rows else 0
    return max(len(data_rows) - 1, 0), cols


def _fixed_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return _chunk_fixed(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# 后处理：空白清理、小片段合并、超长强制切分、附加 chunk_index
# ---------------------------------------------------------------------------
_MIN_TOKENS = 50
_MAX_TOKENS = 2000


def _postprocess(chunks: list[Chunk], preserve_structure: bool = False) -> list[Chunk]:
    cleaned = _clean_whitespace(chunks)
    if preserve_structure:
        final = cleaned
    else:
        merged = _merge_small(cleaned)
        final = []
        for chunk in merged:
            if chunk.token_count > _MAX_TOKENS:
                final.extend(_force_split(chunk, _MAX_TOKENS))
            else:
                final.append(chunk)
    for idx, chunk in enumerate(final):
        chunk.metadata["chunk_index"] = idx
    return final


def _clean_whitespace(chunks: list[Chunk]) -> list[Chunk]:
    cleaned = []
    for chunk in chunks:
        content = chunk.content.strip()
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        if not content:
            continue
        cleaned.append(
            Chunk(content=content, token_count=_count_tokens(content), metadata=chunk.metadata)
        )
    return cleaned


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """token 数小于 _MIN_TOKENS 的片段与上一相邻 chunk 合并。"""
    merged = []
    for chunk in chunks:
        if merged and chunk.token_count < _MIN_TOKENS:
            prev = merged[-1]
            prev.content = f"{prev.content}\n{chunk.content}"
            prev.token_count = _count_tokens(prev.content)
            prev.metadata.update(chunk.metadata)
        else:
            merged.append(chunk)
    return merged


def _force_split(chunk: Chunk, max_tokens: int) -> list[Chunk]:
    """超长 chunk 在句子边界强制切分为不超过 max_tokens 的块。"""
    result = []
    for i, piece in enumerate(_split_at_sentences(chunk.content, max_tokens)):
        meta = dict(chunk.metadata)
        meta["split"] = i
        result.append(Chunk(content=piece, token_count=_count_tokens(piece), metadata=meta))
    return result
