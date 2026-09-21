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
    # 父子分块：child → parent（小节）的归属（父子分块设计.md §4.1/§4.2）
    section_index: int = 0
    section_path: str | None = None


@dataclass
class Section:
    """文档小节（父子分块的 parent 单元），内容 = 节标题行 + 正文。"""

    index: int
    path: str | None
    text: str


# 降级节（段落聚类/句子聚类）的目标节大小（token）；标题节的边界由标题决定不受此约束
_SECTION_TARGET_TOKENS = 1500


def parse_sections(text: str) -> list[Section]:
    """按结构信号切节（父子分块设计.md §4.2 四级降级）。

    级别 1：Markdown 标题树（# ~ ######，路径随层级累积）
    级别 2：双换行段落聚类（无标题，合并小段到目标节大小）
    级别 3：句子贪心聚类（无空行仅句末标点，按目标节大小聚合切节）
    级别 4：单节（无任何信号，超预算由上下文注入截断兜底）

    降级级别的节是**原文区间切片**（不做 split+join 重组），保证零结构破坏。
    """
    if not text.strip():
        return []
    sections = _sections_by_heading(text)
    if sections:
        return sections
    para_spans = _split_spans(text, r"\n\s*\n")
    if len(para_spans) >= 2:
        return _cluster_sections_by_spans(text, para_spans)
    sent_spans = _split_spans(text, r"(?<=[。！？!?.])[ \t]*|(?<=\n)(?=\S)")
    if len(sent_spans) >= 2:
        return _cluster_sections_by_spans(text, sent_spans)
    return [Section(index=0, path=None, text=text)]


def _split_spans(text: str, pattern: str) -> list[tuple[int, int]]:
    """按分隔模式把文本切为非空区间 [(start, end)]，区间为原文精确切片。"""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(pattern, text):
        if text[pos : m.start()].strip():
            spans.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        spans.append((pos, len(text)))
    return spans


def _cluster_sections_by_spans(text: str, spans: list[tuple[int, int]]) -> list[Section]:
    """把原文区间贪心聚合到目标节大小切节，节内容为原文连续切片。"""
    sections: list[Section] = []
    buf_start: int | None = None
    buf_end = 0
    buf_tokens = 0
    for s, e in spans:
        span_tokens = _count_tokens(text[s:e])
        if buf_start is not None and buf_tokens + span_tokens > _SECTION_TARGET_TOKENS:
            sections.append(Section(index=len(sections), path=None, text=text[buf_start:buf_end]))
            buf_start, buf_tokens = None, 0
        if buf_start is None:
            buf_start = s
        buf_end = e
        buf_tokens += span_tokens
    if buf_start is not None:
        sections.append(Section(index=len(sections), path=None, text=text[buf_start:buf_end]))
    return sections


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _sections_by_heading(text: str) -> list[Section]:
    """按 Markdown 标题树切节；标题路径随层级累积（"4 核心设计 > 4.2 检索"）。

    代码围栏内的 `#` 是注释不是标题，行扫描时跳过（围栏开关跟踪）。
    """
    lines = text.split("\n")
    heading_lines: dict[int, tuple[int, str]] = {}  # 行号 → (层级, 标题)
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_LINE_RE.match(stripped)
        if m:
            heading_lines[i] = (len(m.group(1)), m.group(2).strip())
    if not heading_lines:
        return []

    sections: list[Section] = []
    # 首个标题前的正文归入无路径引导节（若有内容）
    first_line = min(heading_lines)
    if first_line > 0 and "\n".join(lines[:first_line]).strip():
        sections.append(Section(index=0, path=None, text="\n".join(lines[:first_line])))

    stack: list[str] = []  # 各级标题栈（索引 = 层级-1）
    ordered = sorted(heading_lines.items())
    for idx, (line_no, (level, title)) in enumerate(ordered):
        stack[level - 1 :] = [title]
        path = " > ".join(s for s in stack[:level] if s)
        end_line = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(lines)
        sections.append(
            Section(index=len(sections), path=path, text="\n".join(lines[line_no:end_line]))
        )
    return sections


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
    if strategy == "qa":
        # Q/A 标记是比标题更强的结构信号（问题-答案对不可被节边界拆散），整文档执行
        raw = _dispatch(strategy, text, chunk_size, chunk_overlap)
    else:
        # 父子分块：其余策略在节（parent 单元）内执行，chunk 携带 section 归属
        raw = []
        for sec in parse_sections(text):
            sec_chunks = _dispatch(strategy, sec.text, chunk_size, chunk_overlap)
            for c in sec_chunks:
                c.section_index = sec.index
                c.section_path = sec.path
            raw.extend(sec_chunks)
    # qa/table 结构语义必须保留，不做小片段合并与强制切分
    return _postprocess(raw, preserve_structure=strategy in ("qa", "table"), chunk_size=chunk_size)


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
# 固定长度分块（结构感知装箱）
# ---------------------------------------------------------------------------
# 文本流式取数时剩余预算低于该值即结算当前块（防止为凑预算产出碎块）
_MIN_TEXT_STEP = 50


def _chunk_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """结构感知固定分块：围栏/表格作为不可分割原子，与相邻文本在预算内贪心装箱。

    块结构完整（原子不切断）与块大小（装箱接近预算）同时保证——独立原子块信息量
    小，检索时"高分不命中"（检索基线实测原子独立成块后 Recall 回退）。装箱模式下
    相邻块不重叠：原子边界是天然锚点，token 级 overlap 的连续性收益不再必要。
    """
    chunks: list[Chunk] = []
    parts: list[str] = []
    parts_tokens = 0

    def settle() -> None:
        nonlocal parts, parts_tokens
        if parts:
            content = "\n".join(parts)
            chunks.append(Chunk(content=content, token_count=_count_tokens(content)))
        parts, parts_tokens = [], 0

    for seg_type, seg in _scan_atomic_segments(text):
        seg_tokens = _count_tokens(seg)
        if seg_type != "atomic":
            rest = seg
            while rest:
                if chunk_size - parts_tokens < _MIN_TEXT_STEP:
                    settle()
                piece, rest = _take_budget_text(rest, chunk_size - parts_tokens)
                if piece is None:
                    settle()
                    continue
                parts.append(piece)
                parts_tokens += _count_tokens(piece)
        elif seg_tokens > chunk_size:
            # 超预算原子（超长围栏/表格）结算后按行贪心切分独立成块
            settle()
            chunks.extend(_atomic_chunks(seg, chunk_size))
        elif parts_tokens + seg_tokens <= chunk_size:
            parts.append(seg)
            parts_tokens += seg_tokens
        else:
            settle()
            parts, parts_tokens = [seg], seg_tokens
    settle()
    return chunks


def _scan_atomic_segments(text: str) -> list[tuple[str, str]]:
    """行扫描产出有序段序列：("text", 文本段) 与 ("atomic", 围栏/表格整体)。"""
    lines = text.split("\n")
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(("```", "~~~")):
            fence, i = _collect_fence(lines, i)
            if buffer:
                segments.append(("text", "\n".join(buffer)))
                buffer = []
            segments.append(("atomic", fence))
            continue
        if stripped.startswith("|") or stripped.lower().startswith("<table"):
            table_text, i = _collect_table(lines, i)
            if buffer:
                segments.append(("text", "\n".join(buffer)))
                buffer = []
            if table_text:
                segments.append(("atomic", table_text))
            continue
        buffer.append(lines[i])
        i += 1
    if buffer:
        segments.append(("text", "\n".join(buffer)))
    return segments


def _take_budget_text(text: str, budget: int) -> tuple[str | None, str]:
    """从文本头部截取不超过 budget token 的内容（句边界优先），返回 (截取, 剩余)。

    decode(token 前缀) 与原文前缀逐字符一致（BPE 确定性），剩余部分用字符切片
    而非 token 重编码，避免 token 边界切字符产生 U+FFFD 污染。
    """
    if budget <= 0:
        return None, text
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= budget:
        return text, ""
    content, _, _win_end = _decode_window(enc, tokens, 0, budget)
    cut = _sentence_cut(content)
    if cut is not None and cut < len(content.rstrip()):
        return content[:cut], text[cut:]
    return content, text[len(content) :]


def _collect_fence(lines: list[str], start: int) -> tuple[str, int]:
    """收集一个代码围栏块（开闭标记间全部行；未闭合则到文本尾）。"""
    body = [lines[start]]
    i = start + 1
    while i < len(lines):
        body.append(lines[i])
        if lines[i].strip().startswith(("```", "~~~")):
            i += 1
            break
        i += 1
    return "\n".join(body), i


def _atomic_chunks(text: str, chunk_size: int) -> list[Chunk]:
    """原子结构块（围栏/表格）：预算内整块；超预算按行贪心（行完整性优先于 token 预算）。"""
    if _count_tokens(text) <= chunk_size:
        return [Chunk(content=text, token_count=_count_tokens(text))]
    chunks = []
    current = ""
    for line in text.split("\n"):
        candidate = line if not current else current + "\n" + line
        if _count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(Chunk(content=current, token_count=_count_tokens(current)))
            current = line
    if current:
        chunks.append(Chunk(content=current, token_count=_count_tokens(current)))
    return chunks


def _tokens_upto_chars(enc, tokens: list, start: int, end: int, char_pos: int) -> int:
    """二分定位窗口内 decode 长度达到 char_pos 的最小 token 前缀长度（相对 start）。

    孤立 encode(文本片段) 与全文上下文编码的 token 数不一致，直接用片段 token 数
    推进游标会漂移（累积后产生大量碎块）；本映射基于真实 token 序列，精确可靠。
    """
    lo, hi = 0, end - start
    while lo < hi:
        mid = (lo + hi) // 2
        if len(enc.decode(tokens[start : start + mid])) >= char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


# token 边界借用的最大次数（原文本身含 replacement char 时防御死循环）
_MAX_TOKEN_BORROW = 3


def _decode_window(enc, tokens: list, start: int, end: int) -> tuple[str, int, int]:
    """decode token 窗口；边界切断多字节字符时借相邻 token 对齐到字符边界。

    tiktoken 字节级 BPE 的 token 可跨字符字节边界，直接 decode 半个字符的 token
    会产生 U+FFFD 乱码（如中文+符号混合语料）。返回 (内容, 借用后窗口起点, 终点)。
    """
    borrowed = 0
    while start > 0 and borrowed < _MAX_TOKEN_BORROW:
        content = enc.decode(tokens[start:end])
        if not content or content[0] != "\ufffd":
            break
        start -= 1
        borrowed += 1
    borrowed = 0
    while end < len(tokens) and borrowed < _MAX_TOKEN_BORROW:
        content = enc.decode(tokens[start:end])
        if not content or content[-1] != "\ufffd":
            break
        end += 1
        borrowed += 1
    return enc.decode(tokens[start:end]), start, end


def _sentence_cut(content: str) -> int | None:
    """在内容尾部（后 25%）从后向前找句末/换行边界，返回截断点；太靠前返回 None。"""
    text = content.rstrip()
    if not text:
        return None
    floor = int(len(text) * 0.75)
    for idx in range(len(text) - 1, floor - 1, -1):
        if text[idx] == "\n" or text[idx] in _SENTENCE_END:
            return idx + 1
    return None


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
    """按句末标点/换行切成小段后贪心重组为不超过 chunk_size 的文本块。

    切分保留换行在前段尾部（句号后只吃空格/Tab、换行处零宽切分），重组时换行
    不丢失——否则 Markdown 表格行/代码行会被拼接到上一行造成粘连。
    """
    pieces = [p for p in re.split(r"(?<=[。！？!?.])[ \t]*|(?<=\n)(?=\S)", text) if p.strip()]
    chunks = []
    current = ""
    for piece in pieces:
        if _count_tokens(piece) > chunk_size:
            # 无标点长段（emoji/连续符号）无法按句切分 → token 硬切兜底（含字符边界对齐）
            if current:
                chunks.append(current)
                current = ""
            hard: list[Chunk] = []
            _hard_split(piece, chunk_size, 0, hard)
            chunks.extend(c.content for c in hard)
            continue
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
    # 仍有超大片 → 用下一级分隔符继续细化；split 会吃掉分隔符，细化时必须拼回
    # 头部，否则分隔符内容静默丢失（连续同分隔符的空段只还原一个，可接受的折中）
    for i, p in enumerate(parts):
        _recursive_split(p if i == 0 else sep + p, separators[1:], chunk_size, chunk_overlap, out)


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
    """无分隔符可用时按 token 硬切（同样经字符边界对齐防乱码）。"""
    enc = _get_encoder()
    tokens = enc.encode(text)
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        content, _, win_end = _decode_window(enc, tokens, start, end)
        out.append(Chunk(content=content, token_count=_count_tokens(content)))
        start = max(win_end - chunk_overlap, start + 1)


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
    cols = len(list(data_rows[0].strip().strip("|").split("|"))) if data_rows else 0
    return max(len(data_rows) - 1, 0), cols


def _fixed_text_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    return _chunk_fixed(text, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# 后处理：空白清理、小片段合并、超长强制切分、附加 chunk_index
# ---------------------------------------------------------------------------
_MIN_TOKENS = 50
_MAX_TOKENS = 2000


def _postprocess(
    chunks: list[Chunk], preserve_structure: bool = False, chunk_size: int = 800
) -> list[Chunk]:
    cleaned = _clean_whitespace(chunks)
    if preserve_structure:
        final = cleaned
    else:
        merged = _merge_small(cleaned, chunk_size)
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
            Chunk(
                content=content,
                token_count=_count_tokens(content),
                metadata=chunk.metadata,
                # 重建 Chunk 时必须保留小节归属，否则写入侧拿到的全是默认值
                section_index=chunk.section_index,
                section_path=chunk.section_path,
            )
        )
    return cleaned


def _merge_small(chunks: list[Chunk], chunk_size: int) -> list[Chunk]:
    """token 数小于 _MIN_TOKENS 的片段与上一相邻 chunk 合并。

    合并受 chunk_size 预算约束：前块已达预算不再并入（小块独立成块），
    防止碎片密集语料（emoji/无标点）级联合并出远超预算的巨块。
    """
    merged = []
    for chunk in chunks:
        if (
            merged
            and chunk.token_count < _MIN_TOKENS
            # 跨节合并会让 child 内容混入相邻小节，破坏 child→parent 映射（父子分块设计 D5）
            and merged[-1].section_index == chunk.section_index
            and merged[-1].token_count + chunk.token_count <= chunk_size
        ):
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
        result.append(
            Chunk(
                content=piece,
                token_count=_count_tokens(piece),
                metadata=meta,
                section_index=chunk.section_index,
                section_path=chunk.section_path,
            )
        )
    return result
