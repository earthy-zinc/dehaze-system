"""知识库分块评估语料抽样器。

从 dehaze-doc/docs 按 规模(short/medium/long) × 结构主导类型
(heading_text/table_dense/code_dense/mixed) 两维分层，每格确定性等距抽样
per_cell 篇；另构造 8 个内存对抗样本（不落盘，检索评估阶段按 key 重建）。
抽样清单由 write_manifest 落 reports/corpus_manifest.json，保证离线分块评估
与检索链路评估复用同一批语料、两阶段可比。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 规模分桶阈值（字节）
SIZE_SHORT_BYTES = 5 * 1024
SIZE_LONG_BYTES = 30 * 1024

# 结构分类阈值（占非空行比例）：
# - 表格行/代码行任一占比 ≥ 0.35 即视为该结构主导（表格密集/代码密集文档特征极强）
# - 两者合计 < 0.12 视为纯标题+正文（设计/需求/说明类文档）
# - 其余为混合结构
TABLE_DENSE_RATIO = 0.35
CODE_DENSE_RATIO = 0.35
PLAIN_MAX_RATIO = 0.12

# 非正文目录：模板配置、静态资源、评估产物自身
SKIP_DIR_NAMES = {"reports", ".vitepress", "public", "node_modules", "__pycache__"}


@dataclass
class CorpusItem:
    path: str
    size_bucket: str  # short / medium / long
    struct_type: str  # heading_text / table_dense / code_dense / mixed
    content: str
    adversarial: bool = False


def _size_bucket(nbytes: int) -> str:
    if nbytes < SIZE_SHORT_BYTES:
        return "short"
    if nbytes <= SIZE_LONG_BYTES:
        return "medium"
    return "long"


def classify_struct(content: str) -> str:
    """按特征行占比判定结构主导类型。

    代码行 = 成对 ``` / ~~~ 围栏内的所有行（含围栏标记行本身），
    围栏内的 |行| 属于代码示例而非表格，必须排除。
    表格行 = 围栏外以 | 开头的行（含 |---| 分隔行，它本就是表格一部分）。
    """
    nonempty = table = code = 0
    in_fence = False
    for raw in content.split("\n"):
        s = raw.strip()
        if not s:
            continue
        nonempty += 1
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
            code += 1
            continue
        if in_fence:
            code += 1
        elif s.startswith("|"):
            table += 1
    if nonempty == 0:
        return "heading_text"
    t, c = table / nonempty, code / nonempty
    if t >= TABLE_DENSE_RATIO:
        return "table_dense"
    if c >= CODE_DENSE_RATIO:
        return "code_dense"
    if t + c < PLAIN_MAX_RATIO:
        return "heading_text"
    return "mixed"


def _evenly(sorted_docs: list, n: int) -> list:
    """确定性等距抽样：排序后按等步长取 n 个点，两次运行结果一致。"""
    if len(sorted_docs) <= n:
        return list(sorted_docs)
    step = len(sorted_docs) / n
    return [sorted_docs[int(i * step)] for i in range(n)]


def select_corpus(doc_root: str | Path, per_cell: int = 4) -> list[CorpusItem]:
    """分层抽样：扫描 doc_root 下全部 md，3×4 矩阵每格取 per_cell 篇。"""
    doc_root = Path(doc_root)
    buckets: dict[tuple[str, str], list] = {}
    for md in sorted(doc_root.rglob("*.md")):
        if any(part in SKIP_DIR_NAMES for part in md.parts):
            continue
        content = md.read_text(encoding="utf-8")
        key = (_size_bucket(md.stat().st_size), classify_struct(content))
        buckets.setdefault(key, []).append((md, content))
    items = []
    for (size_b, struct_t), docs in sorted(buckets.items()):
        for md, content in _evenly(docs, per_cell):
            items.append(CorpusItem(str(md), size_b, struct_t, content))
    items.extend(build_adversarial())
    return items


def build_adversarial() -> list[CorpusItem]:
    """构造 8 个对抗样本（内存，不落盘）。

    每个样本都足够长（远超 512 token）以触发多块切分，才能真正暴露
    token 硬切乱码、句子割裂、围栏/表格切断等切分缺陷，而非摆设。
    """
    base = (
        "这是一段用于知识库分块质量评估的中文正文，包含若干个结构完整的句子。"
        "知识库在入库前会经过解析与分块，分块质量直接影响检索召回效果。"
        "本段文本会被重复拼接多段，用于凑足触发切分所需的长度。"
    )

    # 零宽字符：模拟网页/Office 复制粘贴残留的 U+200B，逐段错位插入
    paras = [base[: i % 20] + "\u200b" + base[i % 20 :] for i in range(12)]
    zero_width = "\n\n".join(paras)

    crlf = "\r\n\r\n".join(base for _ in range(12))
    bom = "\ufeff" + "\n\n".join(base for _ in range(12))

    # 超长单行：>5000 字符无换行；前半有句号，后半完全无标点（逼出硬切行为）
    long_line = (
        "超长单行测试，这一行没有任何换行符直到段落结束。" * 130
        + "后半部分完全没有标点符号用于观察分块引擎在无边界文本上的硬切行为" * 90
    )

    # 中英混杂无标点：连续输入形态，考察 tokenizer 与句子边界识别
    mixed = (
        "这是一个中文短语 followed by English words without any delimiter"
        "然后继续中文混入 code_variable_name 和 123456 数字连续输入"
    ) * 40

    emoji = ("情感标注结果为积极 😀 继续下一句测试 🚀 表情符号夹杂在中文里 🎉\n\n") * 30

    # 仅表格：60 行 × 8 列，无任何正文，表格总 token 远超单块容量
    rows = [
        "| 字段 | 类型 | 允许空 | 默认值 | 说明 | 示例 | 索引 | 关联 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in range(60):
        rows.append(
            f"| col_{i} | varchar(64) | 否 | '' | 第{i}个字段的详细说明内容 | value_{i} | idx_{i} | fk_table_{i % 5} |"
        )
    table_only = "\n".join(rows)

    # 仅代码块：单围栏长代码（多函数定义），考察围栏是否被块边界切断
    code_lines = ["```python", "import os", ""]
    for i in range(40):
        code_lines += [
            f"def handler_{i}(payload: dict) -> dict:",
            f'    """处理第 {i} 类请求，校验参数后返回标准响应结构。"""',
            "    if not payload:",
            '        raise ValueError("empty payload")',
            "    result = {",
            '        "code": 0,',
            f'        "message": "handler {i} ok",',
            '        "data": {k: v for k, v in payload.items() if k != "secret"},',
            "    }",
            "    return result",
            "",
        ]
    code_lines.append("```")
    code_only = "\n".join(code_lines)

    samples = [
        ("zero_width", zero_width),
        ("crlf", crlf),
        ("bom", bom),
        ("long_line", long_line),
        ("mixed_no_punct", mixed),
        ("emoji_dense", emoji),
        ("table_only", table_only),
        ("code_fence_only", code_only),
    ]
    items = []
    for key, content in samples:
        nbytes = len(content.encode("utf-8"))
        items.append(
            CorpusItem(
                path=f"adversarial:{key}",
                size_bucket=_size_bucket(nbytes),
                struct_type=classify_struct(content),
                content=content,
                adversarial=True,
            )
        )
    return items


def write_manifest(items: list[CorpusItem], doc_root: str | Path, per_cell: int, out_path: str | Path):
    """抽样清单落盘：记录路径与分类信息（不含全文），附 sha256 前缀供语料漂移校验。"""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "doc_root": str(doc_root),
        "per_cell": per_cell,
        "doc_count": sum(1 for i in items if not i.adversarial),
        "adversarial_keys": [i.path for i in items if i.adversarial],
        "items": [
            {
                "path": i.path,
                "size_bucket": i.size_bucket,
                "struct_type": i.struct_type,
                "adversarial": i.adversarial,
                "content_bytes": len(i.content.encode("utf-8")),
                "sha256_16": hashlib.sha256(i.content.encode("utf-8")).hexdigest()[:16],
            }
            for i in items
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
