"""检索质量评估测试集构建器（结构即标注，零人工）。

对语料清单（reports/corpus_manifest.json）中每篇真实文档解析 Markdown
标题结构（#/##/### 层级），每个**内容小节**（标题下有 >=200 字符正文）
生成一条测试用例：

- query: "{模块上下文}的{小节标题}"（模块上下文 = 文档一级标题，无 H1 时回退文件名）
- expected_text: 小节正文（去标题行、去代码围栏标记行，保留代码内容）

标注零人工的定位说明：query 是"文档主题 + 小节标题"的拼接而非自然问句，
考察的是向量检索对贴近文档语汇的查询的召回，与真实用户提问有差距，
解读报告时需注意。对抗样本不参与（它们测分块健壮性，无标题结构）。

用法（dehaze-test 目录下，校验用例规模）：
    ../dehaze-python/.venv/bin/python -m kb_eval.testset_builder
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 内容小节正文（去空白后）最低字符数：低于此视为标题占位/目录，不构成可检索目标
MIN_SECTION_CHARS = 200
# 每篇文档最多取的小节数（按标题顺序均匀取，控制全量用例在 ~100-140 条）
MAX_SECTIONS_PER_DOC = 5


@dataclass
class TestCase:
    """单条检索测试用例（doc_path/strategy 无关，可直接跨策略复用）。"""

    doc_path: str  # 语料文档绝对路径（对应 manifest items[].path）
    doc_title: str  # 上传用唯一标题（相对路径去 .md，避免同名"后端实现.md"混淆归属）
    size_bucket: str  # short / medium / long（分层用，来自 manifest）
    struct_type: str  # heading_text / table_dense / code_dense / mixed
    module_context: str  # 文档一级标题（query 的模块上下文）
    section_title: str  # 小节标题（保留编号/英文标识）
    query: str
    expected_text: str


def _strip_fence_markers(lines: list[str]) -> list[str]:
    """去掉 ```/~~~ 围栏标记行，保留围栏内代码内容（expected_text 只去标记）。"""
    return [ln for ln in lines if not ln.lstrip().startswith(("```", "~~~"))]


def _scan_headings(content: str) -> tuple[list[tuple[int, str, int]], bool]:
    """扫描围栏外的 #/##/### 标题行，返回 [(level, title, line_idx)] 与是否跳过 YAML frontmatter。"""
    lines = content.split("\n")
    headings: list[tuple[int, str, int]] = []
    in_fence = False
    in_frontmatter = lines and lines[0].strip() == "---"
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not in_frontmatter and (s.startswith("```") or s.startswith("~~~")):
            in_fence = not in_fence
            continue
        if in_fence or in_frontmatter:
            # frontmatter 结束于第二条 --- 行
            if in_frontmatter and idx > 0 and s == "---":
                in_frontmatter = False
            continue
        if s.startswith("#") and len(s) > 1 and s.lstrip("#").startswith(" "):
            level = len(s) - len(s.lstrip("#"))
            if level <= 3:
                headings.append((level, s[level:].strip(), idx))
    return headings, bool(lines and lines[0].strip() == "---")


def _extract_sections(content: str) -> tuple[str, list[tuple[str, str]]]:
    """解析文档，返回 (模块上下文, [(小节标题, 正文)])。

    小节区间 = 该标题行之后到下一个任意层级标题（或文档结尾）之间，
    正文去掉围栏标记行；区间内的深级标题行不属于本节正文（它们是独立小节）。
    """
    lines = content.split("\n")
    headings, _ = _scan_headings(content)
    module_context = next((t for lv, t, _ in headings if lv == 1), "")
    sections: list[tuple[str, str]] = []
    for i, (level, title, line_idx) in enumerate(headings):
        if level == 1:
            continue  # H1 只作模块上下文，不生成用例（query 会与自身重复）
        end = headings[i + 1][2] if i + 1 < len(headings) else len(lines)
        body_lines = [
            ln for ln in lines[line_idx + 1 : end] if not ln.lstrip().startswith("#")
        ]
        body = "\n".join(_strip_fence_markers(body_lines)).strip()
        sections.append((title, body))
    return module_context, sections


def _evenly_pick(items: list, n: int) -> list:
    """确定性等距抽样（与 corpus_selector._evenly 同规则，保持两阶段抽样一致）。"""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def build_testset(manifest: dict, doc_root: str | Path | None = None) -> list[TestCase]:
    """从语料清单构建检索测试用例。

    Args:
        manifest: reports/corpus_manifest.json 的内容（复用离线评估同一份抽样）
        doc_root: 文档根目录，缺省用 manifest["doc_root"]
    """
    root = Path(doc_root or manifest["doc_root"])
    cases: list[TestCase] = []
    for item in manifest["items"]:
        if item["adversarial"]:
            continue  # 对抗样本测分块健壮性，无标题结构，不参与召回评估
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        content = path.read_text(encoding="utf-8")
        module_context, sections = _extract_sections(content)
        if not module_context:
            module_context = path.stem
        qualified = [(t, b) for t, b in sections if len("".join(b.split())) >= MIN_SECTION_CHARS]
        for title, body in _evenly_pick(qualified, MAX_SECTIONS_PER_DOC):
            # 小节标题与模块上下文重复时不再拼接（避免 "X的X" 式 query）
            query = title if title.startswith(module_context) else f"{module_context}的{title}"
            cases.append(
                TestCase(
                    doc_path=str(path),
                    doc_title=str(path.relative_to(root)).removesuffix(".md"),
                    size_bucket=item["size_bucket"],
                    struct_type=item["struct_type"],
                    module_context=module_context,
                    section_title=title,
                    query=query,
                    expected_text=body,
                )
            )
    return cases


def main() -> None:
    """打印用例规模统计，用于校验 ~100-140 条的目标区间。"""
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "reports" / "corpus_manifest.json").read_text(encoding="utf-8"))
    cases = build_testset(manifest)
    by_doc: dict[str, int] = {}
    for c in cases:
        by_doc[c.doc_path] = by_doc.get(c.doc_path, 0) + 1
    print(f"总用例数: {len(cases)}，覆盖文档数: {len(by_doc)} / {manifest['doc_count']}")
    for doc, n in sorted(by_doc.items(), key=lambda kv: -kv[1]):
        print(f"  {n} 条  {Path(doc).name}")
    print("\n样例:")
    for c in cases[:3]:
        print(f"  query: {c.query}")
        print(f"  expected[:80]: {c.expected_text[:80]!r}")


if __name__ == "__main__":
    main()
