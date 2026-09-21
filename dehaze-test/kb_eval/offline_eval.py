"""知识库分块质量离线评估主入口。

纯离线：仅调用 dehaze-python 分块算法，不起后端、不连 DB/Redis。
用法（在 dehaze-test 目录下）：
    ../dehaze-python/.venv/bin/python -m kb_eval.offline_eval \
        [--strategies fixed,semantic,recursive,qa,table] [--per-cell 4]
产物：reports/corpus_manifest.json（抽样清单）+ reports/offline_baseline_*.md（基线报告）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent  # dehaze-test/kb_eval
WORKSPACE = HERE.parents[1]  # dehaze-system 根
DOC_ROOT = WORKSPACE / "dehaze-doc" / "docs"
sys.path.insert(0, str(WORKSPACE / "dehaze-python"))
sys.path.insert(0, str(HERE.parent))

from app.service.kb.chunking_engine import chunk_text  # noqa: E402
from kb_eval.corpus_selector import (  # noqa: E402
    SKIP_DIR_NAMES,
    CorpusItem,
    select_corpus,
    write_manifest,
)
from kb_eval.metrics import SENTENCE_END, compute_metrics  # noqa: E402

# 用户建库实际配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
# qa/table 结构语义优先（Q/A 对、整表保留），句子完整性指标不适用
SENTENCE_APPLICABLE = {"fixed", "semantic", "recursive"}
DEFAULT_STRATEGIES = ["fixed", "semantic", "recursive", "qa", "table"]
SIZE_ORDER = ["short", "medium", "long"]
STRUCT_ORDER = ["heading_text", "table_dense", "code_dense", "mixed"]


def evaluate(items: list[CorpusItem], strategies: list[str]) -> list[dict]:
    """每篇 × 每策略跑 chunk_text 并计算指标，返回扁平记录列表。"""
    records = []
    for item in items:
        for strat in strategies:
            chunks = chunk_text(item.content, strat, CHUNK_SIZE, CHUNK_OVERLAP)
            metrics = compute_metrics(
                item.content, chunks, CHUNK_SIZE, sentence_applicable=strat in SENTENCE_APPLICABLE
            )
            records.append(
                {
                    "path": item.path,
                    "size": item.size_bucket,
                    "struct": item.struct_type,
                    "adversarial": item.adversarial,
                    "strategy": strat,
                    **metrics,
                }
            )
        print(f"[eval] {item.path}", flush=True)
    return records


def _avg(records: list[dict], key: str) -> float | None:
    vals = [r[key] for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _f(v, nd: int = 3) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _overview_rows(records: list[dict], strategies: list[str]) -> list[dict]:
    rows = []
    for strat in strategies:
        grp = [r for r in records if r["strategy"] == strat]
        if not grp:
            continue
        rows.append(
            {
                "strategy": strat,
                "n": len(grp),
                "chunks": _avg(grp, "chunk_count"),
                "repl": _avg(grp, "replacement_char_count"),
                "sentence": _avg(grp, "sentence_complete_rate"),
                "fence": _avg(grp, "code_fence_broken"),
                "table_row": _avg(grp, "table_row_broken"),
                "p50": _avg(grp, "token_p50"),
                "p95": _avg(grp, "token_p95"),
                "fragment": _avg(grp, "fragment_rate"),
                "oversized": _avg(grp, "oversized_rate"),
                "coverage": _avg(grp, "coverage"),
            }
        )
    return rows


def _layered_rows(records: list[dict], strategies: list[str]) -> list[dict]:
    """策略 × 规模 × 结构 的分层聚合，按固定顺序输出。"""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["strategy"], r["size"], r["struct"]), []).append(r)
    rows = []
    for strat in strategies:
        for size_b in SIZE_ORDER:
            for struct_t in STRUCT_ORDER:
                grp = groups.get((strat, size_b, struct_t))
                if not grp:
                    continue
                rows.append(
                    {
                        "strategy": strat,
                        "size": size_b,
                        "struct": struct_t,
                        "n": len(grp),
                        "sentence": _avg(grp, "sentence_complete_rate"),
                        "fragment": _avg(grp, "fragment_rate"),
                        "oversized": _avg(grp, "oversized_rate"),
                        "coverage": _avg(grp, "coverage"),
                    }
                )
    return rows


def _bottlenecks(records: list[dict], top: int = 5, min_avg_chunks: int = 5):
    """按劣化分挑最差的 策略×分层 组合：fragment + oversized + (1-coverage) + (1-sentence)。

    fence/table_row 是次数不是比率，不进评分，但在表中展示供人工判断。
    平均块数 < min_avg_chunks 的组合剔除：短文档只有 1-2 块，单个块尾噪声即可
    把 sentence 拉到 0，排进榜单无统计意义。
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["strategy"], r["size"], r["struct"]), []).append(r)
    scored = []
    for key, grp in groups.items():
        if (_avg(grp, "chunk_count") or 0) < min_avg_chunks:
            continue
        sent = _avg(grp, "sentence_complete_rate")
        score = (
            (_avg(grp, "fragment_rate") or 0.0)
            + (_avg(grp, "oversized_rate") or 0.0)
            + (1 - (_avg(grp, "coverage") or 1.0))
            + (1 - sent if sent is not None else 0.0)
        )
        scored.append((score, key, grp))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[:top]


def _pick_bad_excerpt(item_by_path: dict[str, CorpusItem], strategy: str, path: str) -> tuple[str, str]:
    """取该组合最差文档（coverage 最低）的典型坏块摘录：乱码 > 碎块 > 块尾非句末 > 首块。"""
    item = item_by_path[path]
    chunks = chunk_text(item.content, strategy, CHUNK_SIZE, CHUNK_OVERLAP)
    for c in chunks:
        if "\ufffd" in c.content:
            return c.content, "乱码块"
    if len(chunks) > 1:
        for c in chunks:
            if c.token_count < 50:
                return c.content, "碎块(token<50)"
        for c in chunks[:-1]:
            tail = c.content.rstrip()
            if tail and tail[-1] not in SENTENCE_END:
                return c.content, "块尾非句末"
    return (chunks[0].content if chunks else ""), "无坏例，摘录首块"


def build_report(
    doc_records: list[dict],
    adv_records: list[dict],
    items: list[CorpusItem],
    strategies: list[str],
    per_cell: int,
) -> str:
    doc_count = sum(1 for i in items if not i.adversarial)
    adv_count = sum(1 for i in items if i.adversarial)
    doc_total = sum(
        1
        for p in DOC_ROOT.rglob("*.md")
        if not any(part in SKIP_DIR_NAMES for part in p.parts)
    )
    lines = [
        "# 知识库分块质量离线基线报告",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- git commit：`{_git_commit()}`（dehaze-system workspace）",
        f"- 语料：dehaze-doc/docs 全库 {doc_total} 篇 md，分层抽样 {doc_count} 篇 + 对抗样本 {adv_count} 个",
        f"- 参数：chunk_size={CHUNK_SIZE}，chunk_overlap={CHUNK_OVERLAP}；"
        f"strategies={','.join(strategies)}；per_cell={per_cell}",
        "- 指标定义见 kb_eval/README.md；qa/table 的 sentence_complete_rate 不适用（结构语义优先）",
        "",
        "## 1. 总览（全语料宏平均，文档等权）",
        "",
        "| 策略 | n | 块数/篇 | repl | sentence | fence_brk | table_row_brk | token_p50 | token_p95 | fragment | oversized | coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in _overview_rows(doc_records, strategies):
        lines.append(
            f"| {row['strategy']} | {row['n']} | {_f(row['chunks'], 1)} | {_f(row['repl'], 2)} "
            f"| {_f(row['sentence'])} | {_f(row['fence'], 2)} | {_f(row['table_row'], 2)} "
            f"| {_f(row['p50'], 0)} | {_f(row['p95'], 0)} | {_f(row['fragment'])} "
            f"| {_f(row['oversized'])} | {_f(row['coverage'], 4)} |"
        )
    lines += [
        "",
        "## 2. 分层表（策略 × 规模 × 结构）",
        "",
        "| 策略 | 规模 | 结构 | n | sentence | fragment | oversized | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in _layered_rows(doc_records, strategies):
        lines.append(
            f"| {row['strategy']} | {row['size']} | {row['struct']} | {row['n']} "
            f"| {_f(row['sentence'])} | {_f(row['fragment'])} | {_f(row['oversized'])} "
            f"| {_f(row['coverage'], 4)} |"
        )
    lines += [
        "",
        "## 3. 对抗样本专项",
        "",
        "| 样本 | 策略 | chunks | repl | fence_brk | table_row_brk | fragment | oversized | coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in adv_records:
        lines.append(
            f"| {r['path']} | {r['strategy']} | {r['chunk_count']} | {r['replacement_char_count']} "
            f"| {r['code_fence_broken']} | {r['table_row_broken']} | {_f(r['fragment_rate'])} "
            f"| {_f(r['oversized_rate'])} | {_f(r['coverage'], 4)} |"
        )
    lines += [
        "",
        "## 4. 瓶颈清单（最差 5 个 策略×分层 组合）",
        "",
        "| 排名 | 策略 | 规模 | 结构 | n | score | sentence | fragment | oversized | coverage | fence_brk | table_row_brk |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    item_by_path = {i.path: i for i in items}
    excerpts = []
    for rank, (score, (strat, size_b, struct_t), grp) in enumerate(_bottlenecks(doc_records), 1):
        worst = min(grp, key=lambda r: r["coverage"])
        lines.append(
            f"| {rank} | {strat} | {size_b} | {struct_t} | {len(grp)} | {_f(score)} "
            f"| {_f(_avg(grp, 'sentence_complete_rate'))} | {_f(_avg(grp, 'fragment_rate'))} "
            f"| {_f(_avg(grp, 'oversized_rate'))} | {_f(_avg(grp, 'coverage'), 4)} "
            f"| {_f(_avg(grp, 'code_fence_broken'), 2)} | {_f(_avg(grp, 'table_row_broken'), 2)} |"
        )
        excerpt, kind = _pick_bad_excerpt(item_by_path, strat, worst["path"])
        flat = excerpt[:200].replace("\n", "⏎")
        excerpts.append(
            f"{rank}. `{strat} × {size_b} × {struct_t}`（{kind}，最差文档 "
            f"`{Path(worst['path']).name}`，coverage={_f(worst['coverage'], 4)}）\n"
            f"   > {flat}"
        )
    lines += ["", "### 坏例摘录（≤200 字符，⏎ 为换行）", ""]
    lines += excerpts
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库分块质量离线评估")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="逗号分隔的策略列表")
    parser.add_argument("--per-cell", type=int, default=4, help="分层矩阵每格抽样篇数")
    args = parser.parse_args()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    items = select_corpus(DOC_ROOT, args.per_cell)
    reports_dir = HERE / "reports"
    write_manifest(items, DOC_ROOT, args.per_cell, reports_dir / "corpus_manifest.json")
    doc_count = sum(1 for i in items if not i.adversarial)
    adv_count = len(items) - doc_count
    print(f"[corpus] {doc_count} 篇文档 + {adv_count} 个对抗样本，清单已写 reports/corpus_manifest.json")

    print(f"[eval] chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP} strategies={strategies}")
    records = evaluate(items, strategies)
    doc_records = [r for r in records if not r["adversarial"]]
    adv_records = [r for r in records if r["adversarial"]]

    report = build_report(doc_records, adv_records, items, strategies, args.per_cell)
    out = reports_dir / f"offline_baseline_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(report, encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    main()
