"""知识库检索质量评估主入口（两阶段：collect 真实检索落盘 / evaluate 离线重放算指标）。

与 offline_eval（纯算法级）互补：collect 阶段把 corpus_manifest.json 精选的真实文档
送入完整产品链路（建 KB → 上传文档 → 后台解析/分块/向量化 → ES 检索），检索结果
原样落盘 reports/raw/{strategy}.jsonl；evaluate 阶段读 raw 离线计算
Recall@K / MRR / NDCG / HitRate 出报告，并与离线分块基线做关联分析。

两阶段解耦的成本约束（为什么这么设计）：
- 检索链路是唯一有成本环节（建库+向量化+检索，单策略分钟级）——每个策略只执行一次，
  落盘后一切调参（命中阈值/指标口径/报告格式）走 evaluate 离线重放，秒级完成；
  严禁为调阈值重新建库或重新检索。
- 断点安全：chunks 在检索前整体落盘；每条 query 检索完立即 flush 写入 jsonl，
  任意时刻中断，已完成行完好；raw 已完整（非空且行数=用例数）的策略直接跳过。
- score_threshold 用建库默认 0.5（真实用户默认行为，不额外调参）；低于阈值的
  检索结果已被服务端丢弃，raw 保存过滤后返回集，空结果率在报告中单独呈现。

其余设计约束（沿用已验证的实现）：
- 每策略串行"建库→检索→删库"：admin 为 level_0 会员，私有库上限 3 个；
  分块策略创建后不可改，不同策略必须各自建库。
- 文档串行上传（传一篇等一篇）：本地 embedding 单实例 CPU 推理，并发上传的
  批量请求在推理锁上排队会连锁超时（实测），且并发入库触发 MySQL 死锁（实测）。
- 删库放 finally：中断也不能遗留 kbeval_ 前缀的库（占满配额且污染开发库）；
  启动时另做一次 kbeval_ 前缀清扫，幂等自愈历史崩溃残留。
- 命中判定用 10-gram shingle 覆盖率（chunk 对 expected_text）双向阈值：
  块侧（|交|/|chunk|，理想命中块≈1.0）或节侧（|交|/|expected|，短小节完整落
  入大块的场景）任一 ≥ threshold 即命中；跨节表格头/套话等局部重合只贡献零星
  shingle（两侧均 <0.15），校准分布见报告附录。阈值参数化（--threshold），
  调整只需重放 raw。
- Recall@K 的分母用"目标文档全部 chunk 中与期望文本重合达阈值的块数"
  （chunks 随 collect 落盘）：期望小节常被切成多块，仅数 Top-K 命中块数
  会低估召回；只统计目标文档而非全库，是因为标注本身指向该文档该小节，
  跨文档重复内容属语料噪声，会在零命中清单中暴露。

用法（dehaze-test 目录下）：
    ../dehaze-python/.venv/bin/python -m kb_eval.retrieval_eval collect \\
        --strategies fixed,semantic [--docs N]   # 阶段一：执行检索落盘 raw（唯一有成本）
    ../dehaze-python/.venv/bin/python -m kb_eval.retrieval_eval evaluate \\
        [--threshold 0.5] [--strategies fixed,semantic]  # 阶段二：离线算指标出报告（零成本）
产物：reports/raw/{strategy}.jsonl + {strategy}.chunks.json（阶段一）、
      reports/retrieval_baseline_YYYYMMDD_HHMM.md（阶段二）
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent  # dehaze-test/kb_eval
WORKSPACE = HERE.parents[1]  # dehaze-system 根
sys.path.insert(0, str(HERE.parent))

from utils import api, auth  # noqa: E402

from kb_eval.testset_builder import TestCase, build_testset  # noqa: E402

BACKEND = "python"

# 建库契约（对齐 KnowledgeBaseCreateForm；只传 embeddingModel，供应商由
# sys_ai_model 注册表推导为 local）；vector 检索隔离 rerank 变量；512/64 与离线基线对齐
KB_FORM = {
    "visibility": "private",
    "embeddingModel": "bge-m3",
    "chunkSize": 512,
    "chunkOverlap": 64,
    "searchStrategy": "vector",
}
TOP_K = 5
DEFAULT_STRATEGIES = ["fixed", "semantic", "recursive", "table"]

# 命中判定常量（依据见模块 docstring 与报告附录校准分布）
SHINGLE_N = 10
HIT_THRESHOLD = 0.5  # 双向阈值默认值：块侧或节侧 shingle 覆盖率任一 ≥0.5 即命中

# 轮询/超时参数
POLL_INTERVAL_S = 2
DOC_PROCESS_TIMEOUT_S = 300  # 单篇处理超时：含推理排队，长文档 40 块 × ~1.5s + 模型冷启动余量
UPLOAD_ALL_TIMEOUT_S = 1800  # 全部文档串行处理的兜底总超时
SLOW_REQ_TIMEOUT_S = 60  # 慢请求超时（检索的 query 向量化 / 文件上传）

SIZE_ORDER = ["short", "medium", "long"]
STRUCT_ORDER = ["heading_text", "table_dense", "code_dense", "mixed"]
_EMPTY_RESULT = {"records": [], "invalid": [], "top1_scores": [], "zero_result": 0, "n_rows": 0}

RAW_DIR = HERE / "reports" / "raw"


def _raw_path(strategy: str) -> Path:
    return RAW_DIR / f"{strategy}.jsonl"


def _chunks_path(strategy: str) -> Path:
    return RAW_DIR / f"{strategy}.chunks.json"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


# ==================== 命中判定 ====================


def _norm(text: str) -> str:
    """去除全部空白（含全角空格）：对齐后端 _clean_text 的空白压缩，消除换行/缩进噪声。"""
    return "".join(text.split())


def _shingle_set(norm_text: str) -> set[str]:
    """10-gram 字符 shingle 集合；不足 N 字符的串整体作为单一 shingle。"""
    if len(norm_text) < SHINGLE_N:
        return {norm_text} if norm_text else set()
    return {norm_text[i : i + SHINGLE_N] for i in range(len(norm_text) - SHINGLE_N + 1)}


def containment(chunk_text: str, expected_text: str) -> float:
    """chunk 内容对期望文本的**块侧** shingle 覆盖率（0-1）= |交|/|chunk shingles|。

    逐字子集块 ≈1，仅套话/表头局部重合的块接近 0；超短块退化为整串包含判断。
    报告附录的重合度分布即此口径。
    """
    cn, en = _norm(chunk_text), _norm(expected_text)
    if not cn or not en:
        return 0.0
    c, e = _shingle_set(cn), _shingle_set(en)
    if len(cn) < SHINGLE_N:
        return 1.0 if cn in en else 0.0
    return len(c & e) / len(c)


def is_hit(chunk_text: str, expected_text: str, threshold: float = HIT_THRESHOLD) -> bool:
    """命中判定：块侧或节侧 shingle 覆盖率任一 ≥ threshold。

    为什么双向：单一"块侧"口径会把**短小节**的真实命中漏掉——250 字符的小节
    完整落入 ~750 字符的分块时，块侧覆盖率仅 ~0.33，但它确是检索应找回的块；
    补上"节侧"口径（|交|/|期望文本 shingles| ≥threshold，即块包含小节过半内容）后，
    短节命中可测。误报风险低：与目标节共享 ≥threshold 节侧内容的异节块，意味着
    大段逐字重复，属语料自身重复而非套话噪声。
    """
    cn, en = _norm(chunk_text), _norm(expected_text)
    if not cn or not en:
        return False
    c, e = _shingle_set(cn), _shingle_set(en)
    if not c or not e:
        return False
    inter = len(c & e)
    if len(cn) < SHINGLE_N:
        return cn in en
    return inter / len(c) >= threshold or inter / len(e) >= threshold


# ==================== API 封装（契约以 app/router/kb.py 为准） ====================


def _create_kb(name: str, strategy: str) -> int:
    resp = api.post(
        "/api/v1/kb", backend=BACKEND, json={"name": name, **KB_FORM, "chunkingStrategy": strategy}
    )
    return resp["data"]["id"]


def _delete_kb(kb_id: int) -> None:
    api.delete(f"/api/v1/kb/{kb_id}", backend=BACKEND)


def _list_kbeval_kbs() -> list[dict]:
    """管理端视角列出 kbeval_ 前缀库（该前缀为本套件专用）。"""
    resp = api.get(
        "/api/v1/kb",
        backend=BACKEND,
        params={"view": "admin", "keyword": "kbeval_", "pageNum": 1, "pageSize": 100},
    )
    return [k for k in resp["data"]["list"] if k["name"].startswith("kbeval_")]


def _sweep_stale_kbs() -> int:
    """清扫历史运行遗留的 kbeval_ 库（幂等自愈，保证不占满私有库配额）。"""
    stale = _list_kbeval_kbs()
    for kb in stale:
        _delete_kb(kb["id"])
        print(f"[cleanup] 删除遗留知识库 {kb['name']}")
    return len(stale)


def _upload_file(md_path: Path) -> int:
    """multipart 上传到 sys_file，返回 fileId（后端按 MD5 去重，重复轮次复用记录）。"""
    resp = api.post(
        "/api/v1/files",
        backend=BACKEND,
        files={"file": (md_path.name, md_path.read_bytes(), "text/markdown")},
        timeout=SLOW_REQ_TIMEOUT_S,
    )
    return resp["data"]["id"]


def _add_document(kb_id: int, file_id: int, title: str) -> int:
    """绑定文件为知识库文档，后台异步处理，返回 documentId。"""
    resp = api.post(
        f"/api/v1/kb/{kb_id}/documents", backend=BACKEND, json={"fileId": file_id, "title": title}
    )
    return resp["data"]["id"]


def _list_documents(kb_id: int) -> list[dict]:
    resp = api.get(
        f"/api/v1/kb/{kb_id}/documents", backend=BACKEND, params={"pageNum": 1, "pageSize": 100}
    )
    return resp["data"]["list"]


def _wait_document(kb_id: int, doc_id: int, deadline: float) -> str | None:
    """轮询单篇文档直至终态，返回失败原因（completed 返回 None，超时/failed 返回原因）。"""
    while time.time() < deadline:
        docs = _list_documents(kb_id)
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if doc is None:
            return "上传后未在文档列表中找到"
        st = doc.get("processingStatus")
        if st == "completed":
            return None
        if st == "failed":
            return doc.get("error") or "processing failed"
        time.sleep(POLL_INTERVAL_S)
    return f"处理超时(>{DOC_PROCESS_TIMEOUT_S}s 未终态)"


def _list_chunks(doc_id: int) -> list[str]:
    """拉取文档全部分块内容（Recall@K 的相关块全集用，分块数 ~15-40，单页 100 足够）。"""
    chunks: list[str] = []
    page = 1
    while True:
        resp = api.get(
            f"/api/v1/kb/documents/{doc_id}/chunks",
            backend=BACKEND,
            params={"pageNum": page, "pageSize": 100},
        )
        data = resp["data"]
        chunks.extend(c["content"] for c in data["list"])
        if len(chunks) >= data["total"] or not data["list"]:
            return chunks
        page += 1


def _retrieve(kb_id: int, query: str) -> list[dict]:
    """单库检索 Top-K（POST /{kb_id}/retrieve/test，走完整 search_service 链路）。"""
    resp = api.post(
        f"/api/v1/kb/{kb_id}/retrieve/test",
        backend=BACKEND,
        json={"query": query, "topK": TOP_K},
        timeout=SLOW_REQ_TIMEOUT_S,
    )
    return resp["data"]["results"] or []


# ==================== 指标 ====================


def _case_metrics(rels: list[bool], relevant_total: int) -> dict:
    """单用例指标。rels 为 Top-K 结果按返回顺序的命中布尔序列。

    Recall@K = 前 K 位命中块数 / 相关块全集（期望小节常跨多块，只有对全集
    归一才反映"该找回的找回了多少"）；NDCG@5 的理想序列取返回集内全部命中
    前置（对返回列表的二元判定求 NDCG，标准做法）。
    """
    hit_ranks = [i + 1 for i, r in enumerate(rels) if r]
    recall = {k: sum(rels[:k]) / relevant_total for k in (1, 3, 5)}
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rels), sum(rels))))
    return {
        "hit_ranks": hit_ranks,
        "recall1": recall[1],
        "recall3": recall[3],
        "recall5": recall[5],
        "mrr": mrr,
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "hit5": bool(hit_ranks),
    }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ==================== 阶段一：collect（唯一有成本环节） ====================


def _slim_manifest(docs_limit: int) -> dict:
    """manifest 精简：每个规模×结构格子取 1 篇（跳过对抗样本），格子按固定顺序排列。

    一篇/格即可覆盖全部分层组合，控制用例总量（格子数 × 每篇 ≤5 节 ≈ 30-50 条）；
    取格内路径最小者，确定性可复现（断点续跑"行数=用例数"判断的前提）。
    """
    manifest = json.loads((HERE / "reports" / "corpus_manifest.json").read_text(encoding="utf-8"))
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for item in manifest["items"]:
        if not item["adversarial"]:
            by_cell.setdefault((item["size_bucket"], item["struct_type"]), []).append(item)
    picked: list[dict] = []
    for size_b in SIZE_ORDER:
        for struct_t in STRUCT_ORDER:
            cell = sorted(by_cell.get((size_b, struct_t), []), key=lambda i: i["path"])
            if cell:
                picked.append(cell[0])
    if docs_limit:
        picked = picked[:docs_limit]
    return {**manifest, "items": picked}


def collect_strategy(strategy: str, cases: list[TestCase], doc_titles: dict[str, str]) -> int:
    """单策略 collect：建库 → 串行上传 → chunks 落盘 → 逐条检索逐条落盘 → 删库。

    返回落盘行数。断点续跑：raw 已完整（非空且行数=用例数）则不建库直接跳过。
    """
    raw_path = _raw_path(strategy)
    done = _count_lines(raw_path)
    if done > 0 and done == len(cases):
        print(f"[{strategy}] raw 已完整（{done}/{len(cases)} 行），跳过（不重新检索）")
        return done

    kb_name = f"kbeval_{strategy}_{datetime.now():%H%M%S}"
    kb_id = _create_kb(kb_name, strategy)
    print(f"[{strategy}] 已建库 {kb_name} (id={kb_id})", flush=True)
    try:
        # 串行上传（传一篇等一篇完成再传下一篇）。
        # 为什么串行：本地 embedding 是单实例 CPU 推理（512-token 块 ~1.3s/条），
        # 并发上传会让各批请求在推理锁上排队，单请求耗时被放大到分钟级，连锁触发
        # 客户端超时（评估实测 4 篇并发全军覆没）；串行同时避免并发分块入库的
        # MySQL 死锁（实测 INSERT sys_knowledge_chunk 出现 1213 死锁）。
        doc_ids: dict[str, int] = {}
        failed: dict[str, str] = {}
        upload_deadline = time.time() + UPLOAD_ALL_TIMEOUT_S
        for i, path in enumerate(doc_titles, 1):
            md_path = Path(path)
            file_id = _upload_file(md_path)
            doc_id = _add_document(kb_id, file_id, doc_titles[path])
            doc_ids[path] = doc_id
            err = _wait_document(kb_id, doc_id, upload_deadline)
            if err:
                failed[path] = err
                print(
                    f"[{strategy}] 上传 {i}/{len(doc_titles)}: {md_path.name} (doc={doc_id}) 失败: {err}",
                    flush=True,
                )
            else:
                print(
                    f"[{strategy}] 上传 {i}/{len(doc_titles)}: {md_path.name} (doc={doc_id}) 完成",
                    flush=True,
                )
        print(f"[{strategy}] 文档处理完成：{len(doc_ids) - len(failed)} 成功 / {len(failed)} 失败")

        # 相关块全集在检索前整体落盘：evaluate 重算 relevant_total（调阈值）必需，
        # 此后即与后端解耦；此处之后任意中断都不影响 chunk 数据完整性。
        chunks_by_title = {
            doc_titles[path]: _list_chunks(did) for path, did in doc_ids.items() if path not in failed
        }
        _chunks_path(strategy).write_text(
            json.dumps(chunks_by_title, ensure_ascii=False), encoding="utf-8"
        )
        n_chunks = sum(len(v) for v in chunks_by_title.values())
        print(
            f"[{strategy}] 分块落盘完成（{len(chunks_by_title)} 篇 / {n_chunks} 块），"
            f"开始检索 {len(cases)} 条用例",
            flush=True,
        )

        # 逐条检索逐条落盘（"w" 打开后逐行 write+flush = 追加语义；
        # 中断时已完成行完好，重跑 collect 会覆盖重写整个策略）。
        n_written = 0
        with raw_path.open("w", encoding="utf-8") as f:
            for i, case in enumerate(cases, 1):
                if case.doc_path in failed:
                    continue  # 上传失败文档的用例无 chunk 全集，不落盘（行数≠用例数 → 下次重跑）
                results = _retrieve(kb_id, case.query)
                row = {
                    "doc_path": case.doc_path,
                    "doc_title": case.doc_title,
                    "size": case.size_bucket,
                    "struct": case.struct_type,
                    "query": case.query,
                    "section_title": case.section_title,
                    "expected_text": case.expected_text,
                    "results": [{"content": r["content"], "score": r["score"]} for r in results],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1
                if not results:
                    print(f"[{strategy}] 空结果({i}/{len(cases)}): {case.query}", flush=True)
                elif i % 10 == 0:
                    print(f"[{strategy}] 检索进度 {i}/{len(cases)}", flush=True)
        return n_written
    finally:
        _delete_kb(kb_id)
        print(f"[{strategy}] 已删库 {kb_name} (id={kb_id})", flush=True)


def cmd_collect(args: argparse.Namespace) -> None:
    """阶段一：每策略执行一次完整检索链路，结果落盘 raw，不做任何指标计算。"""
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    cases = build_testset(_slim_manifest(args.docs))
    doc_titles = {c.doc_path: c.doc_title for c in cases}
    print(f"[corpus] {len(doc_titles)} 篇文档（每规模×结构格子 1 篇）/ {len(cases)} 条用例")
    print(f"[corpus] strategies={strategies}，raw 输出 {RAW_DIR}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    auth.login(backend=BACKEND)  # admin 登录（含探活：captcha → Redis 验证码）
    stale = _sweep_stale_kbs()
    if stale:
        print(f"[cleanup] 清扫历史遗留 kbeval_ 库 {stale} 个")

    errors: dict[str, Exception] = {}
    try:
        for strategy in strategies:
            try:
                n = collect_strategy(strategy, cases, doc_titles)
                print(f"[collect] {strategy}: raw 落盘 {n}/{len(cases)} 行 -> {_raw_path(strategy)}")
            except Exception as exc:  # 单策略失败不拖垮后续策略，raw 已落部分保完整
                errors[strategy] = exc
                print(f"[error] [{strategy}] collect 中断: {exc!r}", flush=True)
    finally:
        try:
            leftover = _list_kbeval_kbs()
            print(f"[final] kbeval_ 前缀残留库：{len(leftover)} 个")
        except Exception as exc:
            print(f"[final] 残留库检查失败（API 异常）: {exc!r}")

    if errors:
        raise SystemExit(1)


# ==================== 阶段二：evaluate（零成本可反复） ====================


def evaluate_strategy(strategy: str, threshold: float) -> dict:
    """读 raw 重放：relevant_total / 命中序列全部按当前 threshold 重算，零检索成本。

    invalid 判定（目标文档无任何达阈值相关块）也随阈值联动——collect 阶段不做
    任何判定，把全部用例的检索结果原样落盘。
    """
    rows = [
        json.loads(ln)
        for ln in _raw_path(strategy).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    chunks_map = json.loads(_chunks_path(strategy).read_text(encoding="utf-8"))
    records: list[dict] = []
    invalid: list[dict] = []
    top1_scores: list[float] = []
    zero_result = 0
    for row in rows:
        expected = row["expected_text"]
        chunks = chunks_map.get(row["doc_title"], [])
        relevant_total = sum(1 for c in chunks if is_hit(c, expected, threshold))
        if relevant_total == 0:
            # 期望文本在索引中找不到任何达阈值的块：无法度量召回（内容丢失/阈值过高）
            invalid.append(
                {
                    "doc_path": row["doc_path"],
                    "section_title": row["section_title"],
                    "reason": f"目标文档无相关块（chunks={len(chunks)}）",
                }
            )
            continue
        results = row["results"]
        if not results:
            zero_result += 1
        rels = [is_hit(r["content"], expected, threshold) for r in results]
        top1_scores.append(containment(results[0]["content"], expected) if results else 0.0)
        m = _case_metrics(rels, relevant_total)
        records.append(
            {
                "strategy": strategy,
                "query": row["query"],
                "section_title": row["section_title"],
                "doc_path": row["doc_path"],
                "doc_title": row["doc_title"],
                "doc_name": Path(row["doc_path"]).name,
                "size": row["size"],
                "struct": row["struct"],
                "relevant_total": relevant_total,
                "n_results": len(results),
                "top1_score": top1_scores[-1],
                **m,
            }
        )
    return {
        "records": records,
        "invalid": invalid,
        "top1_scores": top1_scores,
        "zero_result": zero_result,
        "n_rows": len(rows),
        "n_docs": len({r["doc_title"] for r in rows}),
    }


# ==================== 离线基线关联 ====================


def _load_offline_summary() -> tuple[str, dict, dict] | None:
    """解析最新离线基线报告的总览表与分层表（表格式样固定，按列数分派解析）。

    返回 (文件名, {策略: {fence, table_row, fragment}}, {(策略,规模,结构): {fragment}})。
    """
    reports = sorted(HERE.glob("reports/offline_baseline_*.md"))
    if not reports:
        return None
    path = reports[-1]
    overview: dict[str, dict] = {}
    layered: dict[tuple, dict] = {}
    section = None
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## 1."):
            section = "overview"
        elif line.startswith("## 2."):
            section = "layered"
        elif line.startswith("## "):
            section = None
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]

        def fnum(idx: int, cells=cells) -> float | None:
            try:
                return float(cells[idx])
            except (ValueError, IndexError):
                return None

        # 总览表 12 列：策略 n 块数 repl sentence fence table_row p50 p95 fragment oversized coverage
        if section == "overview" and len(cells) == 12 and cells[0] in DEFAULT_STRATEGIES:
            overview[cells[0]] = {"fence": fnum(5), "table_row": fnum(6), "fragment": fnum(9)}
        # 分层表 8 列：策略 规模 结构 n sentence fragment oversized coverage
        if section == "layered" and len(cells) == 8 and cells[0] in DEFAULT_STRATEGIES:
            layered[(cells[0], cells[1], cells[2])] = {"fragment": fnum(5)}
    return path.name, overview, layered


def _offline_struct_fragment(offline, strategy: str, struct_t: str) -> float | None:
    """该策略×结构在离线分层表的 fragment 均值（跨规模桶）。"""
    frags = [
        v["fragment"]
        for (st, _, stt), v in offline[2].items()
        if st == strategy and stt == struct_t and v["fragment"] is not None
    ]
    return _mean(frags) if frags else None


def _correlation_lines(strategies: list[str], results: dict, offline: tuple) -> list[str]:
    """生成"离线分块缺陷 ↔ 检索召回"的对照表与数据驱动陈述（不预设结论）。"""
    name, overview, _ = offline
    lines = [f"离线基线：`{name}`（同一份 corpus_manifest.json 抽样，参数 512/64 对齐）。"]

    lines += [
        "",
        "### 策略级对照",
        "",
        "| 策略 | 离线 fence_brk(次/篇) | 离线 table_row_brk | 离线 fragment | 检索 Recall@5 | 检索 MRR |",
        "|---|---|---|---|---|---|",
    ]
    for s in strategies:
        ov = overview.get(s, {})
        recs = results[s]["records"]
        lines.append(
            f"| {s} | {ov.get('fence') or 0:.2f} | {ov.get('table_row') or 0:.2f} "
            f"| {ov.get('fragment') or 0:.3f} | {_mean([r['recall5'] for r in recs]):.3f} "
            f"| {_mean([r['mrr'] for r in recs]):.3f} |"
        )

    lines += [
        "",
        "### 结构级对照（检索 Recall@5 vs 该结构分层在离线基线的 fragment 均值）",
        "",
        "| 策略 | 结构 | 检索 Recall@5 | n | 离线 fragment |",
        "|---|---|---|---|---|",
    ]
    for s in strategies:
        for struct_t in STRUCT_ORDER:
            recs = [r for r in results[s]["records"] if r["struct"] == struct_t]
            if not recs:
                continue
            frag = _offline_struct_fragment(offline, s, struct_t)
            frag_txt = f"{frag:.3f}" if frag is not None else "N/A"
            lines.append(
                f"| {s} | {struct_t} | {_mean([r['recall5'] for r in recs]):.3f} | {len(recs)} | {frag_txt} |"
            )

    lines += ["", "### 对照结论", ""]
    fence_sorted = sorted(strategies, key=lambda s: -(overview.get(s, {}).get("fence") or 0))
    recall_sorted = sorted(
        strategies, key=lambda s: _mean([r["recall5"] for r in results[s]["records"]])
    )
    worst_fence, worst_recall = fence_sorted[0], recall_sorted[0]
    lines.append(
        f"- 离线围栏切断最多的策略是 {worst_fence}（{overview[worst_fence]['fence']:.2f} 次/篇）；"
        f"检索 Recall@5 最低的策略是 {worst_recall}"
        f"（{_mean([r['recall5'] for r in results[worst_recall]['records']]):.3f}）。"
        + (
            "两者一致，支持『围栏切断 → 代码块语义破损 → 向量漂移 → 召回下降』的传导方向。"
            if worst_fence == worst_recall
            else "两者不一致：检索召回还叠加 embedding/查询形态/阈值过滤等链路因素，不能单归因于分块缺陷。"
        )
    )
    for s in strategies:
        recs = results[s]["records"]
        by_struct = {
            t: _mean([r["recall5"] for r in recs if r["struct"] == t])
            for t in STRUCT_ORDER
            if any(r["struct"] == t for r in recs)
        }
        if not by_struct:
            continue
        low_struct = min(by_struct, key=lambda t: by_struct[t])
        frag = _offline_struct_fragment(offline, s, low_struct)
        frag_txt = f"，其离线 fragment={frag:.3f}" if frag is not None else ""
        lines.append(f"- {s} 检索 Recall@5 最差的结构分层为 {low_struct}（{by_struct[low_struct]:.3f}）{frag_txt}。")
    return lines


# ==================== 报告 ====================


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_report(strategies: list[str], results: dict, n_docs: int, n_cases: int, threshold: float) -> str:
    raw_meta = "；".join(
        f"{s} {results[s]['n_rows']} 行（collect 完成于 "
        f"{datetime.fromtimestamp(_raw_path(s).stat().st_mtime):%m-%d %H:%M}）"
        for s in strategies
        if results[s].get("n_rows")
    )
    lines = [
        "# 知识库检索质量基线报告",
        "",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- git commit：`{_git_commit()}`（dehaze-system workspace）",
        "- 链路：collect 阶段走 8991 真实后端（bge-m3/local → KB → 解析分块 → ES 向量检索 Top-5），"
        "每策略独立建库串行执行、finally 删库；本报告为 evaluate 阶段离线重放 raw 数据（零检索成本）",
        f"- raw 数据：reports/raw/*.jsonl —— {raw_meta}",
        f"- 语料：corpus_manifest.json 每个规模×结构格子 1 篇，共 {n_docs} 篇真实文档（对抗样本不参与），"
        f"测试用例 {n_cases} 条（结构即标注：query=「一级标题的{{小节标题}}」，expected_text=小节正文）",
        "- 参数：chunk_size=512，chunk_overlap=64，searchStrategy=vector，"
        f"score_threshold=建库默认 0.5，top_k={TOP_K}，strategies={','.join(strategies)}",
        f"- 命中判定：{SHINGLE_N}-gram shingle 覆盖率（chunk 对 expected_text）"
        f"块侧或节侧任一 ≥ {threshold}；"
        "依据：逐字子集块块侧≈1.0、短节完整入块时节侧≈1.0，跨节套话两侧均 <0.15，间隔宽（校准分布见附录）；"
        "阈值在 evaluate 阶段参数化，调整只需重放 raw，无需重新检索",
        "",
        "## 1. 总览（策略 × 指标）",
        "",
        "| 策略 | 有效用例 | 无效用例 | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 | HitRate@5 | 空结果 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in strategies:
        r = results.get(s, _EMPTY_RESULT)
        recs = r["records"]
        lines.append(
            f"| {s} | {len(recs)} | {len(r['invalid'])} "
            f"| {_mean([x['recall1'] for x in recs]):.3f} | {_mean([x['recall3'] for x in recs]):.3f} "
            f"| {_mean([x['recall5'] for x in recs]):.3f} | {_mean([x['mrr'] for x in recs]):.3f} "
            f"| {_mean([x['ndcg'] for x in recs]):.3f} "
            f"| {_mean([1.0 if x['hit5'] else 0.0 for x in recs]):.3f} | {r['zero_result']} |"
        )

    lines += [
        "",
        "## 2. 分层 Recall@5（策略 × 规模 × 结构）",
        "",
        "| 策略 | 规模 | 结构 | n | Recall@5 |",
        "|---|---|---|---|---|",
    ]
    for s in strategies:
        for size_b in SIZE_ORDER:
            for struct_t in STRUCT_ORDER:
                recs = [
                    x for x in results.get(s, _EMPTY_RESULT)["records"]
                    if x["size"] == size_b and x["struct"] == struct_t
                ]
                if recs:
                    lines.append(
                        f"| {s} | {size_b} | {struct_t} | {len(recs)} "
                        f"| {_mean([x['recall5'] for x in recs]):.3f} |"
                    )

    lines += [
        "",
        "## 3. 零命中 query 清单（Top20，直接指示检索弱点）",
        "",
        "| # | 策略 | 文档 | query | 期望节标题 | Recall@5 | Top-1 重合度 |",
        "|---|---|---|---|---|---|---|",
    ]
    zero = [x for s in strategies for x in results.get(s, _EMPTY_RESULT)["records"] if not x["hit5"]]
    if zero:
        for i, x in enumerate(zero[:20], 1):
            lines.append(
                f"| {i} | {x['strategy']} | {x['doc_title']} | {x['query']} "
                f"| {x['section_title']} | {x['recall5']:.3f} | {x['top1_score']:.2f} |"
            )
    else:
        lines.append("| - | - | - | （无零命中用例） | - | - | - |")

    lines += ["", "## 4. 与离线分块基线关联分析", ""]
    offline = _load_offline_summary()
    if offline:
        lines += _correlation_lines(strategies, results, offline)
    else:
        lines.append("未找到离线基线报告（reports/offline_baseline_*.md），跳过关联分析。")

    lines += [
        "",
        "## 附录：命中阈值校准（Top-1 内容重合度分布）",
        "",
        "| 策略 | <0.1 | 0.1-0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.9 | ≥0.9 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in strategies:
        buckets = [0] * 6
        for v in results.get(s, _EMPTY_RESULT)["top1_scores"]:
            idx = 0 if v < 0.1 else 1 if v < 0.3 else 2 if v < 0.5 else 3 if v < 0.7 else 4 if v < 0.9 else 5
            buckets[idx] += 1
        lines.append(f"| {s} | " + " | ".join(str(b) for b in buckets) + " |")
    lines += [
        "",
        "上表为块侧口径（|交|/|chunk|）。理想命中应落在 ≥0.5 区间；0.3-0.5 区间的用例"
        "可能经节侧口径（块包含小节过半内容，短小节场景）判为命中；若 0.1-0.3 区间"
        "显著堆积且对应零命中清单，说明该策略把期望文本切碎/丢失，需回看分块行为。",
        "",
        "## 附录：无效用例明细（期望文本未入库，无法度量召回）",
        "",
    ]
    invalid_rows = [
        f"- [{s}] {Path(i['doc_path']).name} «{i['section_title']}»：{i['reason']}"
        for s in strategies
        for i in results.get(s, _EMPTY_RESULT)["invalid"]
    ]
    lines += invalid_rows if invalid_rows else ["（无）"]
    lines.append("")
    return "\n".join(lines)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """阶段二：读 raw 重放计算指标并出报告；调阈值可多次执行，秒级完成。"""
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = sorted(p.name[: -len(".jsonl")] for p in RAW_DIR.glob("*.jsonl"))
    if not strategies:
        raise SystemExit(f"raw 目录无数据（{RAW_DIR}），先执行 collect 阶段")

    results: dict = {}
    for strategy in strategies:
        results[strategy] = evaluate_strategy(strategy, args.threshold)
    r0 = next(iter(results.values()))
    report = build_report(strategies, results, r0["n_docs"], r0["n_rows"], args.threshold)
    out = HERE / "reports" / f"retrieval_baseline_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(report, encoding="utf-8")

    print(f"[evaluate] threshold={args.threshold}，报告 -> {out}")
    for s in strategies:
        recs = results[s]["records"]
        print(
            f"[{s}] raw {results[s]['n_rows']} 行，有效 {len(recs)} / 无效 {len(results[s]['invalid'])}，"
            f"Recall@5={_mean([r['recall5'] for r in recs]):.3f}，"
            f"MRR={_mean([r['mrr'] for r in recs]):.3f}，空结果 {results[s]['zero_result']} 条"
        )


# ==================== 主流程 ====================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="知识库检索质量评估（两阶段：collect 真实检索落盘 / evaluate 离线重放算指标）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="阶段一：真实链路执行检索，逐条落盘 raw（唯一有成本环节）")
    p_collect.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help="逗号分隔的分块策略")
    p_collect.add_argument("--docs", type=int, default=0, help="格子精选后再截前 N 篇文档（调试用，0=不截）")
    p_collect.set_defaults(func=cmd_collect)

    p_eval = sub.add_parser("evaluate", help="阶段二：读 raw 离线计算指标并出报告（零成本可反复）")
    p_eval.add_argument("--threshold", type=float, default=HIT_THRESHOLD, help="命中阈值（双向 shingle 覆盖率）")
    p_eval.add_argument("--strategies", default="", help="逗号分隔的策略，缺省自动发现 reports/raw/*.jsonl")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    t0 = time.time()
    args.func(args)
    print(f"[done] {args.cmd} 耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
