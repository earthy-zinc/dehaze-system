# kb_eval —— 知识库质量评估

定位：dehaze-test 的**度量型任务**（见 `../README.md`），产出指标报告供人决策，
无 pass/fail 断言；评估发现的硬性不变量（如 replacement_char_count=0）应沉淀回
`dehaze-python/tests`，不留在这里当长期断言。

## 离线分块评估（offline_eval）

对 dehaze-python 分块引擎（`app/service/kb/chunking_engine.py` 的
`chunk_text(text, strategy, chunk_size, chunk_overlap)`）做**纯算法级**质量评估：
不起后端、不连 DB/Redis，语料为 `dehaze-doc/docs` 的 221 篇真实 Markdown。

### 运行

```bash
cd dehaze-test
../dehaze-python/.venv/bin/python -m kb_eval.offline_eval
# 可选参数
#   --strategies fixed,semantic,recursive,qa,table   参与评估的策略
#   --per-cell 4                                     分层矩阵每格抽样篇数
```

### 语料与分层

扫描 `dehaze-doc/docs/**/*.md`，跳过 `.vitepress/`、`public/`、`reports/` 等非正文目录。

**规模**（按字节数）：short < 5KB / medium 5–30KB / long > 30KB。

**结构主导类型**（按占非空行比例判定，阈值见 `corpus_selector.py` 顶部常量）：

| 类型 | 判定规则 | 典型文档 |
|---|---|---|
| table_dense | 表格行（`\|` 开头，围栏外）占比 ≥ 0.35 | API 接口、数据库设计 |
| code_dense | 代码围栏内行（含围栏标记）占比 ≥ 0.35 | 后端实现 |
| heading_text | 表格 + 代码合计占比 < 0.12 | 需求、设计说明 |
| mixed | 其余 | 前端实现（正文+表格+代码混合） |

每格**确定性等距抽样** per_cell 篇（路径排序后等距取点，可复现），格子内文档不足
per_cell 时全取（如 short × heading_text 全库仅 2 篇）；另附 8 个内存构造的
对抗样本（不落盘）：`zero_width`（U+200B）/ `crlf` / `bom` / `long_line`（>5000 字符无换行，
后半无标点）/ `mixed_no_punct`（中英混杂无标点）/ `emoji_dense` / `table_only` / `code_fence_only`。

抽样清单落 `reports/corpus_manifest.json`（含每篇 sha256 前 16 位，用于校验语料漂移）；
检索链路评估（retrieval_eval）复用同一份清单 + 对抗样本 key，保证两阶段可比。

### 指标定义

| 指标 | 定义 | 期望 |
|---|---|---|
| replacement_char_count | 块内容中 U+FFFD（乱码替换符）个数 | **= 0** |
| sentence_complete_rate | 非末块中块尾（rstrip 后）为句末标点（。！？!?；;）的比例；块数 < 2 时无意义 | 越高越好；qa/table 因结构语义优先不适用，标 N/A。本语料以表格/代码/列表行为主，行尾字符多为 `\|`、代码符号，该指标绝对值天然偏低，**看策略间相对差异** |
| code_fence_broken | 块内 ``` 标记出现奇数次的块数（= 围栏被块边界切断的块数） | 越低越好 |
| table_row_broken | 表格行连续性破坏计数 = 块内出现原文中不存在的表格行（被粘连成一行 / 被 token 硬切截断的残行）+ 原文相邻表格行被拆进相邻两块 | 越低越好 |
| token_p50 / p95 | 块 token 数分位数（最近秩法，cl100k_base 口径） | p95 应接近 chunk_size |
| fragment_rate | token < 50 的块占比（阈值与引擎 `_MIN_TOKENS` 对齐） | 越低越好 |
| oversized_rate | token > chunk_size×1.2 的块占比（软预算余量） | fixed/semantic/recursive 越低越好；qa/table 为保留结构可超预算 |
| coverage | 全部块内容（去空白）对原文（去空白）的字符**多重集**覆盖率 | ≈ 1.0 |
| chunk_count | 块总数 | — |

评估口径：`chunk_size=512`、`chunk_overlap=64`（用户建库实际配置）；每篇文档先算指标，
再按组**宏平均**（文档等权）。coverage 取多重集口径：overlap 重复字符无碍（取 min），
丢段、丢行、字符损坏都会拉低它。

### 报告

`reports/offline_baseline_YYYYMMDD_HHMM.md`：

1. 总览表：策略 × 指标均值（全语料）
2. 分层表：策略 × 规模 × 结构 的 sentence / fragment / oversized / coverage
3. 对抗样本专项表
4. 瓶颈清单：劣化分（fragment + oversized + (1-coverage) + (1-sentence)）最差的
   5 个 策略×分层 组合及典型坏例摘录（≤200 字符）；平均块数 < 5 的组合不参与排名
   （短文档块数过少，单块尾噪声无统计意义）

报告头记录 git commit、时间、参数，可与历史报告同表对比（指标口径固定）。

## 在线检索评估（retrieval_eval，两阶段）

对 8991 真实产品链路（建 KB → 上传文档 → 后台解析分块向量化 → ES 检索）做端到端
质量评估。**两阶段解耦：检索执行是唯一有成本环节，每策略只执行一次并落盘 raw；
一切调参（命中阈值/指标口径/报告格式）走 evaluate 离线重放，秒级完成，
严禁为调参重新建库或重新检索。**

### 运行

```bash
cd dehaze-test
../dehaze-python/.venv/bin/python -m kb_eval.retrieval_eval collect --strategies fixed,semantic
# 阶段一：manifest 每个规模×结构格子选 1 篇（跳过对抗样本）→ admin 建私有库上传 →
#   逐条 query 检索，结果逐条追加落盘 reports/raw/{strategy}.jsonl，finally 删库；
#   raw 已完整（非空且行数=用例数）的策略自动跳过（断点续跑）；不产出任何指标
../dehaze-python/.venv/bin/python -m kb_eval.retrieval_eval evaluate [--threshold 0.5]
# 阶段二：读 raw 重放计算指标出报告，调阈值重跑秒级完成
```

### raw 格式与指标

- `reports/raw/{strategy}.jsonl` 每行：
  `{doc_path, doc_title, size, struct, query, section_title, expected_text, results:[{content, score}]}`；
  `{strategy}.chunks.json`：doc_title → 目标文档全部 chunk 内容（evaluate 重算
  Recall@K 分母用），在检索前整体落盘。
- 命中判定：10-gram shingle 覆盖率（chunk 对 expected_text）块侧或节侧任一
  ≥ threshold（默认 0.5，evaluate `--threshold` 可调）。
- 指标：Recall@K 分母 = 目标文档全部 chunk 中与期望文本重合达阈值的块数
  （期望小节常被切成多块，仅数 Top-K 命中块数会低估召回）。

### 报告

`reports/retrieval_baseline_YYYYMMDD_HHMM.md`：总览（Recall@1/3/5、MRR、NDCG@5、
HitRate@5、空结果）、规模×结构分层 Recall@5、零命中 query 清单、与离线分块基线
关联分析、Top-1 重合度分布（阈值校准依据）、无效用例明细。报告头记录 raw 落盘
时间与行数，可与历史报告同表对比。
