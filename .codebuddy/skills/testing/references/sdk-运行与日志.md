# 运行与日志

## 输出格式

测试输出已配置成紧凑 JSON（`test/config/compact-reporter.ts`），stdout 每次运行只输出一行：

- 通过：`{"state":"passed","files":N,"passed":N,"failed":0,"skipped":N,"duration":"Xs"}`
- 失败：追加 `failures` 数组，每条含 `file:行号`、完整用例名、`errors` 数组（语义顺序为断言错误、请求错误信息、堆栈帧）；收集错误与未捕获错误一并放进 `errors`。

同时写两份文件到 `test/logs/<backend>/`，同步轮转保留最近 10 份，`<backend>` 由 `BACKEND_URL` 的端口推导；单元测试写 `test/logs/unit/`：

- `brief.json`：与终端输出一致的简要报告，供大模型查阅。
- `detail.log`：NDJSON 格式，每行一个 JSON 对象，记录全部用例（含通过），每例含完整请求与响应（带时间戳）、单用例耗时、完整错误堆栈，用于性能分析与疑难排查。

## 运行期进度

进度走 stderr，不影响 stdout 的 JSON。全量集成测试串行跑 60 多个文件、可能超过一小时，进度行用来判断是卡住还是正常慢：

- `▶ file`：文件开始执行
- `[i/N] ✓/✗ file 通过x 失败y 跳过z 12.3s`：文件完成
- `✗ file:行 用例名` 与 `⏱ 慢用例 8.4s ...`：失败用例与慢用例（≥5s）即时输出，不必等整轮跑完
- `⏱ 已运行 ... ｜ 当前 file 已 2m05s ｜ 距上次进展 2m05s`：每 30s 一次心跳；距上次进展达 120s 时提示疑似后端不可达或钩子卡住

前台运行时进度与 JSON 都在终端可见；重定向日志文件时进度在 stderr、JSON 在 stdout。

## 用 NDJSON 查问题

每行一个 JSON 对象，第一行是 `type:"summary"` 汇总，之后每行是 `type:"test"` 用例记录。可用 `jq`、`grep` 或 Python `json.loads` 按行解析，支持按 `traceId`、`timestamp`、`url`、`state` 精确过滤，排查并行测试的时序问题尤其好用。

断言失败直接依据 `failures` 里的 `file:行号` 与 `errors` 数组定位；疑难问题查 `detail.log` 还原完整请求与堆栈，无需重跑。需要恢复默认排版输出时，临时加 `--reporter=default`。
