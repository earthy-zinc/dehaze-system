/**
 * 紧凑报告器：面向大模型查阅与疑难排查的双输出
 *
 * 输出到 dehaze-sdk-js/test/logs/<backend>/（brief.json 与 detail.json 同步轮转，保留最近 MAX_LOG_FILES 份）：
 * - brief.json：简要报告（汇总 + 失败/跳过用例精简信息 + 收集错误），供大模型快速查阅
 * - detail.json：NDJSON 格式详细日志（每行一个 JSON 对象），记录全部用例（含通过），
 *   每例输出完整请求/响应（含时间戳）/耗时/完整堆栈，用于性能分析与疑难排查
 *
 * NDJSON 格式优势：
 * - 可用 grep/jq/Python json 按行解析，无需读取整个文件
 * - 支持按 traceId/时间戳/路径/状态精确过滤，适合排查时序问题
 *
 * 请求/响应由 test/config/vitest.setup.ts 的请求拦截器与 transformResponse 捕获，经 onTestFinished
 * 写入 task.meta（所有用例均写入），reporter 收集为 TestRecord[] 后分别由 generateBrief/generateDetailJson 处理。
 *
 * 运行期进度（stderr）：
 * - 起始行、逐文件开始/结束、失败与慢用例、30s 心跳（含当前文件与"距上次进展"时长），
 *   用于全量长跑时判断进度、定位卡在哪个用例或钩子
 * - 进度走 stderr，stdout 保持"仅一行 brief JSON"的契约，便于脚本/大模型解析
 *
 * 日志分组：test/logs/<BACKEND_NAME>/，unit 测试经 SDK_TEST_LOG_GROUP 写入 test/logs/unit/，
 * 避免 unit 运行轮转掉集成测试的报告历史。
 */
import type {
  Reporter,
  SerializedError,
  TestCase,
  TestModule,
  TestResultSkipped,
  TestRunEndReason,
  TestSpecification,
} from "vitest/node";
import fs from "fs";
import path from "path";
import type { RequestData, ResponseData } from "@/types";
import { BACKEND_NAME } from "./constant";

const MAX_VALUE_LENGTH = 300;
const MAX_STACK_LINES = 5;
const MAX_LOG_FILES = 10;
const LOGS_DIR = path.resolve(__dirname, "../logs");

// 心跳间隔；慢用例阈值（超过即逐条输出，用于定位拖慢全量的用例）；无进展时长达 testTimeout 时的告警阈值
const HEARTBEAT_MS = 30_000;
const SLOW_CASE_MS = 5_000;
const STALL_HINT_MS = 120_000;

type TestState = "passed" | "failed" | "skipped" | "pending";

/**
 * 带时间戳的请求记录（测试捕获）
 */
export interface CapturedRequest extends RequestData {
  /** 请求发起时间（ISO 8601） */
  timestamp: string;
}

/**
 * 带时间戳的响应记录（测试捕获）
 */
export interface CapturedResponse<T = any> extends ResponseData<T> {
  /** 响应接收时间（ISO 8601） */
  timestamp: string;
}
/**
 * 统一测试记录：brief 与 detail 共用
 * - detail 形态：errors 为完整 SerializedError[]
 * - brief 形态：errors 为 string[]，按语义顺序：断言错误、请求错误信息、堆栈帧
 * duration/slow 两种形态均携带，brief 展示失败/跳过用例时亦可定位慢用例
 */
interface TestRecord {
  file: string;
  test: string;
  state: TestState;
  /** 单用例耗时，来自 diagnostic().duration，格式如 "234.23ms" */
  duration?: string;
  /** 是否超过 slowTestThreshold，用于定位慢用例 */
  slow?: boolean;
  requests?: CapturedRequest[];
  responses?: CapturedResponse<unknown>[];
  /** detail: 完整错误对象数组；brief: string[]（断言错误 / 请求错误信息 / 堆栈帧） */
  errors?: ReadonlyArray<SerializedError> | string[];
  /** 跳过原因 */
  reason?: string;
}

/** brief.json 顶层结构 */
interface BriefSummary {
  state: TestRunEndReason;
  files: number;
  passed: number;
  failed: number;
  skipped: number;
  duration: string;
  failures?: TestRecord[];
  skippedTests?: TestRecord[];
  errors?: string[];
}

// ---- 类型守卫 ----

/** 区分 detail 形态的 SerializedError[] 与 brief 形态的 string[]：detail 首个元素为对象 */
function isDetailErrors(errors: unknown): errors is ReadonlyArray<SerializedError> {
  return Array.isArray(errors) && errors.length > 0 && typeof errors[0] !== "string";
}

// ---- 工具函数 ----

// Vitest 断言失败信息按终端着色能力内嵌 ANSI 转义序列，JSON/日志输出前需剥离
function stripAnsi(text: string): string {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
}

/** 进度行走 stderr：stdout 留给 brief JSON，保证脚本/大模型解析不被进度污染 */
function progress(message: string): void {
  process.stderr.write(`[${new Date().toTimeString().slice(0, 8)}] ${message}\n`);
}

/** 日志分组：默认按被测后端分目录，unit 配置经 SDK_TEST_LOG_GROUP 指向独立目录 */
function logGroup(): string {
  return process.env.SDK_TEST_LOG_GROUP || BACKEND_NAME;
}

function truncate(value: unknown, maxLength = MAX_VALUE_LENGTH): string | undefined {
  if (value === undefined) return undefined;
  const text = stripAnsi(typeof value === "string" ? value : (JSON.stringify(value) ?? ""));
  if (text.length <= maxLength) return text;
  const ellipsis = "…[truncated]…";
  const half = (maxLength - ellipsis.length) >> 1;
  return `${text.slice(0, half)}${ellipsis}${text.slice(-half)}`;
}

// brief 用：过滤 node_modules/node:internal 后仅保留前 MAX_STACK_LINES 帧
function trimStack(stack: string | undefined): string[] | undefined {
  if (!stack) return undefined;
  const frames = stack
    .split("\n")
    .map((line) => stripAnsi(line.trim()))
    .filter(
      (line) =>
        line.startsWith("at ") &&
        !line.includes("node_modules") &&
        !line.includes("node:internal") &&
        !line.includes("<anonymous>")
    );
  return frames.length > 0 ? frames.slice(0, MAX_STACK_LINES) : undefined;
}

function formatLocation(testModule: TestModule, testCase: TestCase): string {
  return testCase.location
    ? `${testModule.relativeModuleId}:${testCase.location.line}`
    : testModule.relativeModuleId;
}

// ---- 数据收集 ----

/** 遍历全部测试，收集完整记录（含通过用例）与模块/套件级收集错误 */
function collectAll(testModules: ReadonlyArray<TestModule>): {
  records: TestRecord[];
  errors: string[];
} {
  const records: TestRecord[] = [];
  const errors: string[] = [];

  for (const testModule of testModules) {
    for (const error of testModule.errors()) {
      errors.push(`${testModule.relativeModuleId}: ${stripAnsi(error.message)}`);
    }
    for (const suite of testModule.children.allSuites()) {
      for (const error of suite.errors()) {
        errors.push(
          `${testModule.relativeModuleId} > ${suite.fullName}: ${stripAnsi(error.message)}`
        );
      }
    }
    for (const testCase of testModule.children.allTests()) {
      const result = testCase.result();
      const diagnostic = testCase.diagnostic();
      const meta = testCase.meta?.() as Record<string, unknown> | undefined;
      const record: TestRecord = {
        file: formatLocation(testModule, testCase),
        test: testCase.fullName,
        state: result.state,
        slow: diagnostic?.slow ?? false,
      };
      if (diagnostic?.duration !== undefined) {
        record.duration = `${diagnostic.duration.toFixed(2)}ms`;
      }
      const requests = meta?.requests as CapturedRequest[] | undefined;
      const responses = meta?.responses as CapturedResponse<unknown>[] | undefined;
      if (requests && requests.length > 0) record.requests = requests;
      if (responses && responses.length > 0) record.responses = responses;
      if (result.state === "skipped") {
        record.reason = resolveSkipReason(testCase, result);
      } else if (result.state === "failed") {
        record.errors = result.errors;
      }
      records.push(record);
    }
  }

  return { records, errors };
}

// 跳过原因优先级：ctx.skip 备注 > todo/skip 标记 > 套件钩子失败 > only 挤占
function resolveSkipReason(testCase: TestCase, result: TestResultSkipped): string {
  if (result.note) return result.note;
  const mode = testCase.options.mode;
  if (mode === "todo") return "todo 标记（待实现）";
  if (mode === "skip") return "skip/skipIf 标记";
  // mode 为 run/only：用例自身未标记跳过，沿套件链定位钩子或收集失败
  let parent = testCase.parent;
  while (parent.type !== "module") {
    if (parent.errors().length > 0) return `所在套件「${parent.name}」钩子失败，详见 errors`;
    parent = parent.parent;
  }
  if (parent.errors().length > 0) return "测试模块收集失败，详见 errors";
  return "其他用例标记 only 导致未执行";
}

// ---- brief.json 生成（TestRecord → brief 形态的截断/扁平化）----

/** 将完整记录转为 brief 形态：errors 转为语义 string[]（断言/请求/堆栈），截断请求/响应 */
function toBriefRecord(record: TestRecord): TestRecord {
  const brief: TestRecord = {
    file: record.file,
    test: record.test,
    state: record.state,
  };
  if (record.duration !== undefined) brief.duration = record.duration;
  if (record.slow !== undefined) brief.slow = record.slow;

  if (record.state === "skipped") {
    brief.reason = record.reason ?? "";
    return brief;
  }

  // failed：errors 数组 → 语义 string[]：[断言错误, 请求错误信息, ...堆栈帧]
  if (record.state === "failed" && isDetailErrors(record.errors)) {
    const testErrors = record.errors;
    const firstError = testErrors[0];
    const lines: string[] = [];

    // 1) 断言错误信息（有 expected/actual 时）
    const expected = truncate(firstError?.expected);
    const actual = truncate(firstError?.actual);
    if (expected !== undefined || actual !== undefined) {
      const exp = expected ?? "?";
      const act = actual ?? "?";
      lines.push(`[断言失败] expected ${exp} to be ${act}`);
    }

    // 2) 请求错误信息
    const msg = stripAnsi(firstError?.message ?? "unknown error");
    const suffix = testErrors.length > 1 ? ` (+${testErrors.length - 1} more errors)` : "";
    lines.push(msg + suffix);

    // 3) 堆栈帧
    const frames = trimStack(firstError?.stack);
    if (frames) lines.push(...frames);

    brief.errors = lines;
  }

  // 截断请求/响应
  if (record.responses && record.responses.length > 0) {
    brief.responses = record.responses.map((r) => ({
      code: r.code,
      msg: r.msg,
      traceId: r.traceId,
      timestamp: r.timestamp,
      data: truncate(r.data, 1000) ?? "",
    }));
  }
  if (record.requests && record.requests.length > 0) {
    brief.requests = record.requests.map((r) => {
      const req: CapturedRequest = { method: r.method, url: r.url, timestamp: r.timestamp };
      const params = truncate(r.params, 1000);
      const body = truncate(r.body, 1000);
      if (params !== undefined) req.params = params;
      if (body !== undefined) req.body = body;
      return req;
    });
  }

  return brief;
}

/**
 * 生成 brief.json：从全量 TestRecord[] 后处理出精简报告
 * 过滤失败/跳过用例 → toBriefRecord 截断/扁平化，通过用例仅计入计数
 */
function generateBrief(
  records: TestRecord[],
  errors: string[],
  state: TestRunEndReason,
  files: number,
  duration: string
): BriefSummary {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  const failures: TestRecord[] = [];
  const skippedTests: TestRecord[] = [];

  for (const r of records) {
    if (r.state === "passed") {
      passed++;
    } else if (r.state === "skipped") {
      skipped++;
      skippedTests.push(toBriefRecord(r));
    } else if (r.state === "failed") {
      failed++;
      failures.push(toBriefRecord(r));
    }
  }

  return {
    state,
    files,
    passed,
    failed,
    skipped,
    duration,
    ...(failures.length > 0 && { failures }),
    ...(skippedTests.length > 0 && { skippedTests }),
    ...(errors.length > 0 && { errors }),
  };
}

// ---- detail.json 生成（NDJSON 格式，每行一个 JSON 对象，全量完整）----

/**
 * 生成 detail.json：NDJSON（Newline Delimited JSON），每行一个 JSON 对象
 *
 * 结构：
 * - 第 1 行：summary 对象（汇总信息）
 * - 后续每行：单个 TestRecord（含请求/响应/耗时/完整堆栈/时间戳）
 *
 * NDJSON 优势：
 * - 可用 grep/jq/Python json 按行解析，无需读取整个文件
 * - 每行独立 JSON，便于按 traceId/时间戳/路径/状态过滤
 * - 支持流式读取，适合大文件
 */
function generateDetailJson(records: TestRecord[], brief: BriefSummary): string {
  const lines: string[] = [];

  // 第 1 行：汇总信息
  const summary = {
    type: "summary" as const,
    backend: BACKEND_NAME,
    generatedAt: new Date().toISOString(),
    duration: brief.duration,
    state: brief.state,
    files: brief.files,
    passed: brief.passed,
    failed: brief.failed,
    skipped: brief.skipped,
    total: records.length,
    collectErrors: brief.errors ?? [],
  };
  lines.push(JSON.stringify(summary));

  // 后续每行：单个用例记录
  for (const r of records) {
    const record: Record<string, unknown> = {
      type: "test" as const,
      file: r.file,
      test: r.test,
      state: r.state,
      duration: r.duration ?? "0.00ms",
    };
    if (r.slow) record.slow = true;
    if (r.reason) record.reason = r.reason;

    // 完整错误（detail 形态）
    if (isDetailErrors(r.errors)) {
      record.errors = r.errors.map((err) => ({
        message: stripAnsi(err.message ?? ""),
        expected: err.expected !== undefined ? stripAnsi(String(err.expected)) : undefined,
        actual: err.actual !== undefined ? stripAnsi(String(err.actual)) : undefined,
        diff: err.diff ? stripAnsi(String(err.diff)) : undefined,
        stack: err.stack ? stripAnsi(err.stack) : undefined,
      }));
    }

    // 请求/响应配对
    if ((r.requests && r.requests.length > 0) || (r.responses && r.responses.length > 0)) {
      const exchangeCount = Math.max(r.requests?.length ?? 0, r.responses?.length ?? 0);
      const exchanges: unknown[] = [];
      for (let j = 0; j < exchangeCount; j++) {
        const req = r.requests?.[j];
        const res = r.responses?.[j];
        exchanges.push({
          request: req
            ? {
                method: req.method,
                url: req.url,
                params: req.params,
                body: req.body,
                timestamp: req.timestamp,
              }
            : null,
          response: res
            ? {
                code: res.code,
                msg: res.msg,
                traceId: res.traceId,
                data: res.data,
                timestamp: res.timestamp,
              }
            : null,
        });
      }
      record.exchanges = exchanges;
    }

    lines.push(JSON.stringify(record));
  }

  return lines.join("\n");
}

// ---- 文件输出 ----

/** 日志轮转：删除最旧的 {base}{N-1}.ext，依次滚动 {base}{i-1} → {base}{i}，最后 {base}.ext → {base}1.ext */
function rotateFile(dir: string, base: string, ext: string): void {
  const oldest = path.join(dir, `${base}${MAX_LOG_FILES - 1}.${ext}`);
  if (fs.existsSync(oldest)) fs.unlinkSync(oldest);

  for (let i = MAX_LOG_FILES - 2; i >= 1; i--) {
    const src = path.join(dir, i === 1 ? `${base}.${ext}` : `${base}${i - 1}.${ext}`);
    const dst = path.join(dir, `${base}${i}.${ext}`);
    if (fs.existsSync(src)) fs.renameSync(src, dst);
  }
}

function writeReportFiles(brief: BriefSummary, records: TestRecord[]): void {
  const backendDir = path.join(LOGS_DIR, logGroup());
  fs.mkdirSync(backendDir, { recursive: true });

  // brief.json 与 detail.json 同步轮转，保留最近 MAX_LOG_FILES 份历史
  rotateFile(backendDir, "brief", "json");
  rotateFile(backendDir, "detail", "log");

  fs.writeFileSync(path.join(backendDir, "brief.json"), JSON.stringify(brief, null, 2));
  fs.writeFileSync(path.join(backendDir, "detail.log"), generateDetailJson(records, brief));
}

// ---- Reporter ----

class CompactReporter implements Reporter {
  private startTime = 0;
  private lastProgressTime = 0;
  private totalFiles = 0;
  private doneFiles = 0;
  private passed = 0;
  private failed = 0;
  private skipped = 0;
  /** 运行中的文件 → 开始时间；unit 等并行配置下可能有多个 */
  private runningFiles = new Map<string, number>();
  private heartbeat?: NodeJS.Timeout;

  onTestRunStart(specifications: ReadonlyArray<TestSpecification>) {
    this.startTime = Date.now();
    this.lastProgressTime = this.startTime;
    this.totalFiles = specifications.length;
    this.doneFiles = 0;
    this.passed = 0;
    this.failed = 0;
    this.skipped = 0;
    this.runningFiles.clear();
    // watch 模式会重复触发 onTestRunStart，先清理上一轮定时器；unref 保证心跳不阻止进程退出
    clearInterval(this.heartbeat);
    this.heartbeat = setInterval(() => this.reportHeartbeat(), HEARTBEAT_MS).unref();
    progress(`开始 ${this.totalFiles} 个测试文件（${logGroup()}）`);
  }

  // 并行配置下多个文件同时开始，序号只用于完成行，避免出现重复的 [i/N]
  onTestModuleStart(testModule: TestModule) {
    this.runningFiles.set(testModule.relativeModuleId, Date.now());
    progress(`▶ ${testModule.relativeModuleId}`);
  }

  onTestModuleEnd(testModule: TestModule) {
    const startAt = this.runningFiles.get(testModule.relativeModuleId);
    this.runningFiles.delete(testModule.relativeModuleId);

    const states = [...testModule.children.allTests()].map((testCase) => testCase.result().state);
    const passed = states.filter((state) => state === "passed").length;
    const failed = states.filter((state) => state === "failed").length;
    this.passed += passed;
    this.failed += failed;
    this.skipped += states.length - passed - failed;
    this.doneFiles++;
    this.touch();

    const mark = failed > 0 || testModule.state() === "failed" ? "✗" : "✓";
    const cost = startAt === undefined ? "" : ` ${formatDuration(Date.now() - startAt)}`;
    progress(
      `[${this.doneFiles}/${this.totalFiles}] ${mark} ${testModule.relativeModuleId} 通过${passed} 失败${failed} 跳过${states.length - passed - failed}${cost}`
    );
  }

  // 失败用例即时上报，不必等整轮跑完；慢用例单独标记，用于定位拖慢全量的用例
  onTestCaseResult(testCase: TestCase) {
    const location = `${testCase.module.relativeModuleId}:${testCase.location?.line ?? 0}`;
    if (testCase.result().state === "failed") {
      progress(`✗ ${location} ${testCase.fullName}`);
    } else {
      const duration = testCase.diagnostic()?.duration ?? 0;
      if (duration < SLOW_CASE_MS) return;
      progress(`⏱ 慢用例 ${formatDuration(duration)} ${location} ${testCase.fullName}`);
    }
    this.touch();
  }

  onTestRunEnd(
    testModules: ReadonlyArray<TestModule>,
    unhandledErrors: ReadonlyArray<SerializedError>,
    reason: TestRunEndReason
  ) {
    clearInterval(this.heartbeat);

    const { records, errors } = collectAll(testModules);
    for (const error of unhandledErrors) {
      errors.push(stripAnsi(error.message));
    }

    const duration = `${((Date.now() - this.startTime) / 1000).toFixed(1)}s`;
    // brief 是 detail 的精简输出：generateBrief 后处理 TestRecord[]（截断/扁平化）
    const brief = generateBrief(records, errors, reason, testModules.length, duration);
    console.log(JSON.stringify(brief));
    writeReportFiles(brief, records);
    progress(
      `结束 ${reason}：通过${brief.passed} 失败${brief.failed} 跳过${brief.skipped}，耗时 ${brief.duration}`
    );
  }

  private touch() {
    this.lastProgressTime = Date.now();
  }

  private reportHeartbeat() {
    const idle = Date.now() - this.lastProgressTime;
    progress(
      `⏱ 已运行 ${formatDuration(Date.now() - this.startTime)}｜文件 ${this.doneFiles}/${this.totalFiles}｜` +
        `通过${this.passed} 失败${this.failed} 跳过${this.skipped}｜${this.describeRunning()}｜距上次进展 ${formatDuration(idle)}`
    );
    if (idle >= STALL_HINT_MS) {
      progress(
        `⚠️ ${formatDuration(idle)} 无进展（已达 testTimeout/hookTimeout 120s）：检查后端可用性与当前文件的钩子`
      );
    }
  }

  // 串行配置（fileParallelism: false）下即当前文件；并行配置下取开始最早的运行中文件
  private describeRunning(): string {
    if (this.runningFiles.size === 0) return "等待下一个文件";
    const [name, startAt] = [...this.runningFiles].reduce((oldest, entry) =>
      entry[1] < oldest[1] ? entry : oldest
    );
    const parallel = this.runningFiles.size > 1 ? `（+${this.runningFiles.size - 1} 并行）` : "";
    return `当前 ${name} 已 ${formatDuration(Date.now() - startAt)}${parallel}`;
  }
}

export default new CompactReporter();
