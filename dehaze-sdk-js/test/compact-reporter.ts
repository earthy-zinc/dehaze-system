/**
 * 紧凑 JSON 报告器：面向大模型查阅场景，避免默认 reporter 的排版噪声
 *
 * 输出约定（单行 JSON）：
 * - 全部通过：仅 state/files/passed/skipped/duration 汇总字段
 * - 存在失败：追加 failures 数组（文件:行号、完整用例名、错误信息、截断的
 *   expected/actual、过滤 node_modules 后的堆栈帧）
 * - 存在跳过：追加 skippedTests 数组（文件:行号、完整用例名、跳过原因）
 * - 收集错误（语法错误等）与未捕获错误归入 errors 数组
 */
import type {
  Reporter,
  SerializedError,
  TestCase,
  TestModule,
  TestResultSkipped,
  TestRunEndReason,
} from "vitest/node";

const MAX_VALUE_LENGTH = 300;
const MAX_STACK_LINES = 5;

interface FailureEntry {
  file: string;
  test: string;
  error: string;
  expected?: string;
  actual?: string;
  stack?: string[];
}

interface SkippedEntry {
  file: string;
  test: string;
  reason: string;
}

// Vitest 断言失败信息按终端着色能力内嵌 ANSI 转义序列，JSON 输出前需剥离
function stripAnsi(text: string): string {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function truncate(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  const text = stripAnsi(typeof value === "string" ? value : (JSON.stringify(value) ?? ""));
  return text.length > MAX_VALUE_LENGTH ? `${text.slice(0, MAX_VALUE_LENGTH)}…[truncated]` : text;
}

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

class CompactReporter implements Reporter {
  private startTime = 0;

  onTestRunStart() {
    this.startTime = Date.now();
  }

  onTestRunEnd(
    testModules: ReadonlyArray<TestModule>,
    unhandledErrors: ReadonlyArray<SerializedError>,
    reason: TestRunEndReason
  ) {
    let passed = 0;
    let failed = 0;
    let skipped = 0;
    const failures: FailureEntry[] = [];
    const skippedTests: SkippedEntry[] = [];
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
        if (result.state === "passed") {
          passed++;
          continue;
        }
        if (result.state === "skipped") {
          skipped++;
          skippedTests.push(buildSkippedEntry(testModule, testCase, result));
          continue;
        }
        if (result.state !== "failed") continue;
        failed++;
        failures.push(buildFailureEntry(testModule, testCase, result.errors));
      }
    }

    for (const error of unhandledErrors) {
      errors.push(stripAnsi(error.message));
    }

    const summary = {
      state: reason,
      files: testModules.length,
      passed,
      failed,
      skipped,
      duration: `${((Date.now() - this.startTime) / 1000).toFixed(1)}s`,
      ...(failures.length > 0 && { failures }),
      ...(skippedTests.length > 0 && { skippedTests }),
      ...(errors.length > 0 && { errors }),
    };
    console.log(JSON.stringify(summary));
  }
}

function formatLocation(testModule: TestModule, testCase: TestCase): string {
  return testCase.location
    ? `${testModule.relativeModuleId}:${testCase.location.line}`
    : testModule.relativeModuleId;
}

function buildFailureEntry(
  testModule: TestModule,
  testCase: TestCase,
  testErrors: ReadonlyArray<SerializedError>
): FailureEntry {
  const firstError = testErrors[0];
  const entry: FailureEntry = {
    file: formatLocation(testModule, testCase),
    test: testCase.fullName,
    error:
      stripAnsi(firstError?.message ?? "unknown error") +
      (testErrors.length > 1 ? ` (+${testErrors.length - 1} more errors)` : ""),
  };
  const expected = truncate(firstError?.expected);
  const actual = truncate(firstError?.actual);
  const stack = trimStack(firstError?.stack);
  if (expected !== undefined) entry.expected = expected;
  if (actual !== undefined) entry.actual = actual;
  if (stack !== undefined) entry.stack = stack;
  return entry;
}

function buildSkippedEntry(
  testModule: TestModule,
  testCase: TestCase,
  result: TestResultSkipped
): SkippedEntry {
  return {
    file: formatLocation(testModule, testCase),
    test: testCase.fullName,
    reason: resolveSkipReason(testCase, result),
  };
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

export default new CompactReporter();
