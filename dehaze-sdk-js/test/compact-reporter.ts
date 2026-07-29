/**
 * 紧凑 JSON 报告器：面向大模型查阅场景，避免默认 reporter 的排版噪声
 *
 * 输出约定（单行 JSON）：
 * - 全部通过：仅 state/files/passed/skipped/duration 汇总字段
 * - 存在失败：追加 failures 数组（文件:行号、完整用例名、错误信息、截断的
 *   expected/actual、过滤 node_modules 后的堆栈帧）
 * - 收集错误（语法错误等）与未捕获错误归入 errors 数组
 */
import type {
  Reporter,
  SerializedError,
  TestCase,
  TestModule,
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

function truncate(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (text === undefined) return undefined;
  return text.length > MAX_VALUE_LENGTH ? `${text.slice(0, MAX_VALUE_LENGTH)}…[truncated]` : text;
}

function trimStack(stack: string | undefined): string[] | undefined {
  if (!stack) return undefined;
  const frames = stack
    .split("\n")
    .map((line) => line.trim())
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
    const errors: string[] = [];

    for (const testModule of testModules) {
      for (const error of testModule.errors()) {
        errors.push(`${testModule.relativeModuleId}: ${error.message}`);
      }
      for (const suite of testModule.children.allSuites()) {
        for (const error of suite.errors()) {
          errors.push(`${testModule.relativeModuleId} > ${suite.fullName}: ${error.message}`);
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
          continue;
        }
        if (result.state !== "failed") continue;
        failed++;
        failures.push(buildFailureEntry(testModule, testCase, result.errors));
      }
    }

    for (const error of unhandledErrors) {
      errors.push(error.message);
    }

    const summary = {
      state: reason,
      files: testModules.length,
      passed,
      failed,
      skipped,
      duration: `${((Date.now() - this.startTime) / 1000).toFixed(1)}s`,
      ...(failures.length > 0 && { failures }),
      ...(errors.length > 0 && { errors }),
    };
    console.log(JSON.stringify(summary));
  }
}

function buildFailureEntry(
  testModule: TestModule,
  testCase: TestCase,
  testErrors: ReadonlyArray<SerializedError>
): FailureEntry {
  const firstError = testErrors[0];
  const location = testCase.location;
  const entry: FailureEntry = {
    file: location
      ? `${testModule.relativeModuleId}:${location.line}`
      : testModule.relativeModuleId,
    test: testCase.fullName,
    error:
      (firstError?.message ?? "unknown error") +
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

export default new CompactReporter();
