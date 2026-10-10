#!/usr/bin/env node
// PostToolUse hook：编辑回读校验 —— 编辑工具报 success 不等于内容已落盘（实测有秒级延迟与偶发丢失）。
//
// 为什么延迟复核：写入后立刻读盘可能读到旧内容，静默期过后结论才可信 ⇒ 本次登记、下次触发复核上一批；
// 轮末由 verify-on-stop.mjs 以同一函数兜底（否则"最后一次编辑"这一跳无人验收）。
//
// 判据（任一成立即视为已落盘）：
//   ① 文件包含本次写入内容里的一个长标识片段（探针）；
//   ② 文件 mtime 明显晚于该次写入 ⇒ 内容已被后续编辑覆盖，无法再验，不再追究。
// 失败项只报一次（复核即消费，不写回台账），仅写 stderr 并置 exit 2 反馈给 agent。

import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ROOT, appendWrite, consumeWrites, log, restoreWrites, sessionIdOf } from './state.mjs';

// 复核前的静默期：短于此间隔的登记留到下一次触发/Stop，躲开写盘延迟造成的假失败
const RECHECK_MIN_AGE_MS = 2000;
// mtime 晚于登记时刻这么多 ⇒ 文件在该次写入之后又被编辑过，探针可能已被覆盖
const REWRITE_SLACK_MS = 1000;
// 探针＝新内容里最长的标识片段：格式化不会拆开标识符内部，故排除引号/空白/运算符
const PROBE_PATTERN = /[A-Za-z_][A-Za-z0-9_./:-]{11,}/g;

function persisted({ file, probe, at }) {
  const absolute = path.join(ROOT, file);
  try {
    const stat = statSync(absolute);
    if (!stat.isFile()) return false;
    if (stat.mtimeMs > at + REWRITE_SLACK_MS) return true;
    return readFileSync(absolute, 'utf8').includes(probe);
  } catch {
    return false;
  }
}

// 复核所有到期的待验项；未到期的写回台账，失败的返回给调用方报告
export function recheckWrites(id, minAgeMs) {
  const now = Date.now();
  const pending = [];
  const failures = [];
  for (const entry of consumeWrites(id)) {
    if (!entry?.file || !entry?.probe || !Number.isFinite(entry.at)) continue;
    if (now - entry.at < minAgeMs) {
      pending.push(entry);
      continue;
    }
    if (!persisted(entry)) failures.push(entry);
  }
  restoreWrites(id, pending);
  return failures;
}

function pickProbe(text) {
  const candidates = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    // 导入行的标识片段会被格式化器（ruff --fix / spotless removeUnusedImports）整行删除，
    // 拿它当"未落盘"证据会误报
    if (/^(import|from|#include|using\s|require\()/.test(trimmed)) continue;
    const matches = trimmed.match(PROBE_PATTERN);
    if (matches) candidates.push(...matches);
  }
  if (!candidates.length) return null;
  return candidates.reduce((longest, item) => (item.length > longest.length ? item : longest), '');
}

function writeOf(payload) {
  const input = payload?.tool_input ?? {};
  const file = input.file_path ?? input.filePath;
  if (typeof file !== 'string' || !file) return null;
  const absolute = path.isAbsolute(file) ? path.normalize(file) : path.resolve(ROOT, file);
  const relative = path.relative(ROOT, absolute);
  if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) return null;

  const texts = [];
  if (typeof input.content === 'string') texts.push(input.content);
  for (const edit of Array.isArray(input.edits) ? input.edits : []) {
    if (typeof edit?.new_string === 'string') texts.push(edit.new_string);
  }
  for (const key of ['new_string', 'new_str']) {
    if (typeof input[key] === 'string') texts.push(input[key]);
  }

  const probe = texts.map(pickProbe).filter(Boolean).sort((a, b) => b.length - a.length)[0];
  return probe ? { file: relative, probe, at: Date.now() } : null;
}

function render(failures) {
  return [
    '[编辑回读校验] 以下编辑在延迟复核时未读到写入内容（工具报成功 ≠ 已落盘）：',
    ...failures.map((f) => `- ${f.file}：探针 "${f.probe}" 未命中，且文件 mtime 早于该次写入`),
    '请回读这些文件确认内容；确属未落盘时重新执行该次编辑。',
  ].join('\n');
}

// 仅当被直接执行（PostToolUse）时读 stdin：被 verify-on-stop import 时若也抢读 stdin，
// 会让 Stop 侧拿不到 payload 而整体失效
const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    const payload = JSON.parse(readFileSync(0, 'utf8'));
    const id = sessionIdOf(payload);
    const failures = recheckWrites(id, RECHECK_MIN_AGE_MS);
    const entry = writeOf(payload);
    if (entry) appendWrite(id, entry);
    log(`recheck=${failures.length} record=${entry ? entry.file : 'none'}`);
    if (failures.length) {
      process.stderr.write(`${render(failures)}\n`);
      process.exitCode = 2;
    }
  } catch (error) {
    log(`error: ${error?.message ?? error}`);
  }
}
