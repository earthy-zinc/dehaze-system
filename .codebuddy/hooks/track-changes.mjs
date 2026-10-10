#!/usr/bin/env node
// PostToolUse hook：只登记「本轮被写过的文件」，不做别的 —— 格式化与检查统一在 Stop 的 verify-on-stop.mjs 收口。
//
// 为什么登记而不是就地格式化：见 verify-on-stop.mjs 头部说明。此处必须零副作用、零输出、毫秒级返回，
// 因为它随每一次 Write/Edit 触发（含子代理并发触发），任何耗时或 stdout 都会压在 agent 的编辑节奏上。
//
// payload 字段名两套都认：CLI 传 `tool_input.file_path`、IDE 传 `tool_input.filePath`（均为实测）。

import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { ROOT, appendLedger, log, sessionIdOf } from './state.mjs';

// 锁文件由包管理器生成与校验，格式化会破坏其与 package.json 的一致性
const SKIP_FILENAMES = new Set([
  'pnpm-lock.yaml',
  'package-lock.json',
  'npm-shrinkwrap.json',
  'yarn.lock',
  'bun.lockb',
]);

const SKIP_SEGMENTS = new Set([
  'node_modules',
  '.venv',
  '.git',
  'dist',
  'build',
  'target',
  '.dart_tool',
  '__pycache__',
  'coverage',
  'out',
  '.nuxt',
  '.output',
  '.state',
]);

function record(payload, file) {
  const absolute = path.isAbsolute(file) ? path.normalize(file) : path.resolve(ROOT, file);
  const relative = path.relative(ROOT, absolute);
  // 只排除「仓库外」与「仓库根本身」：仓库根下的单段文件（README.md 等）同样登记
  const outside = relative === '' || relative.startsWith('..') || path.isAbsolute(relative);
  const ignorable =
    SKIP_FILENAMES.has(path.basename(absolute)) ||
    relative.split(path.sep).some((segment) => SKIP_SEGMENTS.has(segment));

  if (outside || ignorable || !existsSync(absolute) || !statSync(absolute).isFile()) {
    return log(`skip ${relative || file}（${outside ? 'outside' : 'ignorable'}）`);
  }
  appendLedger(sessionIdOf(payload), relative);
  log(`register ${relative}`);
}

try {
  const payload = JSON.parse(readFileSync(0, 'utf8'));
  const file = payload?.tool_input?.file_path ?? payload?.tool_input?.filePath;
  if (typeof file === 'string' && file) record(payload, file);
} catch (error) {
  log(`error: ${error?.message ?? error}`);
}
