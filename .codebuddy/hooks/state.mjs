#!/usr/bin/env node
// 四个 hook 共用的状态层：仓库根、会话状态读写、改动台账、编辑回读台账、孤儿回收、调试日志。
//
// 状态布局（`.codebuddy/.state/`，已由 .gitignore 排除）：
//   <session>.files  改动台账，每行一个仓库相对路径，**只追加**（O_APPEND 小写入原子）
//   <session>.writes 编辑回读台账，每行一条 JSON `{file,probe,at}`，**只追加**
//   <session>.json   `{ reported: [指纹], chain: 续跑链已反馈次数 }`，只由 Stop 单写
//   debug            标记文件，存在即开调试日志（IDE/CLI 里未必能设环境变量）
//
// 为什么台账与 JSON 分家：PostToolUse 会随子代理**并发**触发（实测子代理与父会话同 session_id），
// 而"读 JSON → 追加 → 整份写回"会互相覆盖 —— 轻则丢一条登记，重则用旧快照把 reported 冲掉。
// 拆成两个文件、每个文件只有一个写者之后，并发只剩追加，天然安全。

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// 仓库根只认脚本自身位置：IDE 实测 payload.cwd 是 /root，环境变量也未必导出
export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const STATE_DIR = path.join(ROOT, '.codebuddy/.state');
const LOG_FILE = path.join(process.env.TMPDIR || '/tmp', 'dehaze-hooks.log');
const STALE_MS = 24 * 60 * 60 * 1000;

const debugEnabled =
  process.env.DEHAZE_HOOK_DEBUG === '1' || existsSync(path.join(STATE_DIR, 'debug'));

export function log(message) {
  if (!debugEnabled) return;
  try {
    appendFileSync(
      LOG_FILE,
      `[${new Date().toISOString()}] ${path.basename(process.argv[1])} ${message}\n`,
    );
  } catch {
    // 日志写不进去不影响 hook 本身
  }
}

export function sessionIdOf(payload) {
  return payload?.session_id || 'default';
}

function statePath(id) {
  return path.join(STATE_DIR, `${id}.json`);
}

function ledgerPath(id) {
  return path.join(STATE_DIR, `${id}.files`);
}

export function appendLedger(id, relative) {
  mkdirSync(STATE_DIR, { recursive: true });
  appendFileSync(ledgerPath(id), `${relative}\n`);
}

// 原子接管台账：先改名再读，这样"读完到删除"之间落进来的并发追加会落进新台账而不是被吞掉
export function consumeLedger(id) {
  const ledger = ledgerPath(id);
  if (!existsSync(ledger)) return [];
  const taken = `${ledger}.consuming`;
  try {
    renameSync(ledger, taken);
  } catch {
    return [];
  }
  try {
    return readFileSync(taken, 'utf8').split('\n').filter(Boolean);
  } finally {
    rmSync(taken, { force: true });
  }
}

function writesPath(id) {
  return path.join(STATE_DIR, `${id}.writes`);
}

export function appendWrite(id, entry) {
  mkdirSync(STATE_DIR, { recursive: true });
  appendFileSync(writesPath(id), `${JSON.stringify(entry)}\n`);
}

// 与 consumeLedger 同款原子接管：并发触发时只有一个进程拿到待复核项，另一个拿到空集（少一次复核，不会误报）
export function consumeWrites(id) {
  const writes = writesPath(id);
  if (!existsSync(writes)) return [];
  const taken = `${writes}.consuming`;
  try {
    renameSync(writes, taken);
  } catch {
    return [];
  }
  try {
    return readFileSync(taken, 'utf8')
      .split('\n')
      .filter(Boolean)
      .flatMap((line) => {
        try {
          return [JSON.parse(line)];
        } catch {
          return [];
        }
      });
  } finally {
    rmSync(taken, { force: true });
  }
}

// 剩余项必须**追加**回原文件：接管窗口内并发写入的条目落在 writesPath 上，覆盖写会把它们冲掉
export function restoreWrites(id, entries) {
  if (!entries.length) return;
  mkdirSync(STATE_DIR, { recursive: true });
  appendFileSync(writesPath(id), entries.map((entry) => `${JSON.stringify(entry)}\n`).join(''));
}

export function readState(id) {
  try {
    return JSON.parse(readFileSync(statePath(id), 'utf8'));
  } catch {
    return {};
  }
}

export function writeState(id, state) {
  mkdirSync(STATE_DIR, { recursive: true });
  writeFileSync(statePath(id), JSON.stringify(state));
}

// 回收孤儿状态：只删「非本会话、且超过 STALE_MS 没动过」的。
// headless（`codebuddy -p`）实测不发 SessionEnd，那种场景只能靠这里兜底。
export function sweepStale(keepId) {
  try {
    for (const name of readdirSync(STATE_DIR)) {
      // 会话 id 不含点，取第一段即可；不含点的条目（debug 标记）与目标会话一并跳过
      const id = name.split('.')[0];
      if (id === name || id === keepId) continue;
      const file = path.join(STATE_DIR, name);
      if (Date.now() - statSync(file).mtimeMs > STALE_MS) rmSync(file, { force: true });
    }
  } catch {
    // 状态目录还不存在：无需回收
  }
}

export function removeSession(id) {
  rmSync(statePath(id), { force: true });
  rmSync(ledgerPath(id), { force: true });
  rmSync(`${ledgerPath(id)}.consuming`, { force: true });
  rmSync(writesPath(id), { force: true });
  rmSync(`${writesPath(id)}.consuming`, { force: true });
}
