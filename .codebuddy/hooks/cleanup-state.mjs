#!/usr/bin/env node
// SessionEnd hook：删除本会话的状态（改动台账 + 反馈指纹），并回收其它会话的孤儿状态。
//
// 注意：headless（`codebuddy -p`）实测**不发** SessionEnd，那种场景只能靠 verify-on-stop 里的
// sweepStale 兜底，所以本脚本不是唯一的清理保障。

import { readFileSync } from 'node:fs';
import { log, removeSession, sessionIdOf, sweepStale } from './state.mjs';

try {
  const payload = JSON.parse(readFileSync(0, 'utf8'));
  const id = sessionIdOf(payload);
  removeSession(id);
  sweepStale(id);
  log(`cleanup session=${id.slice(0, 8)}`);
} catch (error) {
  log(`error: ${error?.message ?? error}`);
}
