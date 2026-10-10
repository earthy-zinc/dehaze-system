#!/usr/bin/env node
// hooks 链路自测：PostToolUse 登记（track-changes）→ Stop 收口（verify-on-stop：批量格式化 + 静态检查 + 反馈）
// → SessionEnd 清理（cleanup-state）。
//
// 断言覆盖：各语言样本被批量格式化、检查阶段只报新增指纹且走 stderr、两套 payload 字段名（CLI file_path /
// IDE filePath）、仓库根文件被登记、并发登记不丢、续跑轮仍复验、续跑链达上限后静默而新一轮归零、
// 无改动/非法输入静默、台账被消费、调试标记文件生效、状态被清理、样本无残留。
//
// 用法：node .codebuddy/hooks/selftest.mjs

import { existsSync, mkdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HOOKS = path.dirname(fileURLToPath(import.meta.url));
const TRACK = path.join(HOOKS, 'track-changes.mjs');
const VERIFY = path.join(HOOKS, 'verify-on-stop.mjs');
const CLEANUP = path.join(HOOKS, 'cleanup-state.mjs');
const ROOT = path.resolve(HOOKS, '../..');
const SESSION = 'selftest';
const STATE_JSON = path.join(ROOT, '.codebuddy/.state', `${SESSION}.json`);
const LEDGER = path.join(ROOT, '.codebuddy/.state', `${SESSION}.files`);
const DEBUG_MARKER = path.join(ROOT, '.codebuddy/.state/debug');
const LOG_FILE = path.join(process.env.TMPDIR || '/tmp', 'dehaze-hooks.log');
const created = [];
let failures = 0;

function create(relativePath, content) {
  const absolute = path.resolve(ROOT, relativePath);
  mkdirSync(path.dirname(absolute), { recursive: true });
  writeFileSync(absolute, content);
  created.push(absolute);
  return absolute;
}

function read(file) {
  return readFileSync(file, 'utf8');
}

function ledger() {
  return existsSync(LEDGER) ? read(LEDGER) : '';
}

function stateJson() {
  return existsSync(STATE_JSON) ? read(STATE_JSON) : '';
}

function hash(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

function hookEnv() {
  return { ...process.env, CLAUDE_PROJECT_DIR: ROOT, DEHAZE_HOOK_DEBUG: '1' };
}

function payloadOf(file, ideStyle = false) {
  return {
    hook_event_name: 'PostToolUse',
    tool_name: 'Write',
    tool_input: ideStyle ? { filePath: file } : { file_path: file },
    cwd: ROOT,
    session_id: SESSION,
  };
}

function runHook(script, payload) {
  return spawnSync(process.execPath, [script], {
    input: JSON.stringify(payload),
    encoding: 'utf8',
    timeout: 300_000,
    env: hookEnv(),
  });
}

function track(file, ideStyle = false) {
  return runHook(TRACK, payloadOf(file, ideStyle));
}

// 真并发登记：台账是 append-only（O_APPEND 小写入原子），不能被"读-改-写"覆盖掉
function trackAsync(file) {
  const child = spawn(process.execPath, [TRACK], { env: hookEnv() });
  child.stdin.end(JSON.stringify(payloadOf(file)));
  return new Promise((resolve) => child.on('close', (code) => resolve({ status: code })));
}

function stop(extra = {}) {
  return runHook(VERIFY, {
    hook_event_name: 'Stop',
    session_id: SESSION,
    stop_hook_active: false,
    cwd: ROOT,
    ...extra,
  });
}

function check(label, condition, detail) {
  if (condition) {
    console.log(`  PASS  ${label}`);
  } else {
    failures += 1;
    console.log(`  FAIL  ${label}${detail ? ` -- ${detail}` : ''}`);
  }
}

function checkSilent(label, result) {
  check(
    label,
    result.status === 0 && result.stdout === '' && result.stderr === '',
    JSON.stringify({ status: result.status, out: result.stdout, err: result.stderr }),
  );
}

// 纯排版语言：登记 → 轮末收口 → 文件被改写，且全程零输出
function checkFormat(name, relativePath, before, expected) {
  console.log(`\n[${name}] ${relativePath}`);
  const file = create(relativePath, before);
  checkSilent('登记 hook 静默退出', track(file));
  check('改动已进台账', ledger().includes(relativePath), ledger().slice(0, 300));
  checkSilent('Stop hook 静默退出', stop());
  check('台账已被消费', !existsSync(LEDGER));
  const formatted = read(file);
  check('文件已被改写', formatted !== before);
  check(`含规范片段 ${JSON.stringify(expected)}`, formatted.includes(expected), `实际内容:\n${formatted}`);
}

for (const file of [STATE_JSON, LEDGER, `${LEDGER}.consuming`]) rmSync(file, { force: true });

console.log('== 轮末批量格式化：各语言/子项目 ==');

checkFormat(
  'markdown',
  'dehaze-doc/docs/hook_selftest_probe.md',
  '# 标题\n\n|A|B|\n|-|-|\n|1|2|\n',
  '| A   | B   |',
);

checkFormat(
  'dart',
  'dehaze_flutter/lib/hook_selftest_probe.dart',
  'int  add( int a ,int b){return a+b;}\n',
  'int add(int a, int b) {',
);

console.log('\n[java 批量] 两个文件合并成一次 maven 调用');
const javaA = create(
  'dehaze-java/src/main/java/com/pei/dehaze/HookSelftestProbe.java',
  'package com.pei.dehaze;\n\npublic class HookSelftestProbe {\n    public int  add( int a,int b ){\n        return a+b ;\n    }\n}\n',
);
const javaB = create(
  'dehaze-java/src/main/java/com/pei/dehaze/HookSelftestProbe2.java',
  'package com.pei.dehaze;\n\npublic class HookSelftestProbe2 {\n    public int  sub( int a,int b ){\n        return a-b ;\n    }\n}\n',
);
track(javaA);
track(javaB);
check('台账已记录两个文件', ledger().includes('HookSelftestProbe2'), ledger().slice(0, 300));
checkSilent('Stop hook 静默退出', stop());
check('文件 A 已格式化', read(javaA).includes('return a + b;'), read(javaA));
check('文件 B 已格式化', read(javaB).includes('return a - b;'), read(javaB));

checkFormat(
  'front-vue ts',
  'dehaze-front-vue/src/hook_selftest_probe.ts',
  'const  add=(a:number,b:number)=>{return a+b}\nexport default add\n',
  'const add = (a: number, b: number) => {',
);

checkFormat(
  'front-vue css',
  'dehaze-front-vue/src/hook_selftest_probe.css',
  '.a{color:red;background : blue}\n',
  'color: red;',
);

checkFormat(
  'front-vue vue',
  'dehaze-front-vue/src/hook_selftest_probe.vue',
  '<template>\n    <div>{{   msg }}</div>\n</template>\n\n<script setup lang="ts">\nconst   msg="hi"\n</script>\n\n<style scoped>\n.a{color:red}\n</style>\n',
  'const msg = "hi";',
);

console.log('\n== 轮末检查：Python（ruff --fix 后仍有 F841 + pyright 类型错） ==');
const pythonProbe = create(
  'dehaze-python/hook_selftest_probe.py',
  "def  add( a: int,b: int)->int:\n    wrong: int = 'not an int'\n    return a+b\n",
);
track(pythonProbe);
const pythonResult = stop();
check('Stop 退出码 2（把问题回灌给 agent）', pythonResult.status === 2, `status=${pythonResult.status}`);
check('stdout 保持空（反馈只走 stderr）', pythonResult.stdout === '', JSON.stringify(pythonResult.stdout));
check('已格式化', read(pythonProbe).includes('def add(a: int, b: int) -> int:'), read(pythonProbe));
check('含 pyright 类型错误', pythonResult.stderr.includes('pyright-error'), pythonResult.stderr);
check('含 ruff 诊断', /\[F\d+\]|\[E\d+\]|\[B\d+\]|\[RUF\d+\]/.test(pythonResult.stderr), pythonResult.stderr);
check('声明非门禁口径', pythonResult.stderr.includes('不是门禁'), pythonResult.stderr);
check('提示限定修复范围', pythonResult.stderr.includes('不要顺手重构'), pythonResult.stderr);
check('反馈指纹已落盘', stateJson().includes('pyright-error'), stateJson().slice(0, 300));

const pythonDigest = hash(pythonProbe);
track(pythonProbe);
checkSilent('复验后无新增指纹则静默（去重成立）', stop());
check('内容未再变化（格式化幂等）', hash(pythonProbe) === pythonDigest);

console.log('\n== 轮末检查：Go（go vet） ==');
const goProbe = create(
  'dehaze-go/hook_selftest_probe/probe.go',
  'package hookselftestprobe\n\nimport "fmt"\n\nfunc  Add( a int,b int) int {\n\tfmt.Printf("%d\\n", "not an int")\n\treturn a+b\n}\n',
);
track(goProbe);
const goResult = stop();
check('Stop 退出码 2', goResult.status === 2, `status=${goResult.status}`);
check('含 go vet 诊断', goResult.stderr.includes('go vet'), goResult.stderr);
check('已 gofmt 格式化', read(goProbe).includes('func Add(a int, b int) int {'), read(goProbe));

console.log('\n== 轮末检查：TypeScript（taro 的 tsc --noEmit） ==');
const tsProbe = create('dehaze-taro/src/hook_selftest_probe.ts', 'export const n: number = "not a number";\n');
track(tsProbe);
const tsResult = stop();
check('Stop 退出码 2', tsResult.status === 2, `status=${tsResult.status}`);
check('含 tsc 诊断（TS 码）', /TS\d{4}/.test(tsResult.stderr), tsResult.stderr);
check('已 prettier 格式化', read(tsProbe).includes('not a number'), read(tsProbe));

console.log('\n== 续跑轮（stop_hook_active=true）仍复验 ==');
const fixProbe = create('dehaze-python/hook_selftest_probe_fix.py', 'def  f( )->str:\n    return 1\n');
track(fixProbe);
const fixResult = stop({ stop_hook_active: true });
check('续跑轮复验出新问题（exit 2）', fixResult.status === 2, `status=${fixResult.status}`);
check('续跑轮同样完成格式化', read(fixProbe).includes('def f() -> str:'), read(fixProbe));

console.log('\n== 续跑链达上限则静默，新一轮归零后可再反馈 ==');
const capProbe = create('dehaze-python/hook_selftest_probe_cap.py', 'def  g( )->bytes:\n    return 1\n');
track(capProbe);
checkSilent('续跑链达上限后静默', stop({ stop_hook_active: true }));
check('但仍完成格式化', read(capProbe).includes('def g() -> bytes:'), read(capProbe));

const freshProbe = create('dehaze-python/hook_selftest_probe_fresh.py', "def  h( )->float:\n    return 'x'\n");
track(freshProbe);
const freshResult = stop();
check('新的用户轮（stop_hook_active=false）归零后可再反馈', freshResult.status === 2, `status=${freshResult.status}`);

console.log('\n== 登记字段名（CLI file_path / IDE filePath）与仓库根文件 ==');
const ideProbe = create('dehaze-doc/hook_selftest_probe_ide.md', '# ide\n');
checkSilent('登记 hook 静默退出（IDE filePath）', track(ideProbe, true));
check('IDE 的 filePath 被登记', ledger().includes('hook_selftest_probe_ide'), ledger().slice(0, 300));
const rootProbe = create('hook_selftest_probe_root.md', '# root\n');
checkSilent('登记 hook 静默退出（仓库根文件）', track(rootProbe));
check('仓库根文件被登记', ledger().includes('hook_selftest_probe_root'), ledger().slice(0, 300));

console.log('\n== 并发登记不丢（台账 append-only） ==');
const parallelFiles = [];
for (let index = 0; index < 8; index += 1) {
  parallelFiles.push(create(`dehaze-doc/hook_selftest_par_${index}.md`, '# p\n'));
}
const parallelResults = await Promise.all(parallelFiles.map(trackAsync));
check('并发登记全部退出码 0', parallelResults.every((result) => result.status === 0), JSON.stringify(parallelResults));
const parallelLedger = ledger();
const lost = parallelFiles.filter((file) => !parallelLedger.includes(path.relative(ROOT, file)));
check('8 路并发登记无丢条', lost.length === 0, `丢失 ${lost.length} 条：${lost.join(', ')}`);
checkSilent('消费这批登记静默退出', stop());

console.log('\n== 无登记改动与非法输入 ==');
checkSilent('无改动时静默退出', stop());
const badInput = spawnSync(process.execPath, [VERIFY], {
  input: 'not-json',
  encoding: 'utf8',
  timeout: 30_000,
  env: { ...process.env, CLAUDE_PROJECT_DIR: ROOT },
});
checkSilent('stdin 非 JSON 时静默退出', badInput);
const outside = create('/tmp/hook_selftest_outside.py', 'def  add( a,b ):\n    return a\n');
track(outside);
check('仓库外文件不登记', !ledger().includes('hook_selftest_outside'), ledger().slice(0, 300));
rmSync(outside, { force: true });
const vanished = create('dehaze-python/hook_selftest_probe_vanished.py', 'x = 1\n');
track(vanished);
rmSync(vanished, { force: true });
checkSilent('登记文件已被删除时静默退出', stop());
check('台账仍被消费', !existsSync(LEDGER));

console.log('\n== 调试开关：标记文件（不依赖环境变量） ==');
rmSync(LOG_FILE, { force: true });
writeFileSync(DEBUG_MARKER, '');
const markerRun = spawnSync(process.execPath, [TRACK], {
  input: JSON.stringify(payloadOf(rootProbe)),
  encoding: 'utf8',
  timeout: 30_000,
  env: { ...process.env, CLAUDE_PROJECT_DIR: ROOT },
});
check(
  '标记文件即可开启调试日志（无需 DEHAZE_HOOK_DEBUG）',
  markerRun.status === 0 && existsSync(LOG_FILE) && read(LOG_FILE).includes('register'),
  existsSync(LOG_FILE) ? read(LOG_FILE).slice(0, 200) : '日志未生成',
);
rmSync(DEBUG_MARKER, { force: true });

console.log('\n== 孤儿状态回收（headless `codebuddy -p` 不发 SessionEnd） ==');
const orphanState = path.join(ROOT, '.codebuddy/.state', 'hook_selftest_orphan.json');
const freshState = path.join(ROOT, '.codebuddy/.state', 'hook_selftest_fresh.json');
writeFileSync(orphanState, '{}');
writeFileSync(freshState, '{}');
const staleTime = new Date(Date.now() - 25 * 60 * 60 * 1000);
utimesSync(orphanState, staleTime, staleTime);
checkSilent('无新增问题的轮次静默退出', stop());
check('25h 未动的其它会话状态被回收', !existsSync(orphanState));
check('未超时的状态不被误删', existsSync(freshState), 'fresh state 被误删');
rmSync(freshState, { force: true });

console.log('\n== SessionEnd 清理 ==');
const cleanup = spawnSync(process.execPath, [CLEANUP], {
  input: JSON.stringify({ hook_event_name: 'SessionEnd', session_id: SESSION, reason: 'other', cwd: ROOT }),
  encoding: 'utf8',
  timeout: 30_000,
  env: { ...process.env, CLAUDE_PROJECT_DIR: ROOT },
});
checkSilent('清理 hook 静默退出', cleanup);
check('状态与台账已删除', !existsSync(STATE_JSON) && !existsSync(LEDGER));

console.log('\n== 清理与残留复核 ==');
for (const file of created) rmSync(file, { force: true });
const goProbeDir = path.resolve(ROOT, 'dehaze-go/hook_selftest_probe');
rmSync(goProbeDir, { force: true, recursive: true });
for (const file of [STATE_JSON, LEDGER, `${LEDGER}.consuming`, DEBUG_MARKER]) rmSync(file, { force: true });

let leftovers = 0;
for (const file of created) if (existsSync(file)) leftovers += 1;
if (existsSync(goProbeDir)) leftovers += 1;
check('临时样本已清理', leftovers === 0, `残留 ${leftovers} 处`);

const status = execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' });
const strayLines = status.split('\n').filter((line) => line.includes('hook_selftest'));
check('git 工作区无 hook_selftest 残留', strayLines.length === 0, strayLines.join('\n'));

console.log(failures === 0 ? '\n全部用例通过' : `\n${failures} 个断言失败`);
process.exit(failures === 0 ? 0 : 1);
