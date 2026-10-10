#!/usr/bin/env node
// Stop hook：轮末收口 —— 先批量格式化「本轮改动文件」，再对它们做静态检查，检出新增问题时把摘要回灌给 agent。
//
// 为什么格式化放在轮末而不是每次编辑后：
//   1) 编辑期不落盘 ⇒ agent 的 old_string 不会失配、中途写下的 import 不会被 --fix 删掉、Python 热重载不被空转触发；
//   2) 单次进程批量处理 ⇒ 前端 eslint 单次启动约 8-10s、mvn 约 2s，逐文件调用是 N 倍开销；
//   3) 与检查同脚本串行 ⇒ 顺序可控，必须 format → check，否则纯排版问题（E501 之类）会淹没真错误。
//
// 本脚本不是门禁：任何异常静默 exit 0；只有检出「新增」问题时才写 stderr 并置退出码 2
// （Stop 的 exit 2 语义是把 stderr 注入下一条消息，不是阻止回合）。判定仍以
// dehaze-python/scripts/check.sh、dehaze-go 的 make test、mvn verify 为准。
//
// 防死循环：续跑轮（stop_hook_active=true，即上一轮被本 hook 反馈过）**同样复验**，否则修复结果
// 无人验收；靠「指纹去重 + 续跑链内最多反馈 MAX_REPORTS_PER_CHAIN 轮」收敛，新一轮用户提问归零。
// 排障：DEHAZE_HOOK_DEBUG=1（需在启动 CLI/IDE 前设置）或建 `.codebuddy/.state/debug` 标记，
// 日志落 `$TMPDIR/dehaze-hooks.log`，含 payload 摘要与各阶段耗时。

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { execFile } from 'node:child_process';
import path from 'node:path';
import { ROOT, consumeLedger, log, readState, sessionIdOf, sweepStale, writeState } from './state.mjs';
import { recheckWrites } from './verify-write.mjs';

// 与 dehaze-python/scripts/check.sh 的 RUFF_SERVICES / PYRIGHT_PROJECTS 对齐
const RUFF_PROJECTS = [
  'dehaze-python',
  'dehaze-service/dehaze-llm',
  'dehaze-service/dehaze-voice',
  'dehaze-service/dehaze-modal',
];

const PYRIGHT_PROJECTS = [
  'dehaze-python',
  'dehaze-mcp-gateway',
  'dehaze-service/dehaze-llm',
  'dehaze-service/dehaze-voice',
  'dehaze-service/dehaze-modal',
];

const NODE_PROJECTS = [
  'dehaze-front-vue',
  'dehaze-front-react',
  'dehaze-taro',
  'dehaze-uniapp',
  'dehaze-sdk-js',
  'dehaze-react-native',
];

// 前端类型检查：纳入口径只有一条 —— **该项目默认分支上类型是干净的**。成本不作为排除理由：
// agent 本来就要自己跑一遍（同样是整项目级 4-27s），hook 跑过就等于省掉一次工具调用与一轮模型往返。
// 因为 tsc/vue-tsc 是全项目级、无法按文件裁剪，所以整项目跑完再按输出过滤到改动文件；
// 只在「本轮改动含该项目的 .ts/.tsx/.vue」时触发。实测：taro 4.7s / uniapp 6.4s / front-vue 26.9s 已在册；
// 其余三项于 2026-10-01 随存量清零在同批在册（front-react 14 处、react-native 29 处＝SDK 契约 `Dataset`→`DatasetVO` 漂移，
// sdk-js 23 处全在 `test/factories/*`）⇒ 下方 TYPE_CHECK_PROJECTS 的 6 项即当前全部在册项，无待纳入项。
const TYPE_CHECK_PROJECTS = {
  'dehaze-sdk-js': ['node_modules/.bin/tsc', ['--noEmit']],
  'dehaze-taro': ['node_modules/.bin/tsc', ['--noEmit']],
  'dehaze-uniapp': ['node_modules/.bin/vue-tsc', ['--noEmit']],
  'dehaze-front-vue': ['node_modules/.bin/vue-tsc', ['--noEmit']],
  'dehaze-front-react': ['node_modules/.bin/tsc', ['--noEmit']],
  'dehaze-react-native': ['node_modules/.bin/tsc', ['--noEmit']],
};

const TYPE_CHECK_EXTENSIONS = ['.ts', '.tsx', '.vue'];

// 各前端 lint-staged 的处置顺序：js 系 eslint→prettier；vue eslint→prettier→stylelint；
// css 系 stylelint→prettier；json/cjs/html 只 prettier。
const NODE_EXT_KIND = {
  '.js': 'script',
  '.jsx': 'script',
  '.mjs': 'script',
  '.ts': 'script',
  '.tsx': 'script',
  '.cjs': 'prettier-only',
  '.json': 'prettier-only',
  '.vue': 'vue',
  '.html': 'prettier-only',
  '.css': 'style',
  '.scss': 'style',
  '.less': 'style',
};

// 同一条续跑链最多反馈几轮：指纹去重之外的第二道死循环保险
const MAX_REPORTS_PER_CHAIN = 2;
const MAX_FINDINGS_IN_REPORT = 20;

// 格式化器只关心成功/失败、不关心输出 ⇒ 与 runCapture 分开，别为无关输出占内存
function runQuiet(command, args, cwd, timeout) {
  return new Promise((resolve) => {
    const started = Date.now();
    execFile(command, args, { cwd, timeout }, (error) => {
      if (error) {
        log(
          `fail ${command} (${Date.now() - started}ms)  (cwd=${cwd}) status=${error.code ?? error.signal}`,
        );
      } else {
        log(`ok   ${command} ${args.length} args (${Date.now() - started}ms)  (cwd=${cwd})`);
      }
      resolve();
    });
  });
}

// 返回诊断文本；工具缺失/超时/被杀返回 null —— 宁可不报，也不让检查拖住回合。
// 异步版：各语言的检查互相独立（只读诊断），用 Promise.all 并行，把"多项目/多语言同轮"的
// 总耗时从 Σ 压到 max（改 SDK 的轮次实测 64.3s → 约 31s）。
function runCapture(command, args, cwd, timeout, env) {
  return new Promise((resolve) => {
    const started = Date.now();
    execFile(
      command,
      args,
      {
        cwd,
        encoding: 'utf8',
        timeout,
        maxBuffer: 8 * 1024 * 1024,
        env: env || process.env,
      },
      (error, stdout, stderr) => {
        if (error) {
          if (error.killed || error.code === 'ETIMEDOUT' || error.code === 'ENOENT') {
            log(`skip ${command}: ${error.code || 'killed'} (${Date.now() - started}ms)`);
            return resolve(null);
          }
          log(`done ${command} (${Date.now() - started}ms)  (cwd=${cwd})`);
          return resolve(`${stdout || ''}${stderr || ''}`);
        }
        log(`ok   ${command} (${Date.now() - started}ms)  (cwd=${cwd})`);
        resolve(stdout || '');
      },
    );
  });
}

// 取某子项目下指定扩展名的文件，路径相对该子项目（各工具都在子项目目录内执行）
function projectFiles(relFiles, project, extensions) {
  const prefix = `${project}/`;
  return relFiles
    .filter((file) => file.startsWith(prefix) && extensions.includes(path.extname(file).toLowerCase()))
    .map((file) => file.slice(prefix.length));
}

function absolute(relative) {
  return path.join(ROOT, relative);
}

// 不同子项目/语言改的是不同文件 ⇒ 可以并行；但**同一项目内部必须严格串行**
// （eslint → prettier → stylelint 的顺序、ruff 的 check --fix → format 的顺序都不能变）
async function formatAll(relFiles) {
  const tasks = [
    formatMarkdown(relFiles),
    formatGo(relFiles),
    formatDart(relFiles),
    formatJava(relFiles),
  ];
  for (const project of RUFF_PROJECTS) tasks.push(formatRuffProject(project, relFiles));
  for (const project of NODE_PROJECTS) tasks.push(formatNodeProject(project, relFiles));
  await Promise.all(tasks);
}

// .md 全仓共用 dehaze-doc 的 prettier 与配置（前端各项目 .prettierignore 均排除 *.md）
async function formatMarkdown(relFiles) {
  const files = relFiles.filter((file) => path.extname(file).toLowerCase() === '.md');
  if (!files.length) return;
  const docDir = path.join(ROOT, 'dehaze-doc');
  const prettier = path.join(docDir, 'node_modules/.bin/prettier');
  if (!existsSync(prettier)) return log('skip markdown: dehaze-doc 未安装 prettier');
  await runQuiet(
    prettier,
    ['--config', path.join(docDir, '.prettierrc.json'), '--write', ...files.map(absolute)],
    docDir,
    120_000,
  );
}

async function formatRuffProject(project, relFiles) {
  const ruff = path.join(ROOT, 'dehaze-python/.venv/bin/ruff');
  if (!existsSync(ruff)) return log('skip python: ruff 未安装（需在 dehaze-python 下 uv sync）');
  const targets = projectFiles(relFiles, project, ['.py']);
  if (!targets.length) return;
  const cwd = path.join(ROOT, project);
  // 显式传路径时 ruff 不再套用 exclude，必须 --force-exclude（dehaze-modal 等依赖它）
  await runQuiet(ruff, ['check', '--fix', '--force-exclude', ...targets], cwd, 60_000);
  await runQuiet(ruff, ['format', '--force-exclude', ...targets], cwd, 60_000);
}

async function formatNodeProject(project, relFiles) {
  const projectDir = path.join(ROOT, project);
  const binDir = path.join(projectDir, 'node_modules/.bin');
  const targets = relFiles
    .filter((file) => NODE_EXT_KIND[path.extname(file).toLowerCase()] && file.startsWith(`${project}/`))
    .map((file) => file.slice(project.length + 1));
  if (!targets.length) return;

  const kindOf = (file) => NODE_EXT_KIND[path.extname(file).toLowerCase()];
  const eslintTargets = targets.filter((file) => ['script', 'vue'].includes(kindOf(file)));
  const stylelintTargets = targets.filter((file) => ['vue', 'style'].includes(kindOf(file)));
  const eslint = path.join(binDir, 'eslint');
  const prettier = path.join(binDir, 'prettier');
  const stylelint = path.join(binDir, 'stylelint');

  if (eslintTargets.length && existsSync(eslint)) {
    await runQuiet(eslint, ['--fix', ...eslintTargets], projectDir, 120_000);
  }
  if (existsSync(prettier)) {
    await runQuiet(prettier, ['--write', ...targets], projectDir, 120_000);
  }
  if (stylelintTargets.length && existsSync(stylelint)) {
    await runQuiet(stylelint, ['--fix', ...stylelintTargets], projectDir, 120_000);
  }
}

async function formatGo(relFiles) {
  const files = relFiles.filter((file) => file.startsWith('dehaze-go/') && file.endsWith('.go'));
  if (!files.length) return;
  await runQuiet('gofmt', ['-w', ...files.map(absolute)], path.join(ROOT, 'dehaze-go'), 60_000);
}

async function formatDart(relFiles) {
  const files = relFiles.filter(
    (file) =>
      file.startsWith('dehaze_flutter/') &&
      file.endsWith('.dart') &&
      !file.endsWith('.g.dart') &&
      !file.endsWith('.freezed.dart') &&
      !file.startsWith('dehaze_flutter/lib/generated/'),
  );
  if (!files.length) return;
  await runQuiet('dart', ['format', ...files.map(absolute)], path.join(ROOT, 'dehaze_flutter'), 60_000);
}

async function formatJava(relFiles) {
  const files = relFiles.filter((file) => file.startsWith('dehaze-java/') && file.endsWith('.java'));
  if (!files.length) return;
  // 一次 maven 启动处理全部文件；spotless 只写源码、不碰 target/classes，与运行中的 spring-boot:run 不冲突
  const targets = files.map(absolute).join(',');
  await runQuiet(
    'mvn',
    ['-q', 'spotless:apply', `-DspotlessFiles=${targets}`],
    path.join(ROOT, 'dehaze-java'),
    150_000,
  );
}

function pushFinding(findings, { file, line, code, message }) {
  findings.push({
    file,
    line,
    code,
    message,
    key: `${file}:${line}:${code}:${message}`,
  });
}

// ruff 只读检查（不加 --fix）：可自动修复的项在格式化阶段已处理，剩下的才是真问题
async function checkRuff(relFiles) {
  const findings = [];
  const ruff = path.join(ROOT, 'dehaze-python/.venv/bin/ruff');
  if (!existsSync(ruff)) return findings;
  for (const project of RUFF_PROJECTS) {
    const targets = projectFiles(relFiles, project, ['.py']);
    if (!targets.length) continue;
    // 整项目跑（0.1s），不按改动文件裁剪：否则"改 A 打挂 B"（未定义名、未用导入等）看不到；
    // "只报新增"由指纹去重承担。concise 只改排版（path:line:col: CODE msg），规则集仍取各项目 pyproject。
    const output = await runCapture(
      ruff,
      ['check', '--output-format=concise', '.'],
      path.join(ROOT, project),
      30_000,
    );
    if (output === null) continue;
    for (const line of output.split('\n')) {
      const matched = line.match(/^(.*?):(\d+):\d+:\s+([A-Z]+\d+)\s+(.*)$/);
      if (matched) {
        pushFinding(findings, {
          file: `${project}/${matched[1]}`,
          line: matched[2],
          code: matched[3],
          message: matched[4].trim(),
        });
      }
    }
  }
  return findings;
}

function pyrightBin() {
  const local = path.join(ROOT, 'dehaze-python/.venv/bin/pyright');
  return existsSync(local) ? local : 'pyright';
}

// pyright 版本口径单点在 check.sh：不同版本的 typeshed 不同，会得出不同结论，不能各写一份
function pyrightVersion() {
  try {
    const script = readFileSync(path.join(ROOT, 'dehaze-python/scripts/check.sh'), 'utf8');
    return script.match(/PYRIGHT_VERSION:-([\d.]+)/)?.[1];
  } catch {
    return undefined;
  }
}

// 只传本轮改动文件：pyright 仍会解析其依赖闭包（结论不打折），但只报这些文件的错，
// 因此天然滤掉存量告警；全项目检查实测 37s，限定文件后 2-4s。
async function checkPyright(relFiles) {
  const findings = [];
  const version = pyrightVersion();
  const env = version ? { ...process.env, PYRIGHT_PYTHON_FORCE_VERSION: version } : process.env;
  for (const project of PYRIGHT_PROJECTS) {
    const targets = projectFiles(relFiles, project, ['.py']);
    if (!targets.length) continue;
    const config = path.join(ROOT, project, 'pyrightconfig.json');
    if (!existsSync(config)) continue;
    const output = await runCapture(
      pyrightBin(),
      ['-p', config, '--warnings', ...targets],
      path.join(ROOT, project),
      60_000,
      env,
    );
    if (output === null) continue;
    for (const line of output.split('\n')) {
      const matched = line.trim().match(/^(.*?):(\d+):\d+ - (error|warning): (.*)$/);
      if (!matched) continue;
      pushFinding(findings, {
        file: path.isAbsolute(matched[1]) ? path.relative(ROOT, matched[1]) : `${project}/${matched[1]}`,
        line: matched[2],
        code: matched[3] === 'error' ? 'pyright-error' : 'pyright-warning',
        message: matched[4].trim(),
      });
    }
  }
  return findings;
}

async function checkGo(relFiles) {
  const findings = [];
  const touched = relFiles.some((file) => file.startsWith('dehaze-go/') && file.endsWith('.go'));
  if (!touched) return findings;
  // 整仓 vet（实测 1.5s）而不是只 vet 改动所在包：改公共结构体打挂别的包也要能看到；
  // "只报新增"由指纹去重承担（存量已清零，见 §5.4）
  const output = await runCapture('go', ['vet', './...'], path.join(ROOT, 'dehaze-go'), 120_000);
  if (output === null) return findings;
  for (const line of output.split('\n')) {
    const matched = line.trim().match(/^(.*?\.go):(\d+):\d+:\s+(.*)$/);
    if (!matched) continue;
    pushFinding(findings, {
      file: `dehaze-go/${matched[1]}`,
      line: matched[2],
      code: 'go vet',
      message: matched[3].trim(),
    });
  }
  return findings;
}

// 全仓扫描/跨端契约型守卫测试：不随 go build/vet 跑（只在 go test 里），必须显式触发。
// 输入面＝go 代码、SQL 种子/schema、前端 mock、python 声明；命中即跑（冷启动约 5s，命中构建缓存更快）。
const GO_GUARDS = [
  {
    pkg: './internal/router/',
    names: [
      'TestRouterPermissionCodesExistInSeed',
      'TestFrontMockPermissionCodesExistInSeed',
      'TestEveryUsedPermissionIsGrantableBySomeRole',
      'TestRoleMenuGrantsPointToExistingMenus',
      'TestTaskWriteRouteNotGatedByPermission',
    ],
  },
  {
    pkg: './internal/model/',
    names: ['TestGormStateColumnsMustNotCarryNonZeroDefault', 'TestGormDefaultDetectorSelfCheck'],
  },
  {
    pkg: './internal/service/ai/',
    names: [
      'TestModelTypeSetEqualAcrossEnds',
      'TestModelTypeGoDeclarationMatchesRuntimeBehavior',
      'TestModelTypeContractDetectsMutation',
    ],
  },
];

function goGuardsTriggered(relFiles) {
  return relFiles.some(
    (file) =>
      file.startsWith('dehaze-go/') ||
      file.startsWith('config/sql/') ||
      /^dehaze-front-(vue|react)\/mock\//.test(file) ||
      file.startsWith('dehaze-python/app/'),
  );
}

async function checkGoGuards(relFiles) {
  const findings = [];
  if (!goGuardsTriggered(relFiles)) return findings;
  const pattern = `^(${GO_GUARDS.flatMap((guard) => guard.names).join('|')})$`;
  const output = await runCapture(
    'go',
    ['test', ...GO_GUARDS.map((guard) => guard.pkg), '-run', pattern, '-count=1'],
    path.join(ROOT, 'dehaze-go'),
    180_000,
  );
  if (output === null) return findings;

  const lines = output.split('\n');
  let parsed = false;
  for (let i = 0; i < lines.length; i++) {
    const failed = lines[i].match(/^--- FAIL: (\S+)/);
    if (!failed) continue;
    parsed = true;
    const detail = [];
    for (let j = i + 1; j < lines.length && !/^(--- |ok\s|FAIL\s)/.test(lines[j]); j++) {
      detail.push(lines[j].trim());
    }
    const guard = GO_GUARDS.find((item) => item.names.includes(failed[1]));
    const location = detail.join(' ').match(/([\w.-]+_test\.go):(\d+):/);
    const dir = guard ? guard.pkg.replace(/^\.\//, '').replace(/\/$/, '') : '';
    const summary = detail.filter(Boolean).slice(0, 2).join(' ').slice(0, 180);
    findings.push({
      file: location ? `dehaze-go/${dir ? `${dir}/` : ''}${location[1]}` : `dehaze-go/${dir}`,
      line: location ? location[2] : 0,
      code: 'go guard',
      message: summary ? `守卫失败：${failed[1]} —— ${summary}` : `守卫失败：${failed[1]}`,
    });
  }
  // 无 --- FAIL 但整轮 FAIL ＝ 编译/依赖失败，守卫根本没跑起来，不能当"通过"
  if (!parsed && /^FAIL/m.test(output)) {
    findings.push({
      file: 'dehaze-go',
      line: 0,
      code: 'go guard',
      message: `守卫测试未能运行：${output.split('\n').find((line) => line.trim())?.slice(0, 200) ?? ''}`,
    });
  }
  return findings;
}

// 仓库级只读守卫脚本：把"只写在文档里、靠人记"的铁律变成机器拦截。
// 触发面＝该守卫管的对象被本轮改动（整仓扫描由脚本自己做，成本秒级）。
const REPO_GUARD_SCRIPTS = [
  {
    script: 'scripts/check_dockerfiles.py',
    code: 'dockerfile-rule',
    triggered: (file) => /(^|\/)Dockerfile(\.[^/]*)?$/.test(file),
  },
];

async function checkRepoGuards(relFiles) {
  const findings = [];
  const python = path.join(ROOT, 'dehaze-python/.venv/bin/python');
  if (!existsSync(python)) return findings;
  for (const guard of REPO_GUARD_SCRIPTS) {
    if (!relFiles.some(guard.triggered)) continue;
    const output = await runCapture(python, [guard.script], ROOT, 60_000);
    if (output === null) continue;
    const lines = output.split('\n');
    // 脚本输出契约（scripts/check_*.py 统一）：`✗ 文件:行  [规则]` + 下一行 message
    for (let i = 0; i < lines.length; i++) {
      const matched = lines[i].match(/^✗ (.*?):(\d+)\s+\[(.*?)\]$/);
      if (!matched) continue;
      pushFinding(findings, {
        file: matched[1],
        line: matched[2],
        code: guard.code,
        message: `${matched[3]}：${(lines[i + 1] || '').trim()}`,
      });
    }
  }
  return findings;
}

// 读某项目声明的本地依赖（workspace: / file:），用来推导"改上游会波及谁"
function localDependencies(project) {
  const manifest = absolute(`${project}/package.json`);
  if (!existsSync(manifest)) return [];
  try {
    const { dependencies = {}, devDependencies = {} } = JSON.parse(readFileSync(manifest, 'utf8'));
    return Object.entries({ ...dependencies, ...devDependencies })
      .filter(([, spec]) => typeof spec === 'string' && /^(workspace:|file:)/.test(spec))
      .map(([name]) => name);
  } catch {
    return [];
  }
}

// 上游库（本仓是被 workspace 依赖的 dehaze-sdk-js）源码变更时，下游**不是按源码、而是按其构建
// 产物 dist 校验类型** ⇒ 必须先重建产物、再把所有依赖它的项目一并检查，否则"改了 SDK 把下游改坏"
// 在下游完全不可见（假绿）。依赖关系从各项目 package.json 推导，不写死。
function changedUpstreams(relFiles) {
  const names = new Set(
    Object.keys(TYPE_CHECK_PROJECTS).concat(NODE_PROJECTS).flatMap(localDependencies),
  );
  return [...names].filter((name) =>
    relFiles.some((file) => file.startsWith(`${name}/src/`) || file === `${name}/package.json`),
  );
}

// 上游产物陈旧：改过 src 但 dist 落后（切分支/重装/上轮未跑 hook）时，下游按旧 d.ts 校验
// ⇒ 同时得到假错（新契约看不到）与假绿（旧契约仍通过）。按 mtime 判定并纳入重建清单。
function newestSourceMtime(project) {
  let newest = 0;
  const walk = (dir) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (/\.(ts|tsx|vue|mjs|cjs|js|json)$/.test(entry.name)) {
        try {
          newest = Math.max(newest, statSync(full).mtimeMs);
        } catch {
          // 遍历中被删除的文件：忽略
        }
      }
    }
  };
  walk(absolute(`${project}/src`));
  return newest;
}

function upstreamsNeedingBuild(relFiles) {
  const names = new Set(changedUpstreams(relFiles));
  for (const name of new Set(Object.keys(TYPE_CHECK_PROJECTS).concat(NODE_PROJECTS))) {
    const dist = absolute(`${name}/dist/index.d.ts`);
    if (!existsSync(dist)) continue;
    if (newestSourceMtime(name) > statSync(dist).mtimeMs) names.add(name);
  }
  return [...names];
}

function dependentsOf(name) {
  return Object.keys(TYPE_CHECK_PROJECTS).filter((project) =>
    localDependencies(project).includes(name),
  );
}

// 本轮要跑类型检查的项目＝改到自己源码的项目 ∪ 被改上游波及的下游
function typeCheckProjects(relFiles) {
  const projects = new Set(
    Object.keys(TYPE_CHECK_PROJECTS).filter((project) =>
      relFiles.some(
        (file) =>
          file.startsWith(`${project}/`) &&
          TYPE_CHECK_EXTENSIONS.includes(path.extname(file).toLowerCase()),
      ),
    ),
  );
  for (const name of changedUpstreams(relFiles)) {
    for (const dependent of dependentsOf(name)) projects.add(dependent);
  }
  return projects;
}

// tsc/vue-tsc 是全项目级、无法按文件裁剪 ⇒ 整项目跑一遍；"只报新增"交给指纹去重，
// 因此**不再按改动文件过滤输出**：否则"改 A 导致 B 报错"这类连带错误会被整体漏掉。
// 各项目之间互相独立（纯只读诊断）⇒ 并行跑，把多项目同轮的总耗时从 Σ 压到 max
async function checkTypes(projects) {
  const results = await Promise.all(
    [...projects].map(async (project) => {
      const findings = [];
      const [bin, args] = TYPE_CHECK_PROJECTS[project] ?? [];
      if (!bin) return findings;
      const executable = path.join(ROOT, project, bin);
      if (!existsSync(executable)) {
        // 表里写错 bin 名时不能静默跳过，否则就是"hook 装了却像没生效"
        log(`skip type check: ${executable} 不存在`);
        return findings;
      }
      const output = await runCapture(executable, args, path.join(ROOT, project), 120_000);
      if (output === null) return findings;
      // 依赖未装或软链悬空时 wrapper 仍存在，执行会报模块缺失 —— 不能当"类型干净"静默放过
      if (/Cannot find module|MODULE_NOT_FOUND|command not found/.test(output)) {
        pushFinding(findings, {
          file: `${project}/package.json`,
          line: 0,
          code: 'tool-unavailable',
          message: `${bin} 不可用（依赖未安装或链接悬空）⇒ 该项目类型检查被跳过，请先 pnpm install`,
        });
        return findings;
      }
      for (const line of output.split('\n')) {
        const matched = line.trim().match(/^(.*?)\((\d+),\d+\): error (TS\d+): (.*)$/);
        if (!matched) continue;
        pushFinding(findings, {
          file: path.isAbsolute(matched[1])
            ? path.relative(ROOT, matched[1])
            : `${project}/${matched[1]}`,
          line: matched[2],
          code: matched[3],
          message: matched[4].trim(),
        });
      }
      return findings;
    }),
  );
  return results.flat();
}

// 与 runCapture 同，但**只取 stdout**：eslint --format=json 的 JSON 必须干净，混入 stderr 会解析失败
function runCaptureOut(command, args, cwd, timeout) {
  return new Promise((resolve) => {
    const started = Date.now();
    execFile(command, args, { cwd, encoding: 'utf8', timeout, maxBuffer: 8 * 1024 * 1024 }, (error, stdout) => {
      if (error && (error.killed || error.code === 'ETIMEDOUT' || error.code === 'ENOENT')) {
        log(`skip ${command}: ${error.code || 'killed'} (${Date.now() - started}ms)`);
        return resolve(null);
      }
      log(`done ${command} (${Date.now() - started}ms)  (cwd=${cwd})`);
      resolve(stdout || '');
    });
  });
}

// eslint 只读检查（只报 error）：格式化阶段只跑 --fix 且丢弃输出 ⇒ 不可自动修复的规则
// （如 front-vue ai-chat 的 no-restricted-imports 四禁）在轮末完全不可见，这里补一组
async function checkEslint(relFiles) {
  const findings = [];
  const projects = new Set(
    relFiles
      .filter((file) => TYPE_CHECK_EXTENSIONS.includes(path.extname(file).toLowerCase()))
      .map((file) => file.split('/')[0])
      .filter((project) => TYPE_CHECK_PROJECTS[project]),
  );
  for (const project of projects) {
    const eslint = path.join(ROOT, project, 'node_modules/.bin/eslint');
    if (!existsSync(eslint)) {
      log(`skip eslint: ${eslint} 不存在`);
      continue;
    }
    const output = await runCaptureOut(eslint, ['--format=json', '--quiet', '.'], path.join(ROOT, project), 120_000);
    if (output === null) continue;
    let results;
    try {
      results = JSON.parse(output);
    } catch {
      log(`eslint ${project}: JSON 解析失败，跳过（避免误报）`);
      continue;
    }
    for (const result of results) {
      for (const message of result.messages ?? []) {
        if (message.severity !== 2) continue;
        pushFinding(findings, {
          file: path.relative(ROOT, result.filePath),
          line: message.line ?? 0,
          code: message.ruleId ?? 'eslint',
          message: message.message,
        });
      }
    }
  }
  return findings;
}

async function checkAll(relFiles) {
  // 七类检查互相独立（纯只读诊断）⇒ 并行；同一项目内部"格式化 → 检查"的先后由 main 保证
  const groups = await Promise.all([
    checkRuff(relFiles),
    checkPyright(relFiles),
    checkGo(relFiles),
    checkGoGuards(relFiles),
    checkRepoGuards(relFiles),
    checkEslint(relFiles),
    checkTypes(typeCheckProjects(relFiles)),
  ]);
  return groups.flat();
}

function renderReport(findings) {
  const shown = findings
    .slice(0, MAX_FINDINGS_IN_REPORT)
    .map((f) => `- ${f.file}:${f.line} [${f.code}] ${f.message}`);
  if (findings.length > MAX_FINDINGS_IN_REPORT) {
    shown.push(`- …另有 ${findings.length - MAX_FINDINGS_IN_REPORT} 处未列出`);
  }
  return [
    '[轮末检查] 本轮改动文件检出以下问题（只查本轮改动文件、只报新增项）：',
    ...shown,
    '这不是门禁结论，最终判定以 dehaze-python/scripts/check.sh、dehaze-go 的 make test、mvn verify 为准。',
    '请只修复上述位置，不要顺手重构或改动与这些位置无关的代码。',
  ].join('\n');
}

function isExistingFile(relative) {
  const target = absolute(relative);
  return existsSync(target) && statSync(target).isFile();
}

async function main() {
  const payload = JSON.parse(readFileSync(0, 'utf8'));
  const id = sessionIdOf(payload);
  const continuation = payload.stop_hook_active === true;
  // 顺手回收其它会话的孤儿状态：headless（codebuddy -p）实测不发 SessionEnd
  sweepStale(id);

  const registered = consumeLedger(id);
  const files = [...new Set(registered)].filter(isExistingFile);
  // 编辑回读的最后一跳：PostToolUse 只在下一次编辑时才复核，轮末必须兜底（否则最后一次编辑无人验收）
  const writeFindings = recheckWrites(id, 0).map((entry) => ({
    file: entry.file,
    line: 0,
    code: 'write-not-persisted',
    message: `延迟复核未读到写入内容（探针 "${entry.probe}" 未命中，且文件 mtime 早于该次写入），请回读确认`,
    key: `${entry.file}:0:write-not-persisted:${entry.probe}`,
  }));
  log(
    `event=${payload.hook_event_name} client=${payload.client} session=${id.slice(0, 8)} ` +
      `continuation=${continuation} registered=${registered.length} writeFailures=${writeFindings.length}`,
  );
  if (!files.length && !writeFindings.length) return;

  const formatStarted = Date.now();
  if (files.length) await formatAll(files);
  const formatMs = Date.now() - formatStarted;

  // chain = 本续跑链已反馈次数：新的用户轮（continuation=false）归零，避免长会话被累计上限永久静默
  const state = readState(id);
  const reported = Array.isArray(state.reported) ? state.reported : [];
  const chain = continuation && Number.isInteger(state.chain) ? state.chain : 0;
  // 只写当前字段：早期版本的 files 等遗留键不再保留（改动台账已由 PostToolUse 单独维护）
  writeState(id, { reported, chain });

  // 续跑轮同样复验：否则「报错 → 修复」的最后一跳没人验收。防死循环靠指纹去重 + 续跑链上限。
  if (chain >= MAX_REPORTS_PER_CHAIN) {
    return log(`skip check: 续跑链反馈已达上限（format ${formatMs}ms）`);
  }

  // 上游源码变更或 dist 落后（切分支/重装/上轮未跑 hook）⇒ 下游会按旧 d.ts 校验类型（假绿），
  // 必须先重建产物；只在"本轮确实要跑类型检查"时做。DTS 阶段失败即"上游不可构建"，dist/ 已 gitignore。
  if (typeCheckProjects(files).size) {
    for (const upstream of upstreamsNeedingBuild(files)) {
      log(`upstream ${upstream} stale/changed → rebuild before checking dependents`);
      // 必须 await：漏掉就会让下游在旧 d.ts 上做检查（假绿）
      await runQuiet('pnpm', ['--filter', upstream, 'build'], ROOT, 180_000);
    }
  }

  const checkStarted = Date.now();
  const findings = files.length ? [...writeFindings, ...(await checkAll(files))] : writeFindings;
  const checkMs = Date.now() - checkStarted;

  const fresh = findings.filter((finding) => !reported.includes(finding.key));
  if (!fresh.length) {
    return log(`no new findings（共 ${findings.length} 条；format ${formatMs}ms check ${checkMs}ms）`);
  }

  writeState(id, {
    reported: [...reported, ...fresh.map((finding) => finding.key)],
    chain: chain + 1,
  });
  log(`report ${fresh.length} findings（format ${formatMs}ms check ${checkMs}ms）`);
  process.stderr.write(`${renderReport(fresh)}\n`);
  process.exitCode = 2;
}

// main 已是 async：同步 try/catch 抓不到它内部的拒绝，必须用 .catch
main().catch((error) => log(`error: ${error?.stack ?? error}`));
