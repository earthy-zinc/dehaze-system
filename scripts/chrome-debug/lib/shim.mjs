// shim 安装：把三个 CLI 名遮蔽成"实例不在就拉起，再转发真身"的自包含脚本
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { delimiter, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HOME, IS_WIN, persistentPathEntries, windowsUserPath, writeWindowsUserPath } from './config.mjs';
import { LOOPBACK, STARTUP_POLL_MS, STARTUP_TIMEOUT_MS, chromeExtraArgs } from './defaults.mjs';
import { render } from './text.mjs';
import { TOOLS, findChrome, findInPath } from './tools.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const RC_MARK = 'chrome-debug shim path';

let templateName = 'shim-posix.tmpl';
if (IS_WIN) {
  templateName = 'shim-windows.tmpl';
}
const SHIM_TEMPLATE = join(HERE, templateName);

export function shimFilePath(config, tool) {
  if (IS_WIN) {
    return join(config.shimDir, `${tool}.cmd`);
  }
  return join(config.shimDir, tool);
}

function rcFilePath() {
  let shell = process.env.SHELL;
  if (shell === undefined) {
    shell = '';
  }
  if (shell.includes('zsh')) {
    return join(HOME, '.zshrc');
  }
  return join(HOME, '.bashrc');
}

function pathEntries() {
  if (process.env.PATH === undefined) {
    return [];
  }
  return process.env.PATH.split(delimiter);
}

// shim 要排在真身（nvm / npm 的 bin）之前才会被命中，否则装了也不生效
export function pathOrderOk(config) {
  const directories = [];
  for (const item of pathEntries()) {
    if (item === '') {
      directories.push(resolve('/'));
      continue;
    }
    directories.push(resolve(item));
  }

  const shimIndex = directories.indexOf(resolve(config.shimDir));
  if (shimIndex === -1) {
    return false;
  }

  const realTool = findInPath('agent-browser', config.shimDir);
  if (realTool === null) {
    return true;
  }
  const realIndex = directories.indexOf(resolve(dirname(realTool)));
  return shimIndex < realIndex;
}

function appendRcPath(config) {
  const rcFile = rcFilePath();
  let current = '';
  if (existsSync(rcFile)) {
    current = readFileSync(rcFile, 'utf8');
  }
  if (current.includes(RC_MARK)) {
    return;
  }

  const explanation = `# 让 shim 目录优先于 nvm / npm 的 bin（${RC_MARK}）`;
  const exportLine = `export PATH="${config.shimDir}:$PATH" # ${RC_MARK}`;
  writeFileSync(rcFile, `${current.trimEnd()}\n\n${explanation}\n${exportLine}\n`);
  console.log(`已在 ${rcFile} 末尾前置 shim 目录（新开 shell 生效）`);
}

// daemon 的连接目标由启动参数决定：不符时会自起隔离实例、绕过登录态
function daemonGuard(config, realToolPath) {
  const wanted = `--browser-url=${config.cdpUrl}`;
  if (IS_WIN) {
    const checkLine = `"${realToolPath}" status 2>nul | findstr /c:"${wanted}" >nul`;
    const fixLine = `if errorlevel 1 "${realToolPath}" start ${wanted} >nul 2>&1`;
    return `${checkLine}\n${fixLine}`;
  }
  const check = `"${realToolPath}" status 2>/dev/null | grep -q -- "${wanted}"`;
  const fix = `"${realToolPath}" start ${wanted} >/dev/null 2>&1`;
  return `${check} || ${fix}`;
}

// shim 自包含：实例不在就地拉起，再转发真身；脚本形态见 lib/shim-*.tmpl
function shimContent(config, tool) {
  const realTool = findInPath(tool, config.shimDir, persistentPathEntries());
  let realToolPath = '';
  if (realTool !== null) {
    realToolPath = realTool;
  }

  // 服务器模式的实例由 start 管理（拉起还要 Xvfb 与远程桌面），shim 只做转发
  let chromePath = '';
  if (!config.remoteDesktop) {
    const found = findChrome(config);
    if (found !== null) {
      chromePath = found;
    }
  }

  let daemonGuardLine = '';
  if (tool === 'chrome-devtools') {
    daemonGuardLine = daemonGuard(config, realToolPath);
  }

  // shim 自包含、不依赖仓库，等待节奏在安装时注入（Windows 用 ping 计时，约 1 秒一次）
  let startupPolls = Math.ceil(STARTUP_TIMEOUT_MS / 1000);
  if (!IS_WIN) {
    startupPolls = Math.ceil(STARTUP_TIMEOUT_MS / STARTUP_POLL_MS);
  }

  const values = {
    tool: tool,
    cdpUrl: config.cdpUrl,
    loopback: LOOPBACK,
    cdpPort: config.cdpPort,
    userDir: config.userDir,
    realToolPath: realToolPath,
    chromePath: chromePath,
    extraArgs: chromeExtraArgs(config).join(' '),
    daemonGuard: daemonGuardLine,
    startupPolls: startupPolls,
    startupPollSeconds: STARTUP_POLL_MS / 1000,
  };
  const text = render(readFileSync(SHIM_TEMPLATE, 'utf8'), values);
  if (IS_WIN) {
    return text.replace(/\n/g, '\r\n');
  }
  return text;
}

function prependWindowsPath(config) {
  const current = windowsUserPath();
  if (current === null) {
    console.log(`未能安全读取用户级 Path，请手工把 ${config.shimDir} 放到最前`);
    return;
  }

  const parts = current.split(';').filter((part) => part !== '');
  const existing = parts.findIndex((part) => part.toLowerCase() === config.shimDir.toLowerCase());
  if (existing === 0) {
    return;
  }
  if (existing > 0) {
    parts.splice(existing, 1);
  }

  parts.unshift(config.shimDir);
  writeWindowsUserPath(parts.join(';'));
  console.log(`已把 ${config.shimDir} 置顶到用户级 Path（重新登录或重开终端后生效）`);
}

function removeWindowsPath(config) {
  const current = windowsUserPath();
  if (current === null) {
    return;
  }

  const parts = current.split(';').filter((part) => part !== '');
  const kept = parts.filter((part) => part.toLowerCase() !== config.shimDir.toLowerCase());
  if (kept.length === parts.length) {
    return;
  }
  writeWindowsUserPath(kept.join(';'));
  console.log(`已从用户级 Path 移除 ${config.shimDir}`);
}

export function installShims(config) {
  mkdirSync(config.shimDir, { recursive: true });
  for (const tool of TOOLS) {
    const filePath = shimFilePath(config, tool);
    writeFileSync(filePath, shimContent(config, tool), { mode: 0o755 });
    console.log(`已安装 shim：${filePath}`);
  }

  if (IS_WIN) {
    prependWindowsPath(config);
    return;
  }
  if (!pathOrderOk(config)) {
    appendRcPath(config);
  }
}

export function uninstallShims(config) {
  for (const tool of TOOLS) {
    const filePath = shimFilePath(config, tool);
    if (existsSync(filePath)) {
      rmSync(filePath);
      console.log(`已移除 shim：${filePath}`);
    }
  }

  if (IS_WIN) {
    removeWindowsPath(config);
    return;
  }
  const rcFile = rcFilePath();
  if (!existsSync(rcFile)) {
    return;
  }
  const lines = readFileSync(rcFile, 'utf8').split('\n');
  const kept = lines.filter((line) => !line.includes(RC_MARK));
  if (kept.length !== lines.length) {
    writeFileSync(rcFile, kept.join('\n'));
    console.log(`已从 ${rcFile} 移除 PATH 行`);
  }
}
