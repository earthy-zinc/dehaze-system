// 外部依赖定位：三个 CLI 的真身、Chrome 可执行文件、chrome-devtools daemon
import { spawnSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { delimiter, join, resolve } from 'node:path';
import { IS_LINUX, IS_WIN } from './config.mjs';
import { VERSION_TIMEOUT_MS } from './defaults.mjs';

export const TOOLS = ['agent-browser', 'playwright-cli', 'chrome-devtools'];

// 同步执行命令并取回输出
export function runCommand(command, args, options) {
  let settings = options;
  if (settings === undefined) {
    settings = {};
  }
  settings.encoding = 'utf8';
  // Windows 的 .cmd / .bat 必须经 shell 执行：Node 20 起不再允许直接 spawn 批处理。
  // 走 shell 时不能再并列传 args（会触发 DEP0190），改为拼成一条带引号的命令
  if (IS_WIN && /\.(cmd|bat)$/i.test(command)) {
    settings.shell = true;
    return spawnSync([command, ...args].map((part) => `"${part}"`).join(' '), settings);
  }
  return spawnSync(command, args, settings);
}

// 命令未能执行时 stdout 可能是 null 或 undefined，统一归一为空串
function commandOutput(result) {
  if (typeof result.stdout !== 'string') {
    return '';
  }
  return result.stdout;
}

function pathEntries() {
  if (process.env.PATH === undefined) {
    return [];
  }
  return process.env.PATH.split(delimiter);
}

// 在 PATH 中找可执行文件；跳过 shim 目录，避免 shim 找到自己造成递归
export function findInPath(commandName, shimDir, entries) {
  let extensions = [''];
  if (IS_WIN) {
    extensions = ['.cmd', '.exe', '.bat', ''];
  }

  let directories = entries;
  if (directories === undefined) {
    directories = pathEntries();
  }

  for (const directory of directories) {
    if (directory === '') {
      continue;
    }
    if (shimDir !== undefined && resolve(directory) === resolve(shimDir)) {
      continue;
    }
    for (const extension of extensions) {
      const candidate = join(directory, commandName + extension);
      if (existsSync(candidate) && statSync(candidate).isFile()) {
        return candidate;
      }
    }
  }
  return null;
}

function chromeCandidates() {
  if (IS_WIN) {
    let programFiles = process.env.PROGRAMFILES;
    if (programFiles === undefined) {
      programFiles = 'C:\\Program Files';
    }
    let localAppData = process.env.LOCALAPPDATA;
    if (localAppData === undefined) {
      localAppData = '';
    }
    return [
      join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    ];
  }
  if (IS_LINUX) {
    return ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'];
  }
  return ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'];
}

// 先看 CHROME_BIN，再按平台候选列表查找；纯命令名走 PATH
export function findChrome(config) {
  const fromEnvironment = process.env.CHROME_BIN;
  if (fromEnvironment !== undefined && existsSync(fromEnvironment)) {
    return fromEnvironment;
  }

  for (const candidate of chromeCandidates()) {
    const isAbsolutePath = candidate.includes('/') || candidate.includes('\\');
    if (isAbsolutePath) {
      if (existsSync(candidate)) {
        return candidate;
      }
      continue;
    }
    const found = findInPath(candidate, config.shimDir);
    if (found !== null) {
      return found;
    }
  }
  return null;
}

// Windows 上 Chrome 是 GUI 程序，--version 既不输出也不退出：版本改从注册表读
function windowsChromeVersion() {
  const output = commandOutput(
    runCommand('reg', ['query', 'HKCU\\Software\\Google\\Chrome\\BLBeacon', '/v', 'version']),
  );
  const matched = output.match(/\d+\.\d+\.\d+\.\d+/);
  if (matched === null) {
    return null;
  }
  return matched[0];
}

export function chromeVersion(chromePath) {
  if (IS_WIN) {
    const fromRegistry = windowsChromeVersion();
    if (fromRegistry !== null) {
      return fromRegistry;
    }
  }

  const probe = runCommand(chromePath, ['--version'], { timeout: VERSION_TIMEOUT_MS });
  const output = commandOutput(probe);
  const matched = output.match(/Google Chrome [\d.]+/);
  if (matched !== null) {
    return matched[0];
  }
  const trimmed = output.trim();
  if (trimmed === '') {
    return '版本未知';
  }
  return trimmed;
}

export function toolVersion(toolPath) {
  const probe = runCommand(toolPath, ['--version'], { timeout: VERSION_TIMEOUT_MS });
  const output = commandOutput(probe);
  const lines = output.trim().split('\n');
  return lines[0];
}

export function daemonStatus(config) {
  const toolPath = findInPath('chrome-devtools', config.shimDir);
  if (toolPath === null) {
    return null;
  }

  const output = commandOutput(runCommand(toolPath, ['status']));
  if (!output.includes('is running')) {
    return { toolPath: toolPath, running: false, args: '' };
  }

  const matched = output.match(/args=\[(.*)\]/);
  let args = '';
  if (matched !== null) {
    args = matched[1];
  }
  return { toolPath: toolPath, running: true, args: args };
}

// daemon 的连接目标由启动参数决定：不符时它会自起隔离实例、绕过登录态
export function syncDaemon(config) {
  const status = daemonStatus(config);
  if (status === null) {
    return false;
  }
  const wanted = `--browser-url=${config.cdpUrl}`;
  if (status.running && status.args.includes(wanted)) {
    return false;
  }
  runCommand(status.toolPath, ['start', wanted]);
  return true;
}
