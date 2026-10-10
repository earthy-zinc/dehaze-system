// 路径与配置：运行时产物统一放在 ~/.chrome-debug，并写入三个 CLI 的端口配置；
// 默认值与固定参数集中在 lib/defaults.mjs
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { delimiter, dirname, join } from 'node:path';
import {
  DEFAULT_CDP_PORT,
  DEFAULT_DISPLAY,
  DEFAULT_SCREEN,
  DEFAULT_SCREEN_TEXT,
  DEFAULT_SIGN_PORT,
  DEFAULT_VNC_PORT,
  DEFAULT_WEB_PORT,
  LOOPBACK,
  RUNTIME_DIR_NAME,
} from './defaults.mjs';
import { pacUrl, resolveGatewayCredentials } from './gateway-sign.mjs';

export const HOME = homedir();
export const IS_WIN = process.platform === 'win32';
export const IS_LINUX = process.platform === 'linux';
export const RUNTIME_DIR = join(HOME, RUNTIME_DIR_NAME);

const STATE_FILE = join(RUNTIME_DIR, 'config.json');

/**
 * 运行时配置：状态文件、命令行选项与默认值合成后的结果
 * @typedef {object} Config
 * @property {number} cdpPort CDP 调试端口
 * @property {string} userDir Chrome profile 目录
 * @property {string} shimDir shim 安装目录
 * @property {boolean} remoteDesktop 是否拉起 Xvfb 与远程桌面（服务器模式）
 * @property {number} vncPort VNC 端口
 * @property {number} webPort noVNC 网页端口
 * @property {string} display Xvfb 显示号
 * @property {{width: number, height: number, depth: number}} screen Xvfb 与 Chrome 窗口尺寸
 * @property {boolean} devtools 是否自动打开 DevTools（默认开；--no-devtools 关闭）
 * @property {string} runtimeDir 运行时产物根目录
 * @property {string} cdpUrl CDP 端点地址
 * @property {string} webUrl 远程桌面网页地址
 * @property {number} signPort 智能网关签名代理端口
 * @property {boolean} signProxy 是否启用签名代理（凭证齐备且未被 --no-sign-proxy 关闭）
 * @property {boolean} signProxyCredentialed 凭证是否齐备（仅用于缺凭证时的提示）
 * @property {string} signPacUrl 签名代理提供的 PAC 地址（Chrome 经 --proxy-pac-url 引用）
 * @property {string[]} positional 当前命令的位置参数
 */

// 平台启动参数已收进 lib/defaults.mjs（chromeExtraArgs / chromeArgs）

// 读 JSON 文件；文件不存在或内容损坏时返回 null
export function readJsonFile(filePath) {
  let content;
  try {
    content = readFileSync(filePath, 'utf8');
  } catch {
    return null;
  }
  try {
    return JSON.parse(content);
  } catch {
    return null;
  }
}

// 取 --name=value 的值；--name（不带值）记为 true；未出现时返回 undefined
function readOption(argv, name) {
  const prefix = `--${name}=`;
  for (const argument of argv) {
    if (argument === `--${name}`) {
      return true;
    }
    if (argument.startsWith(prefix)) {
      return argument.slice(prefix.length);
    }
  }
  return undefined;
}

// 按顺序取第一个有效值，实现"命令行 > 状态文件 > 默认值"的优先级
function pickFirst(...values) {
  for (const value of values) {
    if (value === undefined || value === null || value === '') {
      continue;
    }
    return value;
  }
  return undefined;
}

function defaultShimDir() {
  if (IS_WIN) {
    return join(HOME, 'bin');
  }
  return join(HOME, '.local', 'bin');
}

// 服务器默认启用远程桌面（无桌面环境），工作站默认关闭
function resolveRemoteDesktop(argv, state) {
  if (readOption(argv, 'no-remote-desktop') === true) {
    return false;
  }
  if (typeof state.remoteDesktop === 'boolean') {
    return state.remoteDesktop;
  }
  return IS_LINUX;
}

function emptyObjectWhenNull(value) {
  if (value === null) {
    return {};
  }
  return value;
}

// 屏幕尺寸是 Xvfb、Chrome 窗口与 vnc-check 期望值的同一份取值
function parseScreen(value) {
  const matched = /^(\d+)x(\d+)$/.exec(String(value));
  if (matched === null) {
    console.error(`--screen 需要 <宽>x<高> 形式（如 --screen=1920x1080），收到：${value}`);
    process.exit(2);
  }
  return {
    width: Number(matched[1]),
    height: Number(matched[2]),
    depth: DEFAULT_SCREEN.depth,
  };
}

/** @returns {Config} */
export function resolveConfig(argv) {
  const saved = readJsonFile(STATE_FILE);
  const state = emptyObjectWhenNull(saved);

  const config = {
    cdpPort: Number(pickFirst(readOption(argv, 'cdp-port'), state.cdpPort, DEFAULT_CDP_PORT)),
    userDir: pickFirst(readOption(argv, 'user-dir'), state.userDir, join(RUNTIME_DIR, 'profile')),
    shimDir: pickFirst(readOption(argv, 'shim-dir'), state.shimDir, defaultShimDir()),
    remoteDesktop: resolveRemoteDesktop(argv, state),
    vncPort: Number(pickFirst(readOption(argv, 'vnc-port'), state.vncPort, DEFAULT_VNC_PORT)),
    webPort: Number(pickFirst(readOption(argv, 'web-port'), state.webPort, DEFAULT_WEB_PORT)),
    display: pickFirst(readOption(argv, 'display'), state.display, DEFAULT_DISPLAY),
    screen: parseScreen(pickFirst(readOption(argv, 'screen'), state.screen, DEFAULT_SCREEN_TEXT)),
    devtools: readOption(argv, 'no-devtools') !== true,
    signPort: Number(pickFirst(readOption(argv, 'sign-port'), state.signPort, DEFAULT_SIGN_PORT)),
    runtimeDir: RUNTIME_DIR,
  };
  config.cdpUrl = `http://${LOOPBACK}:${config.cdpPort}`;
  config.webUrl = `http://${LOOPBACK}:${config.webPort}/vnc.html`;
  config.signPacUrl = pacUrl(config);
  // 签名代理只对「没有智能网关」的机器（如 DevCloud）有意义；
  // 凭证缺一则不启用（不影响启动），真正的告警在 start 时打印
  config.signProxyCredentialed = resolveGatewayCredentials() !== null;
  // --no-sign-proxy 为现名；--no-panshi-sign 保留为兼容旧用法
  const signDisabled =
    readOption(argv, 'no-sign-proxy') === true || readOption(argv, 'no-panshi-sign') === true;
  config.signProxy = config.signProxyCredentialed && !signDisabled;
  config.positional = argv.slice(1).filter((argument) => !argument.startsWith('--'));

  return config;
}

export function writeState(config) {
  mkdirSync(RUNTIME_DIR, { recursive: true });
  const state = {
    cdpPort: config.cdpPort,
    userDir: config.userDir,
    shimDir: config.shimDir,
    remoteDesktop: config.remoteDesktop,
    vncPort: config.vncPort,
    webPort: config.webPort,
    display: config.display,
    screen: `${config.screen.width}x${config.screen.height}`,
    signPort: config.signPort,
  };
  writeFileSync(STATE_FILE, `${JSON.stringify(state, null, 2)}\n`);
}

// 写入时保留配置文件里已有的其它项
export function writeToolConfigs(config) {
  const agentBrowserFile = join(HOME, '.agent-browser', 'config.json');
  const agentBrowserSaved = readJsonFile(agentBrowserFile);
  const agentBrowserConfig = emptyObjectWhenNull(agentBrowserSaved);
  mkdirSync(dirname(agentBrowserFile), { recursive: true });
  agentBrowserConfig.cdp = String(config.cdpPort);
  writeFileSync(agentBrowserFile, `${JSON.stringify(agentBrowserConfig, null, 2)}\n`);

  const playwrightFile = join(HOME, '.playwright', 'cli.config.json');
  const playwrightSaved = readJsonFile(playwrightFile);
  const playwrightConfig = emptyObjectWhenNull(playwrightSaved);
  mkdirSync(dirname(playwrightFile), { recursive: true });
  if (playwrightConfig.browser === undefined) {
    playwrightConfig.browser = {};
  }
  playwrightConfig.browser.cdpEndpoint = config.cdpUrl;
  writeFileSync(playwrightFile, `${JSON.stringify(playwrightConfig, null, 2)}\n`);

  console.log(`已写入 agent-browser 配置：cdp=${config.cdpPort}`);
  console.log(`已写入 playwright-cli 配置：cdpEndpoint=${config.cdpUrl}`);
  console.log('chrome-devtools 没有配置文件，连接目标由 daemon 启动参数保证');
}

// Windows 的用户级 / 机器级 PATH 都存在注册表，值类型是 REG_EXPAND_SZ（可能带 %USERPROFILE% 等变量），
// 必须按原样读回、原样写回；setx 会在 1024 字符处截断，不能用。
const WINDOWS_USER_ENVIRONMENT = 'HKCU\\Environment';
const WINDOWS_MACHINE_ENVIRONMENT =
  'HKLM\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment';

// 控制台代码页与 UTF-8 不一致时会解出替换字符，此时按读回值改写会破坏原值，故返回 null
function readWindowsPath(key) {
  const result = spawnSync('reg', ['query', key, '/v', 'Path'], { encoding: 'utf8' });
  if (result.status !== 0 || typeof result.stdout !== 'string') {
    return null;
  }
  const matched = /Path\s+REG_(?:EXPAND_)?SZ\s+(.*)/.exec(result.stdout);
  if (matched === null) {
    return null;
  }
  const value = matched[1].trim();
  if (value.includes('\uFFFD')) {
    return null;
  }
  return value;
}

export function windowsUserPath() {
  return readWindowsPath(WINDOWS_USER_ENVIRONMENT);
}

export function writeWindowsUserPath(value) {
  spawnSync(
    'reg',
    ['add', WINDOWS_USER_ENVIRONMENT, '/v', 'Path', '/t', 'REG_EXPAND_SZ', '/d', value, '/f'],
    { encoding: 'utf8' },
  );
}

// shim 将来是在普通用户环境里被调用的，定位真身必须用持久 PATH：
// 会话 PATH 可能被版本管理器（fnm 等）注入临时目录，烘进 shim 会随会话失效
export function persistentPathEntries() {
  if (!IS_WIN) {
    if (process.env.PATH === undefined) {
      return [];
    }
    return process.env.PATH.split(delimiter);
  }

  const entries = [];
  for (const value of [readWindowsPath(WINDOWS_MACHINE_ENVIRONMENT), windowsUserPath()]) {
    if (value !== null) {
      entries.push(...value.split(';'));
    }
  }
  return entries.filter((entry) => entry !== '');
}
