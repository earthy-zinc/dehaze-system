// 实例生命周期：工作站只起 Chrome；服务器另起 Xvfb、x11vnc、websockify
// （人工可经 noVNC 接管）
import { spawn } from 'node:child_process';
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { HOME } from './config.mjs';
import {
  IBUS_ENGINE,
  IME_CHROME_ENVIRONMENT,
  IME_READY_POLL_MS,
  IME_READY_TIMEOUT_MS,
  LOOPBACK,
  NOVNC_DEFAULT_DIR,
  NOVNC_PATCHED_DIR_NAME,
  PROBE_TIMEOUT_MS,
  SIGN_PORT_ENV,
  STARTUP_POLL_MS,
  STARTUP_TIMEOUT_MS,
  STOP_POLL_MS,
  STOP_TIMEOUT_MS,
  X11VNC_FIXED_ARGS,
  chromeArgs,
  xvfbArgs,
} from './defaults.mjs';
import { GATEWAY_SIGN_SERVER } from './gateway-sign.mjs';
import { findChrome, runCommand } from './tools.mjs';

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

export function isProcessAlive(pid) {
  if (pid === null) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

// pid 与日志文件名带 CDP 端口：同机可并存多个实例而不互相覆盖
function pidKey(config, name) {
  return `${name}-${config.cdpPort}`;
}

export function readPid(config, name) {
  const filePath = join(config.runtimeDir, 'pids', `${pidKey(config, name)}.pid`);
  let content;
  try {
    content = readFileSync(filePath, 'utf8');
  } catch {
    return null;
  }
  const pid = Number(content.trim());
  if (Number.isNaN(pid)) {
    return null;
  }
  return pid;
}

function writePid(config, name, pid) {
  const directory = join(config.runtimeDir, 'pids');
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, `${pidKey(config, name)}.pid`), `${pid}\n`);
}

// 后台拉起进程：日志与 pid 落到运行时目录，进程与调用方解耦
function spawnDetached(command, args, config, name, extraEnvironment) {
  const logDirectory = join(config.runtimeDir, 'logs');
  mkdirSync(logDirectory, { recursive: true });
  const logFile = join(logDirectory, `${pidKey(config, name)}.log`);
  const logHandle = openSync(logFile, 'a');

  const environment = {};
  for (const key of Object.keys(process.env)) {
    environment[key] = process.env[key];
  }
  if (extraEnvironment !== undefined) {
    Object.assign(environment, extraEnvironment);
  }

  const child = spawn(command, args, {
    detached: true,
    stdio: ['ignore', logHandle, logHandle],
    env: environment,
  });
  child.on('error', (error) => {
    console.error(`启动 ${name} 失败：${error.message}`);
  });
  child.unref();
  closeSync(logHandle);
  writePid(config, name, child.pid);
  return child.pid;
}

export async function fetchCdpInfo(config) {
  try {
    const response = await fetch(`${config.cdpUrl}/json/version`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch {
    return null;
  }
}

export async function fetchCdpPages(config) {
  try {
    const response = await fetch(`${config.cdpUrl}/json/list`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!response.ok) {
      return [];
    }
    const targets = await response.json();
    return targets.filter((target) => target.type === 'page');
  } catch {
    return [];
  }
}

// Chrome 启动参数见 lib/defaults.mjs 的 chromeArgs；平台差异也收在那里

export async function startChrome(config) {
  const current = await fetchCdpInfo(config);
  if (current !== null) {
    console.log(`实例已在运行：${config.cdpUrl}（${current.Browser}）`);
    return true;
  }

  const chromePath = findChrome(config);
  if (chromePath === null) {
    const hint = '用发行版软件包安装，或用 CHROME_BIN 指定路径';
    console.error(`未找到 Chrome：${hint}`);
    return false;
  }

  mkdirSync(config.userDir, { recursive: true });
  let extraEnvironment;
  if (config.remoteDesktop) {
    // 让 Chrome 走 ibus：GTK_IM_MODULE 等 + 会话总线地址（ibus-daemon 挂在同一条总线上）
    extraEnvironment = {
      DISPLAY: config.display,
      DBUS_SESSION_BUS_ADDRESS: dbusAddress(config),
      ...IME_CHROME_ENVIRONMENT,
    };
  }
  const pid = spawnDetached(chromePath, chromeArgs(config), config, 'chrome', extraEnvironment);

  const waitAttempts = Math.ceil(STARTUP_TIMEOUT_MS / STARTUP_POLL_MS);
  for (let attempt = 0; attempt < waitAttempts; attempt++) {
    await sleep(STARTUP_POLL_MS);
    const info = await fetchCdpInfo(config);
    if (info !== null) {
      console.log(`Chrome 已就绪：${config.cdpUrl}（pid ${pid}，${info.Browser}）`);
      return true;
    }
  }

  const logFile = join(config.runtimeDir, 'logs', `${pidKey(config, 'chrome')}.log`);
  console.error(`Chrome 启动超时，日志见 ${logFile}`);
  return false;
}

function startXvfb(config) {
  const pid = readPid(config, 'xvfb');
  if (isProcessAlive(pid)) {
    console.log(`Xvfb ${config.display} 已在运行（pid ${pid}）`);
    return;
  }
  const args = xvfbArgs(config);
  const startedPid = spawnDetached('Xvfb', args, config, 'xvfb');
  console.log(`已启动 Xvfb ${config.display}（pid ${startedPid}）`);
}

// ── 中文输入法（仅服务器模式）──
// 输入法走一条独立的 dbus 会话总线：地址固定成运行时目录下的 socket，Chrome 与 ibus 共用同一条。
// ibus-daemon 不带 --daemonize，作为前台子进程常驻，pid 可被本工具管理（--replace 自愈重复拉起）。
function dbusAddress(config) {
  return `unix:path=${join(config.runtimeDir, 'dbus', `bus-${config.cdpPort}`)}`;
}

function startDbus(config) {
  const pid = readPid(config, 'dbus');
  if (isProcessAlive(pid)) {
    console.log(`dbus 会话总线已在运行（pid ${pid}）`);
    return;
  }
  const directory = join(config.runtimeDir, 'dbus');
  mkdirSync(directory, { recursive: true });
  rmSync(join(directory, `bus-${config.cdpPort}`), { force: true });
  const startedPid = spawnDetached(
    'dbus-daemon',
    ['--session', '--nofork', '--nopidfile', `--address=${dbusAddress(config)}`],
    config,
    'dbus',
  );
  console.log(`已启动 dbus 会话总线（pid ${startedPid}）`);
}

// ibus 相关命令共用的环境：同一条会话总线 + 目标 X display
function ibusEnvironment(config) {
  return {
    ...process.env,
    DISPLAY: config.display,
    DBUS_SESSION_BUS_ADDRESS: dbusAddress(config),
  };
}

function startIbus(config) {
  const pid = readPid(config, 'ibus');
  if (isProcessAlive(pid)) {
    console.log(`ibus 已在运行（pid ${pid}）`);
    return;
  }
  // ⚠️ 必须 --daemonize：部分 ibus 版本（如实测的 1.5.28）前台方式会被会话回收，daemonize 才常驻。
  // daemonize 后 daemon 与启动进程脱钩，pid 不取 spawn 结果，稍后从 ibus 地址文件回填（见 recordIbusPid）。
  runCommand('ibus-daemon', ['--daemonize', '--replace', '--xim'], { env: ibusEnvironment(config) });
  console.log(`已启动 ibus（引擎 ${IBUS_ENGINE}）`);
}

// daemonize 后从 ibus 自写的地址文件取真实 pid 记账（客户端也靠这个文件发现 ibus）
function recordIbusPid(config) {
  const directory = join(HOME, '.config', 'ibus', 'bus');
  const suffix = `-unix-${config.display.replace(':', '')}`;
  let entries;
  try {
    entries = readdirSync(directory);
  } catch {
    return;
  }
  for (const name of entries) {
    if (!name.endsWith(suffix)) {
      continue;
    }
    const matched = /IBUS_DAEMON_PID=(\d+)/.exec(readFileSync(join(directory, name), 'utf8'));
    if (matched !== null) {
      writePid(config, 'ibus', Number(matched[1]));
      return;
    }
  }
}

// 切到拼音引擎。⚠️ libpinyin 会以退出码 1 返回（引擎实际已生效），故不看退出码、只当触发切换
function setIbusEngine(config) {
  runCommand('ibus', ['engine', IBUS_ENGINE], { env: ibusEnvironment(config) });
}

function currentIbusEngine(config) {
  const result = runCommand('ibus', ['engine'], { env: ibusEnvironment(config) });
  if (result.status !== 0 || typeof result.stdout !== 'string') {
    return null;
  }
  return result.stdout.trim();
}

// Xvfb 之后按 dbus → ibus → 拼音引擎 拉起；失败只告警，不阻断实例启动
async function startIme(config) {
  startDbus(config);
  startIbus(config);
  const attempts = Math.ceil(IME_READY_TIMEOUT_MS / IME_READY_POLL_MS);
  for (let attempt = 0; attempt < attempts; attempt++) {
    setIbusEngine(config);
    await sleep(IME_READY_POLL_MS);
    if (currentIbusEngine(config) === IBUS_ENGINE) {
      recordIbusPid(config);
      console.log(`输入法就绪：${IBUS_ENGINE}（noVNC 里直接拼音输入；客户端请保持英文输入法）`);
      return;
    }
  }
  console.log(`⚠️ ibus 引擎切换超时，noVNC 里可手动执行：ibus engine ${IBUS_ENGINE}`);
}

function startX11vnc(config) {
  const pid = readPid(config, 'x11vnc');
  if (isProcessAlive(pid)) {
    console.log(`x11vnc 已在运行（pid ${pid}）`);
    return;
  }

  const args = [
    '-display', config.display,
    '-rfbport', String(config.vncPort),
    '-localhost',
    ...X11VNC_FIXED_ARGS,
  ];
  if (process.env.VNC_PASSWORD === undefined) {
    args.push('-nopw');
    console.log('提示：未设置 VNC_PASSWORD，x11vnc 无密码');
    console.log(`（仅监听 ${LOOPBACK}，需 SSH 隧道才可达）`);
  } else {
    const passwordFile = join(config.runtimeDir, '.vncpasswd');
    runCommand('x11vnc', ['-storepasswd', process.env.VNC_PASSWORD, passwordFile]);
    args.push('-rfbauth', passwordFile);
  }

  const startedPid = spawnDetached('x11vnc', args, config, 'x11vnc');
  console.log(`已启动 x11vnc（pid ${startedPid}，rfbport ${config.vncPort}）`);
}

function startWebsockify(config) {
  const pid = readPid(config, 'websockify');
  if (isProcessAlive(pid)) {
    console.log(`websockify 已在运行（pid ${pid}）`);
    return;
  }

  let novncDirectory = process.env.NOVNC_WEB;
  if (novncDirectory === undefined) {
    // 优先用补丁版 noVNC（若已安装）；NOVNC_WEB 仍可显式覆盖
    const patched = join(config.runtimeDir, NOVNC_PATCHED_DIR_NAME);
    novncDirectory = existsSync(patched) ? patched : NOVNC_DEFAULT_DIR;
  }
  const args = [
    '--web', novncDirectory,
    `${LOOPBACK}:${config.webPort}`,
    `${LOOPBACK}:${config.vncPort}`,
  ];
  const startedPid = spawnDetached('websockify', args, config, 'websockify');
  console.log(`已启动 websockify 与 noVNC（pid ${startedPid}，web ${config.webPort}）`);
}

function startVnc(config) {
  startX11vnc(config);
  startWebsockify(config);
}

// 智能网关签名代理：仅 DevCloud 等无智能网关的机器需要；先于 Chrome 拉起（Chrome 启动参数引用其 PAC）
function startSignProxy(config) {
  if (!config.signProxy) {
    return;
  }
  const pid = readPid(config, 'gateway-sign');
  if (isProcessAlive(pid)) {
    console.log(`签名代理已在运行（pid ${pid}，127.0.0.1:${config.signPort}）`);
    return;
  }
  const startedPid = spawnDetached(
    process.execPath,
    [GATEWAY_SIGN_SERVER],
    config,
    'gateway-sign',
    { [SIGN_PORT_ENV]: String(config.signPort) },
  );
  console.log(`已启动签名代理（pid ${startedPid}，127.0.0.1:${config.signPort}）`);
  console.log(`Chrome 经 PAC 将各站点（panshi / imeta / forcem）的 http 流量交给该代理：${config.signPacUrl}`);
}

async function stopSignProxy(config) {
  if (!config.signProxy) {
    return;
  }
  await stopProcess(config, 'gateway-sign', '签名代理');
}

export async function start(config) {
  if (config.remoteDesktop) {
    startXvfb(config);
    await startIme(config);
  }
  if (config.signProxyCredentialed) {
    startSignProxy(config);
  } else {
    console.log('提示：未配置签名代理凭证（STAFF_NAME + 至少一个站点 token），跳过签名代理');
    console.log('（仅 DevCloud 这类没有智能网关的机器直达 panshi/imeta/forcem 时才需要）');
  }
  const started = await startChrome(config);
  if (!started) {
    process.exitCode = 1;
  }
  if (config.remoteDesktop) {
    startVnc(config);
    console.log('');
    console.log('人工接管入口：node scripts/chrome-debug/index.mjs view');
  }
}

async function stopProcess(config, name, label) {
  const pid = readPid(config, name);
  if (!isProcessAlive(pid)) {
    console.log(`${label} 未在运行`);
    return;
  }

  try {
    process.kill(pid, 'SIGTERM');
  } catch {
    // 进程已自行退出，继续清理 pid 文件
  }
  const waitAttempts = Math.ceil(STOP_TIMEOUT_MS / STOP_POLL_MS);
  for (let attempt = 0; attempt < waitAttempts; attempt++) {
    if (!isProcessAlive(pid)) {
      break;
    }
    await sleep(STOP_POLL_MS);
  }
  if (isProcessAlive(pid)) {
    try {
      process.kill(pid, 'SIGKILL');
    } catch {
      // 进程已自行退出，继续清理 pid 文件
    }
  }
  rmSync(join(config.runtimeDir, 'pids', `${pidKey(config, name)}.pid`), { force: true });
  console.log(`已停止 ${label}（pid ${pid}）`);
}

// 没有 pid 记录的实例（手工启动的）只能经 CDP 关闭
async function closeViaCdp(config) {
  const info = await fetchCdpInfo(config);
  if (info === null) {
    return true;
  }
  const socket = new WebSocket(info.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  socket.send(JSON.stringify({ id: 1, method: 'Browser.close' }));
  socket.close();

  // 等待实例真正下线：与停止等待共用同一份超时
  const waitAttempts = Math.ceil(STOP_TIMEOUT_MS / STARTUP_POLL_MS);
  for (let attempt = 0; attempt < waitAttempts; attempt++) {
    await sleep(STARTUP_POLL_MS);
    const stillRunning = await fetchCdpInfo(config);
    if (stillRunning === null) {
      return true;
    }
  }
  return false;
}

async function stopChrome(config) {
  const pid = readPid(config, 'chrome');
  if (isProcessAlive(pid)) {
    await stopProcess(config, 'chrome', 'Chrome');
    return;
  }
  const info = await fetchCdpInfo(config);
  if (info === null) {
    console.log('Chrome 未在运行');
    return;
  }
  console.log('实例没有 pid 记录（非本工具启动），经 CDP 关闭');
  const closed = await closeViaCdp(config);
  if (closed) {
    console.log('已停止 Chrome');
  } else {
    console.log('CDP 关闭未生效，请手工结束进程');
  }
}

export async function stop(config) {
  if (config.remoteDesktop) {
    await stopProcess(config, 'websockify', 'websockify');
    await stopProcess(config, 'x11vnc', 'x11vnc');
  }
  await stopSignProxy(config);
  await stopChrome(config);
  if (config.remoteDesktop) {
    await stopProcess(config, 'ibus', 'ibus');
    await stopProcess(config, 'dbus', 'dbus');
    await stopProcess(config, 'xvfb', 'Xvfb');
  }
}

