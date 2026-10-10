// 自检视图：check（环境与绑定）、status（实例与 daemon）、view（人工接管入口）
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HOME, readJsonFile } from './config.mjs';
import { LOOPBACK, MIN_NODE_MAJOR } from './defaults.mjs';
import { fetchCdpInfo, fetchCdpPages, isProcessAlive, readPid } from './instance.mjs';
import { pathOrderOk, shimFilePath } from './shim.mjs';
import { render } from './text.mjs';
import {
  TOOLS,
  chromeVersion,
  daemonStatus,
  findChrome,
  findInPath,
  toolVersion,
} from './tools.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));

// 从嵌套的配置对象里取出字符串值，取不到时返回 null
function readConfiguredValue(configFile, keys) {
  let current = readJsonFile(configFile);
  if (current === null) {
    return null;
  }
  for (const key of keys) {
    if (typeof current !== 'object' || current === null) {
      return null;
    }
    current = current[key];
  }
  if (current === undefined || current === null) {
    return null;
  }
  return String(current);
}

function createReporter() {
  const counters = { failed: 0, warned: 0 };

  // 带 warn 的项由 setup 自动处理，不计入失败
  function report(ok, label, detail, isWarning) {
    let mark = '✗';
    if (ok) {
      mark = '✓';
    } else if (isWarning === true) {
      mark = '!';
      counters.warned += 1;
    } else {
      counters.failed += 1;
    }

    if (detail === undefined || detail === '') {
      console.log(`${mark} ${label}`);
      return;
    }
    console.log(`${mark} ${label} — ${detail}`);
  }

  return { report: report, counters: counters };
}

export async function check(config) {
  const reporter = createReporter();
  const report = reporter.report;

  const nodeMajor = Number(process.versions.node.split('.')[0]);
  const nodeDetail = `v${process.versions.node}（要求 >= ${MIN_NODE_MAJOR}）`;
  report(nodeMajor >= MIN_NODE_MAJOR, 'Node', nodeDetail, false);

  const chromePath = findChrome(config);
  if (chromePath === null) {
    const hint = '未找到：用发行版软件包安装，或设 CHROME_BIN';
    report(false, 'Chrome', hint, false);
  } else {
    const detail = `${chromeVersion(chromePath)}（${chromePath}）`;
    report(true, 'Chrome', detail, false);
  }

  for (const tool of TOOLS) {
    const toolPath = findInPath(tool, config.shimDir);
    if (toolPath === null) {
      report(false, tool, `未安装：npm install -g ${tool}`, false);
    } else {
      const detail = `${toolVersion(toolPath)}（${toolPath}）`;
      report(true, tool, detail, false);
    }

    const shimPath = shimFilePath(config, tool);
    if (existsSync(shimPath)) {
      report(true, `${tool} shim`, shimPath, true);
    } else {
      report(false, `${tool} shim`, '未安装（setup 或 install 会写入）', true);
    }
  }

  const agentBrowserFile = join(HOME, '.agent-browser', 'config.json');
  let agentBrowserCdp = readConfiguredValue(agentBrowserFile, ['cdp']);
  if (agentBrowserCdp === null) {
    agentBrowserCdp = '未配置';
  }
  const agentBrowserDetail = `cdp=${agentBrowserCdp}（目标 ${config.cdpPort}）`;
  const agentBrowserOk = agentBrowserCdp === String(config.cdpPort);
  report(agentBrowserOk, 'agent-browser 绑定', agentBrowserDetail, true);

  const playwrightFile = join(HOME, '.playwright', 'cli.config.json');
  let playwrightEndpoint = readConfiguredValue(playwrightFile, ['browser', 'cdpEndpoint']);
  if (playwrightEndpoint === null) {
    playwrightEndpoint = '未配置';
  }
  const playwrightDetail = `cdpEndpoint=${playwrightEndpoint}`;
  report(playwrightEndpoint === config.cdpUrl, 'playwright-cli 绑定', playwrightDetail, true);

  const daemon = daemonStatus(config);
  if (daemon === null) {
    report(false, 'chrome-devtools daemon', 'chrome-devtools 未安装', true);
  } else if (!daemon.running) {
    const hint = '未运行（首次调用时按目标端口启动）';
    report(false, 'chrome-devtools daemon', hint, true);
  } else {
    const wanted = `--browser-url=${config.cdpUrl}`;
    const detail = `args=[${daemon.args}]`;
    report(daemon.args.includes(wanted), 'chrome-devtools daemon 连接目标', detail, true);
  }

  const orderOk = pathOrderOk(config);
  if (orderOk) {
    const detail = `${config.shimDir} 排在真身目录之前`;
    report(true, 'shim PATH 优先级', detail, true);
  } else {
    const hint = 'Linux/macOS 由 setup 前置，Windows 需手工调整 Path';
    const detail = `${config.shimDir} 未排在真身之前（${hint}）`;
    report(false, 'shim PATH 优先级', detail, true);
  }

  const info = await fetchCdpInfo(config);
  if (info === null) {
    const hint = '未运行（start 可拉起；调用三个 CLI 时 shim 会自动拉起）';
    report(false, '实例', hint, true);
  } else {
    const detail = `${config.cdpUrl}（${info.Browser}）`;
    report(true, '实例', detail, true);
  }

  if (config.remoteDesktop) {
    for (const commandName of ['Xvfb', 'x11vnc', 'websockify']) {
      const found = findInPath(commandName, config.shimDir);
      if (found === null) {
        const hint = '未安装：apt install xvfb x11vnc novnc websockify';
        report(false, commandName, hint, false);
      } else {
        report(true, commandName, '已安装', false);
      }
    }
  }

  let summary = '检查通过';
  if (reporter.counters.failed > 0) {
    summary = `${reporter.counters.failed} 项未通过`;
  }
  if (reporter.counters.warned > 0) {
    summary = `${summary}，${reporter.counters.warned} 项待 setup 处理`;
  }
  console.log('');
  console.log(summary);
  return reporter.counters.failed;
}

export async function status(config) {
  const info = await fetchCdpInfo(config);
  let pages = [];
  if (info !== null) {
    pages = await fetchCdpPages(config);
  }
  const chromePid = readPid(config, 'chrome');

  console.log(`平台        : ${process.platform}`);
  console.log(`运行时目录  : ${config.runtimeDir}`);
  console.log(`profile     : ${config.userDir}`);

  if (info === null) {
    console.log(`CDP         : ${config.cdpUrl}（无响应，执行 start 启动）`);
  } else {
    const count = pages.length;
    console.log(`CDP         : ${config.cdpUrl}（${info.Browser}，${count} 个页面）`);
  }

  if (pages.length > 0) {
    const titles = [];
    for (const page of pages) {
      if (page.title === undefined || page.title === '') {
        titles.push(page.url);
      } else {
        titles.push(page.title);
      }
    }
    console.log(`打开页面    : ${titles.join(' | ')}`);
  }

  let managed = '—';
  if (isProcessAlive(chromePid)) {
    managed = `本工具启动（pid ${chromePid}）`;
  } else if (info !== null) {
    managed = '外部启动（没有 pid 记录，stop 时经 CDP 关闭）';
  }
  console.log(`实例管理    : ${managed}`);

  const daemon = daemonStatus(config);
  if (daemon === null) {
    console.log('daemon      : chrome-devtools 未安装');
  } else if (daemon.running) {
    console.log(`daemon      : 运行中，args=[${daemon.args}]`);
  } else {
    console.log('daemon      : 未运行');
  }

  if (config.remoteDesktop) {
    const processes = [
      ['Xvfb', 'xvfb'],
      ['x11vnc', 'x11vnc'],
      ['websockify', 'websockify'],
    ];
    for (const entry of processes) {
      const pid = readPid(config, entry[1]);
      let state = '未运行';
      if (isProcessAlive(pid)) {
        state = `运行中（pid ${pid}）`;
      }
      console.log(`${entry[0].padEnd(11)} : ${state}`);
    }
    console.log(`屏幕        : ${config.screen.width}x${config.screen.height}（Xvfb 与 Chrome 窗口）`);
    const hint = '需 SSH 隧道，执行 view 看接入方式';
    console.log(`远程桌面    : ${config.webUrl}（${hint}）`);
  }

  console.log(`日志        : ${join(config.runtimeDir, 'logs')}`);
  console.log(`截图目录    : ${join(config.runtimeDir, 'screenshots')}`);
}

export function view(config) {
  let templateFile = 'view-local.txt';
  if (config.remoteDesktop) {
    templateFile = 'view-remote.txt';
  }

  let sshTarget = process.env.SSH_TARGET;
  if (sshTarget === undefined) {
    sshTarget = '<用户名>@<服务器地址>';
  }

  let auth = `未设 VNC_PASSWORD，无密码，靠 ${LOOPBACK} 绑定与 SSH 隧道保护`;
  if (process.env.VNC_PASSWORD !== undefined) {
    auth = '已启用 VNC 密码';
  }

  const values = {
    cdpUrl: config.cdpUrl,
    cdpPort: config.cdpPort,
    webPort: config.webPort,
    webUrl: config.webUrl,
    userDir: config.userDir,
    sshTarget: sshTarget,
    auth: auth,
    screen: `${config.screen.width}x${config.screen.height}`,
  };
  const template = readFileSync(join(HERE, templateFile), 'utf8');
  process.stdout.write(render(template, values));
}
