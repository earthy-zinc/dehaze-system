#!/usr/bin/env node
// 浏览器调试实例管理与三 CLI 接管工具，支持 macOS / Linux / Windows
// 命令清单与选项见 lib/usage.txt（--help 会原样输出）
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveConfig, writeState, writeToolConfigs } from './lib/config.mjs';
import { LOG_TAIL_LINES } from './lib/defaults.mjs';
import { check, status, view } from './lib/diagnose.mjs';
import { start, stop } from './lib/instance.mjs';
import { installShims, uninstallShims } from './lib/shim.mjs';
import { ensurePatchedNovnc } from './lib/novnc-setup.mjs';
import { syncDaemon } from './lib/tools.mjs';

const LIB_DIR = join(dirname(fileURLToPath(import.meta.url)), 'lib');
const USAGE_FILE = join(LIB_DIR, 'usage.txt');

function showUsage() {
  const content = readFileSync(USAGE_FILE, 'utf8');
  process.stdout.write(content);
}

function childEnvironment(extra) {
  const environment = {};
  for (const key of Object.keys(process.env)) {
    environment[key] = process.env[key];
  }
  if (extra !== undefined) {
    Object.assign(environment, extra);
  }
  return environment;
}

async function main() {
  const argv = process.argv.slice(2);
  let command = argv[0];
  if (command === undefined) {
    command = 'status';
  }

  if (command === 'help' || command === '--help' || command === '-h') {
    showUsage();
    return;
  }

  const config = resolveConfig(argv);

  if (command === 'check') {
    const failedCount = await check(config);
    if (failedCount > 0) {
      process.exit(1);
    }
    return;
  }

  if (command === 'setup') {
    const failedCount = await check(config);
    if (failedCount > 0) {
      console.log('');
      console.log('先修完未通过项再执行 setup');
      process.exit(1);
    }
    console.log('');
    writeState(config);
    await ensurePatchedNovnc(config);
    writeToolConfigs(config);
    installShims(config);
    const daemonRestarted = syncDaemon(config);
    if (daemonRestarted) {
      console.log(`chrome-devtools daemon 已按 ${config.cdpUrl} 重启`);
    }
    console.log('');
    await start(config);
    return;
  }

  if (command === 'install') {
    writeState(config);
    await ensurePatchedNovnc(config);
    writeToolConfigs(config);
    installShims(config);
    const daemonRestarted = syncDaemon(config);
    if (daemonRestarted) {
      console.log(`chrome-devtools daemon 已按 ${config.cdpUrl} 重启`);
    }
    return;
  }

  if (command === 'uninstall') {
    uninstallShims(config);
    return;
  }

  if (command === 'start') {
    await start(config);
    return;
  }

  if (command === 'stop') {
    await stop(config);
    return;
  }

  if (command === 'restart') {
    await stop(config);
    await start(config);
    return;
  }

  if (command === 'status') {
    await status(config);
    return;
  }

  if (command === 'view') {
    view(config);
    return;
  }

  if (command === 'logs') {
    let name = config.positional[0];
    if (name === undefined) {
      name = 'chrome';
    }
    const logFile = join(config.runtimeDir, 'logs', `${name}-${config.cdpPort}.log`);
    if (!existsSync(logFile)) {
      console.error(`日志不存在：${logFile}`);
      process.exit(1);
    }
    if (process.platform === 'win32') {
      const lines = readFileSync(logFile, 'utf8').split('\n');
      console.log(lines.slice(-LOG_TAIL_LINES).join('\n'));
      return;
    }
    const child = spawn('tail', ['-f', logFile], { stdio: 'inherit' });
    child.on('exit', (code) => {
      if (code === null) {
        process.exit(0);
      }
      process.exit(code);
    });
    return;
  }

  if (command === 'smoke') {
    const args = [join(LIB_DIR, 'smoke.mjs')];
    if (config.positional[0] !== undefined) {
      args.push(config.positional[0]);
    }
    if (config.positional[1] !== undefined) {
      args.push(config.positional[1]);
    }
    const extra = { CDP_PORT: String(config.cdpPort) };
    const result = spawnSync('node', args, { stdio: 'inherit', env: childEnvironment(extra) });
    if (result.status === null) {
      process.exit(1);
    }
    process.exit(result.status);
  }

  if (command === 'vnc-check') {
    const args = [join(LIB_DIR, 'vnc-check.mjs')];
    const extra = {
      WEB_PORT: String(config.webPort),
      SCREEN_WIDTH: String(config.screen.width),
      SCREEN_HEIGHT: String(config.screen.height),
    };
    const result = spawnSync('node', args, { stdio: 'inherit', env: childEnvironment(extra) });
    if (result.status === null) {
      process.exit(1);
    }
    process.exit(result.status);
  }

  console.error(`未知命令：${command}（--help 查看用法）`);
  process.exit(2);
}

main().catch((error) => {
  console.error('执行失败：', error.message);
  process.exit(1);
});
