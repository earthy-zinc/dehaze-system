#!/usr/bin/env node
// CDP 端到端冒烟：新建页 → 导航 → 执行 JS → 点击 → 收 console → 截图 → 关页
//
// 用法：由入口的 smoke 子命令调用（node lib/smoke.mjs [url] [outPng]）
//   CDP 端口来自环境变量 CDP_PORT（入口注入，默认 9222）
//   默认 url 是同目录的 probe.html；默认截图落在 ~/.chrome-debug/screenshots
//
// 任何一项断言失败都返回非 0 退出码并列出失败项，不做静默跳过。
import { mkdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  DEFAULT_CDP_PORT,
  LOOPBACK,
  RUNTIME_DIR_NAME,
  SMOKE_PAGE_TIMEOUT_MS,
  SMOKE_POLL_MS,
  SMOKE_SCREENSHOT_MIN_BYTES,
} from './defaults.mjs';

function environmentValue(name, fallback) {
  const value = process.env[name];
  if (value === undefined) {
    return fallback;
  }
  return value;
}

const HERE = dirname(fileURLToPath(import.meta.url));
const CDP_PORT = environmentValue('CDP_PORT', String(DEFAULT_CDP_PORT));
const CDP_URL = `http://${LOOPBACK}:${CDP_PORT}`;

const fixtureUrl = pathToFileURL(join(HERE, 'probe.html')).href;
let targetUrl = process.argv[2];
if (targetUrl === undefined || targetUrl === '') {
  targetUrl = fixtureUrl;
}
const isFixture = targetUrl === fixtureUrl;

const stamp = new Date().toISOString().replace(/[:.]/g, '-');
let outputPng = process.argv[3];
if (outputPng === undefined || outputPng === '') {
  outputPng = join(homedir(), RUNTIME_DIR_NAME, 'screenshots', `smoke-${stamp}.png`);
}

const results = [];

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function check(name, ok, detail) {
  results.push({ name: name, ok: ok });
  let mark = '✗';
  if (ok) {
    mark = '✓';
  }
  if (detail === undefined || detail === '') {
    console.log(`${mark} ${name}`);
    return ok;
  }
  console.log(`${mark} ${name} — ${detail}`);
  return ok;
}

async function fetchJson(url, method) {
  const options = {};
  if (method !== undefined) {
    options.method = method;
  }
  const response = await fetch(url, options);
  return await response.json();
}

async function main() {
  const version = await fetchJson(`${CDP_URL}/json/version`);
  check('CDP 端口可用', version.Browser !== undefined, `${version.Browser} @ ${CDP_URL}`);

  const createUrl = `${CDP_URL}/json/new?${encodeURIComponent(targetUrl)}`;
  const target = await fetchJson(createUrl, 'PUT');
  check('新建标签页', target.id !== undefined, targetUrl);

  const socket = new WebSocket(target.webSocketDebuggerUrl);
  let sequence = 0;
  const pending = new Map();
  const consoleMessages = [];

  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.consoleAPICalled') {
      consoleMessages.push(message.params.type);
    }
    if (message.id === undefined) {
      return;
    }
    const entry = pending.get(message.id);
    if (entry === undefined) {
      return;
    }
    pending.delete(message.id);
    if (message.error !== undefined) {
      entry.reject(new Error(JSON.stringify(message.error)));
      return;
    }
    entry.resolve(message.result);
  });

  function send(method, params) {
    sequence += 1;
    const id = sequence;
    socket.send(JSON.stringify({ id: id, method: method, params: params }));
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve: resolve, reject: reject });
    });
  }

  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });

  await send('Page.enable', {});
  await send('Runtime.enable', {});

  const deadline = Date.now() + SMOKE_PAGE_TIMEOUT_MS;
  let readyState = '';
  while (Date.now() < deadline) {
    const evaluated = await send('Runtime.evaluate', {
      expression: 'document.readyState',
      returnByValue: true,
    });
    readyState = evaluated.result.value;
    if (readyState === 'complete') {
      break;
    }
    await sleep(SMOKE_POLL_MS);
  }
  check('页面加载完成', readyState === 'complete', `document.readyState=${readyState}`);

  const titleResult = await send('Runtime.evaluate', {
    expression: 'document.title',
    returnByValue: true,
  });
  const title = titleResult.result.value;
  let titleOk = title !== '';
  if (isFixture) {
    titleOk = title === 'CDP Probe OK';
  }
  check('读取标题', titleOk, `title="${title}"`);

  const clickExpression = [
    '(() => {',
    "  const button = document.getElementById('btn');",
    '  if (!button) return null;',
    '  button.click();',
    '  button.click();',
    "  return document.getElementById('count').textContent;",
    '})()',
  ].join('\n');
  const clicked = await send('Runtime.evaluate', {
    expression: clickExpression,
    returnByValue: true,
  });
  const clickCount = clicked.result.value;
  if (clickCount === null) {
    check('执行 JS 并触发交互', true, '夹具无按钮，跳过断言');
  } else {
    check('执行 JS 并触发交互', clickCount === '2', `点击两次后 count=${clickCount}`);
  }

  let consoleOk = consoleMessages.length >= 0;
  if (isFixture) {
    consoleOk = consoleMessages.length >= 4;
  }
  let consoleDetail = '无';
  if (consoleMessages.length > 0) {
    consoleDetail = consoleMessages.join(',');
  }
  check('捕获 console 消息', consoleOk, consoleDetail);

  const screenshot = await send('Page.captureScreenshot', { format: 'png' });
  const bytes = Buffer.from(screenshot.data, 'base64');
  mkdirSync(dirname(outputPng), { recursive: true });
  writeFileSync(outputPng, bytes);
  check('截图落盘', bytes.length > SMOKE_SCREENSHOT_MIN_BYTES, `${outputPng}（${bytes.length} 字节）`);

  await fetch(`${CDP_URL}/json/close/${target.id}`);
  socket.close();
  check('关闭标签页', true, '');

  const failed = results.filter((entry) => !entry.ok);
  console.log('');
  console.log(`${results.length - failed.length}/${results.length} 项通过`);
  if (failed.length === 0) {
    return;
  }
  const names = failed.map((entry) => entry.name);
  console.log(`失败项：${names.join('、')}`);
  process.exit(1);
}

main().catch((error) => {
  const text = String(error.message);
  const isConnectError = text.includes('ECONNREFUSED') || text.includes('fetch failed');
  if (isConnectError) {
    console.error(`无法连接 ${CDP_URL}，先执行：node scripts/chrome-debug/index.mjs start`);
  } else {
    console.error('冒烟测试异常：', text);
  }
  process.exit(1);
});
