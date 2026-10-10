#!/usr/bin/env node
// VNC 链路自检：经 websockify 与 x11vnc 走一遍 RFB 握手，确认远程桌面可用
//
// 用法：由入口的 vnc-check 子命令调用（node lib/vnc-check.mjs）
//   web 端口与屏幕尺寸来自环境变量 WEB_PORT / SCREEN_WIDTH / SCREEN_HEIGHT（入口注入）
//
// 校验项：WebSocket 通道、RFB 协议版本、认证方式、屏幕尺寸、桌面名、画面推送
// 任何一项不符合预期即返回非 0 退出码并列出失败项。
import { DEFAULT_SCREEN, DEFAULT_WEB_PORT, LOOPBACK, VNC_READ_TIMEOUT_MS } from './defaults.mjs';

function environmentValue(name, fallback) {
  const value = process.env[name];
  if (value === undefined) {
    return fallback;
  }
  return value;
}

const WEB_PORT = environmentValue('WEB_PORT', String(DEFAULT_WEB_PORT));
const WS_URL = `ws://${LOOPBACK}:${WEB_PORT}/websockify`;
// 期望尺寸与实例的 Xvfb 启动参数同源（--screen）
const SCREEN_WIDTH = Number(environmentValue('SCREEN_WIDTH', String(DEFAULT_SCREEN.width)));
const SCREEN_HEIGHT = Number(environmentValue('SCREEN_HEIGHT', String(DEFAULT_SCREEN.height)));

const results = [];
let buffered = Buffer.alloc(0);
let waiter = null;

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

const socket = new WebSocket(WS_URL);
socket.binaryType = 'arraybuffer';

socket.addEventListener('message', (event) => {
  const chunk = Buffer.from(event.data);
  buffered = Buffer.concat([buffered, chunk]);
  if (waiter === null) {
    return;
  }
  if (buffered.length < waiter.byteCount) {
    return;
  }
  const resolve = waiter.resolve;
  const byteCount = waiter.byteCount;
  waiter = null;
  const result = buffered.subarray(0, byteCount);
  buffered = buffered.subarray(byteCount);
  resolve(result);
});

function readBytes(byteCount, timeoutMs) {
  if (buffered.length >= byteCount) {
    const result = buffered.subarray(0, byteCount);
    buffered = buffered.subarray(byteCount);
    return Promise.resolve(result);
  }
  return new Promise((resolve, reject) => {
    let timeout = timeoutMs;
    if (timeout === undefined) {
      timeout = VNC_READ_TIMEOUT_MS;
    }
    const timer = setTimeout(() => {
      const message = `等待 ${byteCount} 字节超时（已收 ${buffered.length} 字节）`;
      reject(new Error(message));
    }, timeout);
    waiter = {
      byteCount: byteCount,
      resolve: (value) => {
        clearTimeout(timer);
        resolve(value);
      },
    };
  });
}

function connect() {
  return new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', () => {
      reject(new Error(`连接 ${WS_URL} 失败（websockify 未启动？）`));
    }, { once: true });
  });
}

async function main() {
  await connect();
  check('websockify WebSocket 通道', true, WS_URL);

  const banner = (await readBytes(12)).toString('latin1');
  check('RFB 协议握手', /^RFB \d{3}\.\d{3}\n$/.test(banner), banner.trim());

  socket.send(Buffer.from('RFB 003.008\n', 'latin1'));

  const typeCount = (await readBytes(1))[0];
  const securityTypes = Array.from(await readBytes(typeCount));
  const typesDetail = `security types = [${securityTypes.join(',')}]`;
  check('读到认证方式列表', securityTypes.length > 0, typesDetail);

  if (!securityTypes.includes(1)) {
    console.log('- 该实例要求密码认证，本自检未实现密码流程，只确认端口与协议可达');
    socket.close();
    console.log('');
    console.log('链路只验证到认证阶段；如需完整校验，请以 -nopw 重启后再跑');
    process.exit(0);
  }

  socket.send(Buffer.from([1]));
  const authResult = (await readBytes(4)).readUInt32BE(0);
  check('无密码认证通过', authResult === 0, `SecurityResult=${authResult}`);

  // ClientInit：shared
  socket.send(Buffer.from([1]));

  const header = await readBytes(24);
  const width = header.readUInt16BE(0);
  const height = header.readUInt16BE(2);
  const nameLength = header.readUInt32BE(20);
  let desktopName = '';
  if (nameLength > 0) {
    desktopName = (await readBytes(nameLength)).toString('utf8');
  }
  const sizeOk = width === SCREEN_WIDTH && height === SCREEN_HEIGHT;
  const sizeDetail = `${width}x${height}（期望 ${SCREEN_WIDTH}x${SCREEN_HEIGHT}）`;
  check('屏幕尺寸与 Xvfb 一致', sizeOk, sizeDetail);
  check('读到桌面名', desktopName !== '', desktopName);

  // FramebufferUpdateRequest：incremental=0，全屏
  const request = Buffer.alloc(10);
  request[0] = 3;
  request[1] = 0;
  request.writeUInt16BE(0, 2);
  request.writeUInt16BE(0, 4);
  request.writeUInt16BE(width, 6);
  request.writeUInt16BE(height, 8);
  socket.send(request);

  const messageType = (await readBytes(1))[0];
  const rectCount = (await readBytes(3)).readUInt16BE(1);
  const pushDetail = `消息类型=${messageType}，矩形数=${rectCount}`;
  check('服务器正在推送画面', messageType === 0 && rectCount > 0, pushDetail);

  if (rectCount > 0) {
    const rect = await readBytes(12);
    const position = `区域 ${rect.readUInt16BE(0)},${rect.readUInt16BE(2)}`;
    const size = `${rect.readUInt16BE(4)}x${rect.readUInt16BE(6)}`;
    const encoding = rect.readInt32BE(8);
    check('首个矩形信息', true, `${position} ${size} encoding=${encoding}`);
  }

  socket.close();
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
  console.error('VNC 自检异常：', error.message);
  process.exit(1);
});
