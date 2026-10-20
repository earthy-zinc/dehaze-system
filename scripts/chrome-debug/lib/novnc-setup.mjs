// 自建「补丁版 noVNC」：修复中文剪贴板 + 主机↔远端自动同步。
//
// 为什么需要：x11vnc 不支持 ExtendedClipboard，只走传统 ClientCutText/ServerCutText；
// 而 noVNC 传统路径按 ISO 8859-1 处理（发送时非 Latin-1 变 '?'，接收时按 Latin-1 解码）→ 中文乱码；
// 且 noVNC 至今（1.7.0）不做剪贴板自动同步，必须手动往面板里粘。
// 实测 x11vnc 对 cut-text 是原样透传字节，因此把两端改成 UTF-8 即可，另补 UI 的自动同步。
//
// 由 install / setup 自动调用：下载固定版本的 noVNC 到 ~/.chrome-debug/novnc 并打补丁；
// 已装且版本一致则跳过；拿不到源码时只告警不中断（回落到发行版 noVNC，CLI 仍可用）。
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { NOVNC_PATCHED_DIR_NAME } from './defaults.mjs';

// 补丁按此版本的源码片段精确匹配；升级 noVNC 时需同步更新下面的 hunk
export const NOVNC_VERSION = '1.7.0';
// 安装产物的修订号：改动「装什么/怎么装」（补丁、index.html、裁剪清单）时递增，触发重装
const PATCH_REVISION = 'chrome-debug.9';
const SOURCE_URL = `https://github.com/novnc/noVNC/archive/refs/tags/v${NOVNC_VERSION}.tar.gz`;
const MARKER_FILE = '.chrome-debug-patched';

// 上游 release 不含 index.html（发行版包才带），直接访问 / 会看到目录列表，故补一个跳转页
const INDEX_HTML = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=vnc.html" />
    <title>noVNC</title>
  </head>
  <body>
    <a href="vnc.html">noVNC</a>
  </body>
</html>
`;

// 只保留运行所需；剔掉上游的开发/构建/CI 内容，免得目录列表里全是无关项
const DEV_PATHS = [
  '.github',
  '.gitignore',
  '.gitmodules',
  'tests',
  'snap',
  'po',
  'utils',
  'karma.conf.cjs',
  'eslint.config.mjs',
];

export function patchedNovncDir(config) {
  return join(config.runtimeDir, NOVNC_PATCHED_DIR_NAME);
}

// 四个补丁：两处编码（收发改 UTF-8）+ 两处自动同步
const HUNKS = [
  {
    file: 'core/rfb.js',
    name: '发送方向改 UTF-8',
    from: `        } else {
            let length, i;
            let data;

            length = 0;
            // eslint-disable-next-line no-unused-vars
            for (let codePoint of text) {
                length++;
            }

            data = new Uint8Array(length);

            i = 0;
            for (let codePoint of text) {
                let code = codePoint.codePointAt(0);

                /* Only ISO 8859-1 is supported */
                if (code > 0xff) {
                    code = 0x3f; // '?'
                }

                data[i++] = code;
            }

            RFB.messages.clientCutText(this._sock, data);
        }`,
    to: `        } else {
            // 补丁(chrome-debug)：不支持 ExtendedClipboard 的服务端（如 x11vnc）走这里。
            // 原实现只支持 ISO 8859-1，中文等会被替换成 '?'；实测 x11vnc 原样透传字节，故改发 UTF-8。
            const data = new TextEncoder().encode(text);
            RFB.messages.clientCutText(this._sock, data);
        }`,
  },
  {
    file: 'core/rfb.js',
    name: '接收方向按 UTF-8 解码',
    from: `        if (length >= 0) {
            //Standard msg
            const text = this._sock.rQshiftStr(length);`,
    to: `        if (length >= 0) {
            //Standard msg
            // 补丁(chrome-debug)：服务端发来的是 UTF-8 字节，原按 Latin-1 解码会乱码
            const text = decodeUTF8(this._sock.rQshiftStr(length), true);`,
  },
  {
    file: 'app/ui.js',
    name: '剪贴板自动同步（含 Cmd/Ctrl+V）+ 回车发送 + 空值保护',
    from: `    addClipboardHandlers() {
        document.getElementById("noVNC_clipboard_button")
            .addEventListener('click', UI.toggleClipboardPanel);
        document.getElementById("noVNC_clipboard_text")
            .addEventListener('change', UI.clipboardSend);
    },`,
    to: `    addClipboardHandlers() {
        // 补丁(chrome-debug)：空值保护。页面与 ui.js 版本错配时（典型：浏览器缓存了旧 ui.js、
        // 页面却是新版 vnc.html，元素已移除），上游会直接 null.addEventListener 把整个 UI 打崩。
        const bind = (id, event, handler) => {
            const el = document.getElementById(id);
            if (el) { el.addEventListener(event, handler); }
        };
        const sendClipboard = (text) => {
            if (!text) { return; }
            const el = document.getElementById('noVNC_clipboard_text');
            if (el) { el.value = text; }
            UI.clipboardSend();
        };
        // 补丁(chrome-debug)：主机→远端（右键菜单粘贴等原生 paste 会走到这里）
        document.addEventListener('paste', (e) => {
            const text = ((e.clipboardData || window.clipboardData).getData('text') || '');
            if (!text) { return; }
            sendClipboard(text);
        });
        // 补丁(chrome-debug)：Cmd/Ctrl+V。noVNC 的键盘挂在 canvas（非可编辑元素）上，
        // 浏览器不会产生原生 paste，按键还会被当成 Super+V 转发给远端（macOS 上等于没按）。
        // 这里截下来：读本地剪贴板 → 推给远端 → 再让远端执行一次 Ctrl+V。
        document.addEventListener('keydown', (e) => {
            if (!((e.ctrlKey || e.metaKey) && (e.key || '').toLowerCase() === 'v')) { return; }
            e.preventDefault();
            e.stopPropagation();
            if (!(navigator.clipboard && navigator.clipboard.readText)) { return; }
            navigator.clipboard.readText().then((text) => {
                sendClipboard(text);
                const rfb = UI.rfb;
                if (!rfb) { return; }
                rfb.sendKey(0xFFE3, 'ControlLeft', true);
                rfb.sendKey(0x76, 'KeyV', true);
                rfb.sendKey(0x76, 'KeyV', false);
                rfb.sendKey(0xFFE3, 'ControlLeft', false);
            }).catch(() => {});
        }, true);
        // 补丁(chrome-debug)：剪贴板面板里回车即发送（上游绑的是 change，只能靠失焦触发）
        const textarea = document.getElementById('noVNC_clipboard_text');
        if (textarea) {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    UI.clipboardSend();
                }
            });
        }
        bind("noVNC_clipboard_button", 'click', UI.toggleClipboardPanel);
        bind("noVNC_clipboard_text", 'change', UI.clipboardSend);
    },`,
  },
  {
    file: 'app/ui.js',
    name: '远端→主机自动同步',
    from: `    clipboardReceive(e) {
        Log.Debug(">> UI.clipboardReceive: " + e.detail.text.substr(0, 40) + "...");
        document.getElementById('noVNC_clipboard_text').value = e.detail.text;`,
    to: `    clipboardReceive(e) {
        Log.Debug(">> UI.clipboardReceive: " + e.detail.text.substr(0, 40) + "...");
        document.getElementById('noVNC_clipboard_text').value = e.detail.text;
        // 补丁(chrome-debug)：远端→主机自动写入本机剪贴板（需 https 或 localhost）
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(e.detail.text).catch(() => {});
        }`,
  },
  {
    file: 'core/input/keyboard.js',
    name: 'Mac 客户端把 Cmd 当 Ctrl（UA 判定 + ?cmdasctrl 覆盖）',
    from: `export default class Keyboard {`,
    to: `// 补丁(chrome-debug)：Mac 客户端把 Cmd 当 Ctrl。
// Linux 版 Chrome 只有 Ctrl 快捷键方案，而 noVNC 默认把 Cmd(Super) 映射成 Alt：
// 于是 Cmd+C/T/W/L 全部落空，Cmd+←/→ 还会被 Chrome 当成 Alt+←/→（后退/前进），与 Mac 习惯冲突。
// 按 UA 自动判定（Mac 开启），URL 参数 ?cmdasctrl=0/1 可显式覆盖（仅对 Mac/iOS 客户端有意义）。
const CMD_AS_CTRL = (() => {
    const override = new URLSearchParams(window.location.search).get('cmdasctrl');
    if (override === '0') { return false; }
    if (override === '1') { return true; }
    return browser.isMac();
})();

export default class Keyboard {`,
  },
  {
    file: 'core/input/keyboard.js',
    name: 'mac 分支里 Cmd(Super) → Ctrl',
    from: `        if (browser.isMac() || browser.isIOS()) {
            switch (keysym) {
                case KeyTable.XK_Super_L:
                    keysym = KeyTable.XK_Alt_L;
                    break;
                case KeyTable.XK_Super_R:
                    keysym = KeyTable.XK_Super_L;
                    break;`,
    to: `        if (browser.isMac() || browser.isIOS()) {
            switch (keysym) {
                // 补丁(chrome-debug)：Cmd(Super) → Ctrl（CMD_AS_CTRL 默认在 Mac 上开启）
                case KeyTable.XK_Super_L:
                    keysym = CMD_AS_CTRL ? KeyTable.XK_Control_L : KeyTable.XK_Alt_L;
                    break;
                case KeyTable.XK_Super_R:
                    keysym = CMD_AS_CTRL ? KeyTable.XK_Control_L : KeyTable.XK_Super_L;
                    break;`,
  },
];

function applyPatches(directory) {
  let applied = 0;
  for (const hunk of HUNKS) {
    const path = join(directory, hunk.file);
    const source = readFileSync(path, 'utf8');
    if (source.includes(hunk.to)) {
      continue;
    }
    if (!source.includes(hunk.from)) {
      throw new Error(`${hunk.file} 未匹配到「${hunk.name}」片段，可能 noVNC 版本与预期(${NOVNC_VERSION})不符`);
    }
    writeFileSync(path, source.replace(hunk.from, hunk.to));
    applied++;
  }
  return applied;
}

function isInstalled(directory) {
  const marker = join(directory, MARKER_FILE);
  if (!existsSync(marker)) {
    return false;
  }
  return readFileSync(marker, 'utf8').trim() === `${NOVNC_VERSION}+${PATCH_REVISION}`;
}

/**
 * 确保已安装补丁版 noVNC（幂等）
 * @returns {Promise<boolean>} 是否可用（false 表示跳过，需回落发行版 noVNC）
 */
export async function ensurePatchedNovnc(config) {
  const target = patchedNovncDir(config);
  if (isInstalled(target)) {
    console.log(`补丁版 noVNC 已就绪：${target}（v${NOVNC_VERSION}）`);
    return true;
  }

  const staging = `${target}.staging`;
  try {
    rmSync(staging, { recursive: true, force: true });
    mkdirSync(staging, { recursive: true });

    console.log(`下载 noVNC v${NOVNC_VERSION}…（${SOURCE_URL}）`);
    const response = await fetch(SOURCE_URL, { redirect: 'follow' });
    if (!response.ok) {
      throw new Error(`下载失败：HTTP ${response.status}`);
    }
    const tarball = join(staging, 'novnc.tar.gz');
    writeFileSync(tarball, Buffer.from(await response.arrayBuffer()));

    execFileSync('tar', ['xzf', tarball, '-C', staging], { stdio: 'inherit' });
    const extracted = join(staging, `noVNC-${NOVNC_VERSION}`);
    if (!existsSync(extracted)) {
      throw new Error('解压后未找到 noVNC 目录，压缩包结构可能已变');
    }

    const applied = applyPatches(extracted);
    // 裁剪开发/CI 内容 + 补 index.html（否则 / 会落到目录列表）
    for (const rel of DEV_PATHS) {
      rmSync(join(extracted, rel), { recursive: true, force: true });
    }
    writeFileSync(join(extracted, 'index.html'), INDEX_HTML);
    writeFileSync(join(extracted, MARKER_FILE), `${NOVNC_VERSION}+${PATCH_REVISION}\n`);

    // 就地替换，保留 target 目录本身的 inode（不整目录删掉重建）。
    // websockify 启动时会 chdir 进 web 根，若把整个目录换掉，运行中的 websockify 的 cwd 会指向
    // 已删除目录——表现为「端口在监听但所有 HTTP 请求失败」。保留 inode 即可避免；staging 与 target
    // 同父目录（同文件系统），逐项 rename 是原子的。
    if (existsSync(target)) {
      for (const entry of readdirSync(target)) {
        rmSync(join(target, entry), { recursive: true, force: true });
      }
    } else {
      mkdirSync(target, { recursive: true });
    }
    for (const entry of readdirSync(extracted)) {
      renameSync(join(extracted, entry), join(target, entry));
    }
    rmSync(staging, { recursive: true, force: true });
    console.log(`补丁版 noVNC 安装完成：${target}（应用 ${applied} 处补丁）`);
    return true;
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    console.log(`⚠️ 补丁版 noVNC 安装失败：${error.message}`);
    console.log('   noVNC 仍可用，但剪贴板将退回「面板式手动同步」（无自动同步、中文可能乱码）');
    return false;
  }
}
