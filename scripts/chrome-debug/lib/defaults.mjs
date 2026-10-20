// 集中配置：默认值、超时与固定启动参数都在这里，其它模块只引用不重写。
// 需要按环境变化的值一律走命令行选项或环境变量；改这里的常量等于改所有平台的默认行为。
//
// 注意：shim-*.tmpl 生成的是自包含脚本，其中固定的 Chrome 参数（--no-first-run 等）
// 与 CHROME_BASE_ARGS 是同一份事实，改动时需同步；等待次数由 install 渲染时注入。

// ── 运行约束 ──
// smoke / vnc-check 依赖全局 WebSocket，Node 22 起才默认可用
export const MIN_NODE_MAJOR = 22;

// ── 端口 / 显示 / 屏幕 ──
export const DEFAULT_CDP_PORT = 9222;
export const DEFAULT_VNC_PORT = 5900;
export const DEFAULT_WEB_PORT = 6080;
export const DEFAULT_DISPLAY = ':99';
export const DEFAULT_SCREEN = { width: 1920, height: 1080, depth: 24 };
export const DEFAULT_SCREEN_TEXT = `${DEFAULT_SCREEN.width}x${DEFAULT_SCREEN.height}`;

// 只监听回环：不暴露给局域网是本工具的默认安全边界，不做成选项
export const LOOPBACK = '127.0.0.1';

// ── 等待与超时（毫秒）──
export const PROBE_TIMEOUT_MS = 1500; // 单次 CDP 探活
export const VERSION_TIMEOUT_MS = 5000; // 版本探测上限：Windows 上 chrome.exe --version 不会自行退出
export const STARTUP_TIMEOUT_MS = 20000; // 拉起后等待 CDP 就绪
export const STARTUP_POLL_MS = 500;
export const STOP_TIMEOUT_MS = 10000; // SIGTERM 后等待退出，仍未退则 SIGKILL
export const STOP_POLL_MS = 1000;
export const SMOKE_PAGE_TIMEOUT_MS = 10000; // 冒烟：等待页面 readyState=complete
export const SMOKE_POLL_MS = 200;
export const SMOKE_SCREENSHOT_MIN_BYTES = 1000; // 小于此值视为截图失败
export const VNC_READ_TIMEOUT_MS = 8000; // vnc-check 单次读字节
export const IME_READY_TIMEOUT_MS = 6000; // 拉起后等待 ibus 可用并切到拼音引擎
export const IME_READY_POLL_MS = 300;

// ── 运行时目录与外部资源 ──
export const RUNTIME_DIR_NAME = '.chrome-debug';
export const NOVNC_DEFAULT_DIR = '/usr/share/novnc';
// 自建的补丁版 noVNC（修复中文剪贴板 + 主机↔远端自动同步）：装在运行时目录下，存在则优先于发行版目录
export const NOVNC_PATCHED_DIR_NAME = 'novnc';
export const LOG_TAIL_LINES = 50; // Windows 上 logs 命令打印的尾部行数

// ── 智能网关签名代理（仅 DevCloud 等无智能网关的机器需要）──
// 站点相关的域 / 凭证 / 上游协议都在 lib/channels/ 下，本文件只放与站点无关的常量。
// 凭证来源：环境变量优先，其次 ~/.chrome-debug/.env，绝不写进仓库
export const SIGN_KEY_ENV = 'SMART_PROXY_KEY'; // panshi 站点 token
export const SIGN_STAFF_ENV = 'STAFF_NAME'; // 员工英文名（所有站点共用）
export const SIGN_SA_TOKEN_ENV = 'SA_TOKEN'; // imeta 站点 token；有它才启用 imeta 通道
export const SIGN_IOA_TOKEN_ENV = 'IOA_TOKEN'; // forcem 站点 token；有它才启用 forcem 通道
export const SIGN_ENV_FILE = '.env';
export const SIGN_PAC_PATH = '/gateway-sign.pac'; // PAC 由签名代理自身用 http 提供
export const SIGN_PORT_ENV = 'PANSHI_SIGN_PORT';
// DevCloud 不可达、且会**拖住页面加载**（命中即快速失败，避免标签页一直转圈）。
// 都是页面的装饰性外链资源；要真正加载需为其申请 iGate 放通，然后从这里移除。
export const FAIL_FAST_HOSTS = ['rhrc.woa.com'];
export const DEFAULT_SIGN_PORT = 18070;
export const SIGN_PROXY_ERROR_STATUS = 502; // 代理转发失败时返回给浏览器的状态码

// ── Chrome 启动参数 ──
const CHROME_BASE_ARGS = ['--no-first-run', '--no-default-browser-check'];
// 工作站有桌面：全屏（DevTools 由 devtoolsArgs 统一处理）
const WORKSTATION_CHROME_ARGS = ['--start-fullscreen'];

// 自动开 DevTools：便于随时看页面/网络（不自动开的话，事后看不到已经发生的请求）。
// ⚠️ DevTools 默认停靠右侧会占用窗口、缩小页面可视区；要保留宽度可在 DevTools 里拖到底部
// （停靠位置会存进 profile）。不需要可加 --no-devtools。
function devtoolsArgs(config) {
  return config.devtools === false ? [] : ['--auto-open-devtools-for-tabs'];
}

// 平台差异只在这里：服务器无窗口管理器，靠 Xvfb 撑屏并放宽沙箱
export function chromeExtraArgs(config) {
  if (config.remoteDesktop) {
    const size = `${config.screen.width},${config.screen.height}`;
    return ['--no-sandbox', '--disable-dev-shm-usage', `--window-size=${size}`, ...devtoolsArgs(config)];
  }
  return [...WORKSTATION_CHROME_ARGS, ...devtoolsArgs(config)];
}

// 仅当启用签名代理时，让 Chrome 经 PAC 只把各站点的 http 流量交给本地代理（其余直连）。
// PAC 由代理自身提供：实测 Chrome 不采纳 file:// 形式的 PAC，http 形式才生效。
// 同时关掉 HTTPS 升级：这些通道只能走 http（DevCloud 未放通 443），否则 http 会被强升到 https，
// 撞上 SmartGate 的 TLS 拒绝（ERR_SSL_PROTOCOL_ERROR）且不会回退到 http。
export function signProxyArgs(config) {
  if (config.signProxy !== true) {
    return [];
  }
  return [
    `--proxy-pac-url=${config.signPacUrl}`,
    '--disable-features=HttpsUpgrades,HttpsFirstMode,HttpsFirstBalancedMode,HttpsFirstModeV2',
  ];
}

// 不指定启动 URL：指定了会短路 Chrome 自己的会话恢复行为
export function chromeArgs(config) {
  return [
    `--remote-debugging-port=${config.cdpPort}`,
    `--remote-debugging-address=${LOOPBACK}`,
    `--user-data-dir=${config.userDir}`,
    ...CHROME_BASE_ARGS,
    ...chromeExtraArgs(config),
    ...signProxyArgs(config),
  ];
}

// ── 服务器远程桌面参数 ──
const XVFB_FIXED_ARGS = ['-nolisten', 'tcp'];
// DAMAGE 变化检测（不加 -noxdamage）：x11vnc 只编码真正变化的区域，替代整屏轮询比对。
// 整屏轮询既白烧 CPU 又让更新滞后（VNC 画面卡顿的根因）；Xvfb 不支持时 x11vnc 自动回退轮询。
// 注意：本机源的 x11vnc 构建不支持 -quality / -compresslevel（Tight 压缩被裁剪），不要加。
// -noprimary/-nosetprimary：切断 X11 PRIMARY（选中缓冲）双向同步，只保留 CLIPBOARD（显式 Ctrl+C/V）。
// X11 有两套独立剪贴板：PRIMARY 鼠标选中即变，CLIPBOARD 才是显式复制。x11vnc 默认两套都同步给
// VNC 客户端 → VNC 内一选中文字，选中内容就会被推到客户端、再由补丁自动写进本机剪贴板，
// 把刚复制的内容悄悄顶掉；右键粘贴取的是远端 CLIPBOARD，与本机剪贴板对不上即"贴的不是刚复制的"。
// 远端 Chrome 不使用中键粘贴，切断 PRIMARY 无副作用。
export const X11VNC_FIXED_ARGS = ['-shared', '-forever', '-quiet', '-noprimary', '-nosetprimary'];

export function xvfbArgs(config) {
  const screen = `${config.screen.width}x${config.screen.height}x${config.screen.depth}`;
  return [config.display, '-screen', '0', screen, ...XVFB_FIXED_ARGS];
}

// ── 服务器中文输入法（ibus + libpinyin）──
// Xvfb 是纯 X server，不带任何输入法框架；没有 IME 时 noVNC 只能敲英文，中文只能靠剪贴板粘贴。
// 服务器模式下随实例拉起 dbus 会话总线 + ibus-daemon（见 lib/instance.mjs），并给 Chrome 注入
// 下面这些变量，即可在 noVNC 里直接拼音输入、数字键选词。
// ⚠️ 客户端浏览器要保持在英文输入法：中文由服务端 IME 组词，客户端若开着拼音会先被本机吞掉按键。
export const IBUS_ENGINE = 'libpinyin';
export const IME_CHROME_ENVIRONMENT = {
  GTK_IM_MODULE: 'ibus',
  XMODIFIERS: '@im=ibus',
  QT_IM_MODULE: 'ibus',
  CLUTTER_IM_MODULE: 'ibus',
  INPUT_METHOD: 'ibus',
};
