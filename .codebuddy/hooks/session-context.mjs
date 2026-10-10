#!/usr/bin/env node
// 环境认知 hook：注入「本机环境认知」—— agent 据此知道自己处在哪台机器、**是否在内网**、
// **跨网能到什么**、能不能用浏览器工具。同一份脚本供两端注册：CodeBuddy（SessionStart）
// 与 Hermes（pre_llm_call，`--hermes`）。
//
// 与系统注入的分工：系统已给 OS / cwd 等静态事实，这里只补它给不了的动态增量。
//
// 安全纪律（改动前务必先读）：**不做任何跨网探测，也不批量探端口**。
// 会话启动时自动向固定 IP 反复发起请求，在安全侧看就是周期性扫描特征，容易触发安全告警——
// 哪怕是自己的服务器。跨网可达性与服务状态一律留给 agent 按需实测，不在会话启动时自动探测。
// 本脚本只读本机配置 + 访问 127.0.0.1 的环回端口：环回不走网卡，不产生任何网络流量。
//
// 「是否在内网」的判据（按可靠性排序，任一命中即算已接入；都是本机可观测信号）：
// ① 网卡 IP 命中 DevCloud / OA 区域网段 ⇒ 直连内网；
// ② 隧道接入：macOS 解析 `ifconfig` 里带 IPv4 的 utun/ipsec/ppp 接口（iOA 远程办公驱动模式）；
//    Windows 解析 `ipconfig /all`，Description 含 NGN 或 IP 落 192.168.255.0/24 ⇒ NGN 通道。
//    注意 Node 的 os.networkInterfaces() 在 macOS 上看不到 utun、IPv4 路由表也不体现，
//    所以只能解析命令输出——实测：Mac en0 192.168.31.114 + utun4 192.168.255.10；
//    Windows 云桌面 NGNClient Adapter 192.168.255.10；
// ③ 本地代理 127.0.0.1:12639 在监听 ⇒ 存在经代理访问内网的通道；
// ④ 平台身份优先于网络区域：云桌面**不能靠网段识别**（见 NETWORK_ZONES 下方注释），
//    先判平台再判区域。
//
// 区域策略内联在 REACH_STRATEGY 里（agent 不必再去查文档），来源为
// 《DevCloud 与其他区域的网络通路指引》与
// 《办公网 PC 间端口禁令与各网络到 Devnet/Mnet/Git-SVN 策略》。
//
// 部署（用户级注册，指向本文件）：各机 CodeBuddy 的 `~/.codebuddy/settings.json`（SessionStart）
// 与 Hermes 的 `$HERMES_HOME/config.yaml`（pre_llm_call）各注册一次——Mac / dev / pre 直接写
// 仓库检出路径（改动 pull 后即生效），无仓库的机器（earthy-pm）用 scp 的副本
// `~/.codebuddy/hooks/session-context.mjs`。
// 注入方式：CodeBuddy 用 stdout（SessionStart 下 stdout 与 hookSpecificOutput.additionalContext
// 等价，见扩展 `executeSessionStartHooks`）；Hermes 用 `--hermes`——它的注入点只有
// pre_llm_call 且每轮都触发，故脚本自行读载荷、只在会话首轮输出 `{"context": "..."}`。
// 静默纪律：任何异常都不阻塞会话 —— 输出降级文本后正常退出，绝不抛错、绝不非零退出。

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const HOOK_DIR = path.dirname(fileURLToPath(import.meta.url));
const HERMES_MODE = process.argv.includes('--hermes');
const PROBE_TIMEOUT_MS = 1200;
// 只认"带 IPv4 的隧道接口"：utun0-3 这类只有 IPv6 link-local 的不算已建立的 IPv4 隧道
const TUNNEL_INTERFACE_PATTERN = /^(utun|ipsec|ppp)\d+$/;
// CDP 版本号只放行「产品名/版本号」形态，杜绝任意文本经 hook stdout 注入 agent 上下文
const CDP_VERSION_PATTERN = /^[A-Za-z][\w-]*\/[\d.]+$/;

// 区域网段摘自《DevCloud 与其他区域的网络通路指引》（iwiki 36328118）与
// 《IP安全区域归属查询方法》（iwiki 940629719，含"内网 12 大段"）；DevCloud(IDC) 段很多，此处只列主要段
const NETWORK_ZONES = [
  {
    name: 'DevCloud',
    cidrs: [
      '9.134.0.0/15',
      '21.6.0.0/16',
      '21.91.0.0/16',
      '21.214.0.0/16',
      '21.139.128.0/17',
      '21.130.128.0/17',
    ],
  },
  {
    name: 'DevCloud(IDC)',
    cidrs: [
      '11.141.0.0/16',
      '21.0.0.0/16',
      '11.151.192.0/18',
      '11.145.192.0/20',
      '11.153.16.0/20',
      '21.90.0.0/16',
      '21.196.0.0/16',
      '21.215.0.0/16',
      '21.130.0.0/17',
    ],
  },
  {
    // OA 区判断网段，取自《如何查询服务器所在网络区域》——
    // 比《通路指引》里那组更全（含 30.16.0.0/12）
    name: 'OA',
    cidrs: [
      '10.0.0.0/13',
      '10.8.0.0/14',
      '10.13.0.0/16',
      '10.14.0.0/15',
      '10.16.0.0/12',
      '10.32.0.0/12',
      '10.64.0.0/12',
      '10.80.0.0/14',
      '10.84.0.0/16',
      '10.86.0.0/15',
      '10.88.0.0/13',
      '10.96.0.0/14',
      '30.16.0.0/12',
    ],
  },
  {
    // 私有网：标准内网段（《IP安全区域归属查询方法》的"内网 12 大段"）之外的私有/VPC 段，
    // 如 AI agent 服务器 earthy-pm 的 25.2.2.157（用户口径，2026-10-06）
    name: '私有网',
    cidrs: ['25.0.0.0/8'],
  },
];

// 云桌面（IMate 数字同事）**不能靠网段识别**：实测其 IP 21.214.165.118 落在 DevCloud 段
// （sniper 查询：网段 21.214.0.0/16、安全区域 DEVNET），同一段内既有普通 DevCloud 机器也有云桌面。
// ⇒ 只能靠平台特征判，见 isCloudDesktop()。
const PRIVATE_CIDRS = ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'];

// iOA NGN 虚拟网卡的固定网段（Mac 上体现为 utun4 192.168.255.10，Windows 上为 NGNClient Adapter），
// 地址固定，可用于 Windows 侧识别 NGN 通道（Description 解析失败时的兜底）。
const NGN_TUNNEL_CIDR = '192.168.255.0/24';

// 9222、19222 chrome 浏览器调试实例
const CDP_PORTS = [9222, 19222];

// iOA + NGN 系统扩展（「驱动模式」）开的本地内网出口，地址固定不变，故直接硬编码。
// npm 侧体现为 ~/.npmrc 的 `proxy=`，浏览器侧体现为 whistle 规则 `enable://proxyHost`。
const LOCAL_PROXY = { label: '127.0.0.1:12639', port: 12639 };

// 各接入身份对应的可达性要点：内联在此，agent 不必再去别处查策略。
// 端口一律带用途标注——只写数字对外部读者等于没写（如 139/445 是文件共享、3389 是远程桌面）。
// 办公网与外网远程办公的禁用清单不同（前者含 135/139/445），故分列。
// 云桌面是独立区域（独立网段 + 入站封禁），**不与办公网合并**；
// 私有网（标准内网段之外的 VPC 段）同样是独立区域，办公网与 DevCloud 都不能直连它。
// 每项按语义分段、join('') 后即为注入文案，分段只为控制行宽。
const REACH_STRATEGY = {
  DevCloud: [
    '本机在 DevCloud ⇒ 与 Devnet 互通；可访问 IDC 测试专区，',
    '但文件共享 139/445、远程桌面 3389、MNET 登录 36000/56000 等高危端口不通；',
    '访问 IDC 普通区需申请',
  ].join(''),
  'DevCloud(IDC)': [
    '本机在 DevCloud(IDC) ⇒ **禁止访问 DevCloud**；',
    '可访问 IDC 普通区，高危端口（同 DevCloud：139/445 文件共享、',
    '3389 远程桌面、36000/56000 MNET 登录）不通',
  ].join(''),
  OA: [
    '本机在办公网/OA ⇒ 至 DevCloud：放通 ≥1024 且非高危端口，',
    '禁 6379(Redis)/9000(MinIO)/2375(docker)/10250(k8s)/',
    '135·139·445(文件共享)/1099/5601/5858/6066/7077/8500/9001/9292；',
    'PC 之间另禁 1-1023 及 3306(MySQL)/5432(PostgreSQL)/1433(SQL Server)/',
    '27017(MongoDB)/6379(Redis)/11211(Memcached)/8888·8899(代理工具)；',
    '访问 IDC 需申请',
  ].join(''),
  私有网: [
    '本机在私有网段（标准内网段之外的 VPC 段）⇒ 办公网与 DevCloud 均**不能直连本机**',
    '（只能经工蜂代理域名 *.ssh-proxy.devcloud.woa.com 访问）⇒ 别的机器连不进来，',
    '跨机交互只能由本机主动发起；本机对外（内网域名 / 服务）的可达性按需实测',
  ].join(''),
  CloudDesktop: [
    '本机在云桌面（IMate 数字同事：**独立网段 + 入站封禁**，',
    '与办公网不是一个区域）⇒ 出站受限：Windows 版仅 80/8080/8081/443 放通',
    '（其他端口需在 imate.woa.com 申请网关策略）、SSH 须经 MNET 跳板机、',
    '禁止访问办公 PC、浏览器打不开公网网盘；Linux 版为白名单，默认放行 DevCloud。',
    '**入站封禁 ⇒ 别的机器连不进来，跨机交互只能由本机主动发起**',
  ].join(''),
  VPN: [
    '经 VPN / 远程办公 ⇒ 至 DevCloud：<1024 仅 80/443；≥1024 放通，',
    '但禁 6379(Redis)/9000(MinIO)/2375(docker)/10250(k8s)/',
    '5601/5858/6066/7077/8500/9001/9292；访问 IDC 需申请',
  ].join(''),
  proxy: '仅见本地代理 ⇒ 内网可达性依赖该代理（通常覆盖内网域名），具体范围需实测',
  none: [
    'IDC 外区/外网或未收录网段 ⇒ 内网地址一概不可达（勿报成服务故障）；',
    '若本机确在内网（例如云桌面未收录），把其网段或平台特征补入本脚本',
  ].join(''),
};

function ipToInt(ip) {
  let value = 0;
  for (const octet of ip.split('.')) {
    value = ((value << 8) + Number(octet)) >>> 0;
  }
  return value;
}

function inCidr(ip, cidr) {
  const [network, prefixLength] = cidr.split('/');
  const bits = Number(prefixLength);
  // JS 移位按 32 取模：bits 为 0 时 << 32 等于 << 0，会得到全 1 掩码，故必须显式取 0
  let mask = 0;
  if (bits > 0) {
    mask = (0xffffffff << (32 - bits)) >>> 0;
  }
  return (ipToInt(ip) & mask) === (ipToInt(network) & mask);
}

// 命中哪个区域：返回区域名或 null（命中不了不等于不可达 —— VPN / 代理场景下物理网卡本就是私网段）
function zoneNameOf(ip) {
  for (const zone of NETWORK_ZONES) {
    for (const cidr of zone.cidrs) {
      if (inCidr(ip, cidr)) {
        return zone.name;
      }
    }
  }
  return null;
}

function describeZone(ip) {
  const zoneName = zoneNameOf(ip);
  if (zoneName) {
    return `${zoneName}，命中区域网段`;
  }
  const isPrivate = PRIVATE_CIDRS.some((cidr) => inCidr(ip, cidr));
  if (isPrivate) {
    return '私有网段，未命中 DevCloud/OA 段';
  }
  return '未命中已知区域网段';
}

function collectIPv4() {
  const result = [];
  for (const [name, addresses] of Object.entries(os.networkInterfaces())) {
    if (!addresses) {
      continue;
    }
    for (const address of addresses) {
      if (address.family === 'IPv4' && !address.internal) {
        result.push({ name, address: address.address });
      }
    }
  }
  return result;
}

// 云桌面平台识别（任一信号命中即认定；返回命中信号描述，未命中返回 null）。
// ⚠️ 平台身份优先于网络区域：云桌面跑在 DevCloud 网段内，只看 IP 会被当成普通 DevCloud 机器、
// 从而套错策略（云桌面 Windows 版仅 80/8080/8081/443 放通，远严于 DevCloud）。
// 信号按可靠性排序，均在真机验证过（2026-10-03，Windows 云桌面 earthywu）：
//  a) 网卡名含「云桌面」：云桌面管理网卡（NGNClient）被平台重命名为「禁用后无法使用云桌面」，
//     普通 DevCloud 机器不会有这种命名。名字虽是中文，但取自 os.networkInterfaces() 的
//     JS 字符串，不经过 GBK 解码，无乱码问题。【本机实测命中】
//  b) 主机名 TC 模板 /^TC\d{12}$/i：云桌面实例命名模板。
//  c) BIOS SystemManufacturer = 'Tencent Cloud'：云桌面宿主为腾讯云 CVM（SystemProductName=CVM）。
//     【⚠️ 未定论信号：若 DevCloud 的 Windows VM 同样跑在腾讯云母机上会误报成云桌面——
//     误报方向是「可达性说得偏保守」，可接受；待在已知 DevCloud Windows 机上验证一次后再决定去留】
function isCloudDesktop() {
  if (process.platform === 'win32') {
    const nicNames = Object.keys(os.networkInterfaces());
    for (const name of nicNames) {
      if (name.includes('云桌面')) {
        return '网卡名含「云桌面」';
      }
    }
  }
  if (/^TC\d{12}$/i.test(os.hostname())) {
    return '主机名 TC 模板';
  }
  if (process.platform === 'win32' && readBiosManufacturer() === 'Tencent Cloud') {
    return 'BIOS Tencent Cloud';
  }
  return null;
}

// 读 BIOS 厂商（仅 Windows）：reg query 输出稳定 ASCII，不受系统语言影响
function readBiosManufacturer() {
  let output = '';
  try {
    output = execFileSync(
      'reg',
      ['query', 'HKLM\\HARDWARE\\DESCRIPTION\\System\\BIOS', '/v', 'SystemManufacturer'],
      { encoding: 'utf8', timeout: 1000 },
    );
  } catch {
    return '';
  }
  const matched = output.match(/SystemManufacturer\s+REG_SZ\s+(\S+)/i);
  if (!matched) {
    return '';
  }
  return matched[1];
}

// os.networkInterfaces() 在 macOS 上看不到 utun 隧道 ⇒ 只能解析 ifconfig。
function readVpnTunnels() {
  if (process.platform === 'win32') {
    return readWindowsNgnTunnels();
  }

  let output = '';
  try {
    output = execFileSync('ifconfig', ['-a'], { encoding: 'utf8', timeout: 1000 });
  } catch {
    return [];
  }

  const tunnels = [];
  let currentInterface = '';
  for (const line of output.split('\n')) {
    const header = line.match(/^(\S+):\s+flags=/);
    if (header) {
      currentInterface = header[1];
      continue;
    }
    const inet = line.match(/^\s+inet (\d+\.\d+\.\d+\.\d+)/);
    if (!inet) {
      continue;
    }
    if (TUNNEL_INTERFACE_PATTERN.test(currentInterface)) {
      tunnels.push(`${currentInterface} ${inet[1]}`);
    }
  }
  return tunnels;
}

// Windows 侧隧道识别：解析 `ipconfig /all`（中文系统输出为 GBK，非 ASCII 适配器名会乱码，
// 但本函数只依赖 ASCII 正则，不受影响）。命中条件（满足其一）：
//   a) 适配器 Description 含 NGN（iOA NGNClient 虚拟网卡，名称稳定不随系统语言变）；
//   b) IPv4 落在 NGN 固定网段 192.168.255.0/24（Description 解析失败时的兜底）。
function readWindowsNgnTunnels() {
  let output = '';
  try {
    output = execFileSync('ipconfig', ['/all'], { encoding: 'utf8', timeout: 1000 });
  } catch {
    return [];
  }

  const tunnels = [];
  let adapterDescription = '';
  for (const line of output.split(/\r?\n/)) {
    const descriptionLine = line.match(/^\s*(?:Description|描述)[^:]*:\s*(\S.*?)\s*$/i);
    if (descriptionLine) {
      adapterDescription = descriptionLine[1];
      continue;
    }
    const ipv4Line = line.match(/^\s*IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)/);
    if (!ipv4Line) {
      continue;
    }
    const isNgnAdapter = /NGN/i.test(adapterDescription);
    if (isNgnAdapter || inCidr(ipv4Line[1], NGN_TUNNEL_CIDR)) {
      tunnels.push(`${adapterDescription || 'NGN'} ${ipv4Line[1]}`);
    }
  }
  return tunnels;
}

function probeTcp(host, port, timeout = PROBE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const socket = net.connect({ host, port });
    const finish = (connected) => {
      socket.destroy();
      resolve(connected);
    };
    socket.setTimeout(timeout);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
  });
}

// CDP 端口走 HTTP 而非纯 TCP：能顺带取回版本号，比"端口通"更有判断价值
function probeCdp(port, timeout = PROBE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const options = { host: '127.0.0.1', port, path: '/json/version', timeout };

    function handleResponse(res) {
      let body = '';
      res.on('data', (chunk) => {
        body += chunk;
      });
      // 响应流中途出错（ECONNRESET）不会触发 'end' ⇒ 不兜住本 Promise 就永不 settle
      res.on('error', () => resolve(null));
      res.on('end', () => resolve(parseCdpVersion(body)));
    }

    const request = http.get(options, handleResponse);
    request.on('timeout', () => {
      request.destroy();
      resolve(null);
    });
    request.on('error', () => resolve(null));
  });
}

// 该端口的占用者不可信（可能是残留或已被替换的进程）⇒ 只认「产品名/版本号」形态
function parseCdpVersion(body) {
  let browser;
  try {
    browser = JSON.parse(body).Browser;
  } catch {
    return '响应不可解析';
  }
  const looksLikeVersion = typeof browser === 'string' && CDP_VERSION_PATTERN.test(browser);
  if (!looksLikeVersion) {
    return '非 CDP 响应';
  }
  return browser;
}

// 接入身份决定「是否在内网」与「可达性」两处表述，一并在此判定。
// **平台身份优先于网络区域**：云桌面跑在 DevCloud 网段内，只看 IP 会被当成普通 DevCloud 机器、
// 从而套错策略（云桌面 Windows 版仅 80/8080/8081/443 放通，远严于 DevCloud）。
function describeIntranet(interfaces, tunnels, proxyAlive) {
  const cloudDesktopSignal = isCloudDesktop();
  const directInterface = interfaces.find((item) => zoneNameOf(item.address) !== null);

  if (cloudDesktopSignal) {
    let zoneName = '未收录';
    let nicText = '';
    if (directInterface) {
      zoneName = zoneNameOf(directInterface.address);
      nicText = `：${directInterface.name} ${directInterface.address}`;
    }
    const platformText =
      '**是**（云桌面 IMate 数字同事，平台信号：'
      + `${cloudDesktopSignal}；网络区域属 ${zoneName}${nicText}）`;
    return {
      identity: 'CloudDesktop',
      accessText: `${platformText}｜**入站封禁 ⇒ 跨机连不进来**`,
    };
  }

  if (directInterface) {
    const zoneName = zoneNameOf(directInterface.address);
    const addressText = `${directInterface.name} ${directInterface.address}`;
    return {
      identity: zoneName,
      accessText: `**是**（本机网卡直连 ${zoneName} 段：${addressText}）`,
    };
  }

  if (tunnels.length > 0) {
    return {
      identity: 'VPN',
      accessText: `**是**（经 VPN/NGN 隧道：${tunnels.join(' / ')}）`,
    };
  }

  if (proxyAlive) {
    return {
      identity: 'proxy',
      accessText: `**疑似**（仅见本地代理 ${LOCAL_PROXY.label} 在监听 ⇒ 可达性依赖该代理）`,
    };
  }

  return {
    identity: 'none',
    accessText: '**否**（未见内网网段 IP、VPN 隧道与本地代理 ⇒ 按外网环境对待）',
  };
}

function describeInterfaces(interfaces) {
  if (interfaces.length === 0) {
    return '无对外 IPv4';
  }
  const parts = [];
  for (const item of interfaces) {
    parts.push(`${item.name} ${item.address}（${describeZone(item.address)}）`);
  }
  return parts.join('｜');
}

function describeProxy(proxyAlive) {
  if (proxyAlive) {
    return `｜本地代理 ${LOCAL_PROXY.label} ✓`;
  }
  return `｜本地代理 ${LOCAL_PROXY.label} ✗（未监听）`;
}

function describeCdpPorts(cdpResults) {
  const parts = [];
  for (let index = 0; index < CDP_PORTS.length; index += 1) {
    const port = CDP_PORTS[index];
    const version = cdpResults[index];
    if (version) {
      parts.push(`${port} 可用 ${version}`);
    } else {
      parts.push(`${port} 不可用`);
    }
  }
  return parts.join('｜');
}

function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', () => resolve(''));
  });
}

// Hermes 的载荷结构见 agent/shell_hooks.py::_serialize_payload：事件专有字段收在 extra 里。
// 载荷不可解析时按"非首轮"处理——那意味着 Hermes 改了 wire 格式，须改代码而非兜底猜。
async function isHermesFirstTurn() {
  try {
    return JSON.parse(await readStdin()).extra?.is_first_turn === true;
  } catch {
    return false;
  }
}

async function main() {
  // 兜底护栏：万一某个探测既不 settle 也不报错（如 TCP 半开），await 会永久挂住，
  // 进程被 hook 的 15s timeout 强杀且零输出 ⇒ 这里主动提前退出，保证"有输出、不拖会话"。
  // unref ⇒ 正常路径（实测约 0.04s）不阻塞退出；fs.writeSync ⇒ process.exit 会截断异步 stdout。
  const guard = setTimeout(() => {
    process.exit(0);
  }, PROBE_TIMEOUT_MS * 3);
  guard.unref();

  // Hermes 每轮都触发 pre_llm_call：非首轮立即空返回，既不重复注入也省掉本轮探测
  if (HERMES_MODE && !(await isHermesFirstTurn())) {
    process.stdout.write('{}\n');
    return;
  }

  const interfaces = collectIPv4();
  const tunnels = readVpnTunnels();

  // CDP 与代理探测互不依赖，一并并发发起，最坏只等一次探测超时
  const cdpProbes = Promise.all(CDP_PORTS.map((port) => probeCdp(port)));
  const proxyProbe = probeTcp('127.0.0.1', LOCAL_PROXY.port);
  const [cdpResults, proxyAlive] = await Promise.all([cdpProbes, proxyProbe]);

  const { identity, accessText } = describeIntranet(interfaces, tunnels, proxyAlive);
  const interfaceText = describeInterfaces(interfaces) + describeProxy(proxyAlive);

  const lines = [
    '[本机环境]',
    `- 机器：${os.hostname()}｜内网接入：${accessText}`,
    `- 网络明细：${interfaceText}`,
    `- 可达性：${REACH_STRATEGY[identity]}`,
    `- 浏览器 CDP：${describeCdpPorts(cdpResults)}`,
  ];
  const context = lines.join('\n');
  process.stdout.write(HERMES_MODE ? `${JSON.stringify({ context })}\n` : `${context}\n`);
}

main().catch(() => {
  // 失败也必须让 agent 知道"没探到"，并带上目录以定位是哪台机器/哪个仓库的 hook——
  // 静默会被当成环境无异常
  const message = `${HOOK_DIR} hooks 执行失败，请告知用户`;
  process.stdout.write(HERMES_MODE ? `${JSON.stringify({ context: message })}\n` : `${message}\n`);
});
