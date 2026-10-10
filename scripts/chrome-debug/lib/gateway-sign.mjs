// 智能网关签名代理的**共享核心**：凭证解析、签名 / x-tai-identity 生成、PAC 生成、快速失败名单。
// 本文件不含任何「某站点特有」的逻辑——各站点（panshi / imeta / forcem）的差异全部收在 lib/channels/ 下。
//
// 背景：办公网由智能网关自动注入身份头；DevCloud 这类链路上没有网关，故由本代理按同一格式自行生成。
// 凭证只从环境变量或 ~/.chrome-debug/.env 读取，绝不写进仓库。
import { createCipheriv, createHash, randomBytes } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  FAIL_FAST_HOSTS,
  RUNTIME_DIR_NAME,
  SIGN_ENV_FILE,
  SIGN_IOA_TOKEN_ENV,
  SIGN_KEY_ENV,
  SIGN_PAC_PATH,
  SIGN_SA_TOKEN_ENV,
  SIGN_STAFF_ENV,
} from './defaults.mjs';

export const GATEWAY_SIGN_SERVER = join(dirname(fileURLToPath(import.meta.url)), 'gateway-sign-server.mjs');

// 签名派生的固定字段，对齐官方实现 omp-tools/automatic-test/src/lib/generateSessionHeaders.ts：
// staffId 固定 '1'，x-rio-seq 固定全零，x-ext-data 为空
const SIGN_STAFF_ID = '1';
const SIGN_RIO_SEQ = '000000:000000000000:00000';

function runtimeDir() {
  return join(homedir(), RUNTIME_DIR_NAME);
}

// 解析 KEY=VALUE 形式的 env 文件；不存在或读不出时返回空对象
function readEnvFile(filePath) {
  if (!existsSync(filePath)) {
    return {};
  }
  let content;
  try {
    content = readFileSync(filePath, 'utf8');
  } catch {
    return {};
  }
  const values = {};
  for (const line of content.split('\n')) {
    const matched = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/.exec(line);
    if (matched === null) {
      continue;
    }
    values[matched[1]] = matched[2].replace(/^["']|["']$/g, '');
  }
  return values;
}

/**
 * 解析网关凭证：环境变量优先，其次 ~/.chrome-debug/.env。
 * 至少有 STAFF_NAME 与一个站点 token 才有效；各站点按自己需要的 token 自行判断是否启用。
 * @returns {{staffName: string, key?: string, saToken?: string, ioaToken?: string} | null}
 */
export function resolveGatewayCredentials() {
  const fileValues = readEnvFile(join(runtimeDir(), SIGN_ENV_FILE));
  const staffName = process.env[SIGN_STAFF_ENV] || fileValues[SIGN_STAFF_ENV];
  const key = process.env[SIGN_KEY_ENV] || fileValues[SIGN_KEY_ENV];
  const saToken = process.env[SIGN_SA_TOKEN_ENV] || fileValues[SIGN_SA_TOKEN_ENV];
  const ioaToken = process.env[SIGN_IOA_TOKEN_ENV] || fileValues[SIGN_IOA_TOKEN_ENV];
  if (!staffName || (!key && !saToken && !ioaToken)) {
    return null;
  }
  return {
    staffName,
    ...(key ? { key } : {}),
    ...(saToken ? { saToken } : {}),
    ...(ioaToken ? { ioaToken } : {}),
  };
}

/**
 * 按请求现算智能网关签名头（三个站点共用同一公式，只是 token 不同）；时间戳窗口 180s，故必须每请求生成。
 * @param {string} token 该站点的网关共享密钥
 * @param {string} staffName 员工英文名
 */
export function signGatewayHeaders(token, staffName) {
  const timestamp = `${Math.floor(Date.now() / 1000)}`;
  const signature = createHash('sha256')
    .update(`${timestamp}${token}${SIGN_RIO_SEQ},${SIGN_STAFF_ID},${staffName},${timestamp}`)
    .digest('hex')
    .toUpperCase();
  return {
    timestamp,
    staffid: SIGN_STAFF_ID,
    staffname: staffName,
    'x-rio-seq': SIGN_RIO_SEQ,
    'x-ext-data': '',
    signature,
  };
}

function base64Url(buffer) {
  return Buffer.from(buffer).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * 构造 x-tai-identity（智能网关密文身份，JWE compact：dir + A256GCM，密钥=token）。
 * 办公网由智能网关签发；DevCloud 无网关，因此按同一格式自行生成，使后端能解出身份。
 * @param {string} token 该站点的网关共享密钥
 * @param {string} staffName 员工英文名
 * @param {object} [extra] 追加进明文的字段（各站点按后端期望自定）
 */
export function buildTaiIdentity(token, staffName, extra = {}) {
  const header = base64Url(JSON.stringify({ alg: 'dir', enc: 'A256GCM' }));
  const iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', Buffer.from(token), iv);
  cipher.setAAD(Buffer.from(header, 'ascii'));
  const plaintext = JSON.stringify({ LoginName: staffName, StaffId: SIGN_STAFF_ID, ...extra });
  const ciphertext = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  return [header, '', base64Url(iv), base64Url(ciphertext), base64Url(cipher.getAuthTag())].join('.');
}

// 域名后缀匹配：host === domain 或其子域
export function suffixMatch(host, domain) {
  return host === domain || host.endsWith(`.${domain}`);
}

// DevCloud 不可达、命中即快速失败（不让它拖住页面加载）
export function isFailFastHost(host) {
  return FAIL_FAST_HOSTS.includes(host);
}

// PAC：只把各启用通道的 http 流量交给本地代理，其余一律直连（不影响本地 dev 与其它站点）
export function buildPacContent(port, channelHosts = []) {
  // 不可达资源始终纳入（由代理立即快速失败）；否则会在浏览器侧长时间挂起、拖住标签页加载
  const hosts = [...new Set([...channelHosts, ...FAIL_FAST_HOSTS])];
  const conditions = hosts
    .map((name) => `(host === '${name}' || host.slice(-${name.length + 1}) === '.${name}')`)
    .join(' || ');
  return [
    'function FindProxyForURL(url, host) {',
    `  if (url.indexOf('http://') === 0 && (${conditions})) {`,
    `    return 'PROXY 127.0.0.1:${port}';`,
    '  }',
    "  return 'DIRECT';",
    '}',
    '',
  ].join('\n');
}

// PAC 由代理自身经 http 提供（实测 Chrome 不采纳 file:// 形式的 PAC，http 形式才生效），
// 地址带签名端口，同机多实例互不干扰
export function pacUrl(config) {
  return `http://127.0.0.1:${config.signPort}${SIGN_PAC_PATH}`;
}
