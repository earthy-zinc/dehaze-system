// 本地「智能网关签名」代理：由 chrome-debug 以子进程方式拉起。
// 站点无关：按 lib/channels 注册表匹配目标主机 → 注入该站点的签名头 → 按其声明的上游协议转发。
// 仅 DevCloud 等无智能网关的机器需要；凭证来自环境变量或 ~/.chrome-debug/.env。
import { createServer, request as createRequest } from 'node:http';
import { request as createSecureRequest } from 'node:https';
import { CHANNELS } from './channels/index.mjs';
import { SIGN_PAC_PATH, SIGN_PORT_ENV, SIGN_PROXY_ERROR_STATUS } from './defaults.mjs';
import { buildPacContent, isFailFastHost, resolveGatewayCredentials } from './gateway-sign.mjs';

const credentials = resolveGatewayCredentials();
if (credentials === null) {
  console.error(
    'gateway-sign: 缺少凭证（环境变量或 ~/.chrome-debug/.env）：需 STAFF_NAME 及至少一个站点 token（SMART_PROXY_KEY / SA_TOKEN / IOA_TOKEN），代理不启动',
  );
  process.exit(1);
}

const port = Number(process.env[SIGN_PORT_ENV] || 0);
if (!Number.isInteger(port) || port <= 0) {
  console.error(`gateway-sign: 缺少合法端口（${SIGN_PORT_ENV}）`);
  process.exit(1);
}

// 只保留凭证齐备的通道；未配 token 的通道自动降级（不纳入 PAC、不注入头）
const activeChannels = CHANNELS.filter((channel) => channel.enabled(credentials));
const ctx = { credentials, enabled: new Set(activeChannels.map((channel) => channel.name)) };
const pacHosts = [...new Set(activeChannels.flatMap((channel) => channel.pacHosts))];

function respond(res, statusCode, message) {
  res.writeHead(statusCode, { 'content-type': 'text/plain; charset=utf-8' });
  res.end(message);
}

const server = createServer((req, res) => {
  // 源站式请求：Chrome 拉取 PAC（req.url 是路径而非绝对地址）
  if (req.url.startsWith('/')) {
    if (req.url.split('?')[0] === SIGN_PAC_PATH) {
      res.writeHead(200, { 'content-type': 'application/x-ns-proxy-autoconfig; charset=utf-8' });
      res.end(buildPacContent(port, pacHosts));
      return;
    }
    respond(res, 404, 'gateway-sign: 仅提供 PAC');
    return;
  }

  // 作为正向代理时 req.url 是绝对地址（http://host/path）
  let target;
  try {
    target = new URL(req.url);
  } catch {
    respond(res, 400, 'gateway-sign: 非法请求地址');
    return;
  }

  // DevCloud 不可达、且会拖住页面加载的装饰性资源：立即返回 204，别让浏览器长时间挂起（标签页一直转圈）
  if (isFailFastHost(target.hostname)) {
    res.writeHead(204);
    res.end();
    return;
  }

  const channel = activeChannels.find((item) => item.matches(target.hostname));
  const plan = channel ? channel.forRequest(credentials, ctx) : {};
  const headers = { ...req.headers };
  delete headers['proxy-connection'];
  Object.assign(headers, plan.headers || {});

  // 上游协议由通道声明：panshi/imeta 走 http（DevCloud 未放通 443）；forcem 走 https（443 可达、80 会 302）
  const forward = channel?.upstream?.tls === true ? createSecureRequest : createRequest;
  const upstreamPort = channel?.upstream?.port ?? (target.port || 80);

  const upstream = forward(
    {
      hostname: target.hostname,
      port: upstreamPort,
      method: req.method,
      path: `${target.pathname}${target.search}`,
      headers,
    },
    (upstreamResponse) => {
      const responseHeaders = { ...upstreamResponse.headers };
      // 响应改写由通道声明（目前仅 panshi→imeta 的跳转降级）
      if (plan.rewriteLocation && typeof responseHeaders.location === 'string') {
        responseHeaders.location = plan.rewriteLocation(responseHeaders.location);
      }
      if (plan.rewriteBody && String(responseHeaders['content-type'] || '').includes('text/html')) {
        const chunks = [];
        upstreamResponse.on('data', (chunk) => chunks.push(chunk));
        upstreamResponse.on('end', () => {
          const body = Buffer.from(plan.rewriteBody(Buffer.concat(chunks).toString('utf8')), 'utf8');
          delete responseHeaders['content-encoding'];
          responseHeaders['content-length'] = String(body.length);
          res.writeHead(upstreamResponse.statusCode, responseHeaders);
          res.end(body);
        });
        return;
      }
      res.writeHead(upstreamResponse.statusCode, responseHeaders);
      upstreamResponse.pipe(res);
    },
  );

  upstream.on('error', (error) => {
    respond(res, SIGN_PROXY_ERROR_STATUS, `gateway-sign: 转发失败 ${error.message}`);
  });
  req.on('aborted', () => upstream.destroy());
  req.pipe(upstream);
});

server.on('error', (error) => {
  console.error(`gateway-sign: 监听失败 ${error.message}`);
  process.exit(1);
});

server.listen(port, '127.0.0.1', () => {
  const names = activeChannels.map((channel) => channel.name).join('+') || '(无)';
  console.log(`gateway-sign: 代理就绪 127.0.0.1:${port}（staffname=${credentials.staffName}，通道 ${names}）`);
});
