// 站点：panshi（磐石）dev/test。
//
// DevCloud 经 iGate 直达 dev/test.panshi.woa.com 时链路上没有智能网关：网关会因缺 staffname 返回 302、
// 后端会因缺签名返回 401，故需按请求补签名。用的是磐石自己的网关 token（SMART_PROXY_KEY）。
import { signGatewayHeaders, suffixMatch } from '../gateway-sign.mjs';
import { rewriteImetaInHtml, rewriteImetaLocation } from './imeta.mjs';

const PANSHI_HOST = 'panshi.woa.com';

export default {
  name: 'panshi',
  credential: 'SMART_PROXY_KEY',
  pacHosts: [PANSHI_HOST],
  upstream: { tls: false, port: 80 },
  matches: (host) => suffixMatch(host, PANSHI_HOST),
  enabled: (credentials) => typeof credentials.key === 'string' && credentials.key !== '',
  forRequest(credentials, ctx) {
    const headers = {
      ...signGatewayHeaders(credentials.key, credentials.staffName),
      // dev 环境专用：让网关按 dev token 校验
      'X-Smart-Proxy-Auth': 'devpanshi',
    };
    // 只有 imeta 通道启用时才需要改写响应里指向 imeta 的跳转；
    // 改写正文要求上游不压缩，故同时声明 accept-encoding: identity。
    if (!ctx.enabled.has('imeta')) {
      return { headers };
    }
    headers['accept-encoding'] = 'identity';
    return { headers, rewriteLocation: rewriteImetaLocation, rewriteBody: rewriteImetaInHtml };
  },
};
