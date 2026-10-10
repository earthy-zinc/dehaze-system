// 站点：imeta（Meta 经管工具平台，环境切换页所在站点）。
//
// DevCloud 的 iGate 只放通 imeta:80，而 env-gateway 的跳转地址是 https，故需代理 + 改写；
// 且该站要它自己的网关 token（这里用 SA_TOKEN）。
import { buildTaiIdentity, signGatewayHeaders } from '../gateway-sign.mjs';

export const IMETA_HOST = 'imeta.woa.com';

// env-gateway 的跳转是 https://imeta.woa.com/...，需降级成 http（DevCloud 未放通 443）。
// ⚠️ 只由 panshi 通道在改写**它的响应**时调用——绝不可改写 imeta 自身的响应，
// 否则会与 env-switch 页面的回跳形成 http↔https 无限重定向（浏览器疯狂刷新，已踩过）。
export function rewriteImetaLocation(location) {
  const prefix = `https://${IMETA_HOST}/`;
  return location.startsWith(prefix) ? `http://${IMETA_HOST}/${location.slice(prefix.length)}` : location;
}

// 磐石 HTML 里硬编码了 https://imeta.woa.com（env-switch 浮层按钮的跳转目标，走客户端 location.replace，
// 不经过 Location 改写），故由 panshi 通道对**它的响应正文**做一次同域降级替换。
export function rewriteImetaInHtml(text) {
  return text.split(`https://${IMETA_HOST}`).join(`http://${IMETA_HOST}`);
}

export default {
  name: 'imeta',
  credential: 'SA_TOKEN',
  pacHosts: [IMETA_HOST],
  upstream: { tls: false, port: 80 },
  matches: (host) => host === IMETA_HOST,
  enabled: (credentials) => typeof credentials.saToken === 'string' && credentials.saToken !== '',
  forRequest(credentials) {
    return {
      headers: {
        ...signGatewayHeaders(credentials.saToken, credentials.staffName),
        'x-tai-identity': buildTaiIdentity(credentials.saToken, credentials.staffName),
      },
    };
  },
};
