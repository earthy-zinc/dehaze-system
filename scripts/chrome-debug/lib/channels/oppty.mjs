// 站点：business-oppty（forcem），国内 / 国际 × dev / test。
//
// 与 panshi/imeta 不同：该站 **443 可达、80 会 302 到 https**，故上游固定走 https；
// 前端 API 走相对路径（baseURL: '/api'），静态资源由 COS 托管，只有 /api/* 需要网关身份。
// 用的是它自己的网关 token（IOA_TOKEN）。
import { buildTaiIdentity, signGatewayHeaders } from '../gateway-sign.mjs';

export const OPPTY_HOSTS = [
  'forcem-dev.testsite.woa.com',
  'forcem-test.testsite.woa.com',
  'sg-forcem-dev.testsite.woa.com',
  'sg-forcem-test.testsite.woa.com',
];

export default {
  name: 'oppty',
  credential: 'IOA_TOKEN',
  pacHosts: OPPTY_HOSTS,
  upstream: { tls: true, port: 443 },
  matches: (host) => OPPTY_HOSTS.includes(host),
  enabled: (credentials) => typeof credentials.ioaToken === 'string' && credentials.ioaToken !== '',
  forRequest(credentials) {
    return {
      headers: {
        ...signGatewayHeaders(credentials.ioaToken, credentials.staffName),
        // 后端（access / signature 中间件）要求 x-tai-identity 能解出 LoginName/StaffId/open_id/Expiration
        'x-tai-identity': buildTaiIdentity(credentials.ioaToken, credentials.staffName, {
          ChineseName: '',
          Ticket: '',
          open_id: '',
          Expiration: new Date(Date.now() + 3 * 3600 * 1000).toISOString(),
        }),
      },
    };
  },
};
