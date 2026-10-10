// 站点通道注册表。新增一个站点只需：
//   1) 在本目录加一个适配器文件（见 panshi.mjs / imeta.mjs / oppty.mjs 的顶部注释与结构）；
//   2) 在下面的数组里登记它。
// 适配器结构（由 gateway-sign-server 消费）：
//   name       通道名（日志与跨通道判断用）
//   credential 需要的凭证键（文档用；实际是否启用由 enabled() 决定）
//   pacHosts   纳入 PAC 的域（数组，后缀匹配含其子域）
//   upstream   { tls, port } 上游传输方式
//   matches(host)             是否命中该站点
//   enabled(credentials)      凭证是否齐备、该通道是否可用
//   forRequest(credentials, ctx) -> { headers?, rewriteLocation?(loc), rewriteBody?(text) }
import imeta from './imeta.mjs';
import oppty from './oppty.mjs';
import panshi from './panshi.mjs';

export const CHANNELS = [panshi, imeta, oppty];
