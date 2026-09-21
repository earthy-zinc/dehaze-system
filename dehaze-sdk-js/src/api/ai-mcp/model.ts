import { PageQuery } from "@/types";

/** MCP 传输协议：仅 URL 型协议可注册（stdio 无网络端点，拉取/装载/探测均不支持，已退役） */
export type McpProtocolType = "streamable-http" | "sse";

/** Server 健康状态：online-在线 / offline-离线 */
export type McpHealthStatus = "online" | "offline";

/** 注册/更新 MCP Server 表单 */
export interface McpServerForm {
  /** Server 名称（唯一） */
  name: string;
  description?: string;
  /** 传输协议 */
  protocolType: McpProtocolType;
  /** 端点 URL */
  endpoint?: string;
  /** 鉴权方式（none/api_key/oauth2 等） */
  authType?: string;
}

/** MCP Server 视图对象 */
export interface McpServerVO {
  id: number;
  name: string;
  description?: string;
  /** 传输协议：stdio 为历史存量（服务端已不接受注册，仅作不受支持提示） */
  protocolType: McpProtocolType | "stdio";
  endpoint?: string;
  authType?: string;
  /** 状态：1-启用，0-禁用 */
  status: 0 | 1;
  /** 健康状态 */
  health?: McpHealthStatus | null;
  /** 最近一次健康探测时间（手动探测与后台巡检共用） */
  lastCheckTime?: string | null;
  /** 工具数量 */
  toolCount?: number;
  /** 是否已配置凭据（凭据仅写入不回显） */
  credentialConfigured?: boolean;
  createTime?: string;
  updateTime?: string;
}

/** MCP Server 分页查询参数 */
export interface McpServerQuery extends PageQuery {
  keyword?: string;
  /** 状态筛选（1-启用，0-禁用） */
  status?: 0 | 1;
}

/** Server 健康探测结果 */
export interface McpHealthVO {
  status: McpHealthStatus;
  /** 延迟（毫秒） */
  latencyMs?: number;
}

/** MCP 工具视图对象 */
export interface McpToolVO {
  /** 工具名（命名空间内唯一） */
  name: string;
  description?: string;
  /** 参数 schema 概要 */
  inputSchema?: Record<string, unknown>;
}

/** 命名空间视图对象（工具分组） */
export interface McpNamespaceVO {
  name: string;
  toolNames: string[];
}

/** 凭据配置表单（加密存储，仅录入/更新，不回显明文） */
export interface McpCredentialForm {
  /** API Key 等外部服务凭据 */
  apiKey?: string;
  /** 其他凭据字段 */
  extra?: Record<string, string>;
  /** 清除已配置凭据（轮换/吊销场景，与 apiKey/extra 互斥） */
  clear?: boolean;
}

/** MCP 市场预设目录项 */
export interface McpMarketPresetVO {
  /** 预设 ID（市场唯一标识） */
  presetId: string;
  name: string;
  description?: string;
  /** 能力标签 */
  capabilityTags?: string[];
  /** 是否已接入 */
  installed?: boolean;
}

/** 外部 MCP 调用审计记录 */
export interface McpCallVO {
  id: number;
  userId?: number;
  serverId: number;
  serverName?: string;
  toolName: string;
  /** 调用结果（success/failure） */
  result?: string;
  latencyMs?: number;
  createTime: string;
}

/** 外部 MCP 调用审计查询参数 */
export interface McpCallQuery extends PageQuery {
  serverId?: number;
  toolName?: string;
}

/** 外部 MCP 工具试调用表单 */
export interface McpToolTestForm {
  /** 待调用工具名 */
  toolName: string;
  /** 调用参数（JSON 对象） */
  arguments?: Record<string, unknown>;
}

/** 外部 MCP 工具试调用结果 */
export interface McpToolTestResult {
  success: boolean;
  /** 工具返回文本 */
  result?: string;
  /** 失败原因 */
  error?: string;
  /** 调用耗时（毫秒） */
  latencyMs?: number;
}
