import { PageQuery } from "@/types";

/** MCP 传输协议：stdio-本地进程 / streamable-http-流式 HTTP / sse-传统 SSE */
export type McpProtocolType = "stdio" | "streamable-http" | "sse";

/** Server 健康状态：online-在线 / offline-离线 */
export type McpHealthStatus = "online" | "offline";

/** 注册/更新 MCP Server 表单 */
export interface McpServerForm {
  /** Server 名称（唯一） */
  name: string;
  description?: string;
  /** 传输协议 */
  protocolType: McpProtocolType;
  /** 端点 URL（stdio 可为空） */
  endpoint?: string;
  /** 鉴权方式（none/api_key/oauth2 等） */
  authType?: string;
}

/** MCP Server 视图对象 */
export interface McpServerVO {
  id: number;
  name: string;
  description?: string;
  protocolType: McpProtocolType;
  endpoint?: string;
  authType?: string;
  /** 状态：1-启用，0-禁用 */
  status: 0 | 1;
  /** 健康状态 */
  health?: McpHealthStatus | null;
  /** 工具数量 */
  toolCount?: number;
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
  startTime?: string;
  endTime?: string;
}
