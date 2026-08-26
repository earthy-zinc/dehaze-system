import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  McpCallQuery,
  McpCallVO,
  McpCredentialForm,
  McpHealthVO,
  McpMarketPresetVO,
  McpNamespaceVO,
  McpServerForm,
  McpServerQuery,
  McpServerVO,
  McpToolVO,
} from "./model";

/**
 * MCP Server 管理 API（管理端，需 ai:mcp:manage）。
 *
 * 外部 MCP Server 的通用接入中心：注册表/工具与命名空间/凭据/健康/调用审计 + 市场一键接入。
 */
class AiMCPAPI {
  // ==================== Server 注册表 ====================

  /** MCP Server 列表（分页） */
  static listServers(query?: McpServerQuery) {
    return request<PageResult<McpServerVO[]>>({
      url: "/api/v1/ai/mcp/servers",
      method: "get",
      params: query,
    });
  }

  /** 注册外部 MCP Server（注册后自动拉取工具清单） */
  static createServer(data: McpServerForm) {
    return request<McpServerVO>({
      url: "/api/v1/ai/mcp/servers",
      method: "post",
      data,
    });
  }

  /** Server 详情（含工具清单、命名空间） */
  static getServer(id: number) {
    return request<McpServerVO>({
      url: `/api/v1/ai/mcp/servers/${id}`,
      method: "get",
    });
  }

  /** 更新 Server 配置 */
  static updateServer(id: number, data: McpServerForm) {
    return request<McpServerVO>({
      url: `/api/v1/ai/mcp/servers/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除 Server（软删除；校验是否被 Agent 关联，有则提示先解绑） */
  static deleteServer(id: number) {
    return request({
      url: `/api/v1/ai/mcp/servers/${id}`,
      method: "delete",
    });
  }

  /** 启停 Server（status: 1 启用 / 0 禁用） */
  static switchServerStatus(id: number, status: 0 | 1) {
    return request<McpServerVO>({
      url: `/api/v1/ai/mcp/servers/${id}/status`,
      method: "patch",
      data: { status },
    });
  }

  // ==================== 健康 / 工具 / 命名空间 ====================

  /** Server 健康探测 */
  static probeHealth(id: number) {
    return request<McpHealthVO>({
      url: `/api/v1/ai/mcp/servers/${id}/health`,
      method: "get",
    });
  }

  /** Server 工具清单 */
  static getTools(id: number) {
    return request<McpToolVO[]>({
      url: `/api/v1/ai/mcp/servers/${id}/tools`,
      method: "get",
    });
  }

  /** 命名空间列表（工具分组） */
  static getNamespaces(id: number) {
    return request<McpNamespaceVO[]>({
      url: `/api/v1/ai/mcp/servers/${id}/namespaces`,
      method: "get",
    });
  }

  /** 配置命名空间（工具分组覆盖式更新） */
  static updateNamespaces(id: number, data: McpNamespaceVO[]) {
    return request<McpNamespaceVO[]>({
      url: `/api/v1/ai/mcp/servers/${id}/namespaces`,
      method: "put",
      data,
    });
  }

  // ==================== 凭据 ====================

  /** 配置外部服务凭据（加密存储，仅录入/更新，不回显明文） */
  static updateCredentials(id: number, data: McpCredentialForm) {
    return request({
      url: `/api/v1/ai/mcp/servers/${id}/credentials`,
      method: "put",
      data,
    });
  }

  // ==================== 市场 ====================

  /** MCP 市场目录（内置常用 Server 预设，含已接入状态） */
  static getMarket() {
    return request<McpMarketPresetVO[]>({
      url: "/api/v1/ai/mcp/market",
      method: "get",
    });
  }

  /** 从市场一键接入预设 Server */
  static installPreset(presetId: string) {
    return request<McpServerVO>({
      url: `/api/v1/ai/mcp/market/${presetId}/install`,
      method: "post",
    });
  }

  // ==================== 调用审计 ====================

  /** 外部 MCP 工具调用审计（分页） */
  static listCalls(query?: McpCallQuery) {
    return request<PageResult<McpCallVO[]>>({
      url: "/api/v1/ai/mcp/calls",
      method: "get",
      params: query,
    });
  }
}

export default AiMCPAPI;
