import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  AgentCopyForm,
  AgentCreateForm,
  AgentDetail,
  AgentListItem,
  AgentMcpForm,
  AgentPageQuery,
  AgentPublishForm,
  AgentSkillsForm,
  AgentStatusForm,
  AgentSubAgentsForm,
  AgentTestForm,
  AgentTestResult,
  AgentUpdateForm,
  AgentVersionDetail,
  AgentVersionResult,
  EndpointCreateForm,
  EndpointPageQuery,
  EndpointResult,
  EndpointUpdateForm,
  EvalDatasetCreateForm,
  EvalDatasetResult,
  EvalDatasetUpdateForm,
  EvalRunQuery,
  EvalRunResult,
  EvalSampleCreateForm,
  EvalSampleResult,
  EvalSampleUpdateForm,
  VersionResult,
} from "./model";

/** 管理端权限标识（后端拦截，403 由后端返回） */
const MANAGE_PERMISSION = "ai:agent:manage";

/**
 * AI 智能体管理 API
 *
 * 内部 API（`/api/v1/ai`），含 Agent CRUD、关联配置（Skills/MCP/子 Agent）、
 * 版本发布/回滚、评测集/样本/评测执行、外部 A2A 端点管理。
 *
 * 管理端接口（创建/更新/删除/启停/复制/发布/回滚/评测/A2A 端点）需
 * `ai:agent:manage` 权限，普通用户仅可查看启用列表/详情，由后端拦截。
 */
class AiAgentAPI {
  // ==================== Agent CRUD ====================

  /**
   * Agent 列表（分页）。
   * 管理端返回全量分页；普通用户仅返回启用列表（不分页，直接作为 list）。
   */
  static list(query?: AgentPageQuery) {
    return request<PageResult<AgentListItem[]>>({
      url: "/api/v1/ai/agents",
      method: "get",
      params: query,
    });
  }

  /** 可选 Agent 列表（仅启用，供调用方选择，无特殊权限） */
  static listEnabled() {
    return request<AgentListItem[]>({
      url: "/api/v1/ai/agents/enabled",
      method: "get",
    });
  }

  /** 创建 Agent（管理端） */
  static create(data: AgentCreateForm) {
    return request<AgentDetail>({
      url: "/api/v1/ai/agents",
      method: "post",
      data,
    });
  }

  /** Agent 详情 */
  static detail(id: number) {
    return request<AgentDetail>({
      url: `/api/v1/ai/agents/${id}`,
      method: "get",
    });
  }

  /** 更新 Agent（管理端） */
  static update(id: number, data: AgentUpdateForm) {
    return request<AgentDetail>({
      url: `/api/v1/ai/agents/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除 Agent（管理端，软删除） */
  static delete(id: number) {
    return request({
      url: `/api/v1/ai/agents/${id}`,
      method: "delete",
    });
  }

  /** 启停 Agent（管理端） */
  static setStatus(id: number, data: AgentStatusForm) {
    return request({
      url: `/api/v1/ai/agents/${id}/status`,
      method: "patch",
      data,
    });
  }

  /** Agent 测试预览（管理端，同步返回预览结果） */
  static test(id: number, data: AgentTestForm) {
    return request<AgentTestResult>({
      url: `/api/v1/ai/agents/${id}/test`,
      method: "post",
      // 后端 AgentTestForm 为纯 BaseModel，wire 字段为 conversation_config
      data: {
        message: data.message,
        conversation_config: data.conversationConfig ?? null,
      },
    });
  }

  /** 复制 Agent（管理端，生成新 agent_code） */
  static copy(id: number, data: AgentCopyForm) {
    return request<AgentDetail>({
      url: `/api/v1/ai/agents/${id}/copy`,
      method: "post",
      // 后端 AgentCopyForm 为纯 BaseModel（无 camelCase 别名），wire 字段为 agent_code
      data: { agent_code: data.agentCode },
    });
  }

  // ==================== 关联配置（覆盖式） ====================

  /** 设置 Skills（覆盖式，管理端） */
  static setSkills(id: number, data: AgentSkillsForm) {
    return request({
      url: `/api/v1/ai/agents/${id}/skills`,
      method: "put",
      data,
    });
  }

  /** 设置 MCP 命名空间（覆盖式，管理端） */
  static setMcps(id: number, data: AgentMcpForm) {
    return request({
      url: `/api/v1/ai/agents/${id}/mcps`,
      method: "put",
      // 后端 AgentMcpForm 为纯 BaseModel，wire 字段为 mcp_namespaces
      data: { mcp_namespaces: data.mcpNamespaces },
    });
  }

  /** 设置子 Agent（覆盖式，管理端） */
  static setSubAgents(id: number, data: AgentSubAgentsForm) {
    return request({
      url: `/api/v1/ai/agents/${id}/subagents`,
      method: "put",
      // 后端 AgentSubAgentsForm / AgentSubAgentItem 为纯 BaseModel，wire 字段为 agent_id/endpoint_id
      data: {
        subagents: (data.subagents ?? []).map((s) => ({
          agent_id: s.agentId,
          endpoint_id: s.endpointId ?? null,
          priority: s.priority ?? 0,
        })),
      },
    });
  }

  // ==================== 版本管理 ====================

  /** 发布 Agent（管理端，返回新版本号） */
  static publish(id: number, data: AgentPublishForm) {
    return request<VersionResult>({
      url: `/api/v1/ai/agents/${id}/publish`,
      method: "post",
      // 后端 AgentPublishForm 为纯 BaseModel，wire 字段为 change_note
      data: { change_note: data.changeNote ?? "" },
    });
  }

  /** 版本历史（分页） */
  static versions(id: number, query?: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<AgentVersionResult[]>>({
      url: `/api/v1/ai/agents/${id}/versions`,
      method: "get",
      params: query,
    });
  }

  /** 版本差异对比（返回字段级 diff 列表） */
  static versionDiff(id: number, base: number, target: number) {
    return request<Array<Record<string, unknown>>>({
      url: `/api/v1/ai/agents/${id}/versions/diff`,
      method: "get",
      params: { base, target },
    });
  }

  /** 版本快照详情 */
  static versionDetail(id: number, versionNo: number) {
    return request<AgentVersionDetail>({
      url: `/api/v1/ai/agents/${id}/versions/${versionNo}`,
      method: "get",
    });
  }

  /** 回滚到历史版本（管理端，返回新版本号） */
  static rollback(id: number, versionNo: number) {
    return request<VersionResult>({
      url: `/api/v1/ai/agents/${id}/versions/${versionNo}/rollback`,
      method: "post",
    });
  }

  // ==================== 评测 ====================

  /** 创建评测集（管理端） */
  static createEvalDataset(agentId: number, data: EvalDatasetCreateForm) {
    return request<EvalDatasetResult>({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets`,
      method: "post",
      // 后端 EvalDatasetCreate 为纯 BaseModel，wire 字段为 dataset_type
      data: {
        name: data.name,
        description: data.description ?? "",
        dataset_type: data.datasetType,
      },
    });
  }

  /** 评测集列表（管理端） */
  static listEvalDatasets(agentId: number) {
    return request<EvalDatasetResult[]>({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets`,
      method: "get",
    });
  }

  /** 更新评测集（管理端） */
  static updateEvalDataset(agentId: number, datasetId: number, data: EvalDatasetUpdateForm) {
    return request<EvalDatasetResult>({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets/${datasetId}`,
      method: "patch",
      data,
    });
  }

  /** 删除评测集（管理端） */
  static deleteEvalDataset(agentId: number, datasetId: number) {
    return request({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets/${datasetId}`,
      method: "delete",
    });
  }

  /** 创建评测样本（管理端） */
  static createEvalSample(agentId: number, datasetId: number, data: EvalSampleCreateForm) {
    return request<EvalSampleResult>({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets/${datasetId}/samples`,
      method: "post",
      // 后端 EvalSampleCreate 为纯 BaseModel，wire 字段为 snake_case
      data: {
        dataset_id: data.datasetId,
        task_goal: data.taskGoal,
        allowed_input: data.allowedInput ?? null,
        tools: data.tools ?? null,
        expected_process: data.expectedProcess ?? null,
        expected_result: data.expectedResult ?? null,
        forbidden_behavior: data.forbiddenBehavior ?? null,
        risk_level: data.riskLevel ?? "low",
      },
    });
  }

  /** 评测样本列表（管理端） */
  static listEvalSamples(agentId: number, datasetId: number) {
    return request<EvalSampleResult[]>({
      url: `/api/v1/ai/agents/${agentId}/eval/datasets/${datasetId}/samples`,
      method: "get",
    });
  }

  /** 更新评测样本（管理端） */
  static updateEvalSample(agentId: number, sampleId: number, data: EvalSampleUpdateForm) {
    return request<EvalSampleResult>({
      url: `/api/v1/ai/agents/${agentId}/eval/samples/${sampleId}`,
      method: "patch",
      // 后端 EvalSampleUpdate 为纯 BaseModel，wire 字段为 snake_case
      data: {
        task_goal: data.taskGoal,
        allowed_input: data.allowedInput,
        tools: data.tools,
        expected_process: data.expectedProcess,
        expected_result: data.expectedResult,
        forbidden_behavior: data.forbiddenBehavior,
        risk_level: data.riskLevel,
      },
    });
  }

  /** 删除评测样本（管理端） */
  static deleteEvalSample(agentId: number, sampleId: number) {
    return request({
      url: `/api/v1/ai/agents/${agentId}/eval/samples/${sampleId}`,
      method: "delete",
    });
  }

  /** 手动触发评测（回归集，管理端） */
  static runEval(agentId: number) {
    return request<Record<string, unknown>>({
      url: `/api/v1/ai/agents/${agentId}/eval/runs`,
      method: "post",
    });
  }

  /** 评测执行记录（分页，管理端，支持按评测集过滤） */
  static listEvalRuns(agentId: number, query?: EvalRunQuery) {
    return request<PageResult<EvalRunResult[]>>({
      url: `/api/v1/ai/agents/${agentId}/eval/runs`,
      method: "get",
      params: query,
    });
  }

  // ==================== A2A 端点管理 ====================

  /** 注册外部 A2A 端点（管理端） */
  static createEndpoint(data: EndpointCreateForm) {
    return request<EndpointResult>({
      url: "/api/v1/ai/a2a/endpoints",
      method: "post",
      data,
    });
  }

  /** 更新 A2A 端点（管理端） */
  static updateEndpoint(id: number, data: EndpointUpdateForm) {
    return request<EndpointResult>({
      url: `/api/v1/ai/a2a/endpoints/${id}`,
      method: "patch",
      data,
    });
  }

  /** 删除 A2A 端点（管理端） */
  static deleteEndpoint(id: number) {
    return request({
      url: `/api/v1/ai/a2a/endpoints/${id}`,
      method: "delete",
    });
  }

  /** A2A 端点分页列表（管理端） */
  static listEndpoints(query?: EndpointPageQuery) {
    return request<PageResult<EndpointResult[]>>({
      url: "/api/v1/ai/a2a/endpoints",
      method: "get",
      params: query,
    });
  }

  /** 刷新端点 Agent Card（管理端，返回刷新后的 Card） */
  static refreshEndpointCard(id: number) {
    return request<Record<string, unknown>>({
      url: `/api/v1/ai/a2a/endpoints/${id}/refresh-card`,
      method: "post",
    });
  }
}

export default AiAgentAPI;
export { MANAGE_PERMISSION };
