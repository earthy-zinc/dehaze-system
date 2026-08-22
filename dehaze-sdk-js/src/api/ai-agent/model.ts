import { EnabledStatus, PageQuery } from "@/types";

// ==================== Agent 推理参数 / 护栏配置 ====================

/** 推理范式 */
export type ReasoningMode = "auto" | "direct" | "react" | "plan_execute" | "reflexion";

/** 护栏规则（内置规则仅暴露开关，不支持动态新增规则类型） */
export interface GuardrailRule {
  enabled: boolean;
}

/** 护栏配置 */
export interface GuardrailConfig {
  promptInjection?: GuardrailRule;
  unauthorizedAccess?: GuardrailRule;
  sensitiveTopic?: GuardrailRule;
  piiMask?: GuardrailRule;
  factCheck?: GuardrailRule;
  formatCheck?: GuardrailRule;
}

/** Agent 推理参数配置（空值继承系统默认） */
export interface AgentConfig {
  /** 最大推理步数 */
  maxSteps?: number | null;
  /** 单会话 Token 预算上限 */
  tokenBudget?: number | null;
  /** 并行子任务最大数 */
  maxParallel?: number | null;
  /** 单工具调用超时（秒） */
  toolTimeout?: number | null;
  /** 工具调用失败最大重试次数 */
  retryMax?: number | null;
  /** Reflexion 质量达标阈值（0-1） */
  reflexionThreshold?: number | null;
  /** LLM 温度参数 */
  temperature?: number | null;
  guardrails?: GuardrailConfig | null;
}

// ==================== Agent 主表 ====================

/** 创建 Agent 表单 */
export interface AgentCreateForm {
  /** Agent 唯一编码 */
  agentCode: string;
  /** Agent 显示名称 */
  name: string;
  /** Agent 描述 */
  description?: string;
  /** 系统提示词（Markdown） */
  systemPrompt?: string | null;
  /** 关联模型标识 */
  modelId: string;
  /** 推理范式 */
  reasoningMode?: ReasoningMode;
  /** 推理参数配置 */
  config?: AgentConfig | null;
  /** 是否可作为子 Agent */
  isSubagent?: boolean;
  /** 是否为 Team 团队 */
  isTeam?: boolean;
  /** 是否对外暴露为 A2A 子 Agent */
  isExposed?: boolean;
  /** 文件系统权限规则 */
  permissions?: Array<Record<string, unknown>> | null;
  /** 排序序号 */
  sortOrder?: number;
  /** 状态：1-启用，0-禁用 */
  status?: EnabledStatus;
}

/** 更新 Agent 表单 */
export interface AgentUpdateForm {
  name?: string;
  description?: string;
  systemPrompt?: string | null;
  modelId?: string;
  reasoningMode?: ReasoningMode;
  config?: AgentConfig | null;
  isSubagent?: boolean;
  isTeam?: boolean;
  isExposed?: boolean;
  permissions?: Array<Record<string, unknown>> | null;
  sortOrder?: number;
}

/** Agent 列表项 */
export interface AgentListItem {
  id: number;
  /** Agent 唯一编码 */
  agentCode: string;
  name: string;
  description: string;
  /** 关联模型标识 */
  modelId: string;
  /** 推理范式 */
  reasoningMode: string;
  isSubagent: EnabledStatus;
  isTeam: EnabledStatus;
  isExposed: EnabledStatus;
  status: EnabledStatus;
  sortOrder: number;
  createTime?: string | null;
}

/** 子 Agent 项（详情内嵌） */
export interface SubAgentItem {
  /** 子 Agent ID */
  agentId: number;
  agentName: string;
  agentCode: string;
  description: string;
  /** 外部 A2A 端点 ID（NULL 为本地） */
  endpointId?: number | null;
  priority: number;
}

/** Agent 详情 */
export interface AgentDetail extends AgentListItem {
  systemPrompt?: string | null;
  config?: AgentConfig | null;
  permissions?: Array<Record<string, unknown>> | null;
  /** 关联的 Skill 名称 */
  skills: string[];
  /** 关联的 MCP 命名空间 */
  mcpNamespaces: string[];
  /** 关联的子 Agent */
  subagents: SubAgentItem[];
}

/** Agent 分页查询参数 */
export interface AgentPageQuery extends PageQuery {
  /** 关键字（按名称/编码模糊搜索） */
  keyword?: string;
  /** 状态过滤 */
  status?: EnabledStatus;
}

// ==================== 关联设置（覆盖式） ====================

/** 设置 Skills 表单（覆盖式更新） */
export interface AgentSkillsForm {
  /** Skill 名称列表 */
  skills: string[];
}

/** 设置 MCP 命名空间表单（覆盖式更新） */
export interface AgentMcpForm {
  mcpNamespaces: string[];
}

/** 子 Agent 关联项 */
export interface AgentSubAgentItem {
  /** 子 Agent ID（本地影子记录） */
  agentId: number;
  /** 外部 A2A 端点 ID（NULL 为本地子 Agent） */
  endpointId?: number | null;
  /** 优先级（数字越小越优先） */
  priority?: number;
}

/** 设置子 Agent 表单（覆盖式更新） */
export interface AgentSubAgentsForm {
  subagents: AgentSubAgentItem[];
}

// ==================== 启停 / 复制 / 测试 ====================

/** 启停 Agent 表单 */
export interface AgentStatusForm {
  /** 目标状态：1-启用，0-禁用 */
  status: EnabledStatus;
}

/** 复制 Agent 表单 */
export interface AgentCopyForm {
  /** 新 Agent 唯一编码 */
  agentCode: string;
}

/** 测试 Agent 表单 */
export interface AgentTestForm {
  /** 测试消息 */
  message: string;
  /** 会话级配置覆盖 */
  conversationConfig?: AgentConfig | null;
}

/** Agent 测试预览结果（结构因 Agent 而异） */
export type AgentTestResult = Record<string, unknown>;

// ==================== 版本管理 ====================

/** 发布 Agent 表单 */
export interface AgentPublishForm {
  /** 变更说明 */
  changeNote?: string;
}

/** Agent 版本结果 */
export interface AgentVersionResult {
  id: number;
  /** 关联 Agent ID */
  agentId: number;
  /** 版本号 */
  versionNo: number;
  /** 版本状态：1-草稿，2-已发布 */
  status: 1 | 2;
  /** 变更说明 */
  changeNote?: string | null;
  /** 操作人 ID */
  operatorId?: number | null;
  createTime?: string | null;
}

/** Agent 版本详情（含配置快照） */
export interface AgentVersionDetail extends AgentVersionResult {
  /** 配置快照 */
  snapshot: Record<string, unknown>;
}

/** 版本发布/回滚结果 */
export interface VersionResult {
  /** 版本号（后端返回 snake_case） */
  version_no: number;
}

// ==================== 评测 ====================

/** 评测集类型：dev-开发，regression-回归，heldout-保留 */
export type EvalDatasetType = "dev" | "regression" | "heldout";

/** 风险等级 */
export type EvalRiskLevel = "low" | "medium" | "high";

/** 创建评测集表单 */
export interface EvalDatasetCreateForm {
  name: string;
  description?: string;
  datasetType: EvalDatasetType;
}

/** 更新评测集表单 */
export interface EvalDatasetUpdateForm {
  name?: string;
  description?: string;
}

/** 评测集结果 */
export interface EvalDatasetResult {
  id: number;
  /** 关联 Agent ID */
  agentId: number;
  name: string;
  description: string;
  datasetType: string;
  createTime?: string | null;
}

/** 创建评测样本表单 */
export interface EvalSampleCreateForm {
  /** 关联评测集 ID */
  datasetId: number;
  /** 任务目标 */
  taskGoal: string;
  /** 允许输入 */
  allowedInput?: string | null;
  /** 可用工具 */
  tools?: string[] | null;
  /** 期望过程 */
  expectedProcess?: string | null;
  /** 期望结果 */
  expectedResult?: string | null;
  /** 禁止行为 */
  forbiddenBehavior?: string | null;
  /** 风险等级 */
  riskLevel?: EvalRiskLevel;
}

/** 更新评测样本表单 */
export interface EvalSampleUpdateForm {
  taskGoal?: string;
  allowedInput?: string | null;
  tools?: string[] | null;
  expectedProcess?: string | null;
  expectedResult?: string | null;
  forbiddenBehavior?: string | null;
  riskLevel?: EvalRiskLevel;
}

/** 评测样本结果 */
export interface EvalSampleResult {
  id: number;
  /** 关联评测集 ID */
  datasetId: number;
  taskGoal: string;
  allowedInput?: string | null;
  tools?: string[] | null;
  expectedProcess?: string | null;
  expectedResult?: string | null;
  forbiddenBehavior?: string | null;
  riskLevel: string;
  createTime?: string | null;
}

/** 评测运行结果 */
export interface EvalRunResult {
  id: number;
  /** 关联 Agent ID */
  agentId: number;
  /** 关联评测集 ID */
  datasetId: number;
  /** 触发方式：manual/publish */
  triggerType: string;
  /** 执行状态：1-执行中，2-通过，3-失败 */
  status: 1 | 2 | 3;
  /** 四维评分聚合 */
  scoreSummary?: Record<string, unknown> | null;
  /** 样本明细 */
  results?: Array<Record<string, unknown>> | null;
  /** 创建人 ID */
  createBy?: number | null;
  createTime?: string | null;
}

/** 评测运行列表查询参数 */
export interface EvalRunQuery extends PageQuery {
  /** 评测集过滤 */
  datasetId?: number;
}

/** 手动触发评测结果 */
export type EvalRunResultPayload = Record<string, unknown>;

// ==================== A2A 端点管理 ====================

/** A2A 端点认证方式 */
export type A2AAuthType = "apiKey" | "http" | "oauth2" | "openIdConnect" | "mutualTLS";

/** 注册外部 A2A 端点表单 */
export interface EndpointCreateForm {
  /** 端点名称 */
  name: string;
  /** Agent Card 地址 */
  agentCardUrl?: string | null;
  /** A2A 端点地址 */
  baseUrl: string;
  /** 认证方式 */
  authType?: A2AAuthType;
  /** 凭证密文（AES 加密后 base64） */
  credential?: string | null;
  /** 状态：1-启用，0-禁用 */
  status?: EnabledStatus;
}

/** 更新 A2A 端点表单 */
export interface EndpointUpdateForm {
  name?: string;
  agentCardUrl?: string | null;
  authType?: A2AAuthType;
  credential?: string | null;
  status?: EnabledStatus;
}

/** A2A 端点结果 */
export interface EndpointResult {
  id: number;
  name: string;
  agentCardUrl?: string | null;
  /** A2A 端点地址 */
  baseUrl: string;
  authType: string;
  /** 缓存的 Agent Card */
  agentCard?: Record<string, unknown> | null;
  status: EnabledStatus;
  createTime?: string | null;
}

/** A2A 端点分页查询参数 */
export interface EndpointPageQuery extends PageQuery {
  /** 关键字（按名称/地址模糊搜索） */
  keyword?: string;
  /** 状态过滤 */
  status?: EnabledStatus;
}
