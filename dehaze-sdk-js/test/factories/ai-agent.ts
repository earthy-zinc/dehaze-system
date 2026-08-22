import { pageQuery, uniqueCode, uniqueName } from "./common";
import type {
  AgentCreateForm,
  AgentCopyForm,
  AgentMcpForm,
  AgentPageQuery,
  AgentPublishForm,
  AgentSkillsForm,
  AgentStatusForm,
  AgentSubAgentsForm,
  AgentUpdateForm,
  EvalDatasetCreateForm,
  EvalSampleCreateForm,
} from "../../src/api/ai-agent/model";

/** Agent 创建表单工厂（agent_code 前缀 test_agent_） */
export const createAgentForm = (overrides?: Partial<AgentCreateForm>): AgentCreateForm => ({
  agentCode: uniqueCode("test_agent"),
  name: uniqueName("测试Agent"),
  description: "测试智能体",
  systemPrompt: "你是一个图像去雾测试助手。",
  modelId: "gpt-4o-mini",
  reasoningMode: "auto",
  config: { maxSteps: 10, tokenBudget: 8000 },
  isSubagent: false,
  isTeam: false,
  isExposed: false,
  sortOrder: 0,
  status: 1,
  ...overrides,
});

/** Agent 更新表单工厂 */
export const createAgentUpdateForm = (overrides?: Partial<AgentUpdateForm>): AgentUpdateForm => ({
  name: uniqueName("更新Agent"),
  ...overrides,
});

/** Agent 分页查询参数工厂 */
export const createAgentQuery = (overrides?: Partial<AgentPageQuery>): AgentPageQuery =>
  pageQuery<AgentPageQuery>({ ...overrides });

/** 复制表单工厂 */
export const createAgentCopyForm = (overrides?: Partial<AgentCopyForm>): AgentCopyForm => ({
  agentCode: uniqueCode("test_agent_copy"),
  ...overrides,
});

/** 启停表单工厂 */
export const createAgentStatusForm = (overrides?: Partial<AgentStatusForm>): AgentStatusForm => ({
  status: 0,
  ...overrides,
});

/** Skills 表单工厂（覆盖式） */
export const createAgentSkillsForm = (overrides?: Partial<AgentSkillsForm>): AgentSkillsForm => ({
  skills: ["test_skill_report"],
  ...overrides,
});

/** MCP 命名空间表单工厂（覆盖式） */
export const createAgentMcpForm = (overrides?: Partial<AgentMcpForm>): AgentMcpForm => ({
  mcpNamespaces: ["test_ns"],
  ...overrides,
});

/** 子 Agent 表单工厂（覆盖式） */
export const createAgentSubAgentsForm = (
  overrides?: Partial<AgentSubAgentsForm>
): AgentSubAgentsForm => ({
  subagents: [],
  ...overrides,
});

/** 发布表单工厂 */
export const createAgentPublishForm = (
  overrides?: Partial<AgentPublishForm>
): AgentPublishForm => ({
  changeNote: `发布测试-${Date.now()}`,
  ...overrides,
});

/** 评测集创建表单工厂（name 前缀 test_eval_） */
export const createEvalDatasetForm = (
  overrides?: Partial<EvalDatasetCreateForm>
): EvalDatasetCreateForm => ({
  name: uniqueName("test_eval"),
  description: "测试评测集",
  datasetType: "dev",
  ...overrides,
});

/** 评测样本创建表单工厂 */
export const createEvalSampleForm = (
  overrides?: Partial<EvalSampleCreateForm>
): EvalSampleCreateForm => ({
  datasetId: 0, // 由调用方回填
  taskGoal: "对测试输入给出合理的去雾处理方案",
  allowedInput: "text",
  tools: ["file_read"],
  expectedResult: "返回处理方案",
  riskLevel: "low",
  ...overrides,
});
