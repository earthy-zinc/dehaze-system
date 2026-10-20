import { beforeEach, describe, expect, it, vi } from "vitest";
import AiAgentAPI from "@/api/ai-agent";
import type { AgentConfigDefaults } from "@/api/ai-agent/model";

/**
 * 推理参数系统默认值接口契约（GET /api/v1/ai/agents/config-defaults，登录即可读）。
 *
 * python 侧 `AgentConfigDefaults`（models/schema/ai_agent.py）继承 OrmResult → 输出 camelCase；
 * 九个字段与代码常量 REASONING_DEFAULTS（service/ai/strategies/agent_config_resolver.py）同值，
 * Agent 配置表单的"空值继承系统默认"提示依赖本接口，前端不得硬编码。
 *
 * 下方 fixture 按 python 常量逐项写成并显式标注类型：字段名/类型若与后端漂移，
 * `npx tsc --noEmit` 会直接报错（vitest 本身不做类型检查）。
 */

const { requestMock } = vi.hoisted(() => ({ requestMock: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: requestMock }));

/** 与 python REASONING_DEFAULTS 常量逐项一致的响应样本 */
const configDefaults: AgentConfigDefaults = {
  maxStepsReact: 20,
  maxStepsPlan: 30,
  maxStepsReflexion: 15,
  maxIterationsReflexion: 3,
  reflexionThreshold: 0.8,
  maxParallel: 5,
  toolTimeout: 60,
  tokenBudget: 500000,
  retryMax: 2,
};

describe("AiAgentAPI.getConfigDefaults", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockResolvedValue(configDefaults);
  });

  it("请求 GET /api/v1/ai/agents/config-defaults", async () => {
    await AiAgentAPI.getConfigDefaults();
    expect(requestMock).toHaveBeenCalledWith({
      url: "/api/v1/ai/agents/config-defaults",
      method: "get",
    });
  });

  it("九个 camelCase 默认值原样透传", async () => {
    const result = await AiAgentAPI.getConfigDefaults();
    expect(Object.keys(result)).toHaveLength(9);
    expect(result).toEqual(configDefaults);
    expect(result.reflexionThreshold).toBe(0.8);
    expect(result.tokenBudget).toBe(500000);
  });
});
