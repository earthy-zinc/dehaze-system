import { beforeAll, describe, expect, test } from "vitest";
import { AiAgentAPI, AiObservabilityAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import { createAgentQuery } from "#/factories/ai-agent";

/**
 * AI 域跨端对齐补测（对 dehaze-python 运行）：
 *
 * 1. `GET /api/v1/ai/agents/config-defaults`：代码常量 REASONING_DEFAULTS
 *    （service/ai/strategies/agent_config_resolver.py）的对外只读契约，九个 camelCase 字段与常量同值；
 *    Agent 配置表单的"空值继承系统默认"提示依赖它，前端硬编码即漂移。
 * 2. 分页上限：python `BasePageQuery`（models/schema/common.py）为 `pageSize ge=1, le=100`，
 *    越界拒绝 A0400——与 order-edge 同型的负向口径。
 */
describe("AI 域跨端对齐补测", () => {
  beforeAll(async () => {
    await login(USERS.ADMIN.username);
  });

  describe("GET /api/v1/ai/agents/config-defaults - 推理参数系统默认值", () => {
    test("九个字段与 REASONING_DEFAULTS 常量同值", async () => {
      const defaults = await AiAgentAPI.getConfigDefaults();

      expect(defaults).toEqual({
        maxStepsReact: 20,
        maxStepsPlan: 30,
        maxStepsReflexion: 15,
        maxIterationsReflexion: 3,
        reflexionThreshold: 0.8,
        maxParallel: 5,
        toolTimeout: 60,
        tokenBudget: 500000,
        retryMax: 2,
      });
    });

    test("登录即可读：普通用户无需 ai:agent:manage 也能获取", async () => {
      await login(USERS.USER.username);

      const defaults = await AiAgentAPI.getConfigDefaults();
      expect(typeof defaults.maxStepsReact).toBe("number");

      await login(USERS.ADMIN.username);
    });
  });

  describe("分页上限（A0400）", () => {
    test("Agent 列表 pageSize=101 拒绝（契约 le=100）", async () => {
      await login(USERS.ROOT.username);
      await expectBizError(AiAgentAPI.list(createAgentQuery({ pageSize: 101 })), ["A0400"]);
    });

    test("过程链检索 pageSize=101 拒绝", async () => {
      // root 免权限校验，聚焦分页契约本身（免却审计权限在角色上的差异）
      await login(USERS.ROOT.username);
      await expectBizError(AiObservabilityAPI.getTraces({ pageNum: 1, pageSize: 101 }), ["A0400"]);
    });
  });
});
