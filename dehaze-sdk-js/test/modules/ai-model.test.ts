import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiConversationAPI, AiModelAPI, AiProviderAPI } from "../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createAiModelForm,
  createAiModelQuery,
  createAiModelUpdateForm,
  createModelPriceForm,
  createModelPriceQuery,
  createModelPriceUpdateForm,
} from "#/factories/ai-model";
import { createProviderForm } from "#/factories/ai-provider";

/**
 * AI 模型管理（T-MF-001~008 模型 CRUD + T-MF-197~209 价格版本与权限/不可变约束）
 *
 * 用例编号对齐 dehaze-doc/03-模块设计/基础模块/AI模型管理/测试用例.md。
 * 环境：dehaze-python（管理端接口需 ai:model:manage 权限，需管理员账号）。
 * 数据前缀 test_model_ / test_prov_，beforeAll/afterAll 尽力清理（模型为逻辑删除）。
 */
describe("AI 模型管理 - AiModelAPI (T-MF-001~008,197~209)", () => {
  let providerId: number;

  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    // 先建测试供应商，拿到真实 providerId 供模型与价格关联
    const provider = await AiProviderAPI.createProvider(createProviderForm());
    providerId = provider.id;
  });

  afterAll(async () => {
    await login(USERS.ADMIN.username).catch(() => {});
    try {
      await AiProviderAPI.deleteProvider(providerId);
    } catch (e) {
      console.warn("清理供应商失败:", (e as Error)?.message ?? e);
    }
  });

  describe("GET /api/v1/ai/models - 模型列表（管理员）", () => {
    test("T-MF-001 正向：分页结构含 list/total", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiModelAPI.listModels(createAiModelQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        const model = result.list[0]!;
        expect(model.modelId).toBeTruthy();
        expect(model.providerId).toBeGreaterThan(0);
        expect(typeof model.displayName).toBe("string");
      }
    });

    test("T-MF-204 负向：普通用户访问管理列表 → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiModelAPI.listModels(createAiModelQuery()), ["A0301"]);
      await login(USERS.ADMIN.username);
    });
  });

  describe("POST /api/v1/ai/models - 新增模型（管理员）", () => {
    test("T-MF-002 正向：创建模型返回完整结构", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      const result = await AiModelAPI.createModel(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.modelId).toBe(form.modelId);
      expect(result.providerId).toBe(providerId);
      expect(result.displayName).toBe(form.displayName);
      expect(result.supportsToolCall).toBe(1);
      expect(result.status).toBe(1);

      await cleanupModel(form.modelId);
    });

    test("T-MF-003 负向：同 model_id+provider 重复创建 → A0501", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      await AiModelAPI.createModel(form);
      await expectBizError(AiModelAPI.createModel(form), ["A0501"]);
      await cleanupModel(form.modelId);
    });

    test("T-MF-204 负向：普通用户创建模型 → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiModelAPI.createModel(createAiModelForm({ providerId })), ["A0301"]);
      await login(USERS.ADMIN.username);
    });
  });

  describe("PUT /api/v1/ai/models/{modelId} - 更新模型（管理员）", () => {
    let modelId: string;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      await AiModelAPI.createModel(form);
      modelId = form.modelId;
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username).catch(() => {});
      await cleanupModel(modelId);
    });

    test("T-MF-004 正向：更新 displayName/上下文上限（费率已剥离至价格版本）", async () => {
      await login(USERS.ADMIN.username);
      const updated = await AiModelAPI.updateModel(
        modelId,
        createAiModelUpdateForm({ maxContextTokens: 8192 })
      );
      expect(updated.modelId).toBe(modelId);
      expect(updated.maxContextTokens).toBe(8192);
    });

    test("T-MF-208 边界：更新不存在的模型 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(
        AiModelAPI.updateModel("not_exist_model_xxx", createAiModelUpdateForm()),
        ["A0401"]
      );
    });

    test("T-MF-207 负向：更新 model_type（创建后不可改）→ A0502", async () => {
      await login(USERS.ADMIN.username);
      // modelType 不出现在更新表单中，此处显式构造请求体验证后端不可变约束
      await expectBizError(AiModelAPI.updateModel(modelId, { modelType: "embedding" } as never), [
        "A0502",
      ]);
    });
  });

  describe("GET /api/v1/ai/models/enabled - 启用模型列表", () => {
    test("T-MF-013 新建模型出现在启用列表", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId, status: 1, vipLevel: 0 });
      const created = await AiModelAPI.createModel(form);
      expect(created.id).toBeGreaterThan(0);

      const enabled = await AiModelAPI.listEnabledModels();
      expect(Array.isArray(enabled)).toBe(true);
      expect(enabled.find((m) => m.modelId === form.modelId)).toBeDefined();

      await cleanupModel(form.modelId);
    });

    test("T-MF-209 正向：按 modelType 筛选启用模型", async () => {
      await login(USERS.ADMIN.username);
      const enabled = await AiModelAPI.listEnabledModels("chat");
      expect(Array.isArray(enabled)).toBe(true);
      enabled.forEach((m) => expect(m.modelType).toBe("chat"));
    });
  });

  describe("DELETE /api/v1/ai/models/{modelId} - 删除模型（管理员）", () => {
    test("T-MF-008 负向：存在活跃会话使用该模型 → A0504", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      const created = await AiModelAPI.createModel(form);
      expect(created.id).toBeGreaterThan(0);

      // 创建绑定该模型的活跃会话，使删除被拒
      const conv = await bindConversation(form.modelId);
      expect(conv).toBeGreaterThan(0);

      await expectBizError(AiModelAPI.deleteModel(form.modelId), ["A0504"]);

      // 软删会话后可删除模型
      await AiConversationAPI.deleteConversation(conv);
      await AiModelAPI.deleteModel(form.modelId);
      const models = await AiModelAPI.listModels(createAiModelQuery());
      expect(models.list.find((m) => m.modelId === form.modelId)).toBeUndefined();
    });

    test("T-MF-208 边界：删除不存在的模型 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(AiModelAPI.deleteModel("not_exist_model_yyy"), ["A0401"]);
    });
  });

  describe("模型类型与维度扩展（modelType/dimension）", () => {
    test("T-MF-205 正向：创建 embedding 模型含 modelType/dimension", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ modelType: "embedding", dimension: 1024 });
      const result = await AiModelAPI.createModel(form);
      expect(result.modelType).toBe("embedding");
      expect(result.dimension).toBe(1024);
      await cleanupModel(result.modelId);
    });

    test("T-MF-206 负向：embedding 模型缺 dimension → A0400", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(AiModelAPI.createModel(createAiModelForm({ modelType: "embedding" })), [
        "A0400",
      ]);
    });

    test("正向：按 modelType 筛选模型列表", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiModelAPI.listModels(createAiModelQuery({ modelType: "chat" }));
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((m) => expect(m.modelType).toBe("chat"));
    });
  });

  describe("模型用户售价版本（/prices）", () => {
    let modelId: string;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      await AiModelAPI.createModel(form);
      modelId = form.modelId;
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username).catch(() => {});
      await cleanupPrices(modelId);
      await cleanupModel(modelId);
    });

    test("T-MF-197/203 正向：新增价格版本返回版本号与档位明细", async () => {
      await login(USERS.ADMIN.username);
      const form = createModelPriceForm(modelId, providerId);
      const result = await AiModelAPI.createPrice(modelId, form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.modelId).toBe(modelId);
      expect(result.providerId).toBe(providerId);
      expect(result.priceVersion).toBe(1);
      expect(result.unit).toBe("credits_per_million");
      expect(result.status).toBe(1);
      expect(result.details).toHaveLength(3);
      expect(result.details[0]!.tokenType).toBe("input");
      // 后端 Decimal 序列化为字符串，前端计算前需转 Number
      expect(Number(result.details[0]!.unitPrice)).toBe(2);
    });

    test("T-MF-198 正向：同模型再新增生成新版本（版本号递增）", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiModelAPI.createPrice(
        modelId,
        createModelPriceForm(modelId, providerId, { unit: "credits_per_million" })
      );
      expect(result.priceVersion).toBe(2);
    });

    test("正向：价格版本分页列表", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiModelAPI.listPrices(modelId, createModelPriceQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(result.total).toBeGreaterThanOrEqual(2);
      result.list.forEach((p) => expect(p.modelId).toBe(modelId));
    });

    test("T-MF-200 正向：更新价格版本状态", async () => {
      await login(USERS.ADMIN.username);
      const prices = await AiModelAPI.listPrices(modelId, createModelPriceQuery());
      const target = prices.list[0]!;
      const updated = await AiModelAPI.updatePrice(
        modelId,
        target.id,
        createModelPriceUpdateForm()
      );
      expect(updated.id).toBe(target.id);
      expect(updated.status).toBe(0);
    });

    test("T-MF-202 负向：更新不存在的价格版本 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(
        AiModelAPI.updatePrice(modelId, 99999999, createModelPriceUpdateForm()),
        ["A0401"]
      );
    });

    test("T-MF-204 负向：普通用户新增价格版本 → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(
        AiModelAPI.createPrice(modelId, createModelPriceForm(modelId, providerId)),
        ["A0301"]
      );
      await login(USERS.ADMIN.username);
    });

    test("T-MF-201 清理：删除价格版本后列表为空", async () => {
      await login(USERS.ADMIN.username);
      await cleanupPrices(modelId);
      const prices = await AiModelAPI.listPrices(modelId, createModelPriceQuery());
      expect(prices.total).toBe(0);
    });
  });
});

/** 创建绑定指定模型的活跃会话，返回会话 id（失败返回 0） */
async function bindConversation(modelId: string): Promise<number> {
  try {
    const conv = await AiConversationAPI.createConversation({ model: modelId });
    return conv.id;
  } catch {
    return 0;
  }
}

/** 清理模型（忽略已不存在/已删除的错误） */
async function cleanupModel(modelId: string): Promise<void> {
  await AiModelAPI.deleteModel(modelId).catch(() => {});
}

/** 清理模型的全部价格版本 */
async function cleanupPrices(modelId: string): Promise<void> {
  const prices = await AiModelAPI.listPrices(modelId, createModelPriceQuery()).catch(() => null);
  for (const p of prices?.list ?? []) {
    await AiModelAPI.deletePrice(modelId, p.id).catch(() => {});
  }
}
