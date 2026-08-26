import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiConversationAPI, AiProviderAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import { createAiModelForm, createAiModelQuery } from "#/factories/ai-model";
import { createProviderForm } from "#/factories/ai-provider";

/**
 * AI 模型管理（T-MF-001~012 核心正向 + 关键负向）
 *
 * 环境：dehaze-python（管理端 4 接口需 ai:model:manage 权限，需管理员账号）。
 * 数据前缀 test_model_ / test_prov_，afterAll 尽力清理（软删除）。
 */
describe("AI 模型管理 - AiConversationAPI (T-MF-001~012)", () => {
  let providerId: number;

  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    // 先建测试供应商，拿到真实 providerId 供模型关联
    const provider = await AiProviderAPI.createProvider(createProviderForm());
    providerId = provider.id;
  });

  afterAll(async () => {
    // 恢复管理员会话后再清理供应商
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
      const result = await AiConversationAPI.getModels(createAiModelQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        const model = result.list[0]!;
        expect(model.modelId).toBeTruthy();
        expect(model.providerId).toBeGreaterThan(0);
        expect(typeof model.displayName).toBe("string");
      }
    });

    test("T-MF-009 负向：普通用户访问管理列表应 403", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiConversationAPI.getModels(createAiModelQuery()), ["A0301"]);
      await login(USERS.ADMIN.username);
    });
  });

  describe("POST /api/v1/ai/models - 新增模型（管理员）", () => {
    test("T-MF-002 正向：创建模型返回完整结构", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      const result = await AiConversationAPI.createModel(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.modelId).toBe(form.modelId);
      expect(result.providerId).toBe(providerId);
      expect(result.displayName).toBe(form.displayName);
      expect(result.supportsToolCall).toBe(1);
      expect(result.status).toBe(1);

      // 清理：先软删绑定会话再删模型
      const convId = await bindConversation(form.modelId);
      if (convId) await AiConversationAPI.deleteConversation(convId).catch(() => {});
      await cleanupModel(form.modelId);
    });

    test("T-MF-003 负向：同 model_id+provider 重复创建 → A0501", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      await AiConversationAPI.createModel(form);
      await expectBizError(AiConversationAPI.createModel(form), ["A0501"]);
      await cleanupModel(form.modelId);
    });

    test("T-MF-092 负向：普通用户创建模型应 403", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiConversationAPI.createModel(createAiModelForm({ providerId })), [
        "A0403",
        "A0302",
        "A0301",
      ]);
      await login(USERS.ADMIN.username);
    });
  });

  describe("PUT /api/v1/ai/models/{modelId} - 更新模型（管理员）", () => {
    let modelId: string;

    beforeAll(async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      await AiConversationAPI.createModel(form);
      modelId = form.modelId;
    });

    afterAll(async () => {
      await login(USERS.ADMIN.username).catch(() => {});
      await cleanupModel(modelId);
    });

    test("T-MF-004 正向：更新 displayName/上下文上限（费率已剥离至价格版本，见 AI模型管理 §2.12）", async () => {
      await login(USERS.ADMIN.username);
      const displayName = `更新模型_${Date.now()}`;
      const updated = await AiConversationAPI.updateModel(modelId, {
        displayName,
        maxContextTokens: 8192,
      });
      expect(updated.displayName).toBe(displayName);
      expect(updated.maxContextTokens).toBe(8192);
    });

    test("T-MF-005/058 边界：更新不存在的模型 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(
        AiConversationAPI.updateModel("not_exist_model_xxx", { displayName: "x" }),
        ["A0401"]
      );
    });
  });

  describe("GET /api/v1/ai/models/enabled - 启用模型列表", () => {
    test("T-MF-001 新建模型出现在启用列表", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId, status: 1, vipLevel: 0 });
      const created = await AiConversationAPI.createModel(form);
      expect(created.id).toBeGreaterThan(0);

      const enabled = await AiConversationAPI.getEnabledModels();
      expect(Array.isArray(enabled)).toBe(true);
      const found = enabled.find((m) => m.modelId === form.modelId);
      expect(found).toBeDefined();

      await cleanupModel(form.modelId);
    });
  });

  describe("DELETE /api/v1/ai/models/{modelId} - 删除模型（管理员）", () => {
    test("T-MF-008 负向：存在活跃会话使用该模型 → A0504", async () => {
      await login(USERS.ADMIN.username);
      const form = createAiModelForm({ providerId });
      const created = await AiConversationAPI.createModel(form);
      expect(created.id).toBeGreaterThan(0);

      // 创建绑定该模型的活跃会话，使删除被拒
      const conv = await bindConversation(form.modelId);
      expect(conv).toBeGreaterThan(0);

      await expectBizError(AiConversationAPI.deleteModel(form.modelId), ["A0504"]);

      // 软删会话（deleted=1）后可删除模型
      await AiConversationAPI.deleteConversation(conv);
      await AiConversationAPI.deleteModel(form.modelId);
      const models = await AiConversationAPI.getModels(createAiModelQuery());
      expect(models.list.find((m) => m.modelId === form.modelId)).toBeUndefined();
    });

    test("T-MF-005 边界：删除不存在的模型 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(AiConversationAPI.deleteModel("not_exist_model_yyy"), ["A0401"]);
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
  await AiConversationAPI.deleteModel(modelId).catch(() => {});
}

/**
 * 模型类型与维度扩展（AI模型管理 §2.2 model_type/dimension）。
 *
 * 后端尚未实现 model_type/dimension 字段：测试先行契约，字段缺失时正向用例失败暴露，
 * 待后端实现后统一验证。
 */
describe("模型类型与维度扩展（modelType/dimension）", () => {
  test("正向：创建 embedding 模型含 modelType/dimension", async () => {
    await login(USERS.ADMIN.username);
    const form = createAiModelForm({ modelType: "embedding", dimension: 1024 });
    const result = await AiConversationAPI.createModel(form);
    expect(result.modelType).toBe("embedding");
    expect(result.dimension).toBe(1024);
    await cleanupModel(result.modelId);
  });

  test("正向：按 modelType 筛选模型列表", async () => {
    await login(USERS.ADMIN.username);
    const result = await AiConversationAPI.getModels(createAiModelQuery({ modelType: "chat" }));
    expect(Array.isArray(result.list)).toBe(true);
  });

  test("正向：getEnabledModels 按 modelType 筛选", async () => {
    await login(USERS.ADMIN.username);
    const enabled = await AiConversationAPI.getEnabledModels("chat");
    expect(Array.isArray(enabled)).toBe(true);
  });
});
