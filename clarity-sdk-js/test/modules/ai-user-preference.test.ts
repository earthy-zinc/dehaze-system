import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiConversationAPI, AiUserPreferenceAPI } from "../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";

/**
 * 用户偏好默认模型（GET/PUT /api/v1/ai/user-preference）
 *
 * 契约（AI模型管理/API接口.md §2.5）：登录用户本人读写，无权限码；
 * - 未设置偏好时响应不含 `defaultModel` 键（键缺失 ≡ 未设置）；
 * - PUT 全量替换：`defaultModel` 传 null 或省略字段即清除，空串按非法模型 A0601；
 * - 非空值须为本人可见的启用 chat 模型（存在 + status=1 + model_type=chat + 等级内），否则 A0601；
 * - 偏好只影响新建会话（请求未指定模型时），不改变已有会话的模型。
 */
describe("用户偏好默认模型 - AiUserPreferenceAPI", () => {
  // dev/test 库既有启用 chat 模型（见 config/sql/data/sys_ai_model.sql）
  const modelA = "qwen3-0.6b";
  const modelB = "gpt-4o-mini";
  let convId: number | undefined;
  let explicitConvId: number | undefined;

  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: null });
  });

  afterAll(async () => {
    await login(USERS.ADMIN.username).catch(() => {});
    try {
      await AiUserPreferenceAPI.update({ defaultModel: null });
      for (const id of [convId, explicitConvId].filter((v): v is number => !!v)) {
        await AiConversationAPI.deleteConversation(id);
      }
    } catch (e) {
      console.warn("清理偏好/会话失败:", (e as Error)?.message ?? e);
    }
  });

  test("初始未设置：GET 响应不含 defaultModel 键（键缺失 ≡ 未设置）", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: null });
    // 字段声明为可选，此处用 getter 断言键缺失而非 null
    const pref = (await AiUserPreferenceAPI.get()) as Record<string, unknown>;
    expect(pref).not.toHaveProperty("defaultModel");
  });

  test("设置偏好后回读一致", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: modelA });
    const pref = await AiUserPreferenceAPI.get();
    expect(pref.defaultModel).toBe(modelA);
  });

  test("新建会话（不传 model）落库为偏好模型", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: modelA });
    const conv = await AiConversationAPI.createConversation({});
    convId = conv.id;
    expect(conv.model).toBe(modelA);
  });

  test("请求显式指定 model 优先于偏好", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: modelB });
    const conv = await AiConversationAPI.createConversation({ model: modelA });
    explicitConvId = conv.id;
    expect(conv.model).toBe(modelA);
  });

  test("已有会话不被偏好覆盖（偏好改为 modelB 后原会话仍为 modelA）", async () => {
    await login(USERS.ADMIN.username);
    const conv = await AiConversationAPI.getConversation(explicitConvId!);
    expect(conv.model).toBe(modelA);
  });

  test("PUT {defaultModel: null} 清除偏好 → GET 键消失", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: modelA });
    await AiUserPreferenceAPI.update({ defaultModel: null });
    const pref = (await AiUserPreferenceAPI.get()) as Record<string, unknown>;
    expect(pref).not.toHaveProperty("defaultModel");
  });

  test("PUT 省略 defaultModel 字段（{}）同样清除偏好", async () => {
    await login(USERS.ADMIN.username);
    await AiUserPreferenceAPI.update({ defaultModel: modelA });
    // 省略字段视同清除（与传 null 等价），类型层要求显式传参故此处用空对象强转
    await AiUserPreferenceAPI.update({} as { defaultModel: string | null });
    const pref = (await AiUserPreferenceAPI.get()) as Record<string, unknown>;
    expect(pref).not.toHaveProperty("defaultModel");
  });

  test("负向：不存在的模型 → A0601", async () => {
    await login(USERS.ADMIN.username);
    await expectBizError(
      AiUserPreferenceAPI.update({ defaultModel: "no-such-model-x" }),
      "A0601",
      "模型不存在、未启用或当前等级不可用"
    );
  });

  test("负向：空串不是清除，按非法模型 → A0601", async () => {
    await login(USERS.ADMIN.username);
    await expectBizError(AiUserPreferenceAPI.update({ defaultModel: "" }), "A0601");
    const pref = (await AiUserPreferenceAPI.get()) as Record<string, unknown>;
    expect(pref).not.toHaveProperty("defaultModel");
  });

  test("负向：非 chat 模型（embedding）→ A0601", async () => {
    await login(USERS.ADMIN.username);
    await expectBizError(
      AiUserPreferenceAPI.update({ defaultModel: "text-embedding-3-large" }),
      "A0601"
    );
  });
});
