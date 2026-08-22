import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiProviderAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createProviderForm,
  createProviderKeyForm,
  createProviderKeyUpdateForm,
  createProviderQuery,
  createProviderUpdateForm,
} from "#/factories/ai-provider";

/**
 * AI 供应商与 API Key 管理（T-MF-014~019、033~034）
 *
 * 供应商/Key 相关为管理员接口（ai:model:manage），普通用户 403。
 * 数据前缀 test_prov_ / test_key_，afterAll 尽力清理（软删除）。
 */
describe("AI 供应商与 Key 管理 - AiProviderAPI (T-MF-014~019,033~034)", () => {
  const loginAdmin = () => login(USERS.ADMIN.username);

  describe("供应商 CRUD", () => {
    let createdProviderId: number;

    test("T-MF-014 正向：新增供应商返回完整结构", async () => {
      await loginAdmin();
      const form = createProviderForm();
      const result = await AiProviderAPI.createProvider(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.providerCode).toBe(form.providerCode);
      expect(result.displayName).toBe(form.displayName);
      expect(result.apiBaseUrl).toBe(form.apiBaseUrl);
      expect(result.status).toBe(1);
      createdProviderId = result.id;
    });

    test("T-MF-015 负向：provider_code 重复 → A0501", async () => {
      await loginAdmin();
      const form = createProviderForm();
      await AiProviderAPI.createProvider(form);
      await expectBizError(AiProviderAPI.createProvider(form), ["A0501"]);
      await AiProviderAPI.deleteProvider((await findProviderByCode(form.providerCode)).id).catch(
        () => {}
      );
    });

    test("T-MF-019 正向：供应商保存成功（连通性测试后台执行不阻断）", async () => {
      await loginAdmin();
      // apiBaseUrl 指向不可达地址，保存仍成功（连通性测试仅提示不阻断）
      const form = createProviderForm({ apiBaseUrl: "https://10.255.255.1:9999/v1" });
      const result = await AiProviderAPI.createProvider(form);
      expect(result.id).toBeGreaterThan(0);
      await AiProviderAPI.deleteProvider(result.id).catch(() => {});
    });

    test("T-MF-001 正向：供应商分页列表", async () => {
      await loginAdmin();
      const result = await AiProviderAPI.listProviders(createProviderQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("T-MF-001 启用供应商列表", async () => {
      await loginAdmin();
      const enabled = await AiProviderAPI.listEnabledProviders();
      expect(Array.isArray(enabled)).toBe(true);
      if (enabled.length > 0) {
        const p = enabled[0]!;
        expect(p.id).toBeGreaterThan(0);
        expect(p.providerCode).toBeTruthy();
      }
    });

    test("T-MF-004 正向：更新供应商", async () => {
      await loginAdmin();
      const updated = await AiProviderAPI.updateProvider(
        createdProviderId,
        createProviderUpdateForm()
      );
      expect(updated.id).toBe(createdProviderId);
      expect(updated.displayName).toBeTruthy();
    });

    test("T-MF-014 负向：普通用户新增供应商 → 403", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiProviderAPI.createProvider(createProviderForm()), ["A0301"]);
      await loginAdmin();
    });

    test("T-MF-005 清理：删除测试供应商", async () => {
      await loginAdmin();
      await AiProviderAPI.deleteProvider(createdProviderId).catch(() => {});
    });
  });

  describe("API Key 管理", () => {
    let provId: number;
    let keyId: number;

    beforeAll(async () => {
      await loginAdmin();
      const form = createProviderForm();
      const provider = await AiProviderAPI.createProvider(form);
      provId = provider.id;
    });

    afterAll(async () => {
      await loginAdmin().catch(() => {});
      await AiProviderAPI.deleteProvider(provId).catch(() => {});
    });

    test("T-MF-033 正向：创建 Key 响应无明文、有 keyPrefix", async () => {
      await loginAdmin();
      const form = createProviderKeyForm();
      const result = await AiProviderAPI.createKey(provId, form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.providerId).toBe(provId);
      expect(result.name).toBe(form.name);
      // 响应不含明文
      expect((result as any).key).toBeUndefined();
      // 有 keyPrefix 展示字段
      expect(result.keyPrefix).toBeTruthy();
      keyId = result.id;
    });

    test("T-MF-034 负向：同明文 Key 重复录入 → A0501", async () => {
      await loginAdmin();
      const form = createProviderKeyForm();
      await AiProviderAPI.createKey(provId, form);
      await expectBizError(AiProviderAPI.createKey(provId, form), ["A0501", "A0400", "B0001"]);
      // 清理该 key
      await cleanupProviderKeys(provId, form.name);
    });

    test("T-MF-033 正向：Key 列表（无明文）", async () => {
      await loginAdmin();
      const keys = await AiProviderAPI.listKeys(provId);
      expect(Array.isArray(keys)).toBe(true);
      expect(keys.length).toBeGreaterThan(0);
      keys.forEach((k) => {
        expect((k as any).key).toBeUndefined();
      });
    });

    test("T-MF-004 正向：更新 Key", async () => {
      await loginAdmin();
      const updated = await AiProviderAPI.updateKey(provId, keyId, createProviderKeyUpdateForm());
      expect(updated.id).toBe(keyId);
      expect(updated.name).toBeTruthy();
    });

    test("T-MF-016/017/018 连通性测试：返回结构，允许失败状态", async () => {
      await loginAdmin();
      const result = await AiProviderAPI.testConnection(provId);
      // 结果为对象（成功/失败均返回对象），不断言具体成功与否
      expect(typeof result).toBe("object");
    });

    test("T-MF-053 closeCircuit：幂等调用不抛错（无 data 返回 undefined）", async () => {
      await loginAdmin();
      // 手动解除熔断，幂等调用应不抛错；success 无 data 时 SDK 解包为 undefined
      await expect(AiProviderAPI.closeCircuit(provId)).resolves.toBeUndefined();
    });

    test("T-MF-033 清理：删除 Key（唯一启用 Key 需先禁用）", async () => {
      await loginAdmin();
      await cleanupProviderKeys(provId);
    });
  });
});

/** 按 provider_code 查找供应商（列表分页搜，返回首个匹配） */
async function findProviderByCode(code: string) {
  const result = await AiProviderAPI.listProviders({
    pageNum: 1,
    pageSize: 100,
    keyword: code,
  });
  const found = result.list.find((p) => p.providerCode === code);
  if (!found) throw new Error(`未找到供应商 ${code}`);
  return found;
}

/** 清理 Key：唯一启用 Key 需先禁用（status=0）再删除；可指定 name 只清理匹配项 */
async function cleanupProviderKeys(provId: number, name?: string) {
  const keys = await AiProviderAPI.listKeys(provId);
  for (const k of keys) {
    if (name && k.name !== name) continue;
    if (k.status === 1) {
      await AiProviderAPI.updateKey(provId, k.id, { status: 0 }).catch(() => {});
    }
    await AiProviderAPI.deleteKey(provId, k.id).catch(() => {});
  }
}
