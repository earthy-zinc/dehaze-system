import { describe, test, expect, afterAll } from "vitest";
import { ApiKeyAPI, service } from "../../../index";
import type { ApiKeyVO } from "@/api/api-key/model";
import { createApiKeyForm, createGovernedApiKeyForm } from "#/factories/api-key";
import { TestCleanupRegistry } from "#/utils/cleanup";

describe("API密钥管理", () => {
  const cleanup = new TestCleanupRegistry();
  let createdKey: ApiKeyVO;

  function registerCleanup(keyId: number) {
    cleanup.register(() => ApiKeyAPI.delete(keyId));
  }

  afterAll(async () => {
    await cleanup.executeAll();
  });

  describe("创建API密钥", () => {
    test("创建密钥应返回明文key", async () => {
      const form = createApiKeyForm();
      createdKey = await ApiKeyAPI.create(form);

      expect(createdKey.id).toBeGreaterThan(0);
      expect(createdKey.name).toBe(form.name);
      expect(createdKey.apiKey!.startsWith("dhak_")).toBe(true);
      expect(createdKey.keyPrefix).toBeDefined();
      expect(createdKey.status).toBe(1);

      registerCleanup(createdKey.id);
    });

    test("创建带过期时间的密钥", async () => {
      const futureDate = new Date(Date.now() + 365 * 24 * 3600 * 1000).toISOString();
      const form = createApiKeyForm({ expiresAt: futureDate });
      const result = await ApiKeyAPI.create(form);

      expect(result.apiKey!.startsWith("dhak_")).toBe(true);
      expect(result.expiresAt).toBeDefined();

      registerCleanup(result.id);
    });

    test("创建带治理参数(配额)的密钥应返回并透传", async () => {
      const form = createGovernedApiKeyForm({
        dailyQuota: 1000,
        monthlyQuota: 20000,
        rpmLimit: 60,
      }) as any;
      const result: any = await ApiKeyAPI.create(form);

      expect(result.apiKey!.startsWith("dhak_")).toBe(true);
      expect(result.dailyQuota).toBe(1000);
      expect(result.monthlyQuota).toBe(20000);
      expect(result.rpmLimit).toBe(60);

      registerCleanup(result.id);
    });
  });

  describe("获取API密钥列表", () => {
    test("列表应包含已创建的密钥", async () => {
      const list = await ApiKeyAPI.list();

      expect(Array.isArray(list)).toBe(true);
      expect(list.length).toBeGreaterThan(0);

      const found = list.find((k) => k.id === createdKey.id);
      expect(found).toBeDefined();
      expect(found!.name).toBe(createdKey.name);
      expect(found!.keyPrefix).toBe(createdKey.keyPrefix);
      expect(found!.apiKey).toBeUndefined();
    });
  });

  describe("使用API密钥鉴权", () => {
    test("使用API密钥访问受保护接口应成功", async () => {
      const userInfo = (await service.get("/api/v1/auth/me", {
        headers: { Authorization: `Bearer ${createdKey.apiKey}` },
      })) as any;

      expect(userInfo.userId).toBeDefined();
      expect(userInfo.username).toBeDefined();
    });

    test("使用无效API密钥应返回401", async () => {
      await expect(
        service.get("/api/v1/auth/me", {
          headers: { Authorization: "Bearer dhak_invalidkey123456789012345678901234567890" },
        })
      ).rejects.toSatisfy((error: any) => {
        return error.response?.status === 401 || error.code === "ERR_BAD_REQUEST";
      });
    });
  });

  describe("删除API密钥", () => {
    test("删除密钥后应无法再使用", async () => {
      const form = createApiKeyForm();
      const keyToDelete = await ApiKeyAPI.create(form);
      expect(keyToDelete.apiKey).toBeDefined();

      await ApiKeyAPI.delete(keyToDelete.id);

      // 内部语义：软删除设 revoked_at=now()，列表查询（revoked_at IS NULL）不再返回
      const list = await ApiKeyAPI.list();
      const found = list.find((k) => k.id === keyToDelete.id);
      expect(found).toBeUndefined();

      await expect(
        service.get("/api/v1/auth/me", {
          headers: { Authorization: `Bearer ${keyToDelete.apiKey}` },
        })
      ).rejects.toSatisfy((error: any) => {
        return error.response?.status === 401 || error.code === "ERR_BAD_REQUEST";
      });
    });

    test("删除不存在的密钥应返回错误", async () => {
      await expect(ApiKeyAPI.delete(999999)).rejects.toThrow();
    });
  });
});
