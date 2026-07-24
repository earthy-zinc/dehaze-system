import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { ApiKeyAPI, javaService } from "../../../index";
import type { ApiKeyVO } from "@/api/api-key/model";
import { login, logout } from "#/utils/auth";
import { createApiKeyForm } from "#/factories/api-key";
import { TestCleanupRegistry } from "#/utils/cleanup";

describe("API密钥管理", () => {
  const cleanup = new TestCleanupRegistry();
  let createdKey: ApiKeyVO;

  beforeAll(async () => {
    await login();
  }, 30000);

  afterAll(async () => {
    await cleanup.executeAll();
    await logout();
  });

  describe("创建API密钥", () => {
    test("创建密钥应返回明文key", async () => {
      const form = createApiKeyForm();
      createdKey = await ApiKeyAPI.create(form);

      expect(createdKey).toBeDefined();
      expect(createdKey.id).toBeGreaterThan(0);
      expect(createdKey.name).toBe(form.name);
      expect(createdKey.apiKey).toBeDefined();
      expect(createdKey.apiKey!.startsWith("dhak_")).toBe(true);
      expect(createdKey.keyPrefix).toBeDefined();
      expect(createdKey.status).toBe(1);

      cleanup.register(async () => {
        await ApiKeyAPI.delete(createdKey.id);
      });
    });

    test("创建带过期时间的密钥", async () => {
      const futureDate = new Date(Date.now() + 365 * 24 * 3600 * 1000).toISOString();
      const form = createApiKeyForm({ expiresAt: futureDate });
      const result = await ApiKeyAPI.create(form);

      expect(result).toBeDefined();
      expect(result.apiKey).toBeDefined();
      expect(result.apiKey!.startsWith("dhak_")).toBe(true);
      expect(result.expiresAt).toBeDefined();

      cleanup.register(async () => {
        await ApiKeyAPI.delete(result.id);
      });
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
      const response = await javaService.get("/api/v1/auth/me", {
        headers: { Authorization: `Bearer ${createdKey.apiKey}` },
      });

      expect(response.data.code).toBe("00000");
      expect(response.data.data.userId).toBeDefined();
      expect(response.data.data.username).toBeDefined();
    });

    test("使用无效API密钥应返回401", async () => {
      await expect(
        javaService.get("/api/v1/auth/me", {
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

      const list = await ApiKeyAPI.list();
      const found = list.find((k) => k.id === keyToDelete.id);
      expect(found).toBeUndefined();

      await expect(
        javaService.get("/api/v1/auth/me", {
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
