import { describe, it, expect, beforeEach, vi } from "vitest";
import { setActivePinia, createPinia } from "pinia";
import { useUserStore } from "../user";
import { AuthAPI, UserAPI } from "dehaze-sdk-js";
import { TOKEN_KEY } from "@/enums/CacheEnum";

// Mock SDK APIs
vi.mock("dehaze-sdk-js", () => ({
  AuthAPI: {
    login: vi.fn(),
    logout: vi.fn(),
  },
  UserAPI: {
    getInfo: vi.fn(),
  },
}));

// Mock router
vi.mock("@/router", () => ({
  resetRouter: vi.fn(),
}));

describe("useUserStore", () => {
  beforeEach(() => {
    // 清除所有 mocks
    vi.clearAllMocks();

    // 清除 localStorage
    localStorage.clear();
  });

  describe("login", () => {
    it("应该成功登录并保存 token", async () => {
      // Arrange
      const mockLoginData = {
        username: "admin",
        password: "123456",
      };
      const mockResponse = {
        tokenType: "Bearer",
        accessToken: "test-access-token",
      };

      vi.mocked(AuthAPI.login).mockResolvedValue(mockResponse);

      const store = useUserStore();

      // Act
      await store.login(mockLoginData);

      // Assert
      expect(AuthAPI.login).toHaveBeenCalledWith(mockLoginData);
      expect(localStorage.getItem(TOKEN_KEY)).toBe("Bearer test-access-token");
    });

    it("应该在登录失败时抛出错误", async () => {
      // Arrange
      const mockLoginData = {
        username: "admin",
        password: "wrong-password",
      };
      const mockError = new Error("用户名或密码错误");

      vi.mocked(AuthAPI.login).mockRejectedValue(mockError);

      const store = useUserStore();

      // Act & Assert
      await expect(store.login(mockLoginData)).rejects.toThrow(
        "用户名或密码错误"
      );
      expect(localStorage.getItem(TOKEN_KEY)).toBeNull();
    });
  });

  describe("getUserInfo", () => {
    it("应该成功获取用户信息并更新 store", async () => {
      // Arrange
      const mockUserInfo = {
        userId: 1,
        username: "admin",
        nickname: "管理员",
        avatar: "https://example.com/avatar.jpg",
        roles: ["ADMIN", "USER"],
        perms: ["sys:user:view", "sys:user:edit"],
      };

      vi.mocked(UserAPI.getInfo).mockResolvedValue(mockUserInfo);

      const store = useUserStore();

      // Act
      const result = await store.getUserInfo();

      // Assert
      expect(UserAPI.getInfo).toHaveBeenCalled();
      expect(result).toEqual(mockUserInfo);
      expect(store.user).toEqual(mockUserInfo);
    });

    it("应该在用户信息为空时拒绝", async () => {
      // Arrange
      vi.mocked(UserAPI.getInfo).mockResolvedValue(null as any);

      const store = useUserStore();

      // Act & Assert
      await expect(store.getUserInfo()).rejects.toBe(
        "Verification failed, please Login again."
      );
    });

    it("应该在角色为空数组时拒绝", async () => {
      // Arrange
      const mockUserInfo = {
        userId: 1,
        username: "admin",
        nickname: "管理员",
        avatar: "https://example.com/avatar.jpg",
        roles: [],
        perms: [],
      };

      vi.mocked(UserAPI.getInfo).mockResolvedValue(mockUserInfo);

      const store = useUserStore();

      // Act & Assert
      await expect(store.getUserInfo()).rejects.toBe(
        "getUserInfo: roles must be a non-null array!"
      );
    });

    it("应该在 API 调用失败时抛出错误", async () => {
      // Arrange
      const mockError = new Error("网络错误");
      vi.mocked(UserAPI.getInfo).mockRejectedValue(mockError);

      const store = useUserStore();

      // Act & Assert
      await expect(store.getUserInfo()).rejects.toThrow("网络错误");
    });
  });

  describe("logout", () => {
    it("应该成功登出并清除 token", async () => {
      // Arrange
      localStorage.setItem(TOKEN_KEY, "Bearer test-token");
      vi.mocked(AuthAPI.logout).mockResolvedValue(undefined as any);

      // Mock location.reload
      const reloadMock = vi.fn();
      Object.defineProperty(window, "location", {
        value: { reload: reloadMock },
        writable: true,
      });

      const store = useUserStore();

      // Act
      await store.logout();

      // Assert
      expect(AuthAPI.logout).toHaveBeenCalled();
      expect(localStorage.getItem(TOKEN_KEY) || "").toBe("");
      expect(reloadMock).toHaveBeenCalled();
    });

    it("应该在登出 API 失败时抛出错误", async () => {
      // Arrange
      const mockError = new Error("登出失败");
      vi.mocked(AuthAPI.logout).mockRejectedValue(mockError);

      const store = useUserStore();

      // Act & Assert
      await expect(store.logout()).rejects.toThrow("登出失败");
    });
  });

  describe("resetToken", () => {
    it("应该清除 token 并重置路由", async () => {
      // Arrange
      localStorage.setItem(TOKEN_KEY, "Bearer test-token");
      const { resetRouter } = await import("@/router");

      const store = useUserStore();

      // Act
      await store.resetToken();

      // Assert
      expect(localStorage.getItem(TOKEN_KEY) || "").toBe("");
      expect(resetRouter).toHaveBeenCalled();
    });
  });

  describe("user state", () => {
    it("应该初始化为空的用户信息", () => {
      // Arrange & Act
      const store = useUserStore();

      // Assert
      expect(store.user).toEqual({
        roles: [],
        perms: [],
      });
    });

    it("应该在获取用户信息后正确更新用户状态", async () => {
      // Arrange
      const mockUserInfo = {
        userId: 1,
        username: "testuser",
        nickname: "测试用户",
        avatar: "https://example.com/avatar.jpg",
        roles: ["USER"],
        perms: ["sys:user:view"],
      };

      vi.mocked(UserAPI.getInfo).mockResolvedValue(mockUserInfo);

      const store = useUserStore();

      // Act
      await store.getUserInfo();

      // Assert
      expect(store.user.userId).toBe(1);
      expect(store.user.username).toBe("testuser");
      expect(store.user.roles).toEqual(["USER"]);
      expect(store.user.perms).toEqual(["sys:user:view"]);
    });
  });

  describe("完整登录流程", () => {
    it("应该完整执行登录 -> 获取用户信息的流程", async () => {
      // Arrange
      const mockLoginData = {
        username: "admin",
        password: "123456",
      };
      const mockLoginResponse = {
        tokenType: "Bearer",
        accessToken: "test-token",
      };
      const mockUserInfo = {
        userId: 1,
        username: "admin",
        nickname: "管理员",
        avatar: "https://example.com/avatar.jpg",
        roles: ["ADMIN"],
        perms: ["sys:user:view", "sys:user:edit"],
      };

      vi.mocked(AuthAPI.login).mockResolvedValue(mockLoginResponse);
      vi.mocked(UserAPI.getInfo).mockResolvedValue(mockUserInfo);

      const store = useUserStore();

      // Act
      await store.login(mockLoginData);
      await store.getUserInfo();

      // Assert
      expect(localStorage.getItem(TOKEN_KEY)).toBe("Bearer test-token");
      expect(store.user.username).toBe("admin");
      expect(store.user.roles).toEqual(["ADMIN"]);
    });
  });
});
