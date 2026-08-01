import { AuthAPI, UserAPI } from "dehaze-sdk-js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUserStore } from "../user";

vi.mock("dehaze-sdk-js", () => ({
  AuthAPI: {
    login: vi.fn(),
    logout: vi.fn(),
  },
  UserAPI: {
    getInfo: vi.fn(),
  },
}));

vi.mock("@/router", () => ({
  resetRouter: vi.fn(),
}));

describe("useUserStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("login", () => {
    it("应该成功登录", async () => {
      const mockLoginData = {
        username: "admin",
        password: "Dehaze2026",
        rememberMe: true,
      };
      const mockResponse = {
        sessionId: "mock-session-id",
        user: {
          id: 1,
          username: "admin",
          nickname: "管理员",
        },
      };

      vi.mocked(AuthAPI.login).mockResolvedValue(mockResponse);

      const store = useUserStore();

      await store.login(mockLoginData);

      expect(AuthAPI.login).toHaveBeenCalledWith(mockLoginData);
    });

    it("应该在登录失败时抛出错误", async () => {
      const mockLoginData = {
        username: "admin",
        password: "wrong-password",
      };
      const mockError = new Error("用户名或密码错误");

      vi.mocked(AuthAPI.login).mockRejectedValue(mockError);

      const store = useUserStore();

      await expect(store.login(mockLoginData)).rejects.toThrow(
        "用户名或密码错误"
      );
    });
  });

  describe("getUserInfo", () => {
    it("应该成功获取用户信息并更新 store", async () => {
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

      const result = await store.getUserInfo();

      expect(UserAPI.getInfo).toHaveBeenCalled();
      expect(result).toEqual(mockUserInfo);
      expect(store.user).toEqual(mockUserInfo);
    });

    it("应该在角色为空数组时拒绝", async () => {
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

      await expect(store.getUserInfo()).rejects.toBe(
        "getUserInfo: roles must be a non-null array!"
      );
    });
  });

  describe("logout", () => {
    it("应该成功登出", async () => {
      vi.mocked(AuthAPI.logout).mockResolvedValue(undefined as any);

      const reloadMock = vi.fn();
      Object.defineProperty(window, "location", {
        value: { reload: reloadMock },
        writable: true,
      });

      const store = useUserStore();

      await store.logout();

      expect(AuthAPI.logout).toHaveBeenCalled();
      expect(reloadMock).toHaveBeenCalled();
    });
  });

  describe("resetToken", () => {
    it("应该重置路由", async () => {
      const { resetRouter } = await import("@/router");

      const store = useUserStore();

      await store.resetToken();

      expect(resetRouter).toHaveBeenCalled();
    });
  });

  describe("user state", () => {
    it("应该初始化为空的用户信息", () => {
      const store = useUserStore();

      expect(store.user).toEqual({
        roles: [],
        perms: [],
      });
    });

    it("应该在获取用户信息后正确更新用户状态", async () => {
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

      await store.getUserInfo();

      expect(store.user.userId).toBe(1);
      expect(store.user.username).toBe("testuser");
      expect(store.user.roles).toEqual(["USER"]);
      expect(store.user.perms).toEqual(["sys:user:view"]);
    });
  });

  describe("完整登录流程", () => {
    it("应该完整执行登录 -> 获取用户信息的流程", async () => {
      const mockLoginData = {
        username: "admin",
        password: "Dehaze2026",
      };
      const mockLoginResponse = {
        sessionId: "mock-session-id",
        user: {
          id: 1,
          username: "admin",
          nickname: "管理员",
        },
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

      await store.login(mockLoginData);
      await store.getUserInfo();

      expect(store.user.username).toBe("admin");
      expect(store.user.roles).toEqual(["ADMIN"]);
    });
  });
});
