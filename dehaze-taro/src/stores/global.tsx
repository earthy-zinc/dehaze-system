/**
 * 全局认证状态（zustand）
 *
 * 职责：登录态（user / sessionId / perms / roles）与认证方法（login / logout / initAuth），
 * 状态持久化到本地 storage。组件通过选择器订阅，无 Provider 包裹、无全量重渲染。
 */
import { create } from "zustand";
import { storage } from "@/utils/storage";
import type { UserInfo, LoginData } from "dehaze-sdk-js";

/** 认证状态字段 */
interface AuthState {
  user: UserInfo | null;
  sessionId: string | null;
  isAuthenticated: boolean;
  perms: string[];
  roles: string[];
}

interface AuthStore extends AuthState {
  /** 登录：写入 session 与用户信息（持久化后同步内存态） */
  login: (loginData: LoginData) => Promise<UserInfo>;
  /** 登出：调用登出接口并清理本地认证数据 */
  logout: () => Promise<void>;
  /** 应用启动时从本地 storage 恢复认证状态 */
  initAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  sessionId: null,
  isAuthenticated: false,
  perms: [],
  roles: [],

  // SDK 为单入口 bundle（聚合全部 API），动态导入按需分包，避免主包加载整个 SDK
  login: async (loginData) => {
    const { AuthAPI, UserAPI } = await import("dehaze-sdk-js");
    const response = await AuthAPI.login(loginData);
    storage.setSessionId(response.sessionId);

    const userInfo = await UserAPI.getInfo();
    await storage.setUserInfo(userInfo);
    await storage.setPerms(userInfo.perms || []);
    await storage.setRoles(userInfo.roles || []);

    set({
      isAuthenticated: true,
      user: userInfo,
      sessionId: response.sessionId,
      perms: userInfo.perms || [],
      roles: userInfo.roles || [],
    });
    return userInfo;
  },

  logout: async () => {
    try {
      const { AuthAPI } = await import("dehaze-sdk-js");
      await AuthAPI.logout();
    } catch (error) {
      console.error("登出接口调用失败:", error);
    } finally {
      storage.clearAuth();
      set({
        isAuthenticated: false,
        user: null,
        sessionId: null,
        perms: [],
        roles: [],
      });
    }
  },

  initAuth: async () => {
    try {
      const sessionId = storage.getSessionId();
      const userInfo = await storage.getUserInfo();
      const perms = await storage.getPerms();
      const roles = await storage.getRoles();
      if (sessionId && userInfo) {
        set({ isAuthenticated: true, user: userInfo, sessionId, perms, roles });
      }
    } catch (error) {
      console.error("初始化认证状态失败:", error);
    }
  },
}));
