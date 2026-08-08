import '@/config/sdk';
import { AuthAPI, SESSION_KEY } from 'dehaze-sdk-js';
import type { LoginData, LoginResult, AuthUserInfo } from 'dehaze-sdk-js';
import { CacheEnum } from '@/enums/CacheEnum';
import { storage } from '@/utils/storage';
import { sessionStore, setOnSessionInvalid } from '@/utils/tokenStore';
import { Alert } from 'react-native';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface AuthState {
  sessionId: string | null;
  userInfo: AuthUserInfo | null;
  loading: boolean;
  login: (data: LoginData) => Promise<void>;
  logout: () => Promise<void>;
  refreshUserInfo: () => Promise<void>;
  isAuthenticated: boolean;
  hasPerm: (perm: string) => boolean;
  hasRole: (role: string) => boolean;
}

const handleSessionInvalid = () => {
  Alert.alert(
    '登录已失效',
    '您的登录状态已过期，请重新登录',
    [
      {
        text: '重新登录',
        onPress: () => {
          storage.remove(SESSION_KEY);
          storage.remove(CacheEnum.AUTH_INFO);
          sessionStore.clear();
          useAuthStore.getState().logout();
        },
      },
    ],
    { cancelable: false },
  );
};

// 注册 sessionInvalid 回调（模块加载时一次性）
let registered = false;
function ensureSessionInvalidHandler() {
  if (registered) return;
  registered = true;
  setOnSessionInvalid(handleSessionInvalid);
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => {
      ensureSessionInvalidHandler();

      return {
        sessionId: null,
        userInfo: null,
        loading: true,

        get isAuthenticated() {
          return !!get().sessionId;
        },

        login: async (data: LoginData) => {
          const result: LoginResult = await AuthAPI.login(data);
          sessionStore.set(result.sessionId);
          if (data.rememberMe !== false) {
            await storage.set(SESSION_KEY, result.sessionId);
          }
          const userInfo = await AuthAPI.getCurrentUser();
          await storage.set(CacheEnum.AUTH_INFO, userInfo);
          set({ sessionId: result.sessionId, userInfo, loading: false });
        },

        logout: async () => {
          try {
            await AuthAPI.logout();
          } catch {
            // 忽略登出接口错误
          }
          storage.remove(SESSION_KEY);
          storage.remove(CacheEnum.AUTH_INFO);
          sessionStore.clear();
          set({ sessionId: null, userInfo: null, loading: false });
        },

        refreshUserInfo: async () => {
          const userInfo = await AuthAPI.getCurrentUser();
          await storage.set(CacheEnum.AUTH_INFO, userInfo);
          set({ userInfo });
        },

        hasPerm: (perm: string) => {
          const { userInfo } = get();
          return userInfo?.perms?.includes(perm) ?? false;
        },

        hasRole: (role: string) => {
          const { userInfo } = get();
          return userInfo?.roles?.includes(role) ?? false;
        },
      };
    },
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: state => ({
        sessionId: state.sessionId,
        userInfo: state.userInfo,
      }),
      onRehydrateStorage: () => {
        return (state) => {
          // 必须走 setState 触发订阅通知：直接 mutation state 不会让组件重渲染，会一直停在加载态
          useAuthStore.setState({ loading: false });
          if (state?.sessionId) {
            sessionStore.set(state.sessionId);
          }
        };
      },
    },
  ),
);
