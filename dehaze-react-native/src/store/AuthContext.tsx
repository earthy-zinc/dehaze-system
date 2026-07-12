/**
 * 认证上下文
 *
 * 使用 dehaze-sdk-js 的 AuthAPI 进行认证。
 * - 登录时持久化 token 到 AsyncStorage 并同步到 tokenStore（供 SDK 同步读取）
 * - 注销时清空 token 与权限信息
 * - token 失效由 SDK 拦截器触发 triggerTokenInvalid，自动清空状态
 */
import '@/config/sdk'; // 初始化 SDK 配置（副作用导入）
import { AuthAPI } from 'dehaze-sdk-js';
import type { LoginData, LoginResult, AuthUserInfo } from 'dehaze-sdk-js';
import { CacheEnum } from '@/enums/CacheEnum';
import { storage } from '@/utils/storage';
import { tokenStore, setOnTokenInvalid } from '@/utils/tokenStore';
import React, {
  createContext,
  useContext,
  useEffect,
  useReducer,
  useRef,
  type ReactNode,
} from 'react';

interface AuthState {
  token: string | null;
  userInfo: AuthUserInfo | null;
  /** 初始化加载中（从 AsyncStorage 恢复 token） */
  loading: boolean;
}

type AuthAction =
  | { type: 'RESTORE'; token: string | null; userInfo: AuthUserInfo | null }
  | { type: 'LOGIN'; token: string; userInfo: AuthUserInfo }
  | { type: 'SET_USER_INFO'; userInfo: AuthUserInfo }
  | { type: 'LOGOUT' };

const initialState: AuthState = {
  token: null,
  userInfo: null,
  loading: true,
};

function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'RESTORE':
      return {
        token: action.token,
        userInfo: action.userInfo,
        loading: false,
      };
    case 'LOGIN':
      return { token: action.token, userInfo: action.userInfo, loading: false };
    case 'SET_USER_INFO':
      return { ...state, userInfo: action.userInfo };
    case 'LOGOUT':
      return { token: null, userInfo: null, loading: false };
    default:
      return state;
  }
}

interface AuthContextValue {
  state: AuthState;
  isAuthenticated: boolean;
  login: (data: LoginData) => Promise<void>;
  logout: () => Promise<void>;
  refreshUserInfo: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(authReducer, initialState);
  const initialized = useRef(false);

  // 启动时从 AsyncStorage 恢复 token
  useEffect(() => {
    if (initialized.current) {
      return;
    }
    initialized.current = true;

    (async () => {
      const token = await storage.get<string>(CacheEnum.TOKEN);
      if (token) {
        tokenStore.set(token);
      }
      // 权限信息延迟到首页加载时获取（避免登录页加载阻塞）
      dispatch({ type: 'RESTORE', token, userInfo: null });
    })();
  }, []);

  // 注册 token 失效回调
  useEffect(() => {
    const handleInvalid = () => {
      storage.remove(CacheEnum.TOKEN);
      storage.remove(CacheEnum.AUTH_INFO);
      tokenStore.clear();
      dispatch({ type: 'LOGOUT' });
    };
    setOnTokenInvalid(handleInvalid);
    return () => setOnTokenInvalid(null);
  }, []);

  const login = async (data: LoginData) => {
    const result: LoginResult = await AuthAPI.login(data);
    const token = result.accessToken;
    await storage.set(CacheEnum.TOKEN, token);
    if (result.refreshToken) {
      await storage.set(CacheEnum.REFRESH_TOKEN, result.refreshToken);
    }
    tokenStore.set(token);

    // 获取当前用户信息
    const userInfo = await AuthAPI.getCurrentUser();
    await storage.set(CacheEnum.AUTH_INFO, userInfo);
    dispatch({ type: 'LOGIN', token, userInfo });
  };

  const logout = async () => {
    try {
      await AuthAPI.logout();
    } catch {
      // 即使注销接口失败也清空本地状态
    }
    storage.remove(CacheEnum.TOKEN);
    storage.remove(CacheEnum.REFRESH_TOKEN);
    storage.remove(CacheEnum.AUTH_INFO);
    tokenStore.clear();
    dispatch({ type: 'LOGOUT' });
  };

  const refreshUserInfo = async () => {
    const userInfo = await AuthAPI.getCurrentUser();
    await storage.set(CacheEnum.AUTH_INFO, userInfo);
    dispatch({ type: 'SET_USER_INFO', userInfo });
  };

  const value: AuthContextValue = {
    state,
    isAuthenticated: !!state.token,
    login,
    logout,
    refreshUserInfo,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth 必须在 AuthProvider 内使用');
  }
  return ctx;
}
