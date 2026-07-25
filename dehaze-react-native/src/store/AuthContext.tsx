import '@/config/sdk';
import { AuthAPI, SESSION_KEY } from 'dehaze-sdk-js';
import type { LoginData, LoginResult, AuthUserInfo } from 'dehaze-sdk-js';
import { CacheEnum } from '@/enums/CacheEnum';
import { storage } from '@/utils/storage';
import { sessionStore, setOnSessionInvalid } from '@/utils/tokenStore';
import React, {
  createContext,
  useContext,
  useEffect,
  useReducer,
  useRef,
  type ReactNode,
} from 'react';
import { Alert } from 'react-native';

interface AuthState {
  sessionId: string | null;
  userInfo: AuthUserInfo | null;
  loading: boolean;
}

type AuthAction =
  | { type: 'RESTORE'; sessionId: string | null; userInfo: AuthUserInfo | null }
  | { type: 'LOGIN'; sessionId: string; userInfo: AuthUserInfo }
  | { type: 'SET_USER_INFO'; userInfo: AuthUserInfo }
  | { type: 'LOGOUT' };

const initialState: AuthState = {
  sessionId: null,
  userInfo: null,
  loading: true,
};

function authReducer(state: AuthState, action: AuthAction): AuthState {
  switch (action.type) {
    case 'RESTORE':
      return {
        sessionId: action.sessionId,
        userInfo: action.userInfo,
        loading: false,
      };
    case 'LOGIN':
      return { sessionId: action.sessionId, userInfo: action.userInfo, loading: false };
    case 'SET_USER_INFO':
      return { ...state, userInfo: action.userInfo };
    case 'LOGOUT':
      return { sessionId: null, userInfo: null, loading: false };
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

  useEffect(() => {
    if (initialized.current) {
      return;
    }
    initialized.current = true;

    (async () => {
      const sid = await storage.get<string>(SESSION_KEY);
      if (!sid) {
        dispatch({ type: 'RESTORE', sessionId: null, userInfo: null });
        return;
      }
      sessionStore.set(sid);
      try {
        const userInfo = await AuthAPI.getCurrentUser();
        await storage.set(CacheEnum.AUTH_INFO, userInfo);
        dispatch({ type: 'RESTORE', sessionId: sid, userInfo });
      } catch {
        storage.remove(SESSION_KEY);
        storage.remove(CacheEnum.AUTH_INFO);
        sessionStore.clear();
        dispatch({ type: 'RESTORE', sessionId: null, userInfo: null });
      }
    })();
  }, []);

  useEffect(() => {
    const handleInvalid = () => {
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
              dispatch({ type: 'LOGOUT' });
            },
          },
        ],
        { cancelable: false }
      );
    };
    setOnSessionInvalid(handleInvalid);
    return () => setOnSessionInvalid(null);
  }, []);

  const login = async (data: LoginData) => {
    const result: LoginResult = await AuthAPI.login(data);
    sessionStore.set(result.sessionId);
    if (data.rememberMe !== false) {
      await storage.set(SESSION_KEY, result.sessionId);
    }
    const userInfo = await AuthAPI.getCurrentUser();
    await storage.set(CacheEnum.AUTH_INFO, userInfo);
    dispatch({ type: 'LOGIN', sessionId: result.sessionId, userInfo });
  };

  const logout = async () => {
    try {
      await AuthAPI.logout();
    } catch {
    }
    storage.remove(SESSION_KEY);
    storage.remove(CacheEnum.AUTH_INFO);
    sessionStore.clear();
    dispatch({ type: 'LOGOUT' });
  };

  const refreshUserInfo = async () => {
    const userInfo = await AuthAPI.getCurrentUser();
    await storage.set(CacheEnum.AUTH_INFO, userInfo);
    dispatch({ type: 'SET_USER_INFO', userInfo });
  };

  const value: AuthContextValue = {
    state,
    isAuthenticated: !!state.sessionId,
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
