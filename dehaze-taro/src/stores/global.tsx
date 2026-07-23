import React, { createContext, useContext, useReducer } from "react";
import { storage } from "@/utils/storage";
import { type UserInfo, type LoginData, UserAPI } from "dehaze-sdk-js";

// 全局状态类型定义
interface GlobalState {
  auth: AuthState;
  ui: UIState;
}

interface AuthState {
  user: UserInfo | null;
  token: string | null;
  isAuthenticated: boolean;
  perms: string[];
  roles: string[];
  loading: boolean;
}

interface UIState {
  loading: boolean;
  theme: "light" | "dark";
  networkStatus: "online" | "offline";
}

// Action 类型定义
type AuthAction =
  | { type: "LOGIN_START" }
  | {
      type: "LOGIN_SUCCESS";
      payload: {
        user: UserInfo;
        token: string;
        perms: string[];
        roles: string[];
      };
    }
  | { type: "LOGIN_FAILURE" }
  | { type: "LOGOUT" }
  | { type: "UPDATE_USER"; payload: Partial<UserInfo> }
  | {
      type: "INIT_AUTH_SUCCESS";
      payload: {
        user: UserInfo;
        token: string;
        perms: string[];
        roles: string[];
      };
    };

type UIAction =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_THEME"; payload: "light" | "dark" }
  | { type: "SET_NETWORK_STATUS"; payload: "online" | "offline" };

type GlobalAction = AuthAction | UIAction;

// Reducer 函数
const globalReducer = (
  state: GlobalState,
  action: GlobalAction
): GlobalState => {
  switch (action.type) {
    case "LOGIN_START":
      return {
        ...state,
        auth: { ...state.auth, loading: true },
      };

    case "LOGIN_SUCCESS":
      return {
        ...state,
        auth: {
          ...state.auth,
          loading: false,
          isAuthenticated: true,
          user: action.payload.user,
          token: action.payload.token,
          perms: action.payload.perms,
          roles: action.payload.roles,
        },
      };

    case "LOGIN_FAILURE":
      return {
        ...state,
        auth: {
          ...state.auth,
          loading: false,
          isAuthenticated: false,
          user: null,
          token: null,
          perms: [],
          roles: [],
        },
      };

    case "LOGOUT":
      return {
        ...state,
        auth: {
          ...state.auth,
          isAuthenticated: false,
          user: null,
          token: null,
          perms: [],
          roles: [],
        },
      };

    case "UPDATE_USER":
      return {
        ...state,
        auth: {
          ...state.auth,
          user: state.auth.user
            ? { ...state.auth.user, ...action.payload }
            : null,
        },
      };

    case "INIT_AUTH_SUCCESS":
      return {
        ...state,
        auth: {
          ...state.auth,
          isAuthenticated: true,
          user: action.payload.user,
          token: action.payload.token,
          perms: action.payload.perms,
          roles: action.payload.roles,
          loading: false,
        },
      };

    case "SET_LOADING":
      return {
        ...state,
        ui: {
          ...state.ui,
          loading: action.payload,
        },
      };

    case "SET_THEME":
      return {
        ...state,
        ui: {
          ...state.ui,
          theme: action.payload,
        },
      };

    case "SET_NETWORK_STATUS":
      return {
        ...state,
        ui: {
          ...state.ui,
          networkStatus: action.payload,
        },
      };

    default:
      return state;
  }
};

// 初始状态
const initialState: GlobalState = {
  auth: {
    user: null,
    token: null,
    isAuthenticated: false,
    perms: [],
    roles: [],
    loading: false,
  },
  ui: {
    loading: false,
    theme: "light",
    networkStatus: "online",
  },
};

// Context 创建
const GlobalContext = createContext<{
  state: GlobalState;
  dispatch: React.Dispatch<GlobalAction>;
  login: (loginData: LoginData) => Promise<UserInfo>;
  logout: () => Promise<void>;
  initAuth: () => Promise<void>;
} | null>(null);

// Provider 组件
export const GlobalProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(globalReducer, initialState);

  // 登录函数
  const login = async (loginData: LoginData): Promise<UserInfo> => {
    dispatch({ type: "LOGIN_START" });

    try {
      const { AuthAPI } = await import("dehaze-sdk-js");
      const response = await AuthAPI.login(loginData);
      const { tokenType, accessToken } = response;
      const token = `${tokenType} ${accessToken}`;

      // 先保存 token，后续请求拦截器才能读取到
      storage.setToken(token);

      // 获取用户信息（此时请求头会携带 token）
      const userInfo = await UserAPI.getInfo();

      // 存储到本地
      await storage.setUserInfo(userInfo);
      await storage.setPerms(userInfo.perms || []);
      await storage.setRoles(userInfo.roles || []);

      // 一次性更新认证状态（含权限与角色）
      dispatch({
        type: "LOGIN_SUCCESS",
        payload: {
          user: userInfo,
          token,
          perms: userInfo.perms || [],
          roles: userInfo.roles || [],
        },
      });

      return userInfo;
    } catch (error) {
      dispatch({ type: "LOGIN_FAILURE" });
      throw error;
    }
  };

  // 登出函数
  const logout = async (): Promise<void> => {
    try {
      const { AuthAPI } = await import("dehaze-sdk-js");
      await AuthAPI.logout();
    } catch (error) {
      console.error("登出接口调用失败:", error);
    } finally {
      // 清除本地存储
      storage.clearAuth();

      // 更新状态
      dispatch({ type: "LOGOUT" });
    }
  };

  // 初始化认证状态
  const initAuth = async (): Promise<void> => {
    try {
      const token = storage.getToken();
      const userInfo = await storage.getUserInfo();
      const perms = await storage.getPerms();
      const roles = await storage.getRoles();

      if (token && userInfo) {
        dispatch({
          type: "INIT_AUTH_SUCCESS",
          payload: { user: userInfo, token, perms, roles },
        });
      }
    } catch (error) {
      console.error("初始化认证状态失败:", error);
    }
  };

  const contextValue = {
    state,
    dispatch,
    login,
    logout,
    initAuth,
  };

  return (
    <GlobalContext.Provider value={contextValue}>
      {children}
    </GlobalContext.Provider>
  );
};

// 自定义 Hook
export const useGlobalContext = () => {
  const context = useContext(GlobalContext);
  if (!context) {
    throw new Error("useGlobalContext must be used within GlobalProvider");
  }
  return context;
};
