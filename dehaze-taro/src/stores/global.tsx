import React, { createContext, useContext, useReducer } from "react";
import { storage } from "@/utils/storage";
import { type UserInfo, type LoginData, UserAPI } from "dehaze-sdk-js";

interface GlobalState {
  auth: AuthState;
  ui: UIState;
}

interface AuthState {
  user: UserInfo | null;
  sessionId: string | null;
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

type AuthAction =
  | { type: "LOGIN_START" }
  | {
      type: "LOGIN_SUCCESS";
      payload: {
        user: UserInfo;
        sessionId: string;
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
        sessionId: string;
        perms: string[];
        roles: string[];
      };
    };

type UIAction =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_THEME"; payload: "light" | "dark" }
  | { type: "SET_NETWORK_STATUS"; payload: "online" | "offline" };

type GlobalAction = AuthAction | UIAction;

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
          sessionId: action.payload.sessionId,
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
          sessionId: null,
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
          sessionId: null,
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
          sessionId: action.payload.sessionId,
          perms: action.payload.perms,
          roles: action.payload.roles,
          loading: false,
        },
      };

    case "SET_LOADING":
      return {
        ...state,
        ui: { ...state.ui, loading: action.payload },
      };

    case "SET_THEME":
      return {
        ...state,
        ui: { ...state.ui, theme: action.payload },
      };

    case "SET_NETWORK_STATUS":
      return {
        ...state,
        ui: { ...state.ui, networkStatus: action.payload },
      };

    default:
      return state;
  }
};

const initialState: GlobalState = {
  auth: {
    user: null,
    sessionId: null,
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

const GlobalContext = createContext<{
  state: GlobalState;
  dispatch: React.Dispatch<GlobalAction>;
  login: (loginData: LoginData) => Promise<UserInfo>;
  logout: () => Promise<void>;
  initAuth: () => Promise<void>;
} | null>(null);

export const GlobalProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [state, dispatch] = useReducer(globalReducer, initialState);

  const login = async (loginData: LoginData): Promise<UserInfo> => {
    dispatch({ type: "LOGIN_START" });

    try {
      const { AuthAPI } = await import("dehaze-sdk-js");
      const response = await AuthAPI.login(loginData);

      storage.setSessionId(response.sessionId);

      const userInfo = await UserAPI.getInfo();

      await storage.setUserInfo(userInfo);
      await storage.setPerms(userInfo.perms || []);
      await storage.setRoles(userInfo.roles || []);

      dispatch({
        type: "LOGIN_SUCCESS",
        payload: {
          user: userInfo,
          sessionId: response.sessionId,
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

  const logout = async (): Promise<void> => {
    try {
      const { AuthAPI } = await import("dehaze-sdk-js");
      await AuthAPI.logout();
    } catch (error) {
      console.error("登出接口调用失败:", error);
    } finally {
      storage.clearAuth();
      dispatch({ type: "LOGOUT" });
    }
  };

  const initAuth = async (): Promise<void> => {
    try {
      const sessionId = storage.getSessionId();
      const userInfo = await storage.getUserInfo();
      const perms = await storage.getPerms();
      const roles = await storage.getRoles();

      if (sessionId && userInfo) {
        dispatch({
          type: "INIT_AUTH_SUCCESS",
          payload: { user: userInfo, sessionId, perms, roles },
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

export const useGlobalContext = () => {
  const context = useContext(GlobalContext);
  if (!context) {
    throw new Error("useGlobalContext must be used within GlobalProvider");
  }
  return context;
};
