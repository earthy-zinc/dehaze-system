import { useGlobalContext } from "@/stores/global";

export const useAuth = () => {
  const { state, login, logout, initAuth } = useGlobalContext();

  return {
    // 认证状态
    ...state.auth,

    // 认证方法
    login,
    logout,
    initAuth,

    // 便捷属性
    isAuthenticated: state.auth.isAuthenticated,
    isLoading: state.auth.loading,
  };
};
