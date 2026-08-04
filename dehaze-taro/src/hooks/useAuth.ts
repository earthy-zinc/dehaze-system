import { useShallow } from "zustand/react/shallow";
import { useAuthStore } from "@/stores/global";

/**
 * 认证状态与方法的统一入口。
 * useShallow 浅比较：字段未变化时返回稳定引用，可安全放入 useEffect 依赖数组。
 */
export const useAuth = () =>
  useAuthStore(
    useShallow((s) => ({
      user: s.user,
      sessionId: s.sessionId,
      isAuthenticated: s.isAuthenticated,
      perms: s.perms,
      roles: s.roles,
      login: s.login,
      logout: s.logout,
      initAuth: s.initAuth,
    }))
  );
