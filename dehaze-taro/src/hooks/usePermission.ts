import { useCallback } from "react";
import { hasPermission } from "@/utils/permission";
import { useAuth } from "./useAuth";

export const usePermission = () => {
  const { permissions } = useAuth();

  // 检查是否有指定权限
  const hasPermissionFn = useCallback(
    (permission: string | string[]): boolean => {
      return hasPermission(permissions, permission);
    },
    [permissions]
  );

  return {
    // 权限列表
    permissions,

    // 权限检查方法
    hasPermission: hasPermissionFn,
  };
};
