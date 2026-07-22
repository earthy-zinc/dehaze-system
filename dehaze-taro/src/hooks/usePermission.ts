import { useMemo } from "react";
import { hasPermission, hasRole, isSuperAdmin } from "@/utils/permission";
import { useAuth } from "./useAuth";

export const usePermission = () => {
  const { permissions, roles } = useAuth();

  // 检查是否有指定权限
  const hasPermissionFn = useMemo(
    () =>
      (permission: string | string[]): boolean => {
        return hasPermission(permissions, permission);
      },
    [permissions]
  );

  // 检查是否有指定角色
  const hasRoleFn = useMemo(
    () =>
      (role: string | string[]): boolean => {
        return hasRole(roles, role);
      },
    [roles]
  );

  // 检查是否为超级管理员
  const isSuperAdminFn = useMemo(
    () => (): boolean => {
      return isSuperAdmin(roles);
    },
    [roles]
  );

  return {
    // 权限列表
    permissions,
    roles,

    // 权限检查方法
    hasPermission: hasPermissionFn,
    hasRole: hasRoleFn,
    isSuperAdmin: isSuperAdminFn,
  };
};
