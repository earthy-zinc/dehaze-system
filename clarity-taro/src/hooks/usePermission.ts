import { useCallback } from "react";
import { hasPermission } from "@/utils/permission";
import { useAuth } from "./useAuth";

export const usePermission = () => {
  const { perms } = useAuth();

  const hasPermissionFn = useCallback(
    (permission: string | string[]): boolean => {
      return hasPermission(perms, permission);
    },
    [perms]
  );

  return {
    perms,
    hasPermission: hasPermissionFn,
  };
};
