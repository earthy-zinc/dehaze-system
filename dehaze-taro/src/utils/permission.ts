/**
 * 权限工具函数
 */

/**
 * 检查是否有指定权限
 * @param userPermissions 用户权限列表
 * @param requiredPermissions 需要的权限列表
 * @returns 是否有权限
 */
export const hasPermission = (
  userPermissions: string[],
  requiredPermissions: string | string[]
): boolean => {
  if (!userPermissions || userPermissions.length === 0) {
    return false;
  }

  const permissions = Array.isArray(requiredPermissions)
    ? requiredPermissions
    : [requiredPermissions];

  return permissions.some((permission) => userPermissions.includes(permission));
};
