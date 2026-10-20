/**
 * 权限工具函数
 */

/**
 * 单个权限项匹配。
 *
 * - 精确码（`sys:user:add`）：要求用户权限集合中存在同名项。
 * - 模块前缀（`sys:user:*` / 以 `:` 结尾的串）：命中该模块任一具体权限即可，
 *   与 Flutter `matchModulePerm` / Android `hasModulePerm` / Taro 仪表盘口径一致。
 *   后端权限全部是具体码（无通配符），入口显隐只能按模块前缀判定。
 */
function matchOne(userPermissions: string[], required: string): boolean {
  if (!required) return false;
  if (required.endsWith(":*")) {
    const prefix = required.slice(0, -1); // "sys:user:"
    return userPermissions.some(
      (p) => p === prefix.slice(0, -1) || p.startsWith(prefix)
    );
  }
  return userPermissions.includes(required);
}

/**
 * 检查是否有指定权限
 * @param userPermissions 用户权限列表
 * @param requiredPermissions 需要的权限列表（任意命中即通过）
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

  return permissions.some((permission) =>
    matchOne(userPermissions, permission)
  );
};
