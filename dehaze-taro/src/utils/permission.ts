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

  return permissions.some(permission =>
    userPermissions.includes(permission)
  );
};

/**
 * 检查是否有指定角色
 * @param userRoles 用户角色列表
 * @param requiredRoles 需要的角色列表
 * @returns 是否有角色
 */
export const hasRole = (
  userRoles: string[],
  requiredRoles: string | string[]
): boolean => {
  if (!userRoles || userRoles.length === 0) {
    return false;
  }

  const roles = Array.isArray(requiredRoles) ? requiredRoles : [requiredRoles];

  return roles.some(role => userRoles.includes(role));
};

/**
 * 检查是否为超级管理员
 * @param userRoles 用户角色列表
 * @returns 是否为超级管理员
 */
export const isSuperAdmin = (userRoles: string[]): boolean => {
  return userRoles.includes('ROOT');
};

/**
 * 根据权限过滤菜单项
 * @param menus 菜单列表
 * @param permissions 用户权限列表
 * @param roles 用户角色列表
 * @returns 过滤后的菜单列表（不可变，返回新数组）
 */
export const filterMenusByPermission = (
  menus: any[],
  permissions: string[],
  roles: string[]
): any[] => {
  return menus
    .map(menu => {
      // 如果是超级管理员，直接通过
      if (isSuperAdmin(roles)) {
        return menu;
      }

      // 检查菜单权限
      if (menu.meta?.permissions) {
        const hasMenuPermission = hasPermission(permissions, menu.meta.permissions);
        if (!hasMenuPermission) {
          return null;
        }
      }

      // 检查菜单角色
      if (menu.meta?.roles) {
        const hasMenuRole = hasRole(roles, menu.meta.roles);
        if (!hasMenuRole) {
          return null;
        }
      }

      // 递归检查子菜单（不可变，返回新对象）
      if (menu.children && menu.children.length > 0) {
        const filteredChildren = filterMenusByPermission(menu.children, permissions, roles);
        return { ...menu, children: filteredChildren };
      }

      return menu;
    })
    .filter(menu => {
      if (!menu) return false;
      // 如果有子菜单但都被过滤掉了，则不显示
      if (menu.children && menu.children.length === 0) return false;
      return true;
    });
};

/**
 * 生成权限标识符
 * @param module 模块名
 * @param action 操作名
 * @returns 权限标识符
 */
export const generatePermission = (module: string, action: string): string => {
  return `${module}:${action}`;
};

/**
 * 解析权限标识符
 * @param permission 权限标识符
 * @returns 解析后的模块和操作
 */
export const parsePermission = (permission: string): { module: string; action: string } => {
  const [module, action] = permission.split(':');
  return { module, action };
};

/**
 * 批量检查权限
 * @param userPermissions 用户权限列表
 * @param permissionMap 权限映射表
 * @returns 权限检查结果
 */
export const checkMultiplePermissions = (
  userPermissions: string[],
  permissionMap: Record<string, string[]>
): Record<string, boolean> => {
  const result: Record<string, boolean> = {};

  Object.entries(permissionMap).forEach(([key, permissions]) => {
    result[key] = hasPermission(userPermissions, permissions);
  });

  return result;
};
