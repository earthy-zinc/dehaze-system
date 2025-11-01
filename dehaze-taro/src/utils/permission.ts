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
 * 权限装饰器工厂（用于函数）
 * @param permissions 需要的权限
 */
export const requirePermission = (permissions: string | string[]) => {
  return (target: any, propertyKey: string, descriptor: PropertyDescriptor) => {
    const originalMethod = descriptor.value;

    descriptor.value = function(...args: any[]) {
      // 在实际环境中，这里可以检查当前用户的权限
      // 如果没有权限，可以抛出异常或返回错误
      console.log(`检查权限: ${permissions}`);
      return originalMethod.apply(this, args);
    };

    return descriptor;
  };
};

/**
 * 角色装饰器工厂（用于函数）
 * @param roles 需要的角色
 */
export const requireRole = (roles: string | string[]) => {
  return (target: any, propertyKey: string, descriptor: PropertyDescriptor) => {
    const originalMethod = descriptor.value;

    descriptor.value = function(...args: any[]) {
      // 在实际环境中，这里可以检查当前用户的角色
      // 如果没有对应角色，可以抛出异常或返回错误
      console.log(`检查角色: ${roles}`);
      return originalMethod.apply(this, args);
    };

    return descriptor;
  };
};

/**
 * 根据权限过滤菜单项
 * @param menus 菜单列表
 * @param permissions 用户权限列表
 * @param roles 用户角色列表
 * @returns 过滤后的菜单列表
 */
export const filterMenusByPermission = (
  menus: any[],
  permissions: string[],
  roles: string[]
): any[] => {
  return menus.filter(menu => {
    // 如果是超级管理员，直接通过
    if (isSuperAdmin(roles)) {
      return true;
    }

    // 检查菜单权限
    if (menu.meta?.permissions) {
      const hasMenuPermission = hasPermission(permissions, menu.meta.permissions);
      if (!hasMenuPermission) {
        return false;
      }
    }

    // 检查菜单角色
    if (menu.meta?.roles) {
      const hasMenuRole = hasRole(roles, menu.meta.roles);
      if (!hasMenuRole) {
        return false;
      }
    }

    // 递归检查子菜单
    if (menu.children && menu.children.length > 0) {
      menu.children = filterMenusByPermission(menu.children, permissions, roles);
      // 如果所有子菜单都被过滤掉了，那么当前菜单也不显示
      return menu.children.length > 0;
    }

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