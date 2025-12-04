import React, { Suspense } from 'react';
import Taro from '@tarojs/taro';
import { useAuth } from '@/hooks/useAuth';
import { usePermission } from '@/hooks/usePermission';
import PermissionGuard from '@/components/system/PermissionGuard';
import Loading from '@/components/common/Loading';

// 路由元信息定义
export interface RouteMeta {
  title: string;
  requiresAuth?: boolean;
  permissions?: string[];
  roles?: string[];
  keepAlive?: boolean;
  hidden?: boolean;
  icon?: string;
  sort?: number;
}

// 扩展路由对象
export interface AppRouteObject {
  path: string;
  component?: React.ComponentType<any>;
  meta?: RouteMeta;
  children?: AppRouteObject[];
}

// 页面组件（懒加载）
const pages = {
  // 首页
  home: React.lazy(() => import('@/pages/home/index')),

  // 认证相关
  login: React.lazy(() => import('@/pages/login/index')),

  // 数据集管理
  dataset: React.lazy(() => import('@/pages/dataset/index')),

  // 系统管理
  dashboard: React.lazy(() => import('@/pages/dashboard/index')),
  userList: React.lazy(() => import('@/pages/system/user/index')),
  userDetail: React.lazy(() => import('@/pages/system/user/detail')),
  roleList: React.lazy(() => import('@/pages/system/role/index')),
  rolePermission: React.lazy(() => import('@/pages/system/role/permission')),
};

// 公共路由（无需权限）
export const publicRoutes: AppRouteObject[] = [
  {
    path: '/pages/home/index',
    component: pages.home,
    meta: {
      title: '首页',
      requiresAuth: false,
    },
  },
  {
    path: '/pages/dataset/index',
    component: pages.dataset,
    meta: {
      title: '数据集管理',
      requiresAuth: false,
    },
  },
  {
    path: '/pages/login/index',
    component: pages.login,
    meta: {
      title: '登录',
      requiresAuth: false,
    },
  },
];

// 受保护的路由
export const protectedRoutes: AppRouteObject[] = [
  {
    path: '/pages/dashboard/index',
    component: pages.dashboard,
    meta: {
      title: '首页',
      requiresAuth: true,
    },
  },
  {
    path: '/pages/system/user/index',
    component: pages.userList,
    meta: {
      title: '用户管理',
      requiresAuth: true,
      permissions: ['sys:user:list'],
      keepAlive: true,
    },
  },
  {
    path: '/pages/system/user/detail',
    component: pages.userDetail,
    meta: {
      title: '用户详情',
      requiresAuth: true,
      permissions: ['sys:user:add', 'sys:user:edit'],
      hidden: true,
    },
  },
  {
    path: '/pages/system/role/index',
    component: pages.roleList,
    meta: {
      title: '角色管理',
      requiresAuth: true,
      permissions: ['sys:role:list'],
      keepAlive: true,
    },
  },
  {
    path: '/pages/system/role/permission',
    component: pages.rolePermission,
    meta: {
      title: '权限分配',
      requiresAuth: true,
      permissions: ['sys:role:permission'],
      hidden: true,
    },
  },
];

// 路由守卫组件
const RouteGuard: React.FC<{
  route: AppRouteObject;
  children: React.ReactNode;
}> = ({ route, children }) => {
  const { isAuthenticated } = useAuth();

  // 如果需要认证但未登录，跳转到登录页
  if (route.meta?.requiresAuth && !isAuthenticated) {
    Taro.redirectTo({
      url: '/pages/login/index'
    });
    return <Loading />;
  }

  // 如果有权限要求，检查权限
  if (route.meta?.permissions || route.meta?.roles) {
    return (
      <PermissionGuard
        permissions={route.meta.permissions}
        roles={route.meta.roles}
        requireAuth={route.meta.requiresAuth}
      >
        {children}
      </PermissionGuard>
    );
  }

  return <>{children}</>;
};

// 路由渲染组件
const RouteRenderer: React.FC<{ route: AppRouteObject }> = ({ route }) => {
  if (!route.component) {
    return null;
  }

  return (
    <RouteGuard route={route}>
      <Suspense fallback={<Loading />}>
        <route.component />
      </Suspense>
    </RouteGuard>
  );
};

// 路由映射表
export const routeMap = new Map<string, AppRouteObject>();

// 初始化路由映射
[...publicRoutes, ...protectedRoutes].forEach(route => {
  routeMap.set(route.path, route);
});

// 获取当前路由信息
export const getCurrentRoute = (path: string): AppRouteObject | undefined => {
  return routeMap.get(path);
};

// 检查路由权限
export const checkRoutePermission = (route: AppRouteObject): boolean => {
  const { hasPermission, hasRole, isSuperAdmin } = usePermission();

  // 超级管理员拥有所有权限
  if (isSuperAdmin()) {
    return true;
  }

  // 检查权限
  if (route.meta?.permissions && !hasPermission(route.meta.permissions)) {
    return false;
  }

  // 检查角色
  if (route.meta?.roles && !hasRole(route.meta.roles)) {
    return false;
  }

  return true;
};

// 生成菜单数据
export const generateMenuData = () => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return [];
  }

  return protectedRoutes
    .filter(route => !route.meta?.hidden && checkRoutePermission(route))
    .map(route => ({
      id: route.path,
      title: route.meta?.title || route.path,
      icon: route.meta?.icon,
      path: route.path,
      sort: route.meta?.sort || 999,
    }))
    .sort((a, b) => a.sort - b.sort);
};
