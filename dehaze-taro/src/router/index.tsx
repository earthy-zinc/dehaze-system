import React from 'react';

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

  // 图像处理流程
  imageInput: React.lazy(() => import('@/pages/image-input/index')),

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
  {
    path: '/pages/image-input/index',
    component: pages.imageInput,
    meta: {
      title: '图像输入',
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

// 所有路由合并（用于权限查询）
const allRoutes = [...publicRoutes, ...protectedRoutes];

// 根据路径获取路由元信息（纯函数，不调用 Hooks）
export const getRouteMeta = (path: string): RouteMeta | undefined => {
  return allRoutes.find(route => route.path === path)?.meta;
};
