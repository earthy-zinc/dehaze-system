export interface RouterInfo {
  // 路由跳转地址
  path: string;
  // 当前路由的名称，需要唯一
  name: string;
  // 路由组件
  component: React.ReactNode;
  // 用于菜单导航，点击该路由跳转到redirect所指的路径
  redirect?: string;
  // 路由的基本信息
  meta?: RouteMeta;
  // 子路由，如果子路由只有一个，则不会展示在菜单上，点击当前路由会跳转到其唯一子路由
  children?: RouterInfo[];
}

export interface RouteMeta {
  /** 菜单名称 */
  title?: string;
  /** 菜单图标  */
  icon?: React.ReactNode;
  /** 菜单是否隐藏，True则不会展示在菜单上 */
  hidden?: boolean;
  /** 是否固定页签 */
  affix?: boolean;
  /** 是否缓存页面 */
  keepAlive?: boolean;
  /** 是否在面包屑上隐藏 */
  breadcrumb?: boolean;
  /** 拥有菜单权限的角色编码集合 */
  roles?: string[];
}
