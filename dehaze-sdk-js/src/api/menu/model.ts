import { MenuTypeEnum } from "@/enums/MenuTypeEnum";

/**
 * 菜单查询参数类型
 */
export interface MenuQuery {
  keywords?: string;
  perm?: string;
  path?: string;
  /**
   * 菜单类型查询参数。后端接收整数（1=菜单；2=目录；3=外链；4=按钮），
   * 为兼容枚举名也允许传 MenuTypeEnum，故定义为联合类型。
   */
  type?: MenuTypeEnum | number;
  visible?: number;
}

/**
 * 菜单视图对象类型
 */
export interface MenuVO {
  /**
   * 子菜单
   */
  children?: MenuVO[];
  /**
   * 组件路径
   */
  component?: string;
  /**
   * ICON
   */
  icon?: string;
  /**
   * 菜单ID
   */
  id?: number;
  /**
   * 菜单名称
   */
  name?: string;
  /**
   * 父菜单ID
   */
  parentId?: number;
  /**
   * 路由路径
   */
  path?: string;
  /**
   * 按钮权限标识
   */
  perm?: string;
  /**
   * 跳转路径
   */
  redirect?: string;
  /**
   * 菜单排序(数字越小排名越靠前)
   */
  sort?: number;
  /**
   * 菜单类型
   */
  type?: MenuTypeEnum;
  /**
   * 菜单是否可见(1:显示;0:隐藏)
   */
  visible?: number;
}

/**
 * 菜单表单对象类型
 */
export interface MenuForm {
  /**
   * 菜单ID
   */
  id?: number;
  /**
   * 父菜单ID
   */
  parentId?: number;
  /**
   * 菜单名称
   */
  name?: string;
  /**
   * 菜单是否可见(1:是;0:否;)
   */
  visible: number;
  icon?: string;
  /**
   * 排序
   */
  sort: number;
  /**
   * 组件路径
   */
  component?: string;
  /**
   * 路由路径
   */
  path?: string;
  /**
   * 跳转路由路径
   */
  redirect?: string;

  /**
   * 菜单类型
   */
  type: MenuTypeEnum;

  /**
   * 权限标识
   */
  perm?: string;
  /**
   * 【菜单】是否开启页面缓存
   */
  keepAlive?: number;

  /**
   * 【目录】只有一个子路由是否始终显示
   */
  alwaysShow?: number;
}

/**
 * RouteVO，路由对象
 */
export interface RouteVO {
  /**
   * 子路由列表
   */
  children: RouteVO[];
  /**
   * 组件路径，可为字符串路径或动态加载的 React 组件
   */
  component?: any;
  meta?: Meta;
  /**
   * 路由名称
   */
  name?: string;
  /**
   * 路由路径
   */
  path?: string;
  /**
   * 跳转链接
   */
  redirect?: string;
}

/**
 * Meta，路由属性类型
 */
export interface Meta {
  /**
   * 【目录】只有一个子路由是否始终显示
   */
  alwaysShow?: boolean;
  /**
   * 是否隐藏(true-是 false-否)
   */
  hidden?: boolean;
  /**
   * ICON
   */
  icon?: string;
  /**
   * 【菜单】是否开启页面缓存
   */
  keepAlive?: boolean;
  /**
   * 拥有路由权限的角色编码
   */
  roles?: string[];
  /**
   * 路由title
   */
  title?: string;
}
