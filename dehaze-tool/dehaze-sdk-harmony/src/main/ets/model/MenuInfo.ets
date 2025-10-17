/**
 * 菜单类型枚举
 */
export enum MenuTypeEnum {
  /**
   * 目录
   */
  CATALOG = "CATALOG",
  /**
   * 菜单
   */
  MENU = "MENU",
  /**
   * 按钮
   */
  BUTTON = "BUTTON",
  /**
   * 外链
   */
  EXTLINK = "EXTLINK"
}

/**
 * 菜单查询参数类型
 */
export class MenuQuery {
  keywords?: string = undefined;
}

/**
 * 菜单视图对象类型
 */
export class MenuVO {
  /**
   * 子菜单
   */
  children?: Array<MenuVO> = undefined;
  /**
   * 组件路径
   */
  component?: string = undefined;
  /**
   * ICON
   */
  icon?: string = undefined;
  /**
   * 菜单ID
   */
  id?: number = undefined;
  /**
   * 菜单名称
   */
  name?: string = undefined;
  /**
   * 父菜单ID
   */
  parentId?: number = undefined;
  /**
   * 按钮权限标识
   */
  perm?: string = undefined;
  /**
   * 跳转路径
   */
  redirect?: string = undefined;
  /**
   * 路由名称
   */
  routeName?: string = undefined;
  /**
   * 路由相对路径
   */
  routePath?: string = undefined;
  /**
   * 菜单排序(数字越小排名越靠前)
   */
  sort?: number = undefined;
  /**
   * 菜单类型
   */
  type?: MenuTypeEnum = undefined;
  /**
   * 菜单是否可见(1:显示;0:隐藏)
   */
  visible?: number = undefined;
}

/**
 * 菜单表单对象类型
 */
export class MenuForm {
  /**
   * 菜单ID
   */
  id?: string = undefined;
  /**
   * 父菜单ID
   */
  parentId?: number = undefined;
  /**
   * 菜单名称
   */
  name?: string = undefined;
  /**
   * 菜单是否可见(1:是;0:否;)
   */
  visible: number = 1;
  icon?: string = undefined;
  /**
   * 排序
   */
  sort: number = 0;
  /**
   * 组件路径
   */
  component?: string = undefined;
  /**
   * 路由路径
   */
  path?: string = undefined;
  /**
   * 跳转路由路径
   */
  redirect?: string = undefined;

  /**
   * 菜单类型
   */
  type: MenuTypeEnum = MenuTypeEnum.MENU;

  /**
   * 权限标识
   */
  perm?: string = undefined;
  /**
   * 【菜单】是否开启页面缓存
   */
  keepAlive?: number = undefined;

  /**
   * 【目录】只有一个子路由是否始终显示
   */
  alwaysShow?: number = undefined;
}

/**
 * RouteVO，路由对象
 */
export class RouteVO {
  /**
   * 子路由列表
   */
  children: Array<RouteVO> = [];
  /**
   * 组件路径
   */
  component?: string = undefined;
  meta?: Meta = undefined;
  /**
   * 路由名称
   */
  name?: string = undefined;
  /**
   * 路由路径
   */
  path?: string = undefined;
  /**
   * 跳转链接
   */
  redirect?: string = undefined;
}

/**
 * Meta，路由属性类型
 */
export class Meta {
  /**
   * 【目录】只有一个子路由是否始终显示
   */
  alwaysShow?: boolean = undefined;
  /**
   * 是否隐藏(true-是 false-否)
   */
  hidden?: boolean = undefined;
  /**
   * ICON
   */
  icon?: string = undefined;
  /**
   * 【菜单】是否开启页面缓存
   */
  keepAlive?: boolean = undefined;
  /**
   * 拥有路由权限的角色编码
   */
  roles?: Array<string> = undefined;
  /**
   * 路由title
   */
  title?: string = undefined;
}