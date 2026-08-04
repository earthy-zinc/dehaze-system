/**
 * 菜单配置
 *
 * 依据《05-菜单与页面层级规划》：移动端 5 个 Tab 根页面（L1），
 * 与设计稿 dehaze-mobile 底部导航一致。
 */

/** 菜单项数据模型 */
export interface MenuItem {
  /** uview-plus 图标名 */
  icon: string;
  /** 激活状态图标 */
  activeIcon?: string;
  /** 菜单标题 */
  title: string;
  /** 路由路径 */
  route: string;
}

/** 底部 TabBar 配置（5 个顶层目的地，L1 Tab 根页面） */
export const tabBarItems: MenuItem[] = [
  {
    icon: "home",
    activeIcon: "home-fill",
    title: "首页",
    route: "/pages/home/index",
  },
  {
    icon: "grid",
    activeIcon: "grid-fill",
    title: "工具",
    route: "/pages/tools/index",
  },
  {
    icon: "gift",
    activeIcon: "gift-fill",
    title: "去雾",
    route: "/pages/dehaze/index",
  },
  {
    icon: "bell",
    activeIcon: "bell-fill",
    title: "消息",
    route: "/pages/messages/index",
  },
  {
    icon: "account",
    activeIcon: "account-fill",
    title: "我的",
    route: "/pages/user-center/index",
  },
];
