/**
 * 菜单配置
 *
 * 依据《05-菜单与页面层级规划》：移动端 5 个 Tab 根页面（L1），
 * 与设计稿 dehaze-mobile 底部导航一致。底部导航由原生 tabBar 渲染
 * （app.config.ts tabBar 配置），此处仅提供路由集合供跳转判断使用。
 */

/** 菜单项数据模型 */
export interface MenuItem {
  /** 菜单标题 */
  title: string;
  /** 路由路径 */
  route: string;
}

/** 底部 TabBar 配置（5 个顶层目的地，L1 Tab 根页面） */
export const tabBarItems: MenuItem[] = [
  {
    title: "首页",
    route: "/pages/home/index",
  },
  {
    title: "工具",
    route: "/pages/tools/index",
  },
  {
    title: "去雾",
    route: "/pages/dehaze/index",
  },
  {
    title: "消息",
    route: "/pages/messages/index",
  },
  {
    title: "我的",
    route: "/pages/profile/index",
  },
];
