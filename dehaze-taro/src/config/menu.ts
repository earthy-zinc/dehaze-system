/**
 * 菜单配置
 * 统一管理所有菜单项数据，便于维护和修改
 * 与 Flutter 版本 menu_config.dart 保持一致
 */

/** 菜单项数据模型 */
export interface MenuItem {
  /** 图标名（taroify icons） */
  icon: string;
  /** 菜单标题 */
  title: string;
  /** 路由路径 */
  route: string;
}

/** 菜单分组数据模型 */
export interface MenuSection {
  /** 分组标题 */
  title: string;
  /** 分组图标（可选） */
  icon?: string;
  /** 分组下的菜单项 */
  items: MenuItem[];
}

/** 首页菜单项 */
export const homeItem: MenuItem = {
  icon: "home-o",
  title: "首页",
  route: "/pages/home/index",
};

/** 分组菜单配置 */
export const menuSections: MenuSection[] = [
  {
    title: "处理流程",
    icon: "play-circle-o",
    items: [
      {
        icon: "photograph",
        title: "图像输入",
        route: "/pages/image-input/index",
      },
      {
        icon: "bulb-o",
        title: "算法选择",
        route: "/pages/algorithm-select/index",
      },
      {
        icon: "setting-o",
        title: "去雾处理",
        route: "/pages/processing/index",
      },
    ],
  },
  {
    title: "效果对比",
    icon: "apps-o",
    items: [
      {
        icon: "apps-o",
        title: "并排对比",
        route: "/pages/side-by-side/index",
      },
      {
        icon: "photo-o",
        title: "重叠对比",
        route: "/pages/overlay/index",
      },
      {
        icon: "search",
        title: "放大镜",
        route: "/pages/magnifier/index",
      },
      {
        icon: "filter-o",
        title: "滤镜调节",
        route: "/pages/filter/index",
      },
      {
        icon: "bar-chart-o",
        title: "指标评估",
        route: "/pages/metrics/index",
      },
      {
        icon: "info-o",
        title: "算法信息",
        route: "/pages/algorithm/index",
      },
    ],
  },
  {
    title: "数据管理",
    icon: "orders-o",
    items: [
      {
        icon: "orders-o",
        title: "数据集管理",
        route: "/pages/dataset/index",
      },
    ],
  },
  {
    title: "个人中心",
    icon: "manager-o",
    items: [
      {
        icon: "manager-o",
        title: "个人中心",
        route: "/pages/profile/index",
      },
    ],
  },
];

/** 底部 TabBar 配置（移动端核心入口） */
export const tabBarItems: MenuItem[] = [
  {
    icon: "home-o",
    title: "首页",
    route: "/pages/home/index",
  },
  {
    icon: "photograph",
    title: "输入",
    route: "/pages/image-input/index",
  },
  {
    icon: "bulb-o",
    title: "算法",
    route: "/pages/algorithm-select/index",
  },
  {
    icon: "manager-o",
    title: "我的",
    route: "/pages/profile/index",
  },
];

/** TabBar 页面路由列表 */
const tabBarRoutes: string[] = tabBarItems.map((item) => item.route);

/**
 * 获取 TabBar 项的索引
 */
export const getTabBarIndex = (route: string): number => {
  const index = tabBarItems.findIndex((item) => item.route === route);
  return index !== -1 ? index : 0;
};

/**
 * 判断是否为 TabBar 页面
 */
export const isTabBarPage = (route: string): boolean => {
  return tabBarRoutes.includes(route);
};
