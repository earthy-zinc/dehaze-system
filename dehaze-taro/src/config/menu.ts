/**
 * 菜单配置
 * 统一管理所有菜单项数据，便于维护和修改
 * 与 Flutter 版本 menu_config.dart 保持一致
 */

/** 菜单项数据模型 */
export interface MenuItem {
  /** 图标名（taroify icons） */
  icon: string;
  /** 激活状态图标 */
  activeIcon?: string;
  /** 菜单标题 */
  title: string;
  /** 路由路径 */
  route: string;
  /** 角标文字（如 "NEW"、数字等） */
  badge?: string | number;
  /** 是否为新功能 */
  isNew?: boolean;
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
  icon: 'home-o',
  activeIcon: 'wap-home',
  title: '首页',
  route: '/pages/home/index',
};

/** 分组菜单配置 */
export const menuSections: MenuSection[] = [
  {
    title: '处理流程',
    icon: 'play-circle-o',
    items: [
      {
        icon: 'photograph',
        activeIcon: 'photograph',
        title: '图像输入',
        route: '/pages/image-input/index',
      },
      {
        icon: 'bulb-o',
        activeIcon: 'bulb-o',
        title: '算法选择',
        route: '/pages/algorithm-select/index',
      },
      {
        icon: 'setting-o',
        activeIcon: 'setting',
        title: '去雾处理',
        route: '/pages/processing/index',
      },
    ],
  },
  {
    title: '效果对比',
    icon: 'apps-o',
    items: [
      {
        icon: 'apps-o',
        activeIcon: 'apps-o',
        title: '并排对比',
        route: '/pages/side-by-side/index',
      },
      {
        icon: 'photo-o',
        activeIcon: 'photo',
        title: '重叠对比',
        route: '/pages/overlay/index',
      },
      {
        icon: 'search',
        activeIcon: 'search',
        title: '放大镜',
        route: '/pages/magnifier/index',
      },
      {
        icon: 'filter-o',
        activeIcon: 'filter-o',
        title: '滤镜调节',
        route: '/pages/filter/index',
      },
      {
        icon: 'bar-chart-o',
        activeIcon: 'bar-chart-o',
        title: '指标评估',
        route: '/pages/metrics/index',
      },
      {
        icon: 'info-o',
        activeIcon: 'info',
        title: '算法信息',
        route: '/pages/algorithm/index',
      },
    ],
  },
  {
    title: '数据管理',
    icon: 'orders-o',
    items: [
      {
        icon: 'orders-o',
        activeIcon: 'orders-o',
        title: '数据集管理',
        route: '/pages/dataset/index',
      },
    ],
  },
];

/** 底部 TabBar 配置（移动端核心入口） */
export const tabBarItems: MenuItem[] = [
  {
    icon: 'home-o',
    activeIcon: 'wap-home',
    title: '首页',
    route: '/pages/home/index',
  },
  {
    icon: 'photograph',
    activeIcon: 'photograph',
    title: '输入',
    route: '/pages/image-input/index',
  },
  {
    icon: 'bulb-o',
    activeIcon: 'bulb-o',
    title: '算法',
    route: '/pages/algorithm-select/index',
  },
  {
    icon: 'setting-o',
    activeIcon: 'setting',
    title: '处理',
    route: '/pages/processing/index',
  },
  {
    icon: 'apps-o',
    activeIcon: 'apps-o',
    title: '对比',
    route: '/pages/side-by-side/index',
  },
];

/** TabBar 页面路由列表 */
export const tabBarRoutes: string[] = tabBarItems.map((item) => item.route);

/**
 * 获取所有菜单项（平铺，包含首页）
 */
export const getAllMenuItems = (): MenuItem[] => {
  return [homeItem, ...menuSections.flatMap((s) => s.items)];
};

/**
 * 获取所有菜单项（不包含首页）
 */
export const getMenuItemsWithoutHome = (): MenuItem[] => {
  return menuSections.flatMap((s) => s.items);
};

/**
 * 根据路由查找菜单项
 */
export const findMenuItemByRoute = (route: string): MenuItem | undefined => {
  return getAllMenuItems().find((item) => item.route === route);
};

/**
 * 检查路由是否存在于菜单中
 */
export const containsRoute = (route: string): boolean => {
  return findMenuItemByRoute(route) !== undefined;
};

/**
 * 根据路由获取所属分组
 */
export const findSectionByRoute = (route: string): MenuSection | undefined => {
  return menuSections.find((section) => section.items.some((item) => item.route === route));
};

/**
 * 获取菜单项的索引（用于底部导航栏等）
 */
export const getMenuItemIndex = (route: string): number => {
  const items = getAllMenuItems();
  const index = items.findIndex((item) => item.route === route);
  return index !== -1 ? index : 0;
};

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
