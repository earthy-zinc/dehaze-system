/**
 * 菜单配置
 * 统一管理所有菜单项数据，与 Flutter 应用保持一致
 */

export type RouteNames =
  | 'Home'
  | 'ImageInput'
  | 'AlgorithmSelect'
  | 'Processing'
  | 'SideBySide'
  | 'Overlay'
  | 'Magnifier'
  | 'Filter'
  | 'Metrics'
  | 'Algorithm'
  | 'Dataset';

export interface MenuItemData {
  icon: string;
  title: string;
  route: RouteNames;
  badge?: string;
  isNew?: boolean;
}

export interface MenuSection {
  title: string;
  icon?: string;
  items: MenuItemData[];
}

// 首页菜单项
export const homeItem: MenuItemData = {
  icon: 'home-outline',
  title: '首页',
  route: 'Home',
};

// 菜单分组配置
export const menuSections: MenuSection[] = [
  {
    title: '处理流程',
    icon: 'play-circle-outline',
    items: [
      {
        icon: 'image-outline',
        title: '图像输入',
        route: 'ImageInput',
      },
      {
        icon: 'bulb-outline',
        title: '算法选择',
        route: 'AlgorithmSelect',
      },
      {
        icon: 'settings-outline',
        title: '去雾处理',
        route: 'Processing',
      },
    ],
  },
  {
    title: '效果对比',
    icon: 'git-compare-outline',
    items: [
      {
        icon: 'albums-outline',
        title: '并排对比',
        route: 'SideBySide',
      },
      {
        icon: 'layers-outline',
        title: '重叠对比',
        route: 'Overlay',
      },
      {
        icon: 'search-outline',
        title: '放大镜',
        route: 'Magnifier',
      },
      {
        icon: 'options-outline',
        title: '滤镜调节',
        route: 'Filter',
      },
      {
        icon: 'bar-chart-outline',
        title: '指标评估',
        route: 'Metrics',
      },
      {
        icon: 'information-circle-outline',
        title: '算法信息',
        route: 'Algorithm',
      },
    ],
  },
  {
    title: '数据管理',
    icon: 'folder-outline',
    items: [
      {
        icon: 'server-outline',
        title: '数据集管理',
        route: 'Dataset',
      },
    ],
  },
];

// 底部导航栏配置
export const bottomTabs: MenuItemData[] = [
  {
    icon: 'home',
    title: '首页',
    route: 'Home',
  },
  {
    icon: 'image',
    title: '输入',
    route: 'ImageInput',
  },
  {
    icon: 'bulb',
    title: '算法',
    route: 'AlgorithmSelect',
  },
  {
    icon: 'cog',
    title: '处理',
    route: 'Processing',
  },
  {
    icon: 'albums',
    title: '对比',
    route: 'SideBySide',
  },
];

// 获取所有菜单项（平铺）
export const getAllMenuItems = (): MenuItemData[] => {
  const items: MenuItemData[] = [homeItem];
  menuSections.forEach(section => {
    items.push(...section.items);
  });
  return items;
};

// 根据路由查找菜单项
export const findMenuItemByRoute = (
  route: RouteNames,
): MenuItemData | undefined => {
  if (homeItem.route === route) return homeItem;
  for (const section of menuSections) {
    const item = section.items.find(i => i.route === route);
    if (item) return item;
  }
  return undefined;
};

// 根据路由获取所属分组
export const findSectionByRoute = (
  route: RouteNames,
): MenuSection | undefined => {
  for (const section of menuSections) {
    if (section.items.some(item => item.route === route)) {
      return section;
    }
  }
  return undefined;
};
