/**
 * 通用页面布局组件
 * 封装导航栏、侧边栏、底部导航的统一布局
 */
import React, { useState } from 'react';
import { View } from '@tarojs/components';
import Taro from '@tarojs/taro';
import AppNavbar from './navbar';
import AppSidebar from './sidebar';
import AppTabbar from './tabbar';
import './index.less';

interface PageLayoutProps {
  /** 子元素 */
  children: React.ReactNode;
  /** 是否显示底部导航栏 */
  showTabbar?: boolean;
  /** 是否显示顶部导航栏 */
  showNavbar?: boolean;
  /** 页面标题 */
  title?: string;
  /** 当前路由 */
  currentRoute?: string;
  /** 搜索回调 */
  onSearch?: () => void;
  /** 导航回调 */
  onNavigate?: (route: string) => void;
}

const PageLayout: React.FC<PageLayoutProps> = ({
  children,
  showTabbar = true,
  showNavbar = true,
  title = '图像去雾系统',
  currentRoute,
  onSearch,
  onNavigate,
}) => {
  const [sidebarVisible, setSidebarVisible] = useState(false);

  // 获取当前路由
  const getCurrentRoute = (): string => {
    if (currentRoute) return currentRoute;
    const pages = Taro.getCurrentPages();
    if (pages.length > 0) {
      const page = pages[pages.length - 1];
      return '/' + (page.route || 'pages/home/index');
    }
    return '/pages/home/index';
  };

  const route = getCurrentRoute();

  return (
    <View className='page-layout'>
      {/* 顶部导航栏 */}
      {showNavbar && (
        <AppNavbar
          title={title}
          onToggleMenu={() => setSidebarVisible(true)}
          onSearch={onSearch}
        />
      )}

      {/* 侧边菜单 */}
      <AppSidebar
        visible={sidebarVisible}
        currentRoute={route}
        onClose={() => setSidebarVisible(false)}
        onNavigate={onNavigate}
      />

      {/* 主内容区 */}
      <View className={`page-content ${showTabbar ? 'with-tabbar' : ''}`}>
        {children}
      </View>

      {/* 底部导航栏 */}
      {showTabbar && <AppTabbar currentRoute={route} />}
    </View>
  );
};

export default PageLayout;
