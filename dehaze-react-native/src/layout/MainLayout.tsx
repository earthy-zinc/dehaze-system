/**
 * 主布局组件
 *
 * 提供响应式布局，支持：
 * - 移动端：底部导航栏 + 抽屉菜单
 * - 平板/桌面：侧边导航栏
 *
 * 参考 Flutter 应用的整体设计风格
 */
import React, { useState, useCallback, useMemo } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { useNavigation, useRoute, NavigationProp } from '@react-navigation/native';
import { useResponsive } from '@/hooks/useResponsive';
import { colors } from '@/theme/colors';
import { RouteNames } from './MenuConfig';
import {
  AppHeader,
  BottomTabBar,
  DrawerMenu,
  SideNav,
  SIDE_NAV_WIDTH,
} from './components';
import type { RootStackParamList } from '@/routes/types';

interface MainLayoutProps {
  children: React.ReactNode;
  title?: string;
  showBack?: boolean;
  showHeader?: boolean;
  showBottomNav?: boolean;
  headerRightActions?: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  title,
  showBack = false,
  showHeader = true,
  showBottomNav = true,
  headerRightActions,
}) => {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();
  const route = useRoute();
  const { isMobile, isTablet, isDesktop } = useResponsive();
  const [drawerVisible, setDrawerVisible] = useState(false);

  // 判断是否为宽屏设备（平板/桌面）
  const isWideScreen = isTablet || isDesktop;

  // 当前路由名称
  const currentRoute = route.name as RouteNames;

  // 打开抽屉菜单
  const handleOpenDrawer = useCallback(() => {
    setDrawerVisible(true);
  }, []);

  // 关闭抽屉菜单
  const handleCloseDrawer = useCallback(() => {
    setDrawerVisible(false);
  }, []);

  // 导航处理
  const handleNavigate = useCallback(
    (routeName: RouteNames) => {
      navigation.navigate(routeName as keyof RootStackParamList);
    },
    [navigation],
  );

  // 返回处理
  const handleBack = useCallback(() => {
    if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }, [navigation]);

  // 渲染内容区域
  const renderContent = useMemo(
    () => (
      <View style={styles.content}>
        {children}
      </View>
    ),
    [children],
  );

  // 移动端布局
  if (isMobile) {
    return (
      <View style={styles.container}>
        {/* 顶部导航栏 */}
        {showHeader && (
          <AppHeader
            title={title}
            showBack={showBack}
            showMenu={!showBack}
            onBackPress={handleBack}
            onMenuPress={handleOpenDrawer}
            rightActions={headerRightActions}
          />
        )}

        {/* 内容区域 */}
        {renderContent}

        {/* 底部导航栏 */}
        {showBottomNav && (
          <BottomTabBar
            currentRoute={currentRoute}
            onTabPress={handleNavigate}
          />
        )}

        {/* 抽屉菜单 */}
        <DrawerMenu
          visible={drawerVisible}
          currentRoute={currentRoute}
          onClose={handleCloseDrawer}
          onNavigate={handleNavigate}
        />
      </View>
    );
  }

  // 平板/桌面端布局
  return (
    <View style={styles.container}>
      {/* 顶部导航栏 */}
      {showHeader && (
        <AppHeader
          title={title}
          showBack={showBack}
          showMenu={false}
          onBackPress={handleBack}
          rightActions={headerRightActions}
        />
      )}

      {/* 主体区域：侧边栏 + 内容 */}
      <View style={styles.body}>
        {/* 侧边导航栏 */}
        <SideNav currentRoute={currentRoute} onNavigate={handleNavigate} />

        {/* 内容区域 */}
        {renderContent}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.secondary,
  },
  body: {
    flex: 1,
    flexDirection: 'row',
  },
  content: {
    flex: 1,
    backgroundColor: colors.background.secondary,
  },
});

export default MainLayout;
