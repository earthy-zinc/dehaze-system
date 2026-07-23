/**
 * 主布局组件
 *
 * 提供响应式布局，支持：
 * - 移动端：底部导航栏 + 抽屉菜单
 * - 平板/桌面：侧边导航栏
 *
 * 参考 Flutter 应用的整体设计风格
 */
import React, { useState, useCallback } from 'react';
import { View, StyleSheet } from 'react-native';
import { useNavigation, useRoute, NavigationProp } from '@react-navigation/native';
import { useResponsive } from '@/hooks/useResponsive';
import { colors } from '@/theme/colors';
import {
  AppHeader,
  BottomTabBar,
  DrawerMenu,
  SideNav,
} from './components';
import type { RootStackParamList, RouteKeys } from '@/routes/types';

interface MainLayoutProps {
  children: React.ReactNode;
  title?: string;
  showBack?: boolean;
  showBottomNav?: boolean;
}

const MainLayout: React.FC<MainLayoutProps> = ({
  children,
  title,
  showBack = false,
  showBottomNav = true,
}) => {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();
  const route = useRoute();
  const { isMobile } = useResponsive();
  const [drawerVisible, setDrawerVisible] = useState(false);

  // 当前路由名称
  const currentRoute = route.name as RouteKeys;

  // 导航处理
  const handleNavigate = useCallback(
    (routeName: RouteKeys) => {
      navigation.navigate(routeName);
    },
    [navigation],
  );

  // 返回处理
  const handleBack = useCallback(() => {
    if (navigation.canGoBack()) {
      navigation.goBack();
    }
  }, [navigation]);

  // 移动端布局
  if (isMobile) {
    return (
      <View style={styles.container}>
        {/* 顶部导航栏 */}
        <AppHeader
          title={title}
          showBack={showBack}
          showMenu={!showBack}
          onBackPress={handleBack}
          onMenuPress={() => setDrawerVisible(true)}
        />

        {/* 内容区域 */}
        <View style={styles.content}>{children}</View>

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
          onClose={() => setDrawerVisible(false)}
          onNavigate={handleNavigate}
        />
      </View>
    );
  }

  // 平板/桌面端布局
  return (
    <View style={styles.container}>
      {/* 顶部导航栏 */}
      <AppHeader
        title={title}
        showBack={showBack}
        showMenu={false}
        onBackPress={handleBack}
      />

      {/* 主体区域：侧边栏 + 内容 */}
      <View style={styles.body}>
        {/* 侧边导航栏 */}
        <SideNav currentRoute={currentRoute} onNavigate={handleNavigate} />

        {/* 内容区域 */}
        <View style={styles.content}>{children}</View>
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
