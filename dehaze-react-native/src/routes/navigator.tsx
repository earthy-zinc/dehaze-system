import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { useAuth } from '@/store';
import { protectedRoutes, publicRoutes } from './config';
import type { RootStackParamList } from './types';

const Stack = createNativeStackNavigator<RootStackParamList>();

function SplashLoading() {
  return (
    <View style={styles.splash}>
      <ActivityIndicator size="large" />
    </View>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

/**
 * 路由管理器 + 鉴权守卫
 *
 * 根据认证状态条件渲染路由：
 * - loading 中：显示启动屏
 * - 未认证：仅渲染 Login
 * - 已认证：渲染除 Login 外的所有受保护路由
 */
export const RouteManager = () => {
  const { isAuthenticated, state } = useAuth();

  if (state.loading) {
    return <SplashLoading />;
  }

  const routes = isAuthenticated ? protectedRoutes : publicRoutes;

  return (
    <Stack.Navigator
      initialRouteName={isAuthenticated ? 'Home' : 'Login'}
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      {routes.map(route => (
        <Stack.Screen
          key={route.name}
          name={route.name}
          component={route.component}
          options={route.title ? { title: route.title } : undefined}
        />
      ))}
    </Stack.Navigator>
  );
};
