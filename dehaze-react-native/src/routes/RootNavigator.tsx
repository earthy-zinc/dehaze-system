/**
 * RootNavigator — 根导航组件
 *
 * 根据认证状态条件渲染：
 * - loading：启动屏
 * - 未认证：AuthStack（Login/Register）
 * - 已认证：MainTabs（5 Tab 底部导航）
 */
import React from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuthStore } from '@/store';
import LoginScreen from '@/pages/login';
import RegisterScreen from '@/pages/register';
import MainTabs from './MainTabs';
import type { AuthStackParamList } from './types';

const AuthStack = createNativeStackNavigator<AuthStackParamList>();

function AuthNavigator() {
  return (
    <AuthStack.Navigator screenOptions={{ headerShown: false }}>
      <AuthStack.Screen name="Login" component={LoginScreen} />
      <AuthStack.Screen name="Register" component={RegisterScreen} />
    </AuthStack.Navigator>
  );
}

function SplashLoading() {
  return (
    <View style={styles.splash}>
      <ActivityIndicator size="large" />
    </View>
  );
}

export default function RootNavigator() {
  const sessionId = useAuthStore(s => s.sessionId);
  const loading = useAuthStore(s => s.loading);

  if (loading) {
    return <SplashLoading />;
  }

  const isAuthenticated = !!sessionId;

  return (
    <NavigationContainer>
      {isAuthenticated ? <MainTabs /> : <AuthNavigator />}
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  splash: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
