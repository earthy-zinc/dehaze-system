import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import LoginScreen from '../pages/login';
// 导入其他页面组件
import HomeScreen from '../pages/home';

export type RootStackParamList = {
  Login: undefined;
  // 添加其他路由
  Home: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Login">
        <Stack.Screen 
          name="Login" 
          component={LoginScreen} 
          options={{ title: '登录' }} 
        />
        {/* 添加其他屏幕 */}
        <Stack.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{ title: '主页' }} 
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default AppNavigator;