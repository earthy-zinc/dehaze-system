# React Native 路由和页面跳转指南

在 React Native 应用中，路由管理是一个核心功能。本项目使用 React Navigation 作为路由解决方案，它提供了完整的导航功能。

## 1. 安装依赖

我们已经添加了以下依赖到项目中：

```json
{
  "dependencies": {
    "@react-navigation/native": "^6.1.7",
    "@react-navigation/native-stack": "^6.9.13",
    "react-native-screens": "^3.22.1",
    "react-native-gesture-handler": "^2.12.1",
    "react-native-reanimated": "^3.4.2"
  }
}
```

## 2. 路由配置

### 2.1 创建导航容器

我们创建了 [NavigationContainer.tsx](file:///e:/DehazeSystem/dehaze-react-native/src/navigation/NavigationContainer.tsx) 文件来管理应用的路由：

```typescript
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import LoginScreen from '../pages/login';
import HomeScreen from '../pages/home';

// 定义路由参数类型
export type RootStackParamList = {
  Login: undefined;
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
```

### 2.2 在 App.tsx 中使用导航容器

```typescript
import { StatusBar, useColorScheme } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigator from './navigation/NavigationContainer';

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <AppNavigator />
    </SafeAreaProvider>
  );
}

export default App;
```

## 3. 页面跳转

### 3.1 在页面中使用导航功能

在登录页面中，我们通过 navigation 对象实现页面跳转：

```typescript
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../../navigation/NavigationContainer';

type LoginScreenProps = NativeStackScreenProps<RootStackParamList, 'Login'>;

const LoginScreen: React.FC<LoginScreenProps> = ({ navigation }) => {
  const handleLogin = () => {
    // 登录逻辑...
    // 登录成功后跳转到主页
    navigation.navigate('Home');
  };
};
```

### 3.2 常用的导航方法

1. `navigation.navigate('Home')` - 跳转到指定页面
2. `navigation.goBack()` - 返回上一页
3. `navigation.push('Home')` - 推入新页面（即使该页面已在栈中）
4. `navigation.pop()` - 弹出当前页面
5. `navigation.popToTop()` - 返回到栈顶页面

## 4. 传递参数

### 4.1 定义带参数的路由

```typescript
export type RootStackParamList = {
  Login: undefined;
  Home: { userId?: string; userName?: string };
};
```

### 4.2 传递参数

```typescript
navigation.navigate('Home', {
  userId: '123',
  userName: '张三'
});
```

### 4.3 接收参数

```typescript
type HomeScreenProps = NativeStackScreenProps<RootStackParamList, 'Home'>;

const HomeScreen: React.FC<HomeScreenProps> = ({ route }) => {
  const { userId, userName } = route.params;
  // 使用参数...
};
```

## 5. 运行项目

安装新依赖后，需要重新构建项目：

```bash
# 对于 Android
yarn android

# 对于 iOS
yarn ios
```

## 6. 更多导航类型

除了堆栈导航(Stack Navigator)，React Navigation 还提供了其他导航类型：

1. **Tab Navigator** - 底部或顶部标签导航
2. **Drawer Navigator** - 侧边栏导航

可以根据项目需求选择合适的导航类型。