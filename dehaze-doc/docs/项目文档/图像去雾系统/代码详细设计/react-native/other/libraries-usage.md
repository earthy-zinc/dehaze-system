# React Native 库使用指南

本文档详细介绍了项目中使用的各种 React Native 库及其使用方法和示例。

## 1. 导航相关库

### 1.1 @react-navigation/native 和 @react-navigation/native-stack

这两个库是 React Native 应用中最常用的导航库，用于实现页面之间的跳转和导航。

#### 安装

```bash
npm install @react-navigation/native @react-navigation/native-stack
```

#### 基本使用

1. **创建导航容器**

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

2. **在页面中使用导航功能**

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
  
  return (
    // 组件内容
  );
};
```

3. **传递参数**

```typescript
// 定义带参数的路由
export type RootStackParamList = {
  Login: undefined;
  Home: { userId?: string; userName?: string };
};

// 传递参数
navigation.navigate('Home', {
  userId: '123',
  userName: '张三'
});

// 接收参数
type HomeScreenProps = NativeStackScreenProps<RootStackParamList, 'Home'>;
const HomeScreen: React.FC<HomeScreenProps> = ({ route }) => {
  const { userId, userName } = route.params;
  // 使用参数...
};
```

#### 常用导航方法

- `navigation.navigate('Home')` - 跳转到指定页面
- `navigation.goBack()` - 返回上一页
- `navigation.push('Home')` - 推入新页面（即使该页面已在栈中）
- `navigation.pop()` - 弹出当前页面
- `navigation.popToTop()` - 返回到栈顶页面

### 1.2 react-native-screens

这个库用于优化导航性能，通过使用原生屏幕组件来提高应用性能。

#### 安装

```bash
npm install react-native-screens
```

#### 使用

通常与 React Navigation 一起使用，无需额外配置，React Navigation 会自动使用该库。

### 1.3 react-native-safe-area-context

用于处理不同设备的安全区域（如刘海屏、状态栏等），确保内容不会被设备特定区域遮挡。

#### 安装

```bash
npm install react-native-safe-area-context
```

#### 使用

1. **包装应用根组件**

```typescript
import { SafeAreaProvider } from 'react-native-safe-area-context';

function App() {
  return (
    <SafeAreaProvider>
      <AppNavigator />
    </SafeAreaProvider>
  );
}
```

2. **在组件中使用 SafeAreaView**

```typescript
import { SafeAreaView } from 'react-native-safe-area-context';

const MyComponent = () => {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <View>
        <Text>我的内容</Text>
      </View>
    </SafeAreaView>
  );
};
```

## 2. 状态管理库

### 2.1 redux, react-redux 和 redux-thunk

这些库用于管理应用的全局状态，处理异步操作。

#### 安装

```bash
npm install redux react-redux redux-thunk
```

#### 基本使用

1. **创建 Store**

```javascript
import { createStore, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';
import rootReducer from './reducers';

const store = createStore(
  rootReducer,
  applyMiddleware(thunk)
);

export default store;
```

2. **创建 Reducer**

```javascript
// src/reducers/postsReducer.js
const initialState = {
  loading: false,
  posts: [],
  error: ''
};

const postsReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'FETCH_POSTS_REQUEST':
      return {
        ...state,
        loading: true
      };
    case 'FETCH_POSTS_SUCCESS':
      return {
        loading: false,
        posts: action.payload,
        error: ''
      };
    case 'FETCH_POSTS_FAILURE':
      return {
        loading: false,
        posts: [],
        error: action.payload
      };
    default:
      return state;
  }
};

export default postsReducer;
```

3. **创建 Action**

```javascript
// src/actions/postActions.js
export const fetchPostsRequest = () => {
  return {
    type: 'FETCH_POSTS_REQUEST'
  };
};

export const fetchPostsSuccess = (posts) => {
  return {
    type: 'FETCH_POSTS_SUCCESS',
    payload: posts
  };
};

export const fetchPostsFailure = (error) => {
  return {
    type: 'FETCH_POSTS_FAILURE',
    payload: error
  };
};

// 异步 Action (使用 redux-thunk)
export const fetchPosts = () => {
  return (dispatch) => {
    dispatch(fetchPostsRequest());
    fetch('https://jsonplaceholder.typicode.com/posts')
      .then(response => response.json())
      .then(data => dispatch(fetchPostsSuccess(data)))
      .catch(error => dispatch(fetchPostsFailure(error.message)));
  };
};
```

4. **在组件中使用**

```javascript
import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchPosts } from './actions/postActions';

const PostsComponent = () => {
  const dispatch = useDispatch();
  const { loading, posts, error } = useSelector(state => state.posts);

  useEffect(() => {
    dispatch(fetchPosts());
  }, [dispatch]);

  if (loading) return <Text>加载中...</Text>;
  if (error) return <Text>错误: {error}</Text>;

  return (
    <View>
      {posts.map(post => (
        <Text key={post.id}>{post.title}</Text>
      ))}
    </View>
  );
};
```

## 3. 存储库

### 3.1 @react-native-async-storage/async-storage

用于在设备上持久化存储键值对数据，类似于浏览器的 localStorage。

#### 安装

```bash
npm install @react-native-async-storage/async-storage
```

#### 使用

1. **基本操作**

```javascript
import AsyncStorage from '@react-native-async-storage/async-storage';

// 存储数据
const storeData = async (key, value) => {
  try {
    await AsyncStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    // 保存错误
    console.error('保存数据失败:', e);
  }
};

// 读取数据
const getData = async (key) => {
  try {
    const value = await AsyncStorage.getItem(key);
    return value != null ? JSON.parse(value) : null;
  } catch(e) {
    // 读取错误
    console.error('读取数据失败:', e);
    return null;
  }
};

// 删除数据
const removeValue = async (key) => {
  try {
    await AsyncStorage.removeItem(key);
  } catch(e) {
    // 删除错误
    console.error('删除数据失败:', e);
  }
};

// 清空所有数据
const clearAll = async () => {
  try {
    await AsyncStorage.clear();
  } catch(e) {
    // 清空错误
    console.error('清空数据失败:', e);
  }
};
```

2. **使用示例**

```javascript
const UserStorage = {
  // 存储用户信息
  storeUser: async (user) => {
    try {
      await AsyncStorage.setItem('user', JSON.stringify(user));
    } catch (e) {
      console.error('存储用户信息失败:', e);
    }
  },

  // 获取用户信息
  getUser: async () => {
    try {
      const user = await AsyncStorage.getItem('user');
      return user != null ? JSON.parse(user) : null;
    } catch (e) {
      console.error('获取用户信息失败:', e);
      return null;
    }
  },

  // 删除用户信息
  removeUser: async () => {
    try {
      await AsyncStorage.removeItem('user');
    } catch (e) {
      console.error('删除用户信息失败:', e);
    }
  }
};

export default UserStorage;
```

## 4. 手势和动画库

### 4.1 react-native-gesture-handler

提供原生级别的手势处理能力，优化触摸响应的性能和体验。

#### 安装

```bash
npm install react-native-gesture-handler
```

#### 使用

1. **基本使用**

```javascript
import { PanGestureHandler } from 'react-native-gesture-handler';

const MyComponent = () => {
  const onGestureEvent = (event) => {
    // 处理手势事件
    console.log('手势事件:', event.nativeEvent);
  };

  return (
    <PanGestureHandler onGestureEvent={onGestureEvent}>
      <View style={{ width: 100, height: 100, backgroundColor: 'blue' }} />
    </PanGestureHandler>
  );
};
```

2. **与 React Navigation 集成**

在应用入口文件顶部添加导入：

```javascript
import 'react-native-gesture-handler';
```

### 4.2 react-native-reanimated

提供高性能动画库，用于创建复杂的动画和交互。

#### 安装

```bash
npm install react-native-reanimated
```

#### 使用

1. **配置 Babel**

在 babel.config.js 中添加插件：

```javascript
module.exports = {
  presets: ['module:metro-react-native-babel-preset'],
  plugins: [
    'react-native-reanimated/plugin', // 必须放在最后
  ],
};
```

2. **基本动画**

```javascript
import Animated, {
  useSharedValue,
  withTiming,
  useAnimatedStyle,
} from 'react-native-reanimated';

const FadeInView = () => {
  const opacity = useSharedValue(0);

  useEffect(() => {
    // 动画效果：2秒内透明度从0变为1
    opacity.value = withTiming(1, { duration: 2000 });
  }, []);

  const animatedStyle = useAnimatedStyle(() => {
    return {
      opacity: opacity.value,
    };
  });

  return (
    <Animated.View style={[{ width: 100, height: 100, backgroundColor: 'blue' }, animatedStyle]} />
  );
};
```

3. **手势动画**

```javascript
import React from 'react';
import { View } from 'react-native';
import { PanGestureHandler } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  useAnimatedGestureHandler,
} from 'react-native-reanimated';

const DraggableBox = () => {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);

  const gestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = translateX.value;
      context.startY = translateY.value;
    },
    onActive: (event, context) => {
      translateX.value = context.startX + event.translationX;
      translateY.value = context.startY + event.translationY;
    },
    onEnd: () => {
      // 可选：添加回弹动画
    },
  });

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
      ],
    };
  });

  return (
    <PanGestureHandler onGestureEvent={gestureHandler}>
      <Animated.View 
        style={[
          { width: 100, height: 100, backgroundColor: 'blue' }, 
          animatedStyle
        ]} 
      />
    </PanGestureHandler>
  );
};
```

## 5. 图像处理库

### 5.1 react-native-image-picker

允许用户从设备相册选择图片或使用相机拍照。

#### 安装

```bash
npm install react-native-image-picker
```

#### 使用

```javascript
import { launchImageLibrary, launchCamera } from 'react-native-image-picker';

const ImagePickerExample = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  const selectImage = () => {
    const options = {
      mediaType: 'photo',
      quality: 0.8,
    };

    launchImageLibrary(options, (response) => {
      if (response.didCancel) {
        console.log('用户取消了选择');
      } else if (response.error) {
        console.log('选择图片出错:', response.error);
      } else {
        const source = { uri: response.assets[0].uri };
        setSelectedImage(source);
      }
    });
  };

  const takePhoto = () => {
    const options = {
      mediaType: 'photo',
      quality: 0.8,
    };

    launchCamera(options, (response) => {
      if (response.didCancel) {
        console.log('用户取消了拍照');
      } else if (response.error) {
        console.log('拍照出错:', response.error);
      } else {
        const source = { uri: response.assets[0].uri };
        setSelectedImage(source);
      }
    });
  };

  return (
    <View>
      {selectedImage && (
        <Image 
          source={selectedImage} 
          style={{ width: 200, height: 200 }} 
        />
      )}
      <Button title="选择图片" onPress={selectImage} />
      <Button title="拍照" onPress={takePhoto} />
    </View>
  );
};
```

### 5.2 react-native-canvas

允许在 React Native 中使用 Canvas 进行绘图操作。

#### 安装

```bash
npm install react-native-canvas
```

#### 使用

```javascript
import Canvas from 'react-native-canvas';

const CanvasExample = () => {
  const handleCanvas = (canvas) => {
    if (canvas) {
      const ctx = canvas.getContext('2d');
      canvas.width = 200;
      canvas.height = 200;
      
      // 绘制矩形
      ctx.fillStyle = 'blue';
      ctx.fillRect(10, 10, 100, 100);
      
      // 绘制圆形
      ctx.beginPath();
      ctx.arc(150, 150, 50, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
  };

  return (
    <Canvas ref={handleCanvas} />
  );
};
```

## 总结

这些库为 React Native 应用提供了完整的功能支持：

- 导航库提供了页面跳转和路由管理
- 状态管理库处理全局状态和异步操作
- 存储库实现数据持久化
- 手势和动画库提供流畅的交互体验
- 图像处理库支持图片选择和绘制功能

合理使用这些库可以大大提高开发效率和用户体验。
