me# React Native 导航器和扩展库使用指南

本文档详细介绍新增的 React Native 导航器和扩展库的使用方法和示例。

## 1. react-native-webview

用于在 React Native 应用中显示网页内容。

### 安装

```bash
npm install react-native-webview
```

### 使用示例

```typescript
import React from 'react';
import { View, StyleSheet } from 'react-native';
import { WebView } from 'react-native-webview';

const WebViewExample = () => {
  return (
    <View style={styles.container}>
      <WebView
        source={{ uri: 'https://reactnative.dev/' }}
        style={styles.webview}
        onLoadProgress={({ nativeEvent }) => {
          console.log('加载进度:', nativeEvent.progress);
        }}
        onError={(syntheticEvent) => {
          const { nativeEvent } = syntheticEvent;
          console.warn('WebView 错误:', nativeEvent);
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  webview: {
    flex: 1,
  },
});

export default WebViewExample;
```

## 2. react-native-worklets

用于在 React Native 中运行 JavaScript 工作线程。

### 安装

```bash
npm install react-native-worklets
```

### 使用示例

```typescript
import { Worklets } from 'react-native-worklets';

// 创建一个在 UI 线程运行的工作单元
const uiWorklet = Worklets.createRunInUIContext((value) => {
  'worklet';
  console.log('在 UI 线程运行:', value);
});

// 创建一个在 JS 线程运行的工作单元
const jsWorklet = Worklets.createRunInJsContext((value) => {
  'worklet';
  console.log('在 JS 线程运行:', value);
  return value * 2;
});

// 使用示例
const ExampleComponent = () => {
  const handlePress = async () => {
    // 在 JS 线程运行
    const result = await jsWorklet(10);
    console.log('结果:', result);
    
    // 在 UI 线程运行
    uiWorklet(result);
  };

  return (
    // 组件内容
  );
};
```

## 3. react-native-pager-view

提供可滑动的页面视图组件，通常与标签导航器配合使用。

### 安装

```bash
npm install react-native-pager-view
```

### 使用示例

```typescript
import React, { useRef } from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import PagerView from 'react-native-pager-view';

const PagerViewExample = () => {
  const pagerRef = useRef(null);

  const setPage = (pageIndex) => {
    pagerRef.current?.setPage(pageIndex);
  };

  return (
    <View style={styles.container}>
      <View style={styles.buttonContainer}>
        <Button title="第一页" onPress={() => setPage(0)} />
        <Button title="第二页" onPress={() => setPage(1)} />
        <Button title="第三页" onPress={() => setPage(2)} />
      </View>
      
      <PagerView
        style={styles.pagerView}
        initialPage={0}
        ref={pagerRef}>
        <View key="1" style={styles.page}>
          <Text>第一页内容</Text>
        </View>
        <View key="2" style={styles.page}>
          <Text>第二页内容</Text>
        </View>
        <View key="3" style={styles.page}>
          <Text>第三页内容</Text>
        </View>
      </PagerView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 10,
  },
  pagerView: {
    flex: 1,
  },
  page: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default PagerViewExample;
```

## 4. @react-navigation/native-stack - 原生栈导航器

创建原生平台的栈式导航器。

### 使用示例

```typescript
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/HomeScreen';
import DetailsScreen from './screens/DetailsScreen';

// 定义路由参数类型
export type RootStackParamList = {
  Home: undefined;
  Details: { itemId: number; otherParam?: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerShown: true,
          headerTintColor: '#007AFF',
        }}>
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{
            title: '主页',
            headerBackTitleVisible: false,
          }}
        />
        <Stack.Screen
          name="Details"
          component={DetailsScreen}
          options={({ route }) => ({
            title: `详情 - ${route.params.itemId}`,
          })}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default App;
```

## 5. @react-navigation/bottom-tabs - 底部标签导航器

创建底部标签导航界面。

### 使用示例

```typescript
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons'; // 需要安装 @expo/vector-icons
import HomeScreen from './screens/HomeScreen';
import SettingsScreen from './screens/SettingsScreen';

// 定义标签路由参数类型
export type BottomTabParamList = {
  Home: undefined;
  Settings: undefined;
};

const Tab = createBottomTabNavigator<BottomTabParamList>();

function BottomTabs() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;
            
            if (route.name === 'Home') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Settings') {
              iconName = focused ? 'settings' : 'settings-outline';
            }
            
            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'gray',
        })}>
        <Tab.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{
            title: '主页',
            headerShown: false,
          }} 
        />
        <Tab.Screen 
          name="Settings" 
          component={SettingsScreen} 
          options={{
            title: '设置',
          }} 
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

export default BottomTabs;
```

## 6. @react-navigation/drawer - 抽屉导航器

创建侧边栏抽屉导航界面。

### 使用示例

```typescript
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { DrawerContentScrollView, DrawerItemList } from '@react-navigation/drawer';
import HomeScreen from './screens/HomeScreen';
import ProfileScreen from './screens/ProfileScreen';

// 定义抽屉路由参数类型
export type DrawerParamList = {
  Home: undefined;
  Profile: undefined;
};

const Drawer = createDrawerNavigator<DrawerParamList>();

// 自定义抽屉内容
function CustomDrawerContent(props) {
  return (
    <DrawerContentScrollView {...props}>
      <DrawerItemList {...props} />
      {/* 可以添加自定义内容 */}
    </DrawerContentScrollView>
  );
}

function DrawerNavigator() {
  return (
    <NavigationContainer>
      <Drawer.Navigator
        drawerContent={(props) => <CustomDrawerContent {...props} />}
        screenOptions={{
          drawerActiveTintColor: '#007AFF',
          drawerPosition: 'left',
        }}>
        <Drawer.Screen 
          name="Home" 
          component={HomeScreen} 
          options={{
            title: '主页',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="home-outline" color={color} size={size} />
            ),
          }} 
        />
        <Drawer.Screen 
          name="Profile" 
          component={ProfileScreen} 
          options={{
            title: '个人资料',
            drawerIcon: ({ color, size }) => (
              <Ionicons name="person-outline" color={color} size={size} />
            ),
          }} 
        />
      </Drawer.Navigator>
    </NavigationContainer>
  );
}

export default DrawerNavigator;
```

## 7. @react-navigation/material-top-tabs - 顶部标签导航器

创建 Material Design 风格的顶部标签导航器。

### 使用示例

```typescript
import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import { View, Text, StyleSheet } from 'react-native';

// 定义顶部标签路由参数类型
export type MaterialTopTabParamList = {
  Tab1: undefined;
  Tab2: undefined;
  Tab3: undefined;
};

const Tab = createMaterialTopTabNavigator<MaterialTopTabParamList>();

// 示例标签页组件
function Tab1Screen() {
  return (
    <View style={styles.screen}>
      <Text>标签页 1</Text>
    </View>
  );
}

function Tab2Screen() {
  return (
    <View style={styles.screen}>
      <Text>标签页 2</Text>
    </View>
  );
}

function Tab3Screen() {
  return (
    <View style={styles.screen}>
      <Text>标签页 3</Text>
    </View>
  );
}

function MaterialTopTabs() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'gray',
          tabBarIndicatorStyle: {
            backgroundColor: '#007AFF',
          },
          tabBarLabelStyle: {
            fontSize: 14,
            fontWeight: 'bold',
          },
          tabBarStyle: {
            backgroundColor: 'white',
          },
        }}>
        <Tab.Screen 
          name="Tab1" 
          component={Tab1Screen} 
          options={{ title: '标签1' }} 
        />
        <Tab.Screen 
          name="Tab2" 
          component={Tab2Screen} 
          options={{ title: '标签2' }} 
        />
        <Tab.Screen 
          name="Tab3" 
          component={Tab3Screen} 
          options={{ title: '标签3' }} 
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});

export default MaterialTopTabs;
```

## 8. react-native-axios 与 axios 的区别与使用

### 8.1 两者的区别

1. **axios**:
   - 通用的 HTTP 客户端，可在浏览器和 Node.js 环境中使用
   - 在 React Native 中也能正常工作
   - 功能全面，社区支持广泛
   - 项目中已使用

2. **react-native-axios**:
   - 专门为 React Native 环境优化的 axios 版本
   - 针对移动端网络环境进行了优化
   - 修复了一些在 React Native 环境中可能出现的问题
   - 更好地适配 React Native 的网络请求机制

### 8.2 使用方法

#### 使用 axios（项目中已使用）

```javascript
import axios from 'axios';

// 创建实例
const apiClient = axios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// GET 请求
const fetchData = async () => {
  try {
    const response = await apiClient.get('/users');
    console.log(response.data);
  } catch (error) {
    console.error('请求错误:', error);
  }
};

// POST 请求
const postData = async (data) => {
  try {
    const response = await apiClient.post('/users', data);
    console.log(response.data);
  } catch (error) {
    console.error('请求错误:', error);
  }
};

// 添加请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 在发送请求之前做些什么
    console.log('发送请求:', config);
    return config;
  },
  (error) => {
    // 对请求错误做些什么
    return Promise.reject(error);
  }
);

// 添加响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    // 对响应数据做点什么
    return response;
  },
  (error) => {
    // 对响应错误做点什么
    if (error.response?.status === 401) {
      // 处理未授权错误
      console.log('未授权，请重新登录');
    }
    return Promise.reject(error);
  }
);
```

#### 使用 react-native-axios

```javascript
import rnAxios from 'react-native-axios';

// 创建实例
const apiClient = rnAxios.create({
  baseURL: 'https://api.example.com',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// GET 请求
const fetchData = async () => {
  try {
    const response = await apiClient.get('/users');
    console.log(response.data);
  } catch (error) {
    console.error('请求错误:', error);
  }
};

// POST 请求
const postData = async (data) => {
  try {
    const response = await apiClient.post('/users', data);
    console.log(response.data);
  } catch (error) {
    console.error('请求错误:', error);
  }
};
```

### 8.3 在项目中的选择建议

对于当前项目，建议继续使用 **axios**，原因如下：

1. **项目中已集成** - package.json 中已经包含了 axios 依赖
2. **功能满足需求** - axios 已经能够满足项目的网络请求需求
3. **社区支持** - axios 有更大的社区支持和更丰富的文档
4. **兼容性** - axios 在 React Native 中运行良好，无需更换

只有在遇到特定的 React Native 网络请求问题时，才考虑替换为 react-native-axios。

## 9. lottie-react-native - 动画库

### 9.1 用途介绍

Lottie 是一个由 Airbnb 开发的动画库，可以将 After Effects 动画导出为 JSON 格式，并在移动端和 Web 端原生渲染。lottie-react-native 是 Lottie 在 React Native 中的实现。

#### 主要用途

1. **高质量动画** - 渲染复杂的矢量动画，保持高质量和流畅性
2. **文件体积小** - 相比 GIF 动画，JSON 文件体积更小
3. **可控制性** - 可以精确控制动画的播放、暂停、速度等
4. **跨平台** - 同一个动画文件可以在 iOS 和 Android 上运行
5. **设计师友好** - 设计师可以使用熟悉的 After Effects 工具制作动画

### 9.2 安装方法

```bash
# 安装主库
npm install lottie-react-native

# 对于 iOS，还需要安装 lottie-ios
npm install lottie-ios
```

### 9.3 基本使用

#### 1. 准备动画文件

首先需要获取 Lottie 动画文件（JSON 格式），可以从以下途径获取：

- [LottieFiles](https://lottiefiles.com/) - 免费的动画资源网站
- 自行使用 After Effects 制作并导出
- 设计师提供

将动画文件放置在项目中，例如 `assets/animations/loading.json`

#### 2. 基本使用示例

```javascript
import React, { useRef } from 'react';
import { View, StyleSheet, Button } from 'react-native';
import LottieView from 'lottie-react-native';

const LottieExample = () => {
  const animationRef = useRef(null);

  const playAnimation = () => {
    animationRef.current?.play();
  };

  const resetAnimation = () => {
    animationRef.current?.reset();
  };

  return (
    <View style={styles.container}>
      <LottieView
        ref={animationRef}
        source={require('../assets/animations/loading.json')}
        style={styles.animation}
        autoPlay
        loop
      />
      
      <View style={styles.buttonContainer}>
        <Button title="播放" onPress={playAnimation} />
        <Button title="重置" onPress={resetAnimation} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  animation: {
    width: 200,
    height: 200,
  },
  buttonContainer: {
    flexDirection: 'row',
    marginTop: 20,
    gap: 10,
  },
});

export default LottieExample;
```

### 9.4 高级使用方法

#### 1. 控制动画进度

```javascript
import React, { useRef } from 'react';
import { View, StyleSheet, Slider } from 'react-native';
import LottieView from 'lottie-react-native';

const ProgressControlExample = () => {
  const animationRef = useRef(null);

  const onProgressChange = (progress) => {
    animationRef.current?.play(progress * 100, progress * 100);
  };

  return (
    <View style={styles.container}>
      <LottieView
        ref={animationRef}
        source={require('../assets/animations/like.json')}
        style={styles.animation}
      />
      
      <Slider
        style={styles.slider}
        minimumValue={0}
        maximumValue={1}
        onValueChange={onProgressChange}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  animation: {
    width: 200,
    height: 200,
  },
  slider: {
    width: 300,
    height: 40,
    marginTop: 20,
  },
});

export default ProgressControlExample;
```

#### 2. 响应事件的动画

```javascript
import React, { useRef } from 'react';
import { View, StyleSheet, TouchableOpacity, Text } from 'react-native';
import LottieView from 'lottie-react-native';

const InteractiveAnimation = () => {
  const animationRef = useRef(null);
  const [isLiked, setIsLiked] = React.useState(false);

  const toggleLike = () => {
    setIsLiked(!isLiked);
    
    if (!isLiked) {
      animationRef.current?.play(0, 30);
    } else {
      animationRef.current?.play(30, 0);
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={toggleLike}>
        <LottieView
          ref={animationRef}
          source={require('../assets/animations/like.json')}
          style={styles.animation}
          loop={false}
        />
      </TouchableOpacity>
      
      <Text style={styles.text}>
        {isLiked ? '已点赞' : '点击点赞'}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  animation: {
    width: 150,
    height: 150,
  },
  text: {
    marginTop: 20,
    fontSize: 18,
  },
});

export default InteractiveAnimation;
```

#### 3. 使用远程动画文件

```javascript
import React from 'react';
import { View, StyleSheet } from 'react-native';
import LottieView from 'lottie-react-native';

const RemoteAnimation = () => {
  return (
    <View style={styles.container}>
      <LottieView
        source={{ uri: 'https://assets.lottiefiles.com/packages/lf20_w5gqckvh.json' }}
        style={styles.animation}
        autoPlay
        loop
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  animation: {
    width: 200,
    height: 200,
  },
});

export default RemoteAnimation;
```

### 9.5 常用属性和方法

#### 属性

- `source` - 动画文件路径或远程 URL
- `autoPlay` - 是否自动播放
- `loop` - 是否循环播放
- `speed` - 播放速度（默认为 1）
- `duration` - 动画持续时间（毫秒）
- `style` - 样式设置

#### 方法

- `play(startFrame?, endFrame?)` - 播放动画
- `reset()` - 重置动画到第一帧
- `pause()` - 暂停动画
- `resume()` - 恢复动画

### 9.6 在项目中的应用场景

1. **加载动画** - 页面加载时显示精美的加载动画
2. **按钮反馈** - 点赞、收藏等交互按钮的动画反馈
3. **引导页动画** - 应用启动时的引导动画
4. **空状态页面** - 数据为空时的友好提示动画
5. **徽章动画** - 消息提醒、通知等徽章动画

## 10. 其他常用功能库详解

### 10.1 react-native-sqlite-storage - SQLite 数据库

这是一个用于 React Native 应用的 SQLite 存储库，提供了完整的 SQLite 数据库功能。

#### 安装

```bash
npm install react-native-sqlite-storage
```

#### 使用示例

```javascript
import SQLite from 'react-native-sqlite-storage';

// 打开数据库
const db = SQLite.openDatabase(
  {
    name: 'MyDatabase.db',
    location: 'default',
  },
  () => {
    console.log('数据库打开成功');
  },
  error => {
    console.log('数据库打开失败:', error);
  }
);

// 创建表
db.transaction(tx => {
  tx.executeSql(
    'CREATE TABLE IF NOT EXISTS Users (id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR(50), email VARCHAR(50))',
    [],
    () => {
      console.log('表创建成功');
    },
    error => {
      console.log('表创建失败:', error);
    }
  );
});

// 插入数据
const insertUser = (name, email) => {
  db.transaction(tx => {
    tx.executeSql(
      'INSERT INTO Users (name, email) VALUES (?, ?)',
      [name, email],
      (tx, results) => {
        console.log('插入成功，ID:', results.insertId);
      },
      error => {
        console.log('插入失败:', error);
      }
    );
  });
};

// 查询数据
const getUsers = () => {
  db.transaction(tx => {
    tx.executeSql(
      'SELECT * FROM Users',
      [],
      (tx, results) => {
        const users = [];
        for (let i = 0; i < results.rows.length; i++) {
          users.push(results.rows.item(i));
        }
        console.log('查询结果:', users);
      },
      error => {
        console.log('查询失败:', error);
      }
    );
  });
};

// 更新数据
const updateUser = (id, name, email) => {
  db.transaction(tx => {
    tx.executeSql(
      'UPDATE Users SET name = ?, email = ? WHERE id = ?',
      [name, email, id],
      (tx, results) => {
        console.log('更新成功，影响行数:', results.rowsAffected);
      },
      error => {
        console.log('更新失败:', error);
      }
    );
  });
};

// 删除数据
const deleteUser = (id) => {
  db.transaction(tx => {
    tx.executeSql(
      'DELETE FROM Users WHERE id = ?',
      [id],
      (tx, results) => {
        console.log('删除成功，影响行数:', results.rowsAffected);
      },
      error => {
        console.log('删除失败:', error);
      }
    );
  });
};
```

### 10.2 react-native-camera - 相机功能

提供对设备相机的访问和控制功能。

#### 安装

```bash
npm install react-native-camera
```

#### 使用示例

```javascript
import React, { useRef } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { RNCamera } from 'react-native-camera';

const CameraExample = () => {
  const cameraRef = useRef(null);

  // 拍照
  const takePicture = async () => {
    if (cameraRef.current) {
      const options = {
        quality: 0.8,
        base64: true,
        skipProcessing: true,
      };
      
      try {
        const data = await cameraRef.current.takePictureAsync(options);
        console.log('照片数据:', data.uri);
        // 可以将照片数据保存或上传
      } catch (error) {
        console.log('拍照失败:', error);
      }
    }
  };

  // 录制视频
  const recordVideo = async () => {
    if (cameraRef.current) {
      try {
        const data = await cameraRef.current.recordAsync();
        console.log('视频数据:', data.uri);
      } catch (error) {
        console.log('录制失败:', error);
      }
    }
  };

  // 停止录制
  const stopRecording = async () => {
    if (cameraRef.current) {
      try {
        await cameraRef.current.stopRecording();
      } catch (error) {
        console.log('停止录制失败:', error);
      }
    }
  };

  return (
    <View style={styles.container}>
      <RNCamera
        ref={cameraRef}
        style={styles.preview}
        type={RNCamera.Constants.Type.back}
        flashMode={RNCamera.Constants.FlashMode.on}
        androidCameraPermissionOptions={{
          title: '权限请求',
          message: '应用需要访问相机权限',
          buttonPositive: '确定',
          buttonNegative: '取消',
        }}
      />
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity onPress={takePicture} style={styles.captureButton}>
          <Text style={styles.buttonText}>拍照</Text>
        </TouchableOpacity>
        
        <TouchableOpacity onPress={recordVideo} style={styles.recordButton}>
          <Text style={styles.buttonText}>录制</Text>
        </TouchableOpacity>
        
        <TouchableOpacity onPress={stopRecording} style={styles.stopButton}>
          <Text style={styles.buttonText}>停止</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: 'black',
  },
  preview: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
  },
  captureButton: {
    backgroundColor: 'white',
    borderRadius: 35,
    padding: 15,
  },
  recordButton: {
    backgroundColor: 'red',
    borderRadius: 35,
    padding: 15,
  },
  stopButton: {
    backgroundColor: 'gray',
    borderRadius: 35,
    padding: 15,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default CameraExample;
```

### 10.3 react-native-geolocation-service - 地理位置服务

提供对设备地理位置信息的访问。

#### 安装

```bash
npm install react-native-geolocation-service
```

#### 使用示例

```javascript
import Geolocation from 'react-native-geolocation-service';

// 获取当前位置
const getCurrentLocation = () => {
  Geolocation.getCurrentPosition(
    position => {
      const { latitude, longitude } = position.coords;
      console.log('当前位置:', latitude, longitude);
    },
    error => {
      console.log('获取位置失败:', error.code, error.message);
    },
    {
      enableHighAccuracy: true,
      timeout: 15000,
      maximumAge: 10000,
    }
  );
};

// 监听位置变化
const watchLocation = () => {
  const watchId = Geolocation.watchPosition(
    position => {
      const { latitude, longitude } = position.coords;
      console.log('位置更新:', latitude, longitude);
    },
    error => {
      console.log('监听位置失败:', error.code, error.message);
    },
    {
      enableHighAccuracy: true,
      distanceFilter: 10, // 位置变化超过10米时触发
    }
  );
  
  return watchId;
};

// 停止监听位置变化
const clearWatch = (watchId) => {
  Geolocation.clearWatch(watchId);
};

// 检查是否有位置权限
const checkLocationPermission = async () => {
  try {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      {
        title: '位置权限',
        message: '应用需要访问您的位置信息',
        buttonNeutral: '稍后询问',
        buttonNegative: '取消',
        buttonPositive: '确定',
      },
    );
    
    if (granted === PermissionsAndroid.RESULTS.GRANTED) {
      console.log('位置权限已获取');
      getCurrentLocation();
    } else {
      console.log('位置权限被拒绝');
    }
  } catch (err) {
    console.warn(err);
  }
};
```

### 10.4 react-native-push-notification - 推送通知

实现本地和远程推送通知功能。

#### 安装

```bash
npm install react-native-push-notification
```

#### 使用示例

```javascript
import PushNotification from 'react-native-push-notification';

// 配置推送通知
PushNotification.configure({
  // 当应用在前台时触发
  onNotification: function (notification) {
    console.log('通知内容:', notification);
  },
  
  // iOS权限请求
  requestPermissions: true,
});

// 发送本地通知
const sendLocalNotification = () => {
  PushNotification.localNotification({
    title: '本地通知',
    message: '这是一条本地通知消息',
    playSound: true,
    soundName: 'default',
    actions: '["查看", "取消"]',
  });
};

// 发送定时通知
const scheduleNotification = () => {
  PushNotification.localNotificationSchedule({
    title: '定时通知',
    message: '这是一条定时通知消息',
    date: new Date(Date.now() + 60 * 1000), // 1分钟后触发
    playSound: true,
    soundName: 'default',
  });
};

// 取消所有通知
const cancelAllNotifications = () => {
  PushNotification.cancelAllLocalNotifications();
};

// 创建频道 (Android 8.0+)
const createNotificationChannel = () => {
  PushNotification.createChannel(
    {
      channelId: 'default-channel-id',
      channelName: '默认通知频道',
      channelDescription: '默认通知频道描述',
      playSound: true,
      soundName: 'default',
      importance: 4,
      vibrate: true,
    },
    (created) => console.log(`频道创建 ${created ? '成功' : '失败'}`)
  );
};
```

### 10.5 @react-native-clipboard/clipboard - 剪贴板操作

提供对设备剪贴板的读写操作。

#### 安装

```bash
npm install @react-native-clipboard/clipboard
```

#### 使用示例

```javascript
import Clipboard from '@react-native-clipboard/clipboard';
import { ToastAndroid } from 'react-native';

// 复制文本到剪贴板
const copyToClipboard = (text) => {
  Clipboard.setString(text);
  ToastAndroid.show('已复制到剪贴板', ToastAndroid.SHORT);
};

// 从剪贴板读取文本
const fetchCopiedText = async () => {
  const text = await Clipboard.getString();
  console.log('剪贴板内容:', text);
  return text;
};

// 检查剪贴板是否有内容
const checkClipboardContent = async () => {
  const hasContent = await Clipboard.hasString();
  console.log('剪贴板是否有内容:', hasContent);
  return hasContent;
};

// 在组件中使用示例
const ClipboardExample = () => {
  const [copiedText, setCopiedText] = React.useState('');

  const fetchCopiedText = async () => {
    const text = await Clipboard.getString();
    setCopiedText(text);
  };

  return (
    <View>
      <TextInput
        value={copiedText}
        onChangeText={setCopiedText}
        placeholder="剪贴板内容"
      />
      
      <Button
        title="复制文本"
        onPress={() => copyToClipboard('Hello, World!')}
      />
      
      <Button
        title="粘贴文本"
        onPress={fetchCopiedText}
      />
    </View>
  );
};
```

### 10.6 react-native-svg - SVG 图像渲染

在 React Native 中渲染 SVG 图像。

#### 安装

```bash
npm install react-native-svg
```

#### 使用示例

```javascript
import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, {
  Circle,
  Rect,
  Path,
  Line,
  Text as SvgText,
  G,
 Defs,
  LinearGradient,
  Stop
} from 'react-native-svg';

const SvgExample = () => {
  return (
    <View style={styles.container}>
      {/* 基本图形 */}
      <Svg height="100" width="100">
        <Circle cx="50" cy="50" r="40" stroke="blue" strokeWidth="2.5" fill="green" />
      </Svg>
      
      {/* 矩形 */}
      <Svg height="100" width="100">
        <Rect
          x="20"
          y="20"
          width="60"
          height="60"
          fill="red"
          stroke="black"
          strokeWidth="2"
        />
      </Svg>
      
      {/* 路径 */}
      <Svg height="100" width="100">
        <Path
          d="M25 10 L98 65 L70 25 L16 90 L45 35 L75 80 L25 10 Z"
          fill="none"
          stroke="orange"
          strokeWidth="3"
        />
      </Svg>
      
      {/* 渐变 */}
      <Svg height="100" width="100">
        <Defs>
          <LinearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="red" stopOpacity="1" />
            <Stop offset="1" stopColor="blue" stopOpacity="1" />
          </LinearGradient>
        </Defs>
        <Rect x="20" y="20" width="60" height="60" fill="url(#grad)" />
      </Svg>
      
      {/* 组合图形 */}
      <Svg height="100" width="100">
        <G rotation="45" origin="50,50">
          <Rect x="25" y="25" width="50" height="50" fill="purple" />
          <Circle cx="50" cy="50" r="15" fill="yellow" />
        </G>
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 20,
  },
});

export default SvgExample;
```

## 11. 其他常用的 React Native 库

在 React Native 开发中，除了项目中已经使用的库之外，还有许多其他常用的库可以帮助开发者提高效率和应用功能：

### 11.1 网络请求库

1. **axios** - 基于 Promise 的 HTTP 客户端（项目中已使用）

   ```bash
   npm install axios
   ```

2. **react-native-axios** - 专为 React Native 优化的 axios 版本

   ```bash
   npm install react-native-axios
   ```

### 11.2 UI 组件库

1. **react-native-elements** - 丰富的 UI 组件库

   ```bash
   npm install react-native-elements
   ```

2. **native-base** - 基于 Styled System 的 UI 组件库

   ```bash
   npm install native-base
   ```

3. **react-native-paper** - 遵循 Material Design 的 UI 组件库

   ```bash
   npm install react-native-paper
   ```

### 11.3 图像处理库

1. **react-native-fast-image** - 高性能的图像组件，支持缓存

   ```bash
   npm install react-native-fast-image
   ```

2. **react-native-svg** - 支持 SVG 图像渲染

   ```bash
   npm install react-native-svg
   ```

### 11.4 动画库

1. **react-native-lottie** - 渲染 After Effects 动画

   ```bash
   npm install lottie-react-native
   ```

### 11.5 设备功能库

1. **react-native-camera** - 相机功能

   ```bash
   npm install react-native-camera
   ```

2. **react-native-geolocation-service** - 地理位置服务

   ```bash
   npm install react-native-geolocation-service
   ```

3. **react-native-push-notification** - 推送通知

   ```bash
   npm install react-native-push-notification
   ```

4. **@react-native-clipboard/clipboard** - 剪贴板操作

   ```bash
   npm install @react-native-clipboard/clipboard
   ```

### 11.6 数据存储库

1. **realm** - 移动端数据库

   ```bash
   npm install realm
   ```

2. **react-native-sqlite-storage** - SQLite 数据库

   ```bash
   npm install react-native-sqlite-storage
   ```

### 11.7 工具库

1. **lodash** - 实用工具函数库

   ```bash
   npm install lodash
   ```

2. **moment** / **dayjs** - 日期处理库

   ```bash
   # moment
   npm install moment
   
   # 或者更轻量的 dayjs
   npm install dayjs
   ```

3. **react-native-config** - 环境变量管理

   ```bash
   npm install react-native-config
   ```

### 11.8 调试和性能工具

1. **react-native-debugger** - 调试工具

   ```bash
   npm install react-native-debugger
   ```

2. **react-native-performance** - 性能监控

   ```bash
   npm install react-native-performance
   ```

## 总结

这些新增的库和导航器为 React Native 应用提供了丰富的导航和功能支持：

1. **react-native-webview** - 显示网页内容
2. **react-native-worklets** - 处理多线程任务
3. **react-native-pager-view** - 实现滑动页面切换
4. **@react-navigation/native-stack** - 原生栈导航器
5. **@react-navigation/bottom-tabs** - 底部标签导航器
6. **@react-navigation/drawer** - 抽屉导航器
7. **@react-navigation/material-top-tabs** - 顶部标签导航器

通过合理使用这些导航器，可以构建出功能丰富、用户体验良好的移动应用。
