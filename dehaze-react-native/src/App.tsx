import AppNavigator from '@/routes';
import { StatusBar, StyleSheet, useColorScheme } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Logger, ConsoleTransport, RemoteTransport } from 'dehaze-sdk-js';

// 模块加载即初始化 Logger，早于组件渲染，确保全局错误捕获在最前注册
Logger.install({
  app: 'rn',
  appVersion: '0.0.1',
  transports: __DEV__
    ? [new ConsoleTransport()]
    : [new ConsoleTransport(), new RemoteTransport()],
});

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaProvider>
        <StatusBar
          barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        />
        <AppNavigator />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
});

export default App;
