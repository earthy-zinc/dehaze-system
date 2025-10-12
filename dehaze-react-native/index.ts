import { AppRegistry } from 'react-native';
import App from './src/App';
import { name as appName } from './app.json';

// 添加这两行用于支持 React Navigation
import 'react-native-gesture-handler';

AppRegistry.registerComponent(appName, () => App);