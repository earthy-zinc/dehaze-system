/**
 * 环境配置
 * 模拟器用 10.0.2.2 访问宿主机后端；真机用局域网 IP
 */
import { Platform } from 'react-native';

const isDev = __DEV__;

// 开发期宿主机地址：Android 模拟器用 10.0.2.2 映射宿主机；iOS 模拟器与 Mac 共享网络栈，直连 127.0.0.1
const DEV_HOST = Platform.OS === 'android' ? '10.0.2.2' : '127.0.0.1';

export const BASE_URL = isDev ? `http://${DEV_HOST}:8989` : 'https://api.dehaze.com';
