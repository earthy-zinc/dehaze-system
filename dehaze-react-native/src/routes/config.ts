import HomeScreen from '@/pages/home';
import LoginScreen from '@/pages/login';
import ImageInputScreen from '@/pages/image-input';
import AlgorithmSelectScreen from '@/pages/algorithm-select';
import ProcessingScreen from '@/pages/processing';
import DatasetScreen from '@/pages/dataset';
import AlgorithmScreen from '@/pages/algorithm';
import { RouteConfig } from './navigator';

// 创建简单的占位符组件
const PlaceholderScreen = ({ routeName }: { routeName: string }) => {
  const React = require('react');
  const { View, Text } = require('react-native');
  const { SafeAreaView } = require('react-native-safe-area-context');

  return React.createElement(
    SafeAreaView,
    { style: { flex: 1, backgroundColor: '#f5f5f5' } },
    React.createElement(
      View,
      {
        style: {
          flex: 1,
          justifyContent: 'center',
          alignItems: 'center',
          padding: 20,
        }
      },
      React.createElement(
        Text,
        {
          style: {
            fontSize: 24,
            fontWeight: 'bold',
            color: '#333',
            marginBottom: 10,
          }
        },
        routeName
      ),
      React.createElement(
        Text,
        {
          style: {
            fontSize: 16,
            color: '#666',
            textAlign: 'center',
          }
        },
        '此页面正在开发中...'
      )
    )
  );
};

export const routeConfigs: RouteConfig[] = [
  {
    name: 'Login' as const,
    component: LoginScreen,
    options: { title: '登录' },
  },
  {
    name: 'Home' as const,
    component: HomeScreen,
    options: { title: '主页' },
  },
  {
    name: 'ImageInput' as const,
    component: ImageInputScreen,
    options: { title: '图像输入' },
  },
  {
    name: 'AlgorithmSelect' as const,
    component: AlgorithmSelectScreen,
    options: { title: '算法选择' },
  },
  {
    name: 'Processing' as const,
    component: ProcessingScreen,
    options: { title: '图像处理' },
  },
  {
    name: 'SideBySide' as const,
    component: () => PlaceholderScreen({ routeName: '并排对比' }),
    options: { title: '并排对比' },
  },
  {
    name: 'Overlay' as const,
    component: () => PlaceholderScreen({ routeName: '重叠对比' }),
    options: { title: '重叠对比' },
  },
  {
    name: 'Magnifier' as const,
    component: () => PlaceholderScreen({ routeName: '放大镜' }),
    options: { title: '放大镜' },
  },
  {
    name: 'Filter' as const,
    component: () => PlaceholderScreen({ routeName: '滤镜调节' }),
    options: { title: '滤镜调节' },
  },
  {
    name: 'Metrics' as const,
    component: () => PlaceholderScreen({ routeName: '指标评估' }),
    options: { title: '指标评估' },
  },
  {
    name: 'Dataset' as const,
    component: DatasetScreen,
    options: { title: '数据集管理' },
  },
  {
    name: 'Algorithm' as const,
    component: AlgorithmScreen,
    options: { title: '算法详情' },
  },
];
