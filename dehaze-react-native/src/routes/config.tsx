import HomeScreen from '@/pages/home';
import LoginScreen from '@/pages/login';
import ImageInputScreen from '@/pages/image-input';
import AlgorithmSelectScreen from '@/pages/algorithm-select';
import ProcessingScreen from '@/pages/processing';
import DatasetScreen from '@/pages/dataset';
import AlgorithmScreen from '@/pages/algorithm';
import { RouteConfig } from './navigator';
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MainLayout } from '@/layout';

// 创建使用 MainLayout 的占位符组件
const createPlaceholderScreen = (routeName: string, title: string) => {
  const PlaceholderScreen: React.FC = () => {
    return (
      <MainLayout title={title}>
        <View style={placeholderStyles.content}>
          <Text style={placeholderStyles.title}>{routeName}</Text>
          <Text style={placeholderStyles.description}>此页面正在开发中...</Text>
        </View>
      </MainLayout>
    );
  };
  PlaceholderScreen.displayName = `${routeName}Screen`;
  return PlaceholderScreen;
};

const placeholderStyles = StyleSheet.create({
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
});

export const routeConfigs: RouteConfig[] = [
  {
    name: 'Login' as const,
    component: LoginScreen,
    options: { title: '登录', headerShown: false },
  },
  {
    name: 'Home' as const,
    component: HomeScreen,
    options: { title: '主页', headerShown: false },
  },
  {
    name: 'ImageInput' as const,
    component: ImageInputScreen,
    options: { title: '图像输入', headerShown: false },
  },
  {
    name: 'AlgorithmSelect' as const,
    component: AlgorithmSelectScreen,
    options: { title: '算法选择', headerShown: false },
  },
  {
    name: 'Processing' as const,
    component: ProcessingScreen,
    options: { title: '图像处理', headerShown: false },
  },
  {
    name: 'SideBySide' as const,
    component: createPlaceholderScreen('并排对比', '并排对比'),
    options: { title: '并排对比', headerShown: false },
  },
  {
    name: 'Overlay' as const,
    component: createPlaceholderScreen('重叠对比', '重叠对比'),
    options: { title: '重叠对比', headerShown: false },
  },
  {
    name: 'Magnifier' as const,
    component: createPlaceholderScreen('放大镜', '放大镜'),
    options: { title: '放大镜', headerShown: false },
  },
  {
    name: 'Filter' as const,
    component: createPlaceholderScreen('滤镜调节', '滤镜调节'),
    options: { title: '滤镜调节', headerShown: false },
  },
  {
    name: 'Metrics' as const,
    component: createPlaceholderScreen('指标评估', '指标评估'),
    options: { title: '指标评估', headerShown: false },
  },
  {
    name: 'Dataset' as const,
    component: DatasetScreen,
    options: { title: '数据集管理', headerShown: false },
  },
  {
    name: 'Algorithm' as const,
    component: AlgorithmScreen,
    options: { title: '算法详情', headerShown: false },
  },
];
