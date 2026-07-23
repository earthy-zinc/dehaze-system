import AlgorithmScreen from '@/pages/algorithm';
import AlgorithmSelectScreen from '@/pages/algorithm-select';
import SideBySideScreen from '@/pages/compare/SideBySide';
import OverlayScreen from '@/pages/compare/Overlay';
import MagnifierScreen from '@/pages/compare/Magnifier';
import FilterScreen from '@/pages/compare/Filter';
import MetricsScreen from '@/pages/compare/Metrics';
import DatasetScreen from '@/pages/dataset';
import TaskScreen from '@/pages/task';
import HomeScreen from '@/pages/home';
import ImageInputScreen from '@/pages/image-input';
import LoginScreen from '@/pages/login';
import ProcessingScreen from '@/pages/processing';
import ProfileScreen from '@/pages/profile';
import React from 'react';
import type { RootStackParamList } from './types';

export interface RouteConfig {
  name: keyof RootStackParamList;
  component: React.ComponentType<any>;
  /** 路由标题（headerShown 已在 Stack.Navigator 默认 screenOptions 关闭） */
  title?: string;
}

// 公开路由（无需认证）
export const publicRoutes: RouteConfig[] = [
  { name: 'Login', component: LoginScreen, title: '登录' },
];

// 受保护路由（需认证）
export const protectedRoutes: RouteConfig[] = [
  { name: 'Home', component: HomeScreen, title: '主页' },
  { name: 'ImageInput', component: ImageInputScreen, title: '图像输入' },
  { name: 'AlgorithmSelect', component: AlgorithmSelectScreen, title: '算法选择' },
  { name: 'Processing', component: ProcessingScreen, title: '图像处理' },
  { name: 'SideBySide', component: SideBySideScreen, title: '并排对比' },
  { name: 'Overlay', component: OverlayScreen, title: '重叠对比' },
  { name: 'Magnifier', component: MagnifierScreen, title: '放大镜' },
  { name: 'Filter', component: FilterScreen, title: '滤镜调节' },
  { name: 'Metrics', component: MetricsScreen, title: '指标评估' },
  { name: 'Dataset', component: DatasetScreen, title: '数据集管理' },
  { name: 'Task', component: TaskScreen, title: '任务中心' },
  { name: 'Algorithm', component: AlgorithmScreen, title: '算法详情' },
  { name: 'Profile', component: ProfileScreen, title: '个人中心' },
];
