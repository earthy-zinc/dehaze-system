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
  options?: object;
}

// 公开路由（无需认证）
export const publicRoutes: RouteConfig[] = [
  {
    name: 'Login',
    component: LoginScreen,
    options: { title: '登录', headerShown: false },
  },
];

// 受保护路由（需认证）
export const protectedRoutes: RouteConfig[] = [
  {
    name: 'Home',
    component: HomeScreen,
    options: { title: '主页', headerShown: false },
  },
  {
    name: 'ImageInput',
    component: ImageInputScreen,
    options: { title: '图像输入', headerShown: false },
  },
  {
    name: 'AlgorithmSelect',
    component: AlgorithmSelectScreen,
    options: { title: '算法选择', headerShown: false },
  },
  {
    name: 'Processing',
    component: ProcessingScreen,
    options: { title: '图像处理', headerShown: false },
  },
  {
    name: 'SideBySide',
    component: SideBySideScreen,
    options: { title: '并排对比', headerShown: false },
  },
  {
    name: 'Overlay',
    component: OverlayScreen,
    options: { title: '重叠对比', headerShown: false },
  },
  {
    name: 'Magnifier',
    component: MagnifierScreen,
    options: { title: '放大镜', headerShown: false },
  },
  {
    name: 'Filter',
    component: FilterScreen,
    options: { title: '滤镜调节', headerShown: false },
  },
  {
    name: 'Metrics',
    component: MetricsScreen,
    options: { title: '指标评估', headerShown: false },
  },
  {
    name: 'Dataset',
    component: DatasetScreen,
    options: { title: '数据集管理', headerShown: false },
  },
  {
    name: 'Task',
    component: TaskScreen,
    options: { title: '任务中心', headerShown: false },
  },
  {
    name: 'Algorithm',
    component: AlgorithmScreen,
    options: { title: '算法详情', headerShown: false },
  },
  {
    name: 'Profile',
    component: ProfileScreen,
    options: { title: '个人中心', headerShown: false },
  },
];
