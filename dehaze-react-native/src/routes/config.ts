import HomeScreen from '@/pages/home';
import LoginScreen from '@/pages/login';
import { RouteConfig } from './navigator';

export const routeConfigs: RouteConfig[] = [
  {
    name: 'Login',
    component: LoginScreen,
    options: { title: '登录' },
  },
  {
    name: 'Home',
    component: HomeScreen,
    options: { title: '主页' },
  },
];
