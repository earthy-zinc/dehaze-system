import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import { routeConfigs } from './config';

// 定义所有可能的路由键
type RouteKeys =
  | 'Login'
  | 'Home'
  | 'ImageInput'
  | 'AlgorithmSelect'
  | 'Processing'
  | 'SideBySide'
  | 'Overlay'
  | 'Magnifier'
  | 'Filter'
  | 'Metrics'
  | 'Dataset'
  | 'Algorithm';

export interface RouteConfig {
  name: RouteKeys;
  component: React.ComponentType<any>;
  options?: any;
  initialParams?: any;
}

export type RootStackParamList = {
  Login: undefined;
  Home: undefined;
  ImageInput: undefined;
  AlgorithmSelect: undefined;
  Processing: undefined;
  SideBySide: undefined;
  Overlay: undefined;
  Magnifier: undefined;
  Filter: undefined;
  Metrics: undefined;
  Dataset: undefined;
  Algorithm: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export const RouteManager = () => {
  return (
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerShown: false, // 使用自定义 Header
        animation: 'slide_from_right',
      }}
    >
      {routeConfigs.map(route => (
        <Stack.Screen
          key={route.name}
          name={route.name}
          component={route.component}
          options={route.options}
          initialParams={route.initialParams}
        />
      ))}
    </Stack.Navigator>
  );
};
