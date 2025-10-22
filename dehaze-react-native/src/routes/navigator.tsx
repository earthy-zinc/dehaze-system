import React from 'react';
import { routeConfigs } from './config';
import { createNativeStackNavigator } from '@react-navigation/native-stack';


export interface RouteConfig {
    name: string;
    component: React.ComponentType<any>;
    options?: any;
    initialParams?: any;
}

export type RootStackParamList = {
    [key in RouteConfig['name']]: RouteConfig['initialParams'];
}


const Stack = createNativeStackNavigator<RootStackParamList>();

export const RouteManager = () => {
  return (
    <Stack.Navigator initialRouteName="Login">
      {routeConfigs.map((route) => (
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
