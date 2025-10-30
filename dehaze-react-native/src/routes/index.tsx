import { NavigationContainer } from '@react-navigation/native';
import React from 'react';
import { RouteManager } from './navigator';

function AppNavigator() {
  return (
    <NavigationContainer>
      <RouteManager />
    </NavigationContainer>
  );
}

export default AppNavigator;
