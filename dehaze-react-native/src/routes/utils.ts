import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList, RouteKeys } from './types';

export type StackNavigation = NativeStackNavigationProp<RootStackParamList>;

export function useNavigator() {
  return useNavigation<StackNavigation>();
}

export type { RootStackParamList, RouteKeys };
