import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from './navigator';

export type StackNavigation = NativeStackNavigationProp<RootStackParamList>;

export function useNavigator() {
  return useNavigation<StackNavigation>();
}
