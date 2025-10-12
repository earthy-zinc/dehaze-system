import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from './navigator';
import { useNavigation } from '@react-navigation/native';

export type StackNavigation = NativeStackNavigationProp<RootStackParamList>;

export function useNavigator() {
  return useNavigation<StackNavigation>();
}
