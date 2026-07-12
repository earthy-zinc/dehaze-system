import type { EvaluationMetrics } from '@/types/evaluation';
import type { SelectedImage } from '@/types/image';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

/**
 * 路由参数类型定义
 */
export type RootStackParamList = {
  Login: undefined;
  Home: undefined;
  ImageInput: undefined;
  AlgorithmSelect: { image?: SelectedImage } | undefined;
  Processing:
    | { algorithmId: number; image?: SelectedImage }
    | undefined;
  SideBySide:
    | { originalUrl: string; processedUrl: string }
    | undefined;
  Overlay: { originalUrl: string; processedUrl: string } | undefined;
  Magnifier: { originalUrl: string; processedUrl: string } | undefined;
  Filter: { originalUrl: string; processedUrl: string } | undefined;
  Metrics:
    | {
        originalUrl: string;
        processedUrl: string;
        metrics?: EvaluationMetrics;
        algorithmId?: number;
      }
    | undefined;
  Dataset: undefined;
  Task: undefined;
  Algorithm: { algorithmId: number } | undefined;
};

export type RouteKeys = keyof RootStackParamList;

export type ScreenProps<K extends RouteKeys> = NativeStackScreenProps<
  RootStackParamList,
  K
>;
