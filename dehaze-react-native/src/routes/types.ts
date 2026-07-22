import type { EvaluationMetrics } from '@/types/evaluation';
import type { SelectedImage } from '@/types/image';
import type { InputMethod } from '@/pages/image-input/types/imageInput';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

/**
 * 路由参数类型定义
 */
export type RootStackParamList = {
  Login: undefined;
  Home: undefined;
  ImageInput: { initialMethod?: InputMethod } | undefined;
  AlgorithmSelect: { image?: SelectedImage } | undefined;
  Processing:
    | { algorithmId: number; image?: SelectedImage }
    | undefined;
  SideBySide:
    | { originalUrl: string; processedUrl: string; cleanUrl?: string; algorithmId?: number }
    | undefined;
  Overlay:
    | { originalUrl: string; processedUrl: string; cleanUrl?: string; algorithmId?: number }
    | undefined;
  Magnifier:
    | { originalUrl: string; processedUrl: string; cleanUrl?: string; algorithmId?: number }
    | undefined;
  Filter:
    | { originalUrl: string; processedUrl: string; cleanUrl?: string; algorithmId?: number }
    | undefined;
  Metrics:
    | {
        originalUrl: string;
        processedUrl: string;
        /** GT 参考图（清晰图）URL，用于指标评估 */
        cleanUrl?: string;
        metrics?: EvaluationMetrics;
        algorithmId?: number;
      }
    | undefined;
  Dataset: undefined;
  Task: undefined;
  Algorithm: { algorithmId: number } | undefined;
  Profile: undefined;
};

export type RouteKeys = keyof RootStackParamList;

export type ScreenProps<K extends RouteKeys> = NativeStackScreenProps<
  RootStackParamList,
  K
>;
