import type { EvaluationMetrics } from '@/types/evaluation';
import type { SelectedImage } from '@/types/image';
import type { InputMethod } from '@/pages/image-input/types/imageInput';

/**
 * 对比模式共享参数（SideBySide/Overlay/Magnifier/Filter/Metrics 共用）
 */
export interface CompareRouteParams {
  originalUrl: string;
  processedUrl: string;
  /** GT 参考图（清晰图）URL，用于指标评估 */
  cleanUrl?: string;
  algorithmId?: number;
}

/**
 * 路由参数类型定义
 */
export type RootStackParamList = {
  Login: undefined;
  Register: undefined;
  Home: undefined;
  ImageInput: { initialMethod?: InputMethod } | undefined;
  AlgorithmSelect: { image?: SelectedImage } | undefined;
  Processing:
    | { algorithmId: number; image?: SelectedImage }
    | undefined;
  SideBySide: CompareRouteParams | undefined;
  Overlay: CompareRouteParams | undefined;
  Magnifier: CompareRouteParams | undefined;
  Filter: CompareRouteParams | undefined;
  Metrics:
    | (CompareRouteParams & {
        metrics?: EvaluationMetrics;
      })
    | undefined;
  Dataset: undefined;
  Task: undefined;
  Algorithm: { algorithmId: number } | undefined;
  Profile: undefined;
};

export type RouteKeys = keyof RootStackParamList;
