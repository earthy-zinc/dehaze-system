/**
 * 图像上下文
 *
 * 维护当前输入图像与去雾处理结果，跨「图像输入→算法选择→去雾处理→效果对比」流程共享。
 */
import type { EvaluationMetrics } from '@/types/evaluation';
import type { SelectedImage } from '@/types/image';
import React, {
  createContext,
  useContext,
  useReducer,
  type ReactNode,
} from 'react';

interface ImageState {
  /** 当前输入图像 */
  currentImage: SelectedImage | null;
  /** 去雾处理后结果图像 */
  processedImage: SelectedImage | null;
  /** 评估指标 */
  metrics: EvaluationMetrics | null;
}

type ImageAction =
  | { type: 'SET_CURRENT_IMAGE'; image: SelectedImage | null }
  | { type: 'SET_PROCESSED_IMAGE'; image: SelectedImage | null }
  | { type: 'SET_METRICS'; metrics: EvaluationMetrics | null }
  | { type: 'CLEAR' };

const initialState: ImageState = {
  currentImage: null,
  processedImage: null,
  metrics: null,
};

function imageReducer(state: ImageState, action: ImageAction): ImageState {
  switch (action.type) {
    case 'SET_CURRENT_IMAGE':
      return { ...state, currentImage: action.image };
    case 'SET_PROCESSED_IMAGE':
      return { ...state, processedImage: action.image };
    case 'SET_METRICS':
      return { ...state, metrics: action.metrics };
    case 'CLEAR':
      return { ...initialState };
    default:
      return state;
  }
}

interface ImageContextValue {
  state: ImageState;
  setCurrentImage: (image: SelectedImage | null) => void;
  setProcessedImage: (image: SelectedImage | null) => void;
  setMetrics: (metrics: EvaluationMetrics | null) => void;
  clearImageState: () => void;
}

const ImageContext = createContext<ImageContextValue | undefined>(undefined);

export function ImageProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(imageReducer, initialState);

  const value: ImageContextValue = {
    state,
    setCurrentImage: image => dispatch({ type: 'SET_CURRENT_IMAGE', image }),
    setProcessedImage: image =>
      dispatch({ type: 'SET_PROCESSED_IMAGE', image }),
    setMetrics: metrics => dispatch({ type: 'SET_METRICS', metrics }),
    clearImageState: () => dispatch({ type: 'CLEAR' }),
  };

  return <ImageContext.Provider value={value}>{children}</ImageContext.Provider>;
}

export function useImageContext(): ImageContextValue {
  const ctx = useContext(ImageContext);
  if (!ctx) {
    throw new Error('useImageContext 必须在 ImageProvider 内使用');
  }
  return ctx;
}
