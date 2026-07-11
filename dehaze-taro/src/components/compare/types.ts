import Taro from '@tarojs/taro';
import type { Algorithm, PredictionResultVO } from 'dehaze-sdk-js';

/**
 * 对比模式类型（含放大镜和滤镜）
 */
export type CompareMode = 'side-by-side' | 'overlay' | 'metrics' | 'magnifier' | 'filter';

/**
 * 统一的图片数据类型
 */
export interface CompareImageData {
  url: string;
  name: string;
  width?: number;
  height?: number;
  size?: number;
}

/**
 * 对比模式配置
 */
export interface CompareModeConfig {
  key: CompareMode;
  label: string;
  path: string;
}

/**
 * 所有对比模式列表
 */
export const COMPARE_MODES: CompareModeConfig[] = [
  { key: 'side-by-side', label: '并排对比', path: '/pages/side-by-side/index' },
  { key: 'overlay', label: '重叠对比', path: '/pages/overlay/index' },
  { key: 'magnifier', label: '放大镜', path: '/pages/magnifier/index' },
  { key: 'filter', label: '滤镜', path: '/pages/filter/index' },
  { key: 'metrics', label: '指标对比', path: '/pages/metrics/index' },
];

/**
 * 对比上下文：从 storage 加载的统一数据
 */
export interface CompareContext {
  originImage: CompareImageData | null;
  result: PredictionResultVO | null;
  algorithm: Algorithm | null;
}

/**
 * 从 storage 加载对比数据
 */
export const loadCompareContext = (): CompareContext => {
  let originImage: CompareImageData | null = null;
  let result: PredictionResultVO | null = null;
  let algorithm: Algorithm | null = null;

  try {
    const imgStr = Taro.getStorageSync('current_image');
    if (imgStr) originImage = JSON.parse(imgStr);
  } catch { /* ignore */ }

  try {
    const resStr = Taro.getStorageSync('prediction_result');
    if (resStr) result = JSON.parse(resStr);
  } catch { /* ignore */ }

  try {
    const algoStr = Taro.getStorageSync('selected_algorithm');
    if (algoStr) algorithm = JSON.parse(algoStr);
  } catch { /* ignore */ }

  return { originImage, result, algorithm };
};

/**
 * 格式化处理时间
 */
export const formatTime = (ms: number): string => {
  if (ms < 1000) return ms + ' ms';
  return (ms / 1000).toFixed(2) + ' s';
};
