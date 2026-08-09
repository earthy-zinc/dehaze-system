import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";
import { useProcessStore } from "@/stores/process";

/**
 * 对比模式类型（含放大镜和滤镜）
 */
export type CompareMode =
  "side-by-side" | "overlay" | "metrics" | "magnifier" | "filter";

/**
 * 统一的图片数据类型
 */
export interface CompareImageData {
  url: string;
  name: string;
  width?: number;
  height?: number;
  size?: number;
  cleanUrl?: string;
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
  { key: "side-by-side", label: "并排对比", path: "/pages/side-by-side/index" },
  { key: "overlay", label: "重叠对比", path: "/pages/overlay/index" },
  { key: "magnifier", label: "放大镜", path: "/pages/magnifier/index" },
  { key: "filter", label: "滤镜", path: "/pages/filter/index" },
  { key: "metrics", label: "指标对比", path: "/pages/metrics/index" },
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
 * 从内存 store 加载对比数据
 */
export const loadCompareContext = (): CompareContext => {
  const { image, algorithm, result } = useProcessStore.getState();
  return { originImage: image, result, algorithm };
};
