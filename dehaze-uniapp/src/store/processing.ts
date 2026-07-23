/**
 * 处理流程状态管理
 *
 * 管理从图像输入到去雾处理的完整流程状态：
 * - 当前选中图片（含 fileId）
 * - 选中算法
 * - 处理参数
 * - 处理状态与结果
 */

import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ImageData } from "@/pages/image-input/data/imageInputData";
import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";

/** 处理状态 */
export type ProcessingStatus =
  | "idle" // 空闲
  | "selected" // 已选图片
  | "algorithm" // 已选算法
  | "uploading" // 上传中
  | "processing" // 处理中
  | "completed" // 已完成
  | "failed"; // 失败

/** 去雾处理参数 */
export interface DehazeParams {
  /** 去雾强度 0-100 */
  strength: number;
  /** 色彩饱和度 0-200 */
  saturation: number;
  /** 对比度 0-200 */
  contrast: number;
  /** 锐化程度 0-100 */
  sharpness: number;
}

/** 默认处理参数（与产品文档需求规格.md 默认值对齐） */
export const DEFAULT_DEHAZE_PARAMS: DehazeParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpness: 30,
};

export const useProcessingStore = defineStore("processing", () => {
  // ==================== 状态 ====================

  /** 当前处理状态 */
  const status = ref<ProcessingStatus>("idle");

  /** 当前选中图片 */
  const currentImage = ref<ImageData | null>(null);

  /** 上传后获取的文件 ID */
  const fileId = ref<number | null>(null);

  /** 选中算法 */
  const selectedAlgorithm = ref<Algorithm | null>(null);

  /** 处理参数 */
  const params = ref<DehazeParams>({ ...DEFAULT_DEHAZE_PARAMS });

  /** 处理结果 */
  const result = ref<PredictionResultVO | null>(null);

  /** 错误信息 */
  const errorMessage = ref("");

  // ==================== 计算属性 ====================

  /** 是否可以开始处理 */
  const canProcess = computed(
    () =>
      status.value === "algorithm" &&
      currentImage.value &&
      selectedAlgorithm.value
  );

  /** 是否有图片 */
  const hasImage = computed(() => !!currentImage.value);

  /** 是否有算法 */
  const hasAlgorithm = computed(() => !!selectedAlgorithm.value);

  /** 是否已完成 */
  const isCompleted = computed(() => status.value === "completed");

  /** 原始图 URL */
  const originUrl = computed(() => currentImage.value?.url || "");

  // ==================== 方法 ====================

  /** 设置图片（图像输入页选中后调用） */
  function setImage(image: ImageData) {
    currentImage.value = image;
    status.value = "selected";
    fileId.value = image.fileId || null;
  }

  /** 设置上传后的文件 ID */
  function setFileId(id: number) {
    fileId.value = id;
    if (currentImage.value) {
      currentImage.value.fileId = id;
    }
  }

  /** 选择算法 */
  function setAlgorithm(algorithm: Algorithm) {
    selectedAlgorithm.value = algorithm;
    status.value = "algorithm";
  }

  /** 更新处理参数 */
  function updateParams(newParams: Partial<DehazeParams>) {
    Object.assign(params.value, newParams);
  }

  /** 开始上传 */
  function startUploading() {
    status.value = "uploading";
  }

  /** 开始处理 */
  function startProcessing() {
    status.value = "processing";
    errorMessage.value = "";
  }

  /** 处理完成 */
  function complete(resultData: PredictionResultVO) {
    status.value = "completed";
    result.value = resultData;
  }

  /** 处理失败 */
  function fail(error: string) {
    status.value = "failed";
    errorMessage.value = error;
  }

  /** 重置状态 */
  function reset() {
    status.value = "idle";
    currentImage.value = null;
    fileId.value = null;
    selectedAlgorithm.value = null;
    params.value = { ...DEFAULT_DEHAZE_PARAMS };
    result.value = null;
    errorMessage.value = "";
  }

  return {
    // 状态
    status,
    currentImage,
    fileId,
    selectedAlgorithm,
    params,
    result,
    errorMessage,

    // 计算属性
    canProcess,
    hasImage,
    hasAlgorithm,
    isCompleted,
    originUrl,

    // 方法
    setImage,
    setFileId,
    setAlgorithm,
    updateParams,
    startUploading,
    startProcessing,
    complete,
    fail,
    reset,
  };
});
