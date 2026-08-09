/**
 * 处理流程状态管理
 *
 * 管理从图像输入到去雾处理的完整流程状态：
 * - 当前选中图片（含 fileId）
 * - 选中算法
 * - 处理参数
 * - 处理状态与结果
 * - runPrediction：配额校验 + 提交预测 + 递增重试 + 取消 + 耗时计时
 */

import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ImageData } from "@/pages/image-input/data/imageInputData";
import { ModelAPI } from "dehaze-sdk-js";
import type { Algorithm, PredictionResultVO } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

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

/** 重试间隔（毫秒）：2s → 3s → 5s */
const RETRY_DELAYS = [2000, 3000, 5000];
const MAX_RETRIES = 3;

/** runPrediction 参数 */
export interface RunPredictionOptions {
  algorithmId: number;
  fileId?: number;
  imageUrl?: string;
  params?: string;
  /** 配额耗尽回调（页面弹"去充值"并跳转） */
  onQuotaExhausted?: (quota: { used: number; total: number }) => void;
}

/** runPrediction 返回值 */
export interface RunPredictionResult {
  ok: boolean;
  result?: PredictionResultVO;
  error?: string;
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

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

  /** 是否正在执行 runPrediction */
  const isProcessing = ref(false);

  /** 已耗时（毫秒），runPrediction 期间每 100ms 递增 */
  const elapsedTime = ref(0);

  /** 是否已取消（由 cancelProcessing 设置） */
  const cancelled = ref(false);

  /** 耗时计时器实例 */
  let elapsedTimer: ReturnType<typeof setInterval> | null = null;

  // ==================== 计算属性 ====================

  /** 是否有图片 */
  const hasImage = computed(() => !!currentImage.value);

  /** 是否有算法 */
  const hasAlgorithm = computed(() => !!selectedAlgorithm.value);

  /** 是否已完成 */
  const isCompleted = computed(() => status.value === "completed");

  /** 原始图 URL */
  const originUrl = computed(() => currentImage.value?.url || "");

  // ==================== 计时器 ====================

  /** 启动耗时计时器（每 100ms 递增 elapsedTime） */
  function startElapsedTimer() {
    stopElapsedTimer();
    elapsedTime.value = 0;
    elapsedTimer = setInterval(() => {
      elapsedTime.value += 100;
    }, 100);
  }

  /** 停止耗时计时器 */
  function stopElapsedTimer() {
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
  }

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
    isProcessing.value = false;
    elapsedTime.value = 0;
    cancelled.value = false;
    stopElapsedTimer();
  }

  // ==================== runPrediction ====================

  /**
   * 单次预测尝试（带递增重试）
   *
   * 成功返回 {ok:true, result}；重试耗尽返回 {ok:false, error}；
   * 取消返回 {ok:false}（cancelProcessing 已处理 fail）。
   */
  async function attempt(
    opts: RunPredictionOptions,
    attemptNumber: number
  ): Promise<RunPredictionResult> {
    if (cancelled.value) return { ok: false };

    try {
      const res = await ModelAPI.predictAndWait({
        algorithmId: opts.algorithmId,
        fileId: opts.fileId,
        imageUrl: opts.imageUrl,
        params: opts.params,
      });

      if (cancelled.value) return { ok: false };

      if (res.status === 3) {
        throw new Error(res.errorMessage || "处理失败");
      }

      return { ok: true, result: res };
    } catch (error) {
      if (cancelled.value) return { ok: false };

      const errMsg = getErrorMessage(error, "处理失败");

      if (attemptNumber < MAX_RETRIES) {
        const delay = RETRY_DELAYS[attemptNumber] || 2000;
        uni.showToast({
          title: `${errMsg}，${delay / 1000}秒后重试（${attemptNumber + 1}/${MAX_RETRIES}）`,
          icon: "none",
          duration: delay,
        });
        await sleep(delay);
        if (cancelled.value) return { ok: false };
        return attempt(opts, attemptNumber + 1);
      }

      return { ok: false, error: errMsg };
    }
  }

  /**
   * 执行去雾预测（配额校验 + 提交 + 递增重试 + 耗时计时）
   *
   * 行为：
   * 1. isProcessing 为 true 时直接返回 {ok:false}
   * 2. 配额校验：remaining===0 时回调 onQuotaExhausted 并返回 {ok:false}
   * 3. 设置 isProcessing=true、cancelled=false，启动计时器
   * 4. 递增重试调用 predictAndWait；成功返回 {ok:true,result}，重试耗尽返回 {ok:false,error}
   * 5. cancelProcessing 可随时中止
   */
  async function runPrediction(
    opts: RunPredictionOptions
  ): Promise<RunPredictionResult> {
    if (isProcessing.value) return { ok: false };

    // 配额校验（失败则忽略，继续处理）
    try {
      const quota = await ModelAPI.getQuota();
      if (quota.remaining === 0) {
        opts.onQuotaExhausted?.({ used: quota.used, total: quota.total });
        return { ok: false };
      }
    } catch {
      // 配额查询失败，允许继续
    }

    isProcessing.value = true;
    cancelled.value = false;
    startProcessing();
    startElapsedTimer();

    const ret = await attempt(opts, 0);

    stopElapsedTimer();
    isProcessing.value = false;

    if (ret.ok && ret.result) {
      complete(ret.result);
    } else if (!ret.ok && ret.error) {
      fail(ret.error);
    }
    // 取消情况：cancelProcessing 已调用 fail

    return ret;
  }

  /** 取消正在进行的处理 */
  function cancelProcessing() {
    cancelled.value = true;
    stopElapsedTimer();
    isProcessing.value = false;
    fail("已取消");
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
    isProcessing,
    elapsedTime,
    cancelled,

    // 计算属性
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

    // 预测执行
    runPrediction,
    cancelProcessing,
  };
});
