/**
 * 处理流程状态管理
 *
 * 管理从图像输入到去雾处理的完整流程状态：
 * - 当前选中图片（含 fileId）
 * - 选中图像模型（数据源为启用模型列表）
 * - 处理参数
 * - 处理状态与结果（提交-轮询-取消，对齐 /api/v1/images/* 任务模式）
 */

import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ImageData } from "@/pages/image-input/data/imageInputData";
import { AiImageAPI, AiModelAPI } from "dehaze-sdk-js";
import type { ImageTaskResponse } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

/** 处理状态 */
export type ProcessingStatus =
  | "idle" // 空闲
  | "selected" // 已选图片
  | "model" // 已选模型
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

/** 可选图像模型（选择场景只需标识与展示名） */
export interface ImageModelOption {
  /** 模型业务标识（slug），提交图像任务时作为 model */
  modelId: string;
  displayName: string;
}

/** 重试间隔（毫秒）：2s → 3s → 5s */
const RETRY_DELAYS = [2000, 3000, 5000];
const MAX_RETRIES = 3;

/** 查找历史最近使用模型时回看的历史任务条数 */
const RECENT_TASK_PAGE_SIZE = 20;

/** runImageTask 参数 */
export interface RunImageTaskOptions {
  fileId?: number;
  imageUrl?: string;
  params?: string;
}

/** runImageTask 返回值 */
export interface RunImageTaskResult {
  ok: boolean;
  result?: ImageTaskResponse;
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

  /** 可用图像编辑模型列表 */
  const imageModels = ref<ImageModelOption[]>([]);

  /** 当前选中模型 */
  const selectedModel = ref<ImageModelOption | null>(null);

  /** 处理参数 */
  const params = ref<DehazeParams>({ ...DEFAULT_DEHAZE_PARAMS });

  /** 处理结果 */
  const result = ref<ImageTaskResponse | null>(null);

  /** 提交后返回的任务 ID（取消任务用） */
  const currentTaskId = ref<number | null>(null);

  /** 错误信息 */
  const errorMessage = ref("");

  /** 是否正在执行 runImageTask */
  const isProcessing = ref(false);

  /** 已耗时（毫秒），runImageTask 期间每 100ms 递增 */
  const elapsedTime = ref(0);

  /** 是否已取消（由 cancelProcessing 设置） */
  const cancelled = ref(false);

  /** 耗时计时器实例 */
  let elapsedTimer: ReturnType<typeof setInterval> | null = null;

  // ==================== 计算属性 ====================

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

  // ==================== 模型 ====================

  /**
   * 加载启用的图像编辑模型
   *
   * 预选次序：历史最近使用（且仍在启用列表内）→ 后端标记的平台默认模型
   * （`isDefault` ← 字典 `ai_model.default_model.image_edit`）；后端未配置平台默认时不预选
   * （由用户在模型下拉中显式选择），不再回落到"启用列表首项"猜值；
   * 已有选择仍有效时保留，避免用户从对比页返回后被重置。
   */
  async function loadImageModels() {
    const list = await AiModelAPI.listEnabledModels("image_edit");
    imageModels.value = list.map((m) => ({
      modelId: m.modelId,
      displayName: m.displayName,
    }));

    const stillEnabled = imageModels.value.some(
      (m) => m.modelId === selectedModel.value?.modelId
    );
    if (stillEnabled) return;

    const platformDefault = list.find((m) => m.isDefault);
    selectedModel.value =
      (await findRecentlyUsedModel()) ??
      (platformDefault
        ? {
            modelId: platformDefault.modelId,
            displayName: platformDefault.displayName,
          }
        : null);
  }

  /** 历史最近使用的模型（按任务时间倒序取首个仍在启用列表内的） */
  async function findRecentlyUsedModel(): Promise<ImageModelOption | null> {
    try {
      const tasks = await AiImageAPI.listImageTasks({
        pageNum: 1,
        pageSize: RECENT_TASK_PAGE_SIZE,
      });
      for (const task of tasks.list) {
        const match = imageModels.value.find((m) => m.modelId === task.model);
        if (match) return match;
      }
    } catch {
      // 历史查询失败时回落启用列表首项
    }
    return null;
  }

  /** 选择模型 */
  function setModel(model: ImageModelOption) {
    selectedModel.value = model;
    status.value = "model";
  }

  // ==================== 方法 ====================

  /** 设置图片（图像输入页选中后调用） */
  function setImage(image: ImageData) {
    currentImage.value = image;
    status.value = "selected";
    fileId.value = image.fileId || null;
  }

  /** 更新处理参数 */
  function updateParams(newParams: Partial<DehazeParams>) {
    Object.assign(params.value, newParams);
  }

  /** 开始上传 */
  function startUploading() {
    status.value = "uploading";
  }

  /** 处理完成 */
  function complete(resultData: ImageTaskResponse) {
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
    selectedModel.value = null;
    params.value = { ...DEFAULT_DEHAZE_PARAMS };
    result.value = null;
    currentTaskId.value = null;
    errorMessage.value = "";
    isProcessing.value = false;
    elapsedTime.value = 0;
    cancelled.value = false;
    stopElapsedTimer();
  }

  // ==================== runImageTask ====================

  /**
   * 单次提交尝试（带递增重试）
   *
   * 成功返回 {ok:true, result}；重试耗尽返回 {ok:false, error}；
   * 取消返回 {ok:false}（cancelProcessing 已处理 fail）。
   */
  async function attempt(
    opts: RunImageTaskOptions,
    attemptNumber: number
  ): Promise<RunImageTaskResult> {
    if (cancelled.value) return { ok: false };

    try {
      const res = await AiImageAPI.submitAndWait(
        {
          model: selectedModel.value!.modelId,
          fileId: opts.fileId,
          imageUrl: opts.imageUrl,
          params: opts.params,
        },
        {
          onPoll: (_status, taskId) => {
            if (taskId) currentTaskId.value = taskId;
          },
        }
      );

      if (cancelled.value) return { ok: false };

      if (res.status === 4) return { ok: false };
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
   * 执行去雾处理（提交图像任务 + 递增重试 + 耗时计时）
   *
   * 行为：
   * 1. 未选模型或 isProcessing 为 true 时直接返回 {ok:false}
   * 2. 设置 isProcessing=true、cancelled=false，启动计时器
   * 3. 递增重试调用 submitAndWait；成功返回 {ok:true,result}，重试耗尽返回 {ok:false,error}
   * 4. cancelProcessing 可随时中止
   */
  async function runImageTask(
    opts: RunImageTaskOptions
  ): Promise<RunImageTaskResult> {
    if (isProcessing.value) return { ok: false };
    if (!selectedModel.value) {
      fail("请先选择模型");
      return { ok: false, error: "请先选择模型" };
    }

    isProcessing.value = true;
    cancelled.value = false;
    currentTaskId.value = null;
    status.value = "processing";
    errorMessage.value = "";
    startElapsedTimer();

    const ret = await attempt(opts, 0);

    stopElapsedTimer();
    isProcessing.value = false;

    if (ret.ok && ret.result) {
      complete(ret.result);
    } else if (!ret.ok && ret.error) {
      fail(ret.error);
    }
    // 取消或任务已处于取消态：cancelProcessing 已调用 fail

    return ret;
  }

  /** 取消正在进行的处理（已提交则同时取消后端任务） */
  async function cancelProcessing() {
    cancelled.value = true;
    stopElapsedTimer();
    isProcessing.value = false;
    if (currentTaskId.value) {
      // 取消为幂等操作，失败不影响前端已取消的终态
      await AiImageAPI.cancelImageTask(currentTaskId.value).catch(() => {});
    }
    fail("已取消");
  }

  return {
    // 状态
    status,
    currentImage,
    fileId,
    imageModels,
    selectedModel,
    params,
    result,
    errorMessage,
    isProcessing,
    elapsedTime,

    // 计算属性
    isCompleted,
    originUrl,

    // 方法
    loadImageModels,
    setModel,
    setImage,
    updateParams,
    startUploading,
    complete,
    reset,

    // 任务执行
    runImageTask,
    cancelProcessing,
  };
});
