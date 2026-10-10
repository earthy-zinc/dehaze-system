import { PageResult } from "@/types";
import request from "@/utils/request";
import { DEFAULT_POLL_INTERVAL_MS, DEFAULT_POLL_TIMEOUT_MS, sleep } from "@/utils/poll";
import {
  ImageEditForm,
  ImageTaskQuery,
  ImageTaskResponse,
  ImageTaskVO,
  PollOptions,
} from "./model";

/** 终态状态集合（COMPLETED=2 / FAILED=3 / CANCELED=4） */
const TERMINAL_STATUSES = new Set([2, 3, 4]);

class AiImageAPI {
  /** 提交图像编辑任务（去雾等），返回 taskId + status */
  static submitImageTask(data: ImageEditForm) {
    return request<ImageTaskResponse>({
      url: "/api/v1/images/edits",
      method: "post",
      data,
    });
  }

  /** 查询图像任务状态（仅任务归属用户可访问） */
  static getImageTask(taskId: number) {
    return request<ImageTaskResponse>({
      url: `/api/v1/images/tasks/${taskId}`,
      method: "get",
    });
  }

  /** 取消图像任务（幂等，对终态任务直接返回当前状态） */
  static cancelImageTask(taskId: number) {
    return request<ImageTaskResponse>({
      url: `/api/v1/images/tasks/${taskId}/cancel`,
      method: "post",
    });
  }

  /** 图像任务历史分页列表 */
  static listImageTasks(query?: ImageTaskQuery) {
    return request<PageResult<ImageTaskVO[]>>({
      url: "/api/v1/images/tasks",
      method: "get",
      params: query,
    });
  }

  /**
   * 提交图像任务并等待结果（封装 POST + 轮询 GET）
   *
   * - POST 立即返回，若 status=2（COMPLETED，缓存命中）直接返回
   * - status=1（PROCESSING）时按 intervalMs 轮询 GET，直到 COMPLETED/FAILED/CANCELED 或超时
   *
   * 默认：间隔 2s，超时 120s
   */
  static async submitAndWait(
    data: ImageEditForm,
    options?: PollOptions
  ): Promise<ImageTaskResponse> {
    const result = await this.submitImageTask(data);
    options?.onPoll?.(result.status, result.taskId);
    if (result.status !== 1) {
      return result;
    }
    return this.pollImageTask(result.taskId!, options);
  }

  /** 轮询图像任务直到终态（COMPLETED=2 / FAILED=3 / CANCELED=4）或超时 */
  private static async pollImageTask(
    taskId: number,
    options?: PollOptions
  ): Promise<ImageTaskResponse> {
    const interval = options?.intervalMs ?? DEFAULT_POLL_INTERVAL_MS;
    const timeout = options?.timeoutMs ?? DEFAULT_POLL_TIMEOUT_MS;
    const deadline = Date.now() + timeout;

    while (Date.now() < deadline) {
      await sleep(interval);
      const result = await this.getImageTask(taskId);
      options?.onPoll?.(result.status, result.taskId);
      if (TERMINAL_STATUSES.has(result.status)) {
        return result;
      }
    }
    throw new Error(`图像任务 ${taskId} 超时（${timeout}ms）`);
  }
}

export default AiImageAPI;
