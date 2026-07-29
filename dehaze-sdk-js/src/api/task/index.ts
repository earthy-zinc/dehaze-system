import { PageResult } from "@/types";
import request from "@/utils/request";
import { TaskCreateForm, TaskQuery, TaskVO } from "./model";

class TaskAPI {
  /**
   * 创建任务
   *
   * @param data 任务创建表单
   * @param idempotencyKey 客户端幂等键（可选），相同键返回已有任务
   */
  static create(data: TaskCreateForm, idempotencyKey?: string) {
    return request<TaskVO>({
      url: "/api/v1/tasks",
      method: "post",
      data,
      headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined,
    });
  }

  /**
   * 查询任务状态
   *
   * @param taskId 任务ID
   */
  static getStatus(taskId: string) {
    return request<TaskVO>({
      url: `/api/v1/tasks/${taskId}`,
      method: "get",
    });
  }

  /**
   * 下载任务结果（返回 Blob）
   *
   * @param taskId 任务ID
   */
  static download(taskId: string) {
    return request<Blob>({
      url: `/api/v1/tasks/${taskId}/download`,
      method: "get",
      responseType: "blob",
    });
  }

  /**
   * 取消任务
   *
   * @param taskId 任务ID
   */
  static cancel(taskId: string) {
    return request<void>({
      url: `/api/v1/tasks/${taskId}/cancel`,
      method: "post",
    });
  }

  /**
   * 重试失败的任务
   *
   * @param taskId 任务ID
   */
  static retry(taskId: string) {
    return request<TaskVO>({
      url: `/api/v1/tasks/${taskId}/retry`,
      method: "post",
    });
  }

  /**
   * 任务列表分页查询
   *
   * @param query 查询参数
   */
  static getPage(query?: TaskQuery) {
    return request<PageResult<TaskVO[]>>({
      url: "/api/v1/tasks",
      method: "get",
      params: query,
    });
  }
}

export default TaskAPI;
