import { PageResult } from "@/types";
import request from "@/utils/request";
import { TaskCreateForm, TaskQuery, TaskVO } from "./model";

class TaskAPI {
  /**
   * 创建任务
   *
   * @param data 任务创建表单
   */
  static create(data: TaskCreateForm) {
    return request<any, TaskVO>({
      url: "/api/v1/tasks",
      method: "post",
      data,
    });
  }

  /**
   * 查询任务状态
   *
   * @param taskId 任务ID
   */
  static getStatus(taskId: string) {
    return request<any, TaskVO>({
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
    return request<any, Blob>({
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
    return request<any, void>({
      url: `/api/v1/tasks/${taskId}`,
      method: "delete",
    });
  }

  /**
   * 任务列表分页查询
   *
   * @param query 查询参数
   */
  static getPage(query?: TaskQuery) {
    return request<any, PageResult<TaskVO[]>>({
      url: "/api/v1/tasks",
      method: "get",
      params: query,
    });
  }
}

export default TaskAPI;
