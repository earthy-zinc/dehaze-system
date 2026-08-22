import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  NextTimesPreview,
  RunAcceptedResult,
  RunHistoryItem,
  ScheduleCreateForm,
  SchedulePageQuery,
  ScheduleStatusForm,
  ScheduleUpdateForm,
  ScheduledTaskDetail,
  ScheduledTaskListItem,
} from "./model";

/**
 * AI 定时调度 API（F-M08-009）
 *
 * 内部 API（`/api/v1/ai/scheduled-tasks`），用户级操作，无独立权限标识，
 * 归属校验由后端基于登录用户完成。含任务 CRUD、Cron 下次触发预览、
 * 启停、手动触发与执行历史。
 */
class AiScheduleAPI {
  // ==================== 任务 CRUD ====================

  /** 创建定时任务（保存后返回下次触发时间预览） */
  static create(data: ScheduleCreateForm) {
    return request<ScheduledTaskDetail>({
      url: "/api/v1/ai/scheduled-tasks",
      method: "post",
      data,
    });
  }

  /** 定时任务列表（分页，按下次触发时间排序，含最近执行结果摘要） */
  static list(query?: SchedulePageQuery) {
    return request<PageResult<ScheduledTaskListItem[]>>({
      url: "/api/v1/ai/scheduled-tasks",
      method: "get",
      params: query,
    });
  }

  /**
   * Cron 解释与下次执行时间预览。
   * 集合级路径，不归属任何任务。
   */
  static previewNextTimes(cron: string, count = 5) {
    return request<NextTimesPreview>({
      url: "/api/v1/ai/scheduled-tasks/next-times",
      method: "get",
      params: { cron, count },
    });
  }

  /** 定时任务详情 */
  static detail(id: number) {
    return request<ScheduledTaskDetail>({
      url: `/api/v1/ai/scheduled-tasks/${id}`,
      method: "get",
    });
  }

  /** 更新定时任务（触发规则/输入来源/输出目标，变更后重算下次触发时间） */
  static update(id: number, data: ScheduleUpdateForm) {
    return request<ScheduledTaskDetail>({
      url: `/api/v1/ai/scheduled-tasks/${id}`,
      method: "put",
      data,
    });
  }

  /** 删除定时任务（软删除） */
  static delete(id: number) {
    return request({
      url: `/api/v1/ai/scheduled-tasks/${id}`,
      method: "delete",
    });
  }

  // ==================== 启停 / 手动触发 ====================

  /** 启停定时任务（熔断停用后可重新启用，后端会重置连续失败计数） */
  static setStatus(id: number, data: ScheduleStatusForm) {
    return request({
      url: `/api/v1/ai/scheduled-tasks/${id}/status`,
      method: "patch",
      data,
    });
  }

  /**
   * 手动触发一次执行（验证配置/补跑），不改变原定时规则。
   * 无人值守执行可耗时分钟级，受理即返回 `{accepted: true}`，
   * 实际结果由后台写入执行历史并通知用户。
   */
  static run(id: number) {
    return request<RunAcceptedResult>({
      url: `/api/v1/ai/scheduled-tasks/${id}/run`,
      method: "post",
    });
  }

  // ==================== 执行历史 ====================

  /** 执行历史分页（结果/消耗积分/耗时/失败原因/跳过原因） */
  static history(id: number, query?: { pageNum?: number; pageSize?: number }) {
    return request<PageResult<RunHistoryItem[]>>({
      url: `/api/v1/ai/scheduled-tasks/${id}/history`,
      method: "get",
      params: query,
    });
  }
}

export default AiScheduleAPI;
