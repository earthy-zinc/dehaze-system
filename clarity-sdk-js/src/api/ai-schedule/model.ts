import { PageQuery } from "@/types";

// ==================== 定时任务（F-M08-009） ====================

/** 输入来源类型：fixed-固定输入，dynamic-动态输入 */
export type ScheduleInputType = "fixed" | "dynamic";

/** 输出目标类型：message-站内消息，callback-回调 */
export type ScheduleOutputType = "message" | "callback";

/** 任务状态：1-正常，2-熔断停用 */
export type ScheduleStatus = 1 | 2;

/** 用户启停：1-启用，0-停用 */
export type ScheduleEnabled = 0 | 1;

/** 执行结果：0-执行中，1-成功，2-失败，3-跳过 */
export type RunResultStatus = 0 | 1 | 2 | 3;

/** 输入来源 JSON（type 为 fixed/dynamic，其余字段按类型不同） */
export interface ScheduleInputConfig {
  type: ScheduleInputType;
  [key: string]: unknown;
}

/** 输出目标 JSON（type 为 message/callback，其余字段按类型不同） */
export interface ScheduleOutputConfig {
  type: ScheduleOutputType;
  [key: string]: unknown;
}

/** 创建定时任务表单 */
export interface ScheduleCreateForm {
  /** 任务名称 */
  name: string;
  /** Cron 触发规则（5 位表达式，如 "0 9 * * *"） */
  cron: string;
  /** 任务时区（默认 Asia/Shanghai） */
  timezone?: string;
  /** 输入来源 JSON */
  input?: ScheduleInputConfig | null;
  /** 输出目标 JSON */
  output?: ScheduleOutputConfig | null;
}

/** 更新定时任务表单 */
export interface ScheduleUpdateForm {
  name?: string;
  cron?: string;
  timezone?: string;
  input?: ScheduleInputConfig | null;
  output?: ScheduleOutputConfig | null;
  /** 用户启停：1-启用，0-停用 */
  enabled?: ScheduleEnabled;
}

/** 启停定时任务表单 */
export interface ScheduleStatusForm {
  /** 目标启停状态：1-启用，0-停用 */
  enabled: ScheduleEnabled;
}

/** 定时任务详情 */
export interface ScheduledTaskDetail {
  id: number;
  /** 归属用户 ID */
  userId: number;
  name: string;
  cron: string;
  timezone: string;
  input?: ScheduleInputConfig | null;
  output?: ScheduleOutputConfig | null;
  /** 用户启停：1-启用，0-停用 */
  enabled: ScheduleEnabled;
  /** 任务状态：1-正常，2-熔断停用 */
  status: ScheduleStatus;
  /** 连续失败计数 */
  circuitStreak: number;
  /** 下次触发时间 */
  nextTriggerTime?: string | null;
  createTime?: string | null;
}

/** 最近一次执行摘要（列表聚合展示） */
export interface RunSummary {
  /** 执行结果：0-执行中，1-成功，2-失败，3-跳过 */
  status: RunResultStatus;
  /** 跳过原因 */
  skipReason?: string | null;
  /** 消耗积分 */
  credits?: number | null;
  /** 耗时（毫秒） */
  durationMs?: number | null;
  /** 失败原因 */
  errorMsg?: string | null;
  /** 关联会话 ID */
  conversationId?: number | null;
  /** 执行时间 */
  createTime?: string | null;
}

/** 定时任务列表项（含最近执行摘要） */
export interface ScheduledTaskListItem extends ScheduledTaskDetail {
  /** 最近一次执行摘要 */
  lastRun?: RunSummary | null;
}

/** 定时任务分页查询参数 */
export interface SchedulePageQuery extends PageQuery {
  /** 关键字（按名称模糊搜索） */
  keyword?: string;
}

/** 执行历史项 */
export interface RunHistoryItem {
  id: number;
  /** 关联定时任务 ID */
  scheduleId: number;
  /** 执行结果：0-执行中，1-成功，2-失败，3-跳过 */
  status: RunResultStatus;
  /** 跳过原因 */
  skipReason?: string | null;
  /** 消耗积分 */
  credits?: number | null;
  /** 耗时（毫秒） */
  durationMs?: number | null;
  /** 失败原因 */
  errorMsg?: string | null;
  /** 关联会话 ID */
  conversationId?: number | null;
  /** 调用链路 ID */
  requestId?: string | null;
  /** 触发窗口 */
  windowStart?: string | null;
  /** 执行时间 */
  createTime?: string | null;
}

/** Cron 解释与下次执行时间预览 */
export interface NextTimesPreview {
  /** Cron 的人类可读描述 */
  description: string;
  /** 接下来 N 次触发时间（ISO） */
  nextTimes: string[];
}

/** 手动触发受理结果 */
export interface RunAcceptedResult {
  accepted: boolean;
}
