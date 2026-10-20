import { pageQuery, uniqueName } from "./common";
import type {
  ScheduleCreateForm,
  SchedulePageQuery,
  ScheduleStatusForm,
  ScheduleUpdateForm,
} from "../../src/api/ai-schedule/model";

/** 定时任务创建表单工厂（name 前缀 test_task_，cron 为合法 5 位表达式） */
export const createScheduleForm = (
  overrides?: Partial<ScheduleCreateForm>
): ScheduleCreateForm => ({
  name: uniqueName(TEST_SCHEDULE_NAME_PREFIX),
  cron: "0 9 * * *",
  timezone: "Asia/Shanghai",
  input: { type: "fixed", content: "每天上午9点执行一次去雾图像批处理" },
  output: { type: "message" },
  ...overrides,
});

/** 定时任务测试名统一前缀：任务转定时执行时后端会据此生成会话标题，用可读中文前缀，
 * 集测被中断而残留时在会话列表里也能一眼看出是测试数据（不再出现 test_task_updated_xxx 这类机器名） */
export const TEST_SCHEDULE_NAME_PREFIX = "定时任务测试";

/** 定时任务更新表单工厂 */
export const createScheduleUpdateForm = (
  overrides?: Partial<ScheduleUpdateForm>
): ScheduleUpdateForm => ({
  name: uniqueName(TEST_SCHEDULE_NAME_PREFIX),
  ...overrides,
});

/** 定时任务分页查询参数工厂 */
export const createScheduleQuery = (overrides?: Partial<SchedulePageQuery>): SchedulePageQuery =>
  pageQuery<SchedulePageQuery>({ ...overrides });

/** 启停表单工厂 */
export const createScheduleStatusForm = (
  overrides?: Partial<ScheduleStatusForm>
): ScheduleStatusForm => ({
  enabled: 1,
  ...overrides,
});
