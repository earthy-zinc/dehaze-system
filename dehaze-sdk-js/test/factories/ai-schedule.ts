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
  name: uniqueName("test_task"),
  cron: "0 9 * * *",
  timezone: "Asia/Shanghai",
  input: { type: "fixed", content: "每天上午9点执行一次去雾图像批处理" },
  output: { type: "message" },
  ...overrides,
});

/** 定时任务更新表单工厂 */
export const createScheduleUpdateForm = (
  overrides?: Partial<ScheduleUpdateForm>
): ScheduleUpdateForm => ({
  name: uniqueName("test_task_updated"),
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
