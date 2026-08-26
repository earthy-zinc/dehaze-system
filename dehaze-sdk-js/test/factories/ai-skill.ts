import { pageQuery } from "./common";
import { uniqueName } from "./common";
import type { SkillForm, SkillQuery, SkillTestForm } from "../../src/api/ai-skill/model";

/** Skill 创建表单工厂（名称前缀 test_skill_ 便于清理） */
export const createSkillForm = (overrides?: Partial<SkillForm>): SkillForm => ({
  name: uniqueName("test_skill"),
  description: "SKILL 管理契约测试",
  scene: "通用",
  instruction: "# 测试 Skill 指令\n按步骤执行测试流程",
  ...overrides,
});

/** Skill 分页查询参数工厂 */
export const createSkillQuery = (overrides?: Partial<SkillQuery>) =>
  pageQuery<SkillQuery>({ ...overrides });

/** Skill 试运行表单工厂 */
export const createSkillTestForm = (overrides?: Partial<SkillTestForm>): SkillTestForm => ({
  inputData: { text: "测试输入" },
  ...overrides,
});
