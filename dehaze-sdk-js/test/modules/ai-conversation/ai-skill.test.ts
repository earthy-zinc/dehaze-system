import { describe, test, expect } from "vitest";
import { AiSkillAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import { createSkillForm, createSkillQuery, createSkillTestForm } from "#/factories/ai-skill";

/**
 * SKILL 管理（F-M08-006 §2.6.11/§2.6.14，管理操作需 ai:skill:manage）。
 *
 * 后端尚未实现 SKILL 管理路由：测试先行契约（以 dehaze-doc API接口.md §2.14 为行为断言依据），
 * 接口 404 时正向用例失败暴露，待后端实现后统一验证。
 * 数据前缀 test_skill_，普通用户 403（A0301）。
 */
describe("SKILL 管理 - AiSkillAPI (T-MF-086~089)", () => {
  describe("Skill CRUD", () => {
    test("T-MF-082 正向：创建 Skill 返回完整结构", async () => {
      await login(USERS.ADMIN.username);
      const form = createSkillForm();
      const result = await AiSkillAPI.createSkill(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.name).toBe(form.name);
      expect(result.instruction).toBe(form.instruction);
      expect(result.status).toBe(1);
    });

    test("T-MF-001 正向：Skill 列表（管理员全量）", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiSkillAPI.listSkills(createSkillQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("T-MF-004 正向：更新 Skill", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      const updated = await AiSkillAPI.updateSkill(created.id, { description: "updated-desc" });
      expect(updated.id).toBe(created.id);
      expect(updated.description).toBe("updated-desc");
    });

    test("T-MF-086 正向：删除 Skill（软删除）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      await AiSkillAPI.deleteSkill(created.id);
    });

    test("T-MF-089 负向：普通用户创建 Skill → A0301", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiSkillAPI.createSkill(createSkillForm()), ["A0301"]);
    });
  });

  describe("启停 / 试运行 / 市场", () => {
    test("T-MF-083 正向：启停 Skill", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      const disabled = await AiSkillAPI.switchSkillStatus(created.id, 0);
      expect(disabled.status).toBe(0);
      const enabled = await AiSkillAPI.switchSkillStatus(created.id, 1);
      expect(enabled.status).toBe(1);
    });

    test("T-MF-087 正向：Skill 试运行（不入库不推送）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      const result = await AiSkillAPI.testSkill(created.id, createSkillTestForm());
      expect(result).toBeDefined();
    });

    test("T-MF-088 正向：SKILL 市场目录可浏览", async () => {
      await login(USERS.ADMIN.username);
      const market = await AiSkillAPI.getMarket();
      expect(Array.isArray(market)).toBe(true);
    });

    test("T-MF-088 正向：共享 Skill 至市场（需先启用）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      const shared = await AiSkillAPI.shareToMarket(created.id);
      expect(shared.id).toBe(created.id);
      expect(shared.marketShared).toBe(1);
    });
  });
});
