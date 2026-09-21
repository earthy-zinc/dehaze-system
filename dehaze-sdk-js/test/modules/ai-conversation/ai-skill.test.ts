import { describe, test, expect } from "vitest";
import { AiSkillAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  buildSkillZip,
  createSkillForm,
  createSkillQuery,
  createSkillTestForm,
  createSkillZipFile,
} from "#/factories/ai-skill";

/**
 * SKILL 管理（F-M08-006 §2.6.11/§2.6.14，管理操作需 ai:skill:manage）。
 *
 * 断言依据 dehaze-doc API接口.md §2.12；数据前缀 test_skill_，普通用户 403（A0301），
 * 禁用 Skill 对普通用户按不存在处理（A0401）。
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
      // 新建 Skill 默认禁用（status=0），需显式启用
      expect(result.status).toBe(0);
    });

    test("T-MF-001 正向：Skill 列表（管理员全量）", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiSkillAPI.listSkills(createSkillQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("T-MF-001 正向：管理员状态筛选（status=0 仅含禁用项）", async () => {
      await login(USERS.ADMIN.username);
      const enabled = await AiSkillAPI.createSkill(createSkillForm());
      // 创建默认禁用，需显式启用才能进入 status=1 侧
      await AiSkillAPI.switchSkillStatus(enabled.id, 1);
      const disabled = await AiSkillAPI.createSkill(createSkillForm());
      await AiSkillAPI.switchSkillStatus(disabled.id, 0);
      try {
        const result = await AiSkillAPI.listSkills({ status: 0, pageNum: 1, pageSize: 100 });
        const names = result.list.map((s) => s.id);
        expect(names).toContain(disabled.id);
        expect(names).not.toContain(enabled.id);
      } finally {
        await AiSkillAPI.deleteSkill(enabled.id).catch(() => {});
        await AiSkillAPI.deleteSkill(disabled.id).catch(() => {});
      }
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

    test("T-MF-084a 越权：禁用 Skill 对普通用户按不存在处理 → A0401", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      await AiSkillAPI.switchSkillStatus(created.id, 0);
      try {
        // 管理员仍可见
        const adminView = await AiSkillAPI.getSkill(created.id);
        expect(adminView.status).toBe(0);
        // 普通用户按不存在处理（与列表"仅启用"口径一致）
        await login(USERS.USER.username);
        await expectBizError(AiSkillAPI.getSkill(created.id), ["A0401"]);
      } finally {
        await login(USERS.ADMIN.username);
        await AiSkillAPI.deleteSkill(created.id).catch(() => {});
      }
    });

    test("T-MF-087 正向：Skill 试运行（不入库不推送）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.createSkill(createSkillForm());
      // 试运行要求启用态：新建默认禁用
      await AiSkillAPI.switchSkillStatus(created.id, 1);
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
      // python 契约：共享要求启用态，禁用 Skill 共享 → A0400
      await AiSkillAPI.switchSkillStatus(created.id, 1);
      const shared = await AiSkillAPI.shareToMarket(created.id);
      expect(shared.id).toBe(created.id);
      expect(shared.marketShared).toBe(1);
    });
  });

  describe("SKILL zip 上传（Agent Skills 规范）", () => {
    test("T-MF-092 正向：zip 上传解析 frontmatter 与文件清单", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.uploadSkill(createSkillZipFile());
      expect(created.id).toBeGreaterThan(0);
      expect(created.name).toMatch(/^skillzip[a-z0-9]+$/);
      expect(created.license).toBe("Apache-2.0");
      expect(created.metadata).toEqual({ version: "1.0" });
      expect(created.description).toContain("提取 PDF");
      expect(Array.isArray(created.files)).toBe(true);
      const paths = (created.files ?? []).map((f) => f.path);
      expect(paths).toContain("script/extract.py");
      expect(paths).toContain("reference/REFERENCE.md");
      const py = created.files?.find((f) => f.path === "script/extract.py");
      expect(py?.fileSize).toBeGreaterThan(0);
      expect(py?.fileType).toBe("text/x-python");
      await AiSkillAPI.deleteSkill(created.id);
    });

    test("T-MF-093 正向：读取 SKILL 资源文件内容（对象存储）", async () => {
      await login(USERS.ADMIN.username);
      const created = await AiSkillAPI.uploadSkill(createSkillZipFile());
      const blob = await AiSkillAPI.getSkillFile(created.id, "reference/REFERENCE.md");
      expect(blob).toBeInstanceOf(Blob);
      expect(await blob.text()).toContain("参考文档");
      await AiSkillAPI.deleteSkill(created.id);
    });

    test("T-MF-094 正向：zip 校验失败返回业务错误（缺 SKILL.md）", async () => {
      await login(USERS.ADMIN.username);
      const badZip = buildSkillZip({ "readme-only/README.md": "# 无 SKILL.md" });
      // Node Buffer 的 ArrayBufferLike 不满足 BlobPart，转成 Uint8Array（字节等价）
      const file = new File([new Uint8Array(badZip)], "bad.zip", { type: "application/zip" });
      await expectBizError(AiSkillAPI.uploadSkill(file), ["A0400"]);
    });
  });
});
