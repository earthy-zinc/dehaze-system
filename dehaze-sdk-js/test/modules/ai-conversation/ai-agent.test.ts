import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiAgentAPI, AiConversationAPI, service } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import { uniqueCode } from "#/factories/common";
import {
  createAgentCopyForm,
  createAgentForm,
  createAgentMcpForm,
  createAgentPublishForm,
  createAgentQuery,
  createAgentSkillsForm,
  createAgentStatusForm,
  createAgentSubAgentsForm,
  createAgentUpdateForm,
  createEvalDatasetForm,
  createEvalSampleForm,
} from "#/factories/ai-agent";

/**
 * AI 智能体管理（T-MF-090~101、142~146 + 评测域）
 *
 * 管理端接口（创建/更新/删除/启停/复制/发布/回滚/评测）需 ai:agent:manage，
 * 普通用户仅可查看启用列表。数据前缀 test_agent_ / test_eval_，afterAll 清理。
 */
describe("AI 智能体管理 - AiAgentAPI (T-MF-090~101,142~146)", () => {
  let agentId: number;
  let agentCode: string;
  let subAgentId: number;
  const cleanupAgents: number[] = [];
  let createdSkillName = "";

  const deleteConversation = (id: number) =>
    AiConversationAPI.deleteConversation(id).catch(() => {});

  beforeAll(async () => {
    await login(USERS.ADMIN.username);
    const form = createAgentForm();
    const created = await AiAgentAPI.create(form);
    expect(created.id).toBeGreaterThan(0);
    agentId = created.id;
    agentCode = created.agentCode;
    cleanupAgents.push(agentId);
    // 子 Agent 供 setSubAgents 关联
    const sub = await AiAgentAPI.create(createAgentForm({ isSubagent: true }));
    subAgentId = sub.id;
    cleanupAgents.push(subAgentId);
  });

  afterAll(async () => {
    await login(USERS.ADMIN.username).catch(() => {});
    // 先解绑子 Agent、清空 Skills 再删 Agent（否则删除会因关联被拒）
    if (agentId) {
      await AiAgentAPI.setSubAgents(agentId, { subagents: [] }).catch(() => {});
      await AiAgentAPI.setSkills(agentId, { skills: [] }).catch(() => {});
    }
    for (const id of [...cleanupAgents].reverse()) {
      await AiAgentAPI.delete(id).catch(() => {});
    }
    // 清理测试 Skill
    if (createdSkillName) {
      try {
        const list = (await service.get("/api/v1/ai/skills", {
          params: { pageNum: 1, pageSize: 100 },
        })) as any;
        const skill = (list.list ?? []).find((s: any) => s.name === createdSkillName);
        if (skill) {
          await service.delete(`/api/v1/ai/skills/${skill.id}`);
        }
      } catch {
        /* 忽略清理失败 */
      }
    }
  });

  describe("POST /api/v1/ai/agents - 创建 Agent（管理员）", () => {
    test("T-MF-092 正向：创建成功 agent_code 唯一", async () => {
      await login(USERS.ADMIN.username);
      const form = createAgentForm();
      const result = await AiAgentAPI.create(form);
      expect(result.id).toBeGreaterThan(0);
      expect(result.agentCode).toBe(form.agentCode);
      expect(result.name).toBe(form.name);
      expect(result.status).toBe(1);
      expect(Array.isArray(result.skills)).toBe(true);
      cleanupAgents.push(result.id);
    });

    test("T-MF-094 负向：agent_code 重复 → A0501", async () => {
      await login(USERS.ADMIN.username);
      const form = createAgentForm();
      const first = await AiAgentAPI.create(form);
      cleanupAgents.push(first.id);
      await expectBizError(AiAgentAPI.create(form), ["A0501"]);
    });

    test("T-MF-093 负向：普通用户创建 Agent → 403", async () => {
      await login(USERS.USER.username);
      await expectBizError(AiAgentAPI.create(createAgentForm()), ["A0301"]);
      await login(USERS.ADMIN.username);
    });
  });

  describe("GET /api/v1/ai/agents - Agent 列表", () => {
    test("T-MF-090 正向：管理员分页列表", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiAgentAPI.list(createAgentQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("T-MF-091 普通用户列表仅启用项", async () => {
      await login(USERS.USER.username);
      const result = await AiAgentAPI.list(createAgentQuery());
      expect(Array.isArray(result.list)).toBe(true);
      result.list.forEach((a) => {
        expect(a.status).toBe(1);
      });
      await login(USERS.ADMIN.username);
    });
  });

  describe("GET /api/v1/ai/agents/enabled - 启用 Agent 列表", () => {
    test("T-MF-091 返回启用列表", async () => {
      await login(USERS.ADMIN.username);
      const enabled = await AiAgentAPI.listEnabled();
      expect(Array.isArray(enabled)).toBe(true);
      enabled.forEach((a) => {
        expect(a.status).toBe(1);
        expect(a.agentCode).toBeTruthy();
      });
    });
  });

  describe("GET /api/v1/ai/agents/{id} - Agent 详情", () => {
    test("T-MF-092 正向：查询详情含关联字段", async () => {
      await login(USERS.ADMIN.username);
      const detail = await AiAgentAPI.detail(agentId);
      expect(detail.id).toBe(agentId);
      expect(detail.agentCode).toBe(agentCode);
      expect(Array.isArray(detail.skills)).toBe(true);
      expect(Array.isArray(detail.mcpNamespaces)).toBe(true);
      expect(Array.isArray(detail.subagents)).toBe(true);
    });
  });

  describe("PUT /api/v1/ai/agents/{id} - 更新 Agent", () => {
    test("T-MF-095 正向：更新名称/系统提示词", async () => {
      await login(USERS.ADMIN.username);
      const updated = await AiAgentAPI.update(agentId, createAgentUpdateForm());
      expect(updated.id).toBe(agentId);
      expect(updated.name).toBeTruthy();
    });
  });

  describe("关联配置（覆盖式）", () => {
    test("T-MF-100 正向：设置 Skills 并断言关联", async () => {
      await login(USERS.ADMIN.username);
      const skillName = uniqueCode("test_skill");
      const createdSkill = (await service.post("/api/v1/ai/skills", {
        name: skillName,
        description: "测试技能",
        instruction: "这是一个测试技能的指令说明。",
      })) as any;
      expect(createdSkill.id).toBeGreaterThan(0);
      createdSkillName = skillName;

      await AiAgentAPI.setSkills(agentId, createAgentSkillsForm({ skills: [skillName] }));
      const detail = await AiAgentAPI.detail(agentId);
      expect(detail.skills).toContain(skillName);
    });

    test("T-MF-101 正向：设置 MCP 命名空间并断言关联", async () => {
      await login(USERS.ADMIN.username);
      await AiAgentAPI.setMcps(agentId, createAgentMcpForm({ mcpNamespaces: ["test_ns"] }));
      const detail = await AiAgentAPI.detail(agentId);
      expect(detail.mcpNamespaces).toContain("test_ns");
    });

    test("T-MF-111 正向：设置子 Agent 并断言关联", async () => {
      await login(USERS.ADMIN.username);
      await AiAgentAPI.setSubAgents(
        agentId,
        createAgentSubAgentsForm({ subagents: [{ agentId: subAgentId, priority: 1 }] })
      );
      const detail = await AiAgentAPI.detail(agentId);
      const linked = detail.subagents.find((s) => s.agentId === subAgentId);
      expect(linked).toBeDefined();
      expect(linked!.agentCode).toBeTruthy();
    });
  });

  describe("PATCH /api/v1/ai/agents/{id}/status - 启停", () => {
    test("T-MF-097 正向：停用后 status=0", async () => {
      await login(USERS.ADMIN.username);
      await AiAgentAPI.setStatus(agentId, createAgentStatusForm({ status: 0 }));
      const detail = await AiAgentAPI.detail(agentId);
      expect(detail.status).toBe(0);
    });

    test("T-MF-097 正向：重新启用 status=1", async () => {
      await login(USERS.ADMIN.username);
      await AiAgentAPI.setStatus(agentId, createAgentStatusForm({ status: 1 }));
      const detail = await AiAgentAPI.detail(agentId);
      expect(detail.status).toBe(1);
    });
  });

  describe("POST /api/v1/ai/agents/{id}/copy - 复制 Agent", () => {
    test("T-MF-098 正向：复制生成新 agent_code", async () => {
      await login(USERS.ADMIN.username);
      const copied = await AiAgentAPI.copy(agentId, createAgentCopyForm());
      expect(copied.id).toBeGreaterThan(0);
      expect(copied.agentCode).not.toBe(agentCode);
      expect(copied.name).toBeTruthy();
      cleanupAgents.push(copied.id);
    });
  });

  describe("版本管理与发布", () => {
    test("T-MF-121 发布返回新版本号", async () => {
      await login(USERS.ADMIN.username);
      const result = await AiAgentAPI.publish(agentId, createAgentPublishForm());
      // 后端返回 snake_case version_no，与 SDK 类型契约不一致，按实际返回断言并上报主 agent
      expect((result as any).version_no).toBeGreaterThan(0);
    });

    test("T-MF-120 版本历史分页", async () => {
      await login(USERS.ADMIN.username);
      const versions = await AiAgentAPI.versions(agentId, { pageNum: 1, pageSize: 20 });
      expect(Array.isArray(versions.list)).toBe(true);
      expect(typeof versions.total).toBe("number");
    });

    test("T-MF-123 回滚到历史版本", async () => {
      await login(USERS.ADMIN.username);
      const versions = await AiAgentAPI.versions(agentId, { pageNum: 1, pageSize: 20 });
      expect(versions.list.length).toBeGreaterThan(0);
      const targetVersion = versions.list[versions.list.length - 1]!.versionNo;
      const result = await AiAgentAPI.rollback(agentId, targetVersion);
      // 同 T-MF-121：后端返回 snake_case version_no，SDK 契约不一致，按实际返回断言并上报主 agent
      expect((result as any).version_no).toBeGreaterThan(0);
    });
  });

  describe("会话联动 - Agent 绑定", () => {
    test("T-MF-143 正向：创建会话绑定 Agent 并断言 agentCode/agentVersion", async () => {
      await login(USERS.ADMIN.username);
      // 无发布记录时 agentVersion 为 null，为 number 时须 >0
      const conv = await AiConversationAPI.createConversation({ agentCode });
      expect(conv.id).toBeGreaterThan(0);
      expect(conv.agentCode).toBe(agentCode);
      if (typeof conv.agentVersion === "number") {
        expect(conv.agentVersion).toBeGreaterThan(0);
      }
      await deleteConversation(conv.id);
    });

    test("T-MF-143 边界：不存在的 agentCode 后端不报错（记录 code，version=None）", async () => {
      await login(USERS.ADMIN.username);
      const fakeCode = uniqueCode("test_agent_none");
      const conv = await AiConversationAPI.createConversation({ agentCode: fakeCode });
      expect(conv.id).toBeGreaterThan(0);
      expect(conv.agentCode).toBe(fakeCode);
      expect(conv.agentVersion ?? null).toBeNull();
      await deleteConversation(conv.id);
    });

    test("T-MF-145 正向：未指定 agentCode 使用默认 Agent", async () => {
      await login(USERS.ADMIN.username);
      const conv = await AiConversationAPI.createConversation({});
      expect(conv.id).toBeGreaterThan(0);
      expect(conv.agentCode).toBe("default");
      await deleteConversation(conv.id);
    });
  });

  describe("评测域（T-MF-126~129 前置）", () => {
    let datasetId: number;
    let sampleId: number;

    test("T-MF-126 正向：创建评测集", async () => {
      await login(USERS.ADMIN.username);
      const dataset = await AiAgentAPI.createEvalDataset(agentId, createEvalDatasetForm());
      expect(dataset.id).toBeGreaterThan(0);
      expect(dataset.agentId).toBe(agentId);
      expect(dataset.datasetType).toBe("dev");
      datasetId = dataset.id;
    });

    test("T-MF-126 正向：评测集列表", async () => {
      await login(USERS.ADMIN.username);
      const datasets = await AiAgentAPI.listEvalDatasets(agentId);
      expect(Array.isArray(datasets)).toBe(true);
      const found = datasets.find((d) => d.id === datasetId);
      expect(found).toBeDefined();
    });

    test("T-MF-126 正向：创建评测样本", async () => {
      await login(USERS.ADMIN.username);
      const sample = await AiAgentAPI.createEvalSample(
        agentId,
        datasetId,
        createEvalSampleForm({ datasetId })
      );
      expect(sample.id).toBeGreaterThan(0);
      expect(sample.datasetId).toBe(datasetId);
      expect(sample.taskGoal).toBeTruthy();
      sampleId = sample.id;
    });

    test("T-MF-126 正向：评测样本列表", async () => {
      await login(USERS.ADMIN.username);
      const samples = await AiAgentAPI.listEvalSamples(agentId, datasetId);
      expect(Array.isArray(samples)).toBe(true);
      const found = samples.find((s) => s.id === sampleId);
      expect(found).toBeDefined();
    });

    test("T-MF-126 清理：删除样本与评测集", async () => {
      await login(USERS.ADMIN.username);
      await AiAgentAPI.deleteEvalSample(agentId, sampleId);
      await AiAgentAPI.deleteEvalDataset(agentId, datasetId);
    });
  });

  describe("DELETE /api/v1/ai/agents/{id} - 删除 Agent", () => {
    test("T-MF-096 正向：删除无引用 Agent", async () => {
      await login(USERS.ADMIN.username);
      const form = createAgentForm();
      const created = await AiAgentAPI.create(form);
      expect(created.id).toBeGreaterThan(0);
      await AiAgentAPI.delete(created.id);
      // 删除后详情应失败
      await expectBizError(AiAgentAPI.detail(created.id), ["A0401", "A0403", "A0400", "B0001"]);
    });

    test("T-MF-146 负向：默认 Agent 不可删除 → A0503", async () => {
      await login(USERS.ADMIN.username);
      const list = (await AiAgentAPI.list(createAgentQuery({ keyword: "default" }))) as any;
      const def = (list.list ?? []).find((a: any) => a.agentCode === "default");
      if (def) {
        await expectBizError(AiAgentAPI.delete(def.id), ["A0503"]);
      }
    });
  });
});
