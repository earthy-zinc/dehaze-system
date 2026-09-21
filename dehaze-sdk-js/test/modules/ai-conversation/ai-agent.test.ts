import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { AiAgentAPI, AiConversationAPI, AiMCPAPI, service } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import { uniqueCode } from "#/factories/common";
import { createMcpServerForm } from "#/factories/ai-mcp";
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

/** 轮询异步评测任务至终态：超时抛错而非跳过（任务不进入终态即环境故障，必须暴露） */
async function pollEvalTask(agentId: number, taskId: string) {
  for (let round = 0; round < 40; round++) {
    const task = await AiAgentAPI.getEvalTask(agentId, taskId);
    if (task.status === "succeeded" || task.status === "failed") return task;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`评测任务未在 10s 内进入终态：${taskId}`);
}

/**
 * AI 智能体管理（T-MF-090~101、142~146 + 评测域）
 *
 * 管理端接口（创建/更新/删除/启停/复制/发布/回滚/评测）需 ai:agent:manage，
 * 普通用户仅可查看启用列表。数据前缀 test_agent_ / test_eval_，afterAll 清理。
 *
 * 异步评测任务（POST /runs → task_id；GET /tasks/{taskId}）的 taskId 由后端生成，
 * 无法在离线单测中覆盖，归此处集成用例。
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
      // 命名空间必须已在注册 MCP Server 下声明（后端引用完整性校验），故先注册 Server 再声明命名空间
      const server = await AiMCPAPI.createServer(createMcpServerForm());
      const namespace = uniqueCode("test_ns");
      let bound = false;
      try {
        await AiMCPAPI.updateNamespaces(server.id, [{ name: namespace, toolNames: [] }]);
        await AiAgentAPI.setMcps(agentId, createAgentMcpForm({ mcpNamespaces: [namespace] }));
        bound = true;
        const detail = await AiAgentAPI.detail(agentId);
        expect(detail.mcpNamespaces).toContain(namespace);
      } finally {
        // 先解绑再删 Server，否则 Server 删除会因命名空间被 Agent 引用而拒绝
        if (bound) {
          await AiAgentAPI.setMcps(agentId, { mcpNamespaces: [] }).catch(() => {});
        }
        await AiMCPAPI.deleteServer(server.id).catch(() => {});
      }
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
      // VersionResult 契约为 snake_case version_no（与后端一致）
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
      // 更新 Agent 即写入草稿版本（status=1），用于下方草稿不可回滚的负向断言
      await AiAgentAPI.update(agentId, createAgentUpdateForm());
      const versions = await AiAgentAPI.versions(agentId, { pageNum: 1, pageSize: 20 });
      expect(versions.list.length).toBeGreaterThan(0);

      // 草稿是发布链路的评测中间态，不可作为回滚目标
      const draft = versions.list.find((v) => v.status === 1);
      expect(draft).toBeDefined();
      await expectBizError(AiAgentAPI.rollback(agentId, draft!.versionNo), ["A0502"]);

      const published = versions.list.find((v) => v.status === 2);
      expect(published).toBeDefined();
      const result = await AiAgentAPI.rollback(agentId, published!.versionNo);
      // VersionResult 契约为 snake_case version_no（与后端一致）
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

    test("T-MF-127 正向：异步触发评测返回 taskId 并轮询到终态", async () => {
      await login(USERS.ADMIN.username);
      // 专用 Agent：无回归集时评测平凡放行，不进入样本执行（不依赖模型可用性）
      const evalAgent = await AiAgentAPI.create(createAgentForm());
      cleanupAgents.push(evalAgent.id);

      const ack = await AiAgentAPI.runEvalAsync(evalAgent.id);
      expect(typeof ack.taskId).toBe("string");
      expect(ack.taskId.length).toBeGreaterThan(0);

      const task = await pollEvalTask(evalAgent.id, ack.taskId);
      expect(task.taskId).toBe(ack.taskId);
      expect(task.status).toBe("succeeded");
      expect(task.progress.total).toBeGreaterThanOrEqual(0);
      expect(task.progress.done).toBeLessThanOrEqual(task.progress.total);
      // 无回归集：平凡放行且不产生评测执行记录
      expect(task.error).toBeFalsy();
      expect(task.result?.passed).toBe(true);
      expect(task.result?.degraded).toBe(false);
      expect(task.result?.insufficientEval).toBe(false);
      expect(task.result?.failedSamples).toEqual([]);
      // 无回归集未产生评测记录：runId 字段缺失（后端空字段不下发），非显式 null
      expect(task.result?.runId).toBeUndefined();
    });

    test("T-MF-127 负向：任务 ID 不存在 → A0401", async () => {
      await login(USERS.ADMIN.username);
      await expectBizError(AiAgentAPI.getEvalTask(agentId, "not-exist-task-id"), ["A0401"]);
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
