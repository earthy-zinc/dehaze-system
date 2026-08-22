import { AiConversationAPI, AiBillingAPI } from "../../../index";
import { expectBizError } from "#/utils/assertion";
import { login } from "#/utils/auth";
import { USERS } from "#/factories/constants";
import {
  createConversationForm,
  createConversationQuery,
  createConversationUpdateForm,
  createEditMessageForm,
  createFeedbackForm,
  createMemoryForm,
  createMemoryQuery,
} from "#/factories/ai-conversation";
import type { MessageStreamHandlers } from "@/api/ai-conversation";
import type { MessageEndEvent, MessageStartEvent } from "@/api/ai-conversation/model";

/**
 * 启动 SSE 流并等待结束（message.end / onClose / onNetworkError / 超时）。
 * 统一记录事件序列与网络错误，供各用例断言，避免重复编写 Promise + 超时逻辑。
 */
async function collectStream(
  launch: (handlers: Partial<MessageStreamHandlers>) => AbortController | void,
  capture: {
    onStart?: (data: MessageStartEvent) => void;
    onEnd?: (data: MessageEndEvent) => void;
  } = {},
  timeoutMs = 60000
): Promise<{ events: string[]; networkError: Error | null; closed: boolean }> {
  const events: string[] = [];
  let networkError: Error | null = null;
  let closed = false;
  await new Promise<void>((resolve, reject) => {
    let controller: AbortController | undefined;
    const timeout = setTimeout(() => {
      controller?.abort();
      reject(new Error("SSE 流式请求超时"));
    }, timeoutMs);

    const launched = launch({
      onStart: (data) => {
        events.push("start");
        capture.onStart?.(data);
      },
      onContentBlockDelta: () => events.push("delta"),
      onContentBlockStop: () => events.push("block_stop"),
      onEnd: (data) => {
        events.push("end");
        capture.onEnd?.(data);
        clearTimeout(timeout);
        resolve();
      },
      onNetworkError: (error) => {
        networkError = error;
        clearTimeout(timeout);
        resolve();
      },
      onClose: () => {
        closed = true;
        clearTimeout(timeout);
        resolve();
      },
    });
    if (launched) controller = launched;
  });
  return { events, networkError, closed };
}

/** 启动流式调用并等待 onNetworkError/onClose 结束，返回捕获到的网络错误（可能为 null）。 */
async function expectStreamNetworkError(
  launch: (handlers: Partial<MessageStreamHandlers>) => AbortController | void,
  timeoutMs = 15000
): Promise<Error | null> {
  const { networkError } = await collectStream(launch, {}, timeoutMs);
  return networkError;
}

/** 发送消息并等待 message.end，返回 assistant 消息 ID；网络错误或业务错误时 reject。 */
function sendMessageCompleted(conversationId: number, content: string): Promise<number> {
  return new Promise((resolve, reject) => {
    let controller: AbortController | undefined;
    let msgId = 0;
    const timeout = setTimeout(() => {
      controller?.abort();
      reject(new Error(`SSE 发送消息超时（60s）：${content.slice(0, 30)}`));
    }, 60000);

    controller = AiConversationAPI.sendMessage(
      conversationId,
      { content },
      {
        onStart: (data) => {
          msgId = data.messageId;
        },
        onEnd: () => {
          clearTimeout(timeout);
          resolve(msgId);
        },
        onNetworkError: (error) => {
          clearTimeout(timeout);
          reject(error);
        },
        onError: (data) => {
          clearTimeout(timeout);
          reject(new Error(`SSE 业务错误: ${data.code} ${data.message}`));
        },
        onClose: () => {
          clearTimeout(timeout);
          reject(new Error("SSE 流在未收到 message.end 时关闭"));
        },
      }
    );
  });
}

describe("AI 对话模块接口测试 - AiConversationAPI", () => {
  const createdConversationIds: number[] = [];
  const createdMemoryIds: number[] = [];

  afterAll(async () => {
    for (const id of createdConversationIds.reverse()) {
      try {
        await AiConversationAPI.deleteConversation(id);
      } catch (e) {
        console.warn(`清理会话失败:`, e);
      }
    }
    for (const id of createdMemoryIds.reverse()) {
      try {
        await AiConversationAPI.deleteMemory(id);
      } catch (e) {
        console.warn(`清理记忆失败:`, e);
      }
    }
  });

  /** 创建一个测试会话并登记清理 */
  async function createTestConversation(overrides?: Parameters<typeof createConversationForm>[0]) {
    const conv = await AiConversationAPI.createConversation(createConversationForm(overrides));
    createdConversationIds.push(conv.id);
    return conv;
  }

  /** 创建一个测试记忆并登记清理 */
  async function createTestMemory(overrides?: Parameters<typeof createMemoryForm>[0]) {
    const memory = await AiConversationAPI.createMemory(createMemoryForm(overrides));
    createdMemoryIds.push(memory.id);
    return memory;
  }

  // ===== 会话管理 =====

  describe("POST /api/v1/ai/conversations - 创建会话 [T-CV-001]", () => {
    test("正向测试：创建会话，标题默认'新对话'", async () => {
      const result = await createTestConversation();
      expect(result.id).toBeGreaterThan(0);
      expect(result.title).toBe("新对话");
      expect(result.model).toBeTruthy();
      expect(result.status).toBe(1);
      expect(result.messageCount).toBe(0);
    });

    test("正向测试：创建会话指定模型和配置", async () => {
      const result = await createTestConversation({
        model: "gpt-4o",
        modelConfig: { temperature: 0.7, maxOutputTokens: 2048 },
        systemPrompt: "你是一个图像去雾助手",
      });
      expect(result.id).toBeGreaterThan(0);
      expect(result.model).toBe("gpt-4o");
      expect(result.modelConfig?.temperature).toBe(0.7);
      expect(result.systemPrompt).toBe("你是一个图像去雾助手");
    });
  });

  describe("GET /api/v1/ai/conversations - 会话列表 [T-CV-002]", () => {
    test("正向测试：分页查询会话列表", async () => {
      const result = await AiConversationAPI.getConversations(createConversationQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("正向测试：按关键字搜索会话 [T-CV-003]", async () => {
      const conv = await createTestConversation();
      const newTitle = `search_test_${Date.now()}`;
      await AiConversationAPI.updateConversation(
        conv.id,
        createConversationUpdateForm({ title: newTitle })
      );

      const result = await AiConversationAPI.getConversations(
        createConversationQuery({ keyword: "search_test" })
      );
      expect(result.list.length).toBeGreaterThan(0);
    });

    test("正向测试：按状态筛选（活跃）", async () => {
      const result = await AiConversationAPI.getConversations(
        createConversationQuery({ status: 1 })
      );
      result.list.forEach((conv) => expect(conv.status).toBe(1));
    });

    test("验证：会话列表按置顶+最后消息时间倒序", async () => {
      // 置顶会话 B 并验证其排在未置顶会话 A 之前；keyword 限定到本次创建的两个会话，
      // 避免历史测试数据把新会话挤出第一页
      const marker = `sort_test_${Date.now()}`;
      const convA = await createTestConversation();
      const convB = await createTestConversation();
      await AiConversationAPI.updateConversation(
        convA.id,
        createConversationUpdateForm({ title: `${marker}_A` })
      );
      await AiConversationAPI.updateConversation(
        convB.id,
        createConversationUpdateForm({ title: `${marker}_B` })
      );
      await AiConversationAPI.pinConversation(convB.id);

      const result = await AiConversationAPI.getConversations(
        createConversationQuery({ keyword: marker, pageSize: 20 })
      );
      const idxA = result.list.findIndex((c) => c.id === convA.id);
      const idxB = result.list.findIndex((c) => c.id === convB.id);
      expect(idxA).toBeGreaterThanOrEqual(0);
      expect(idxB).toBeGreaterThanOrEqual(0);
      expect(idxB).toBeLessThan(idxA);
      expect(result.list[idxB]!.pinned).toBe(1);
    });
  });

  describe("GET /api/v1/ai/conversations/{id} - 会话详情 [T-CV-004]", () => {
    let testConvId: number;

    beforeAll(async () => {
      const result = await createTestConversation();
      testConvId = result.id;
    });

    test("正向测试：查询会话详情含模型配置和消息数", async () => {
      const detail = await AiConversationAPI.getConversation(testConvId);
      expect(detail.id).toBe(testConvId);
      expect(detail.model).toBeTruthy();
      expect(typeof detail.messageCount).toBe("number");
    });

    test("边界：查询不存在的会话应失败 [T-CV-009]", async () => {
      await expectBizError(AiConversationAPI.getConversation(99999999), ["A0401"]);
    });
  });

  describe("PATCH /api/v1/ai/conversations/{id} - 更新会话", () => {
    let testConvId: number;

    beforeAll(async () => {
      const result = await createTestConversation();
      testConvId = result.id;
    });

    test("正向测试：修改标题（titleSource=manual）[T-CV-005]", async () => {
      const newTitle = `updated_${Date.now()}`;
      const updated = await AiConversationAPI.updateConversation(
        testConvId,
        createConversationUpdateForm({ title: newTitle })
      );
      expect(updated.title).toBe(newTitle);
      expect(updated.titleSource).toBe("manual");
    });

    test("正向测试：置顶/取消置顶（pinned 为 number）[T-CV-006]", async () => {
      const pinned = await AiConversationAPI.updateConversation(
        testConvId,
        createConversationUpdateForm({ pinned: 1 })
      );
      expect(pinned.pinned).toBe(1);

      const unpinned = await AiConversationAPI.updateConversation(
        testConvId,
        createConversationUpdateForm({ pinned: 0 })
      );
      expect(unpinned.pinned).toBe(0);
    });

    test("正向测试：归档会话（status=2）[T-CV-007]", async () => {
      const archived = await AiConversationAPI.updateConversation(
        testConvId,
        createConversationUpdateForm({ status: 2 })
      );
      expect(archived.status).toBe(2);
      // 归档后不在活跃列表中
      const active = await AiConversationAPI.getConversations(
        createConversationQuery({ status: 1 })
      );
      expect(active.list.find((c) => c.id === testConvId)).toBeUndefined();
    });

    test("边界：更新不存在的会话应失败", async () => {
      await expectBizError(AiConversationAPI.updateConversation(99999999, { title: "test" }), [
        "A0401",
      ]);
    });
  });

  describe("DELETE /api/v1/ai/conversations/{id} - 删除会话 [T-CV-008]", () => {
    test("正向测试：软删除会话", async () => {
      const created = await createTestConversation();
      await AiConversationAPI.deleteConversation(created.id);
      await expectBizError(AiConversationAPI.getConversation(created.id), ["A0401"]);
    });

    test("边界：删除不存在的会话应失败", async () => {
      await expectBizError(AiConversationAPI.deleteConversation(99999999), ["A0401"]);
    });

    test("安全：越权删除他人会话应失败 [T-CV-009]", async () => {
      const created = await createTestConversation();
      await login(USERS.USER.username);
      try {
        await expectBizError(AiConversationAPI.deleteConversation(created.id), ["A0401"]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });
  });

  // ===== 会话能力扩展 =====

  describe("会话能力扩展 - scene/agentCode", () => {
    test("正向测试：创建会话传 scene，systemPrompt 采用场景模板 [T-CT-082]", async () => {
      const conv = await createTestConversation({ scene: "image_dispatch" });
      expect(conv.systemPrompt).toContain("图像处理");
      expect(conv.systemPrompt).toContain("调度");
    });

    test("正向测试：agentCode 传不存在的编码不报错，agentCode 原样记录 [T-CT-081]", async () => {
      const fakeCode = `test_agent_${Date.now()}`;
      const conv = await createTestConversation({ agentCode: fakeCode });
      expect(conv.agentCode).toBe(fakeCode);
    });
  });

  describe("会话恢复/置顶/已读", () => {
    test("正向测试：软删除后恢复 [T-CV-013]", async () => {
      const conv = await createTestConversation();
      await AiConversationAPI.deleteConversation(conv.id);
      await expectBizError(AiConversationAPI.getConversation(conv.id), ["A0401"]);

      const restored = await AiConversationAPI.restoreConversation(conv.id);
      expect(restored.id).toBe(conv.id);
      const detail = await AiConversationAPI.getConversation(conv.id);
      expect(detail.id).toBe(conv.id);
    });

    test("正向测试：pinConversation/unpinConversation（pinned=1/0）", async () => {
      const conv = await createTestConversation();
      const pinned = await AiConversationAPI.pinConversation(conv.id);
      expect(pinned.pinned).toBe(1);

      const unpinned = await AiConversationAPI.unpinConversation(conv.id);
      expect(unpinned.pinned).toBe(0);
    });

    test("正向测试：markConversationRead 标记已读，unreadCount 归零 [T-CV-019]", async () => {
      const conv = await createTestConversation();
      const read = await AiConversationAPI.markConversationRead(conv.id);
      // 空会话无消息：lastReadMessageId 为 null，unreadCount 归零
      expect(read.unreadCount).toBe(0);
      expect(read.lastReadMessageId == null).toBe(true);
    });
  });

  // ===== 消息管理（非流式） =====

  describe("GET /api/v1/ai/conversations/{id}/messages - 消息列表", () => {
    let testConvId: number;

    beforeAll(async () => {
      const result = await createTestConversation();
      testConvId = result.id;
    });

    test("正向测试：查询会话消息列表（分页）", async () => {
      const result = await AiConversationAPI.getMessages(testConvId, { pageNum: 1, pageSize: 20 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
    });

    test("验证：空会话消息列表为空", async () => {
      const result = await AiConversationAPI.getMessages(testConvId, { pageNum: 1, pageSize: 20 });
      expect(result.list.length).toBe(0);
      expect(result.total).toBe(0);
    });

    test("边界：查询不存在会话的消息应失败", async () => {
      await expectBizError(AiConversationAPI.getMessages(99999999, { pageNum: 1, pageSize: 20 }), [
        "A0401",
      ]);
    });
  });

  describe("GET /api/v1/ai/messages/{id} - 消息详情", () => {
    test("边界：查询不存在的消息应失败", async () => {
      await expectBizError(AiConversationAPI.getMessageDetail(99999999), ["A0401"]);
    });
  });

  describe("PUT /api/v1/ai/messages/{id} - 编辑消息（流式）[T-CV-061]", () => {
    test("边界：编辑不存在的消息应失败（流式 onNetworkError）", async () => {
      const networkError = await expectStreamNetworkError(
        (handlers) => AiConversationAPI.editMessage(99999999, createEditMessageForm(), handlers),
        10000
      );
      expect(networkError).not.toBeNull();
    }, 15000);
  });

  describe("POST /api/v1/ai/messages/{id}/stop - 停止消息", () => {
    test("边界：停止不存在的消息应失败", async () => {
      await expectBizError(AiConversationAPI.stopMessage(99999999), ["A0401"]);
    });
  });

  describe("POST /api/v1/ai/messages/{id}/resume - 恢复推理", () => {
    test("边界：对无中断点的消息调用 resume 应失败", async () => {
      const conv = await createTestConversation();
      const msgId = await sendMessageCompleted(conv.id, "1+1=?");
      expect(msgId).toBeGreaterThan(0);

      const networkError = await expectStreamNetworkError(
        (handlers) => AiConversationAPI.resumeMessage(msgId, { confirm: true }, handlers),
        15000
      );
      expect(networkError).not.toBeNull();
    }, 30000);
  });

  // ===== 上下文：产物与记忆 =====

  describe("GET /api/v1/ai/conversations/{id}/artifacts - 产物列表（分页）[T-CT-051]", () => {
    let testConvId: number;

    beforeAll(async () => {
      const result = await createTestConversation();
      testConvId = result.id;
    });

    test("正向测试：查询会话中间产物列表（分页结构）", async () => {
      const result = await AiConversationAPI.getArtifacts(testConvId, { pageNum: 1, pageSize: 20 });
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        result.list.forEach((a) => {
          expect(a.id).toBeGreaterThan(0);
          expect(a.conversationId).toBe(testConvId);
        });
      }
    });

    test("边界：查询不存在会话的产物应失败", async () => {
      await expectBizError(AiConversationAPI.getArtifacts(99999999, { pageNum: 1, pageSize: 20 }), [
        "A0401",
      ]);
    });
  });

  describe("GET /api/v1/ai/artifacts/{id}/detail - 产物详情 [T-CT-052]", () => {
    test("边界：查询不存在的产物应失败（路径 /artifacts/{id}/detail）", async () => {
      await expectBizError(AiConversationAPI.getArtifactDetail(99999999), ["A0401"]);
    });
  });

  describe("GET /api/v1/ai/messages/{id}/artifacts - 消息关联产物 [T-CT-050]", () => {
    test("边界：查询不存在消息的关联产物应失败", async () => {
      await expectBizError(AiConversationAPI.getMessageArtifacts(99999999), ["A0401"]);
    });
  });

  describe("GET /api/v1/ai/artifacts/by-ref - 按业务引用反查 [T-CT-050]", () => {
    test("正向测试：按引用反查产物（无匹配返回空数组）", async () => {
      const result = await AiConversationAPI.getArtifactsByRef("sys_pred_log", 99999999);
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(0);
    });
  });

  describe("GET /api/v1/ai/memories - 长期记忆列表（分页）[T-CT-040]", () => {
    test("正向测试：查询当前用户长期记忆列表（分页结构）", async () => {
      const result = await AiConversationAPI.getMemories(createMemoryQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        const m = result.list[0]!;
        expect(m.id).toBeGreaterThan(0);
        expect(m.content).toBeDefined();
        expect(typeof m.importance).toBe("number");
      }
    });

    test("正向测试：按 memoryType 筛选", async () => {
      await createTestMemory({ memoryType: "procedural" });
      const result = await AiConversationAPI.getMemories(
        createMemoryQuery({ memoryType: "procedural" })
      );
      result.list.forEach((m) => expect(m.memoryType).toBe("procedural"));
    });

    test("正向测试：按 source 筛选", async () => {
      await createTestMemory({ source: "manual" });
      const result = await AiConversationAPI.getMemories(createMemoryQuery({ source: "manual" }));
      result.list.forEach((m) => expect(m.source).toBe("manual"));
    });
  });

  describe("记忆 CRUD 正向闭环 [T-CT-037]", () => {
    test("createMemory→updateMemory→deleteMemory", async () => {
      const content = `test_mem_content_${Date.now()}`;
      const created = await createTestMemory({
        content,
        memoryType: "episodic",
        source: "manual",
        importance: 30,
      });
      expect(created.id).toBeGreaterThan(0);
      expect(created.content).toBe(content);
      expect(created.memoryType).toBe("episodic");
      expect(created.status).toBe(1);

      const newContent = `${content}_updated`;
      const updated = await AiConversationAPI.updateMemory(created.id, {
        content: newContent,
        importance: 80,
      });
      expect(updated.content).toBe(newContent);
      expect(updated.importance).toBe(80);

      await AiConversationAPI.deleteMemory(created.id);
      const listAfter = await AiConversationAPI.getMemories(createMemoryQuery());
      expect(listAfter.list.find((m) => m.id === created.id)).toBeUndefined();
    });

    test("关键词搜索记忆", async () => {
      const created = await createTestMemory({ content: "search_target_memory" });
      const results = await AiConversationAPI.searchMemories("search_target_memory");
      expect(results.some((m) => m.id === created.id)).toBe(true);
    });
  });

  describe("记忆归档查看 [T-CT-038]", () => {
    test("正向测试：查询归档记忆列表（分页结构）", async () => {
      const result = await AiConversationAPI.getArchivedMemories(createMemoryQuery());
      expect(Array.isArray(result.list)).toBe(true);
      expect(typeof result.total).toBe("number");
      if (result.list.length > 0) {
        result.list.forEach((m) => expect(m.archived).toBe(1));
      }
    });
  });

  describe("记忆批量清空 [T-CT-047]", () => {
    test("参数校验：未带 confirm 清空应失败", async () => {
      await expectBizError(AiConversationAPI.clearMemories({}, false), ["A0400"]);
    });

    test("正向测试：清空指定记忆（confirm=true）", async () => {
      const created = await createTestMemory({ source: "manual" });
      const createTime = created.createTime;

      // 用时间范围精确圈定该条记忆，避免误清他人数据
      const range = { memoryType: "semantic", start: createTime, end: createTime };
      const cleared = await AiConversationAPI.clearMemories(range, true);
      expect(cleared).toBeGreaterThanOrEqual(1);
      const listAfter = await AiConversationAPI.getMemories(createMemoryQuery());
      expect(listAfter.list.find((m) => m.id === created.id)).toBeUndefined();
    });

    test("正向测试：恢复软删记忆", async () => {
      const created = await createTestMemory({ source: "manual" });
      const createTime = created.createTime;
      const range = { memoryType: "semantic", start: createTime, end: createTime };
      await AiConversationAPI.clearMemories(range, true);

      const restored = await AiConversationAPI.restoreMemories(range, true);
      expect(restored).toBeGreaterThanOrEqual(1);
      // 按同一时间范围圈定查询，避免累积的历史记忆把目标挤出默认分页
      const listRestored = await AiConversationAPI.getMemories(
        createMemoryQuery({
          memoryType: "semantic",
          start: createTime,
          end: createTime,
          pageSize: 100,
        })
      );
      expect(listRestored.list.find((m) => m.id === created.id)).toBeDefined();
    });
  });

  describe("记忆导出 [T-CT-048]", () => {
    test("正向测试：导出 JSON 返回 Blob", async () => {
      await createTestMemory();
      const blob = await AiConversationAPI.exportMemories("json");
      expect(blob.size).toBeGreaterThan(0);
    });

    test("正向测试：导出 Markdown 返回 Blob", async () => {
      const blob = await AiConversationAPI.exportMemories("markdown");
      expect(blob.size).toBeGreaterThan(0);
    });
  });

  describe("DELETE /api/v1/ai/users/me/memories/{id} - 删除记忆", () => {
    test("边界：删除不存在的记忆应失败", async () => {
      await expectBizError(AiConversationAPI.deleteMemory(99999999), ["A0401"]);
    });
  });

  // ===== 消息反馈 =====

  describe("消息反馈正向闭环 [T-MF-050~056]", () => {
    test("点赞(含标签)→查询→改点踩→删除→查询为 undefined", async () => {
      const conv = await createTestConversation();
      const assistantMsgId = await sendMessageCompleted(conv.id, "请简单介绍一下 RIDCP 算法");
      expect(assistantMsgId).toBeGreaterThan(0);

      const like = await AiConversationAPI.submitFeedback(
        assistantMsgId,
        createFeedbackForm({ rating: 1, tags: ["accurate", "concise"] })
      );
      expect(like.messageId).toBe(assistantMsgId);
      expect(like.rating).toBe(1);
      expect(like.tags).toEqual(expect.arrayContaining(["accurate", "concise"]));
      expect(like.id).toBeGreaterThan(0);
      expect(typeof like.userId).toBe("number");
      expect(like.createTime).toBeTruthy();

      const fetched = await AiConversationAPI.getFeedback(assistantMsgId);
      expect(fetched!.messageId).toBe(assistantMsgId);
      expect(fetched!.rating).toBe(1);
      expect(fetched!.tags).toEqual(expect.arrayContaining(["accurate", "concise"]));
      expect(fetched!.id).toBeGreaterThan(0);
      expect(typeof fetched!.userId).toBe("number");
      expect(fetched!.createTime).toBeTruthy();

      // 改点踩（rating:-1）为 upsert，复用同一条记录
      const dislike = await AiConversationAPI.submitFeedback(assistantMsgId, {
        rating: -1,
        tags: ["too_long"],
        comment: "回复太冗长",
      });
      expect(dislike.rating).toBe(-1);
      expect(dislike.tags).toEqual(expect.arrayContaining(["too_long"]));
      expect(dislike.comment).toBe("回复太冗长");
      expect(dislike.id).toBe(like.id);

      await AiConversationAPI.deleteFeedback(assistantMsgId);
      const afterDelete = await AiConversationAPI.getFeedback(assistantMsgId);
      expect(afterDelete).toBeUndefined();
    }, 60000);

    // 需真实 assistant 消息才能走到标签必选校验（无消息时返回"消息不存在"）
    test("参数校验：点踩不带标签应失败 [T-MF-068]", async () => {
      const conv = await createTestConversation();
      const assistantMsgId = await sendMessageCompleted(conv.id, "1+1=?");
      await expectBizError(
        AiConversationAPI.submitFeedback(assistantMsgId, { rating: -1 }),
        ["A0400"],
        ["点踩必须选择问题标签"]
      );
    }, 60000);

    test("边界：对不存在的消息提交反馈应失败", async () => {
      await expectBizError(AiConversationAPI.submitFeedback(99999999, createFeedbackForm()), [
        "A0401",
      ]);
    });

    test("边界：查询不存在消息的反馈应返回 undefined 或空", async () => {
      const result = await AiConversationAPI.getFeedback(99999999);
      expect(result === undefined || result === null).toBe(true);
    });
  });

  // ===== 权限与数据隔离 =====

  describe("权限与数据隔离", () => {
    test("安全：用户 A 无法访问用户 B 的会话 [T-CV-009]", async () => {
      const created = await createTestConversation();
      await login(USERS.USER.username);
      try {
        await expectBizError(AiConversationAPI.getConversation(created.id), ["A0401"]);
      } finally {
        await login(USERS.ADMIN.username);
      }
    });

    test("安全：用户 A 无法访问用户 B 的长期记忆 [T-CT-037]", async () => {
      await login(USERS.ADMIN.username);
      const adminMemories = await AiConversationAPI.getMemories(createMemoryQuery());

      await login(USERS.USER.username);
      const userMemories = await AiConversationAPI.getMemories(createMemoryQuery());

      const adminIds = new Set(adminMemories.list.map((m) => m.id));
      const overlap = userMemories.list.some((m) => adminIds.has(m.id));
      if (adminMemories.list.length > 0 && userMemories.list.length > 0) {
        expect(overlap).toBe(false);
      }
      await login(USERS.ADMIN.username);
    });
  });

  // ===== SSE 流式消息测试 =====

  describe("SSE 流式消息 - sendMessage [T-CV-020~026]", () => {
    test("正向测试：流式消息发送并接收完整事件流", async () => {
      const conv = await createTestConversation();

      let receivedMessageId = 0;
      const { events } = await collectStream(
        (handlers) =>
          AiConversationAPI.sendMessage(
            conv.id,
            { content: "你好，请简单介绍一下 RIDCP 去雾算法" },
            handlers
          ),
        {
          onStart: (data) => {
            receivedMessageId = data.messageId;
          },
        },
        65000
      );

      expect(events[0]).toBe("start");
      expect(events).toContain("delta");
      expect(events).toContain("end");
      expect(receivedMessageId).toBeGreaterThan(0);
    }, 65000);

    test("正向测试：onStart 返回 messageId/conversationId/model（无 streamSessionId）[T-CV-022]", async () => {
      const conv = await createTestConversation();

      let startData: MessageStartEvent | null = null;
      await collectStream(
        (handlers) => AiConversationAPI.sendMessage(conv.id, { content: "1+1=?" }, handlers),
        {
          onStart: (data) => {
            startData = data;
          },
        },
        65000
      );

      expect(startData).not.toBeNull();
      expect(startData!.messageId).toBeGreaterThan(0);
      expect(startData!.conversationId).toBe(conv.id);
      expect(startData!.model).toBeTruthy();
      // streamSessionId 不再由后端返回，为可选字段
      expect(startData!.streamSessionId).toBeUndefined();
    }, 65000);

    test("正向测试：onEnd 返回 stopReason 和 token 用量 [T-CV-026]", async () => {
      const conv = await createTestConversation();

      let endData: MessageEndEvent | null = null;
      await collectStream(
        (handlers) => AiConversationAPI.sendMessage(conv.id, { content: "1+1=?" }, handlers),
        {
          onEnd: (data) => {
            endData = data;
          },
        },
        65000
      );

      expect(endData).not.toBeNull();
      expect(endData!.stopReason).toBeTruthy();
      if (endData!.usage) {
        expect(typeof endData!.usage.inputTokens).toBe("number");
        expect(typeof endData!.usage.outputTokens).toBe("number");
      }
    }, 65000);

    test("边界：向不存在的会话发送消息应触发 onNetworkError [T-CV-071]", async () => {
      const networkError = await expectStreamNetworkError(
        (handlers) => AiConversationAPI.sendMessage(99999999, { content: "测试" }, handlers),
        15000
      );
      expect(networkError).not.toBeNull();
    }, 20000);
  });

  describe("SSE 流式消息 - regenerate [T-CV-060]", () => {
    test("正向测试：重新生成消息并接收事件流", async () => {
      const conv = await createTestConversation();

      let firstMessageId = 0;
      await collectStream(
        (handlers) => AiConversationAPI.sendMessage(conv.id, { content: "1+1=?" }, handlers),
        {
          onStart: (data) => {
            firstMessageId = data.messageId;
          },
        }
      );

      const { events } = await collectStream((handlers) =>
        AiConversationAPI.regenerate(firstMessageId, handlers)
      );
      expect(events).toContain("start");
      expect(events).toContain("end");
    }, 90000);
  });

  // ===== 配额联动（改用 AiBillingAPI）=====

  describe("配额联动 - AiBillingAPI 余额验证", () => {
    // SSE 连续请求后 Node fetch 偶发 socket hang up，余额查询做有限重试，不掩盖业务断言
    async function fetchBalanceWithRetry(): Promise<
      Awaited<ReturnType<typeof AiBillingAPI.getBalance>>
    > {
      let lastErr: unknown;
      for (let i = 0; i < 3; i++) {
        try {
          return await AiBillingAPI.getBalance();
        } catch (e) {
          lastErr = e;
          await new Promise((r) => setTimeout(r, 500));
        }
      }
      throw lastErr;
    }

    test("正向测试：对话后产生计费记录（balance 字段合理）", async () => {
      const balanceBefore = await fetchBalanceWithRetry();
      expect(typeof balanceBefore.dailyUsed).toBe("number");
      expect(typeof balanceBefore.dailyLimit).toBe("number");
      expect(typeof balanceBefore.monthlyUsed).toBe("number");
      expect(typeof balanceBefore.monthlyLimit).toBe("number");

      const conv = await createTestConversation();
      await sendMessageCompleted(conv.id, "1+1=?");

      // 等待计费结算
      await new Promise((r) => setTimeout(r, 3000));

      const balanceAfter = await fetchBalanceWithRetry();
      // 对话后已用积分应不小于对话前（配额消耗非负）
      expect(balanceAfter.dailyUsed).toBeGreaterThanOrEqual(balanceBefore.dailyUsed);
      expect(balanceAfter.monthlyUsed).toBeGreaterThanOrEqual(balanceBefore.monthlyUsed);
    }, 60000);
  });

  // ===== LLM 标题自动生成 =====

  describe("LLM 标题自动生成 [T-CV-010]", () => {
    test("正向测试：对话后标题 titleSource=auto 且非'新对话'", async () => {
      const conv = await createTestConversation();
      expect(conv.title).toBe("新对话");

      await sendMessageCompleted(conv.id, "请介绍一下 RIDCP 图像去雾算法的原理和适用场景");

      // 等待后端异步生成标题
      await new Promise((r) => setTimeout(r, 8000));

      const updated = await AiConversationAPI.getConversation(conv.id);
      expect(updated.title).not.toBe("新对话");
      expect(updated.titleSource).toBe("auto");
    }, 90000);
  });

  // ===== 流式断线重连 =====

  describe("SSE 断线重连 - reconnectStream", () => {
    // 后端 message.start 不再返回 streamSessionId，只能用过期会话验证：无缓存事件时流直接结束
    test("边界：用过期 streamSessionId 重连，流应结束（onClose 或 onNetworkError）", async () => {
      const conv = await createTestConversation();
      const { closed, networkError } = await collectStream(
        (handlers) =>
          AiConversationAPI.reconnectStream(conv.id, "expired_session_id", "0", handlers),
        {},
        10000
      );
      expect(closed || networkError !== null).toBe(true);
    }, 15000);
  });

  // ===== 多步推理工具调用事件 =====

  // 依赖真实多步推理 LLM 才能触发工具调用事件，当前模型无此能力，保持 skip
  describe.skip("多步推理 - 工具调用事件 [T-RS-110]（依赖真实多步推理 LLM）", () => {
    test("正向测试：发送消息可收到 thought/plan/suggestions 事件（视场景而定）", async () => {
      const conv = await createTestConversation();

      let startSeen = false;
      let endSeen = false;
      const thoughtStatuses: number[] = [];

      await new Promise<void>((resolve, reject) => {
        let controller: AbortController | null = null;
        const timeout = setTimeout(() => {
          controller?.abort();
          reject(new Error("SSE 超时：未收到 onEnd"));
        }, 60000);

        controller = AiConversationAPI.sendMessage(
          conv.id,
          { content: "1+1=?" },
          {
            onStart: () => {
              startSeen = true;
            },
            onThought: (data) => {
              // ThoughtEvent.status 为 number
              expect(typeof data.status).toBe("number");
              thoughtStatuses.push(data.status);
            },
            onEnd: () => {
              endSeen = true;
              clearTimeout(timeout);
              resolve();
            },
            onNetworkError: (error) => {
              clearTimeout(timeout);
              reject(error);
            },
            onClose: () => {
              clearTimeout(timeout);
              resolve();
            },
          }
        );
      });

      expect(startSeen).toBe(true);
      expect(endSeen).toBe(true);
      // 简单问答不保证触发工具调用，因此不强制断言 thought 数量
      expect(Array.isArray(thoughtStatuses)).toBe(true);
    }, 65000);
  });
});
