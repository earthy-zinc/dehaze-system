// wire → VM 适配单测：status 映射、role 分流、字段缺失容错 + 对抗性边界（空字符串/缺字段/非数组）。
import { describe, expect, it } from "vitest";
import type {
  ChatAssistantMessageVM,
  ChatToolMessageVM,
  ChatUserMessageVM,
} from "../types";
import {
  readConfirmKind,
  toArtifactVM,
  toFeedbackVM,
  toInterruptVM,
  toMemoryVM,
  toMessageVM,
  toPlanVM,
  toThinkingVM,
  toUsageVM,
} from "../adapters/fromSdk";

// ===== 本地 wire 字面量夹具（红线：仅 adapters/fromSdk.ts 可 import dehaze-sdk-js，spec 不得引入）=====
type MessageStatus = 1 | 2 | 3 | 4;

/** 仅声明适配层实际读取的字段，与 SDK AiMessageVO 结构兼容（可直接传入 toMessageVM） */
interface WireMessage {
  id: number;
  conversationId: number;
  role: "user" | "assistant" | "system";
  status: MessageStatus;
  content?: string;
  toolCalls?: unknown;
  error?: string;
  inputTokens?: number;
  edited?: number;
  originalContent?: string;
}

type InterruptType = "confirm" | "quota" | "async_wait" | "plan_approve";

interface WireInterruptData {
  confirmKind?: "algorithm_recommend" | "tool_permission" | "dangerous_op";
  recommendation?: {
    recommendationId: number;
    algorithmId: number;
    algorithmName: string;
    reason: string;
  };
  alternatives?: Array<{
    algorithmId: number;
    algorithmName: string;
    matchScore: number;
    reason: string;
  }>;
  plan?: {
    tasks: Array<{
      id?: string;
      description?: string;
      dependsOn?: string[];
      status?: string;
      paradigm?: string;
    }>;
    status?: string;
    revisions: Array<{
      revisionNo: number;
      reason: string;
      changedTaskIds: string[];
    }>;
    phase?: string;
  };
  action?: string;
  upgradeTip?: string;
  usedDaily?: number;
  dailyLimit?: number;
  usedMonthly?: number;
  monthlyLimit?: number;
  reason?: string;
  detail?: string;
  taskId?: string;
  estDuration?: string;
  imageCount?: number;
}

interface WireInterrupt {
  type: InterruptType;
  data: WireInterruptData;
}

/** toMessageVM 的 message 入参形态（适配层不读取其它 wire 键，此处仅声明所需字段） */
type MessageArg = Parameters<typeof toMessageVM>[0]["message"];

const msg = (over: Partial<WireMessage>): MessageArg =>
  ({
    id: 1,
    conversationId: 1,
    role: "assistant",
    status: 2,
    ...over,
  }) as unknown as MessageArg;

const asAssistant = (vm: ReturnType<typeof toMessageVM>) =>
  vm as ChatAssistantMessageVM;

describe("fromSdk 适配", () => {
  describe("status 映射", () => {
    const cases: Array<[MessageStatus, string]> = [
      [1, "streaming"],
      [2, "completed"],
      [3, "failed"],
      [4, "canceled"],
    ];

    it.each(cases)("wire status %i → %s", (status, expected) => {
      const vm = toMessageVM({ message: msg({ status }) });
      expect(asAssistant(vm).status).toBe(expected);
    });

    it("status 缺失 / 未知回退 completed", () => {
      const missing = toMessageVM({
        message: { ...msg({}), status: undefined } as unknown as MessageArg,
      });
      expect(asAssistant(missing).status).toBe("completed");
      const unknown = toMessageVM({
        message: { ...msg({}), status: 9 } as unknown as MessageArg,
      });
      expect(asAssistant(unknown).status).toBe("completed");
    });
  });

  describe("role 分流", () => {
    it("user 消息取 content", () => {
      const vm = toMessageVM({
        message: msg({ role: "user", content: "你好" }),
      }) as ChatUserMessageVM;
      expect(vm).toEqual({
        role: "user",
        id: 1,
        content: "你好",
      });
    });

    it("system 消息归入 tool 分支并归一工具名", () => {
      const vm = toMessageVM({
        message: msg({
          role: "system",
          content: "raw",
          toolCalls: [{ name: "search", arguments: { q: 1 } }],
        }),
      }) as ChatToolMessageVM;
      expect(vm).toMatchObject({
        role: "tool",
        id: 1,
        content: "raw",
        name: "search",
        toolCalls: [{ name: "search", arguments: { q: 1 } }],
      });
    });

    it("content 缺失回退空字符串（含空字符串保持）", () => {
      const missing = toMessageVM({
        message: msg({ role: "user", content: undefined }),
      }) as ChatUserMessageVM;
      expect(missing.content).toBe("");
      const blank = toMessageVM({
        message: msg({ role: "user", content: "" }),
      }) as ChatUserMessageVM;
      expect(blank.content).toBe("");
    });
  });

  describe("字段缺失容错", () => {
    it("无 token 字段则不落 usage（键不存在）", () => {
      const vm = asAssistant(toMessageVM({ message: msg({}) }));
      expect("usage" in vm).toBe(false);
      expect(vm.usage).toBeUndefined();
    });

    it("部分 token 字段则其余补 0", () => {
      const vm = asAssistant(
        toMessageVM({ message: msg({ inputTokens: 10 }) })
      );
      expect(vm.usage).toEqual({
        inputTokens: 10,
        outputTokens: 0,
        cachedInputTokens: 0,
        credits: 0,
      });
    });

    it("subAgents：有数据则映射进 usage.subAgents，无数据不落键", () => {
      const withSub = asAssistant(
        toMessageVM({
          message: msg({ inputTokens: 10 }),
          subAgents: [
            {
              agentCode: "researcher",
              inputTokens: 3,
              outputTokens: 4,
              cachedInputTokens: 1,
              credits: 2,
            },
          ],
        })
      );
      expect(withSub.usage?.subAgents).toEqual([
        {
          agentCode: "researcher",
          inputTokens: 3,
          outputTokens: 4,
          cachedInputTokens: 1,
          credits: 2,
        },
      ]);

      const withoutSub = asAssistant(
        toMessageVM({ message: msg({ inputTokens: 10 }) })
      );
      expect(withoutSub.usage).not.toHaveProperty("subAgents");
    });

    it("无 error 字段则不落 error（键不存在）", () => {
      const vm = asAssistant(toMessageVM({ message: msg({}) }));
      expect("error" in vm).toBe(false);
    });

    it("有 error 字段则透传", () => {
      const vm = asAssistant(toMessageVM({ message: msg({ error: "boom" }) }));
      expect(vm.error).toBe("boom");
    });

    it("缺省集合字段回退空数组 / null", () => {
      const vm = asAssistant(toMessageVM({ message: msg({}) }));
      expect(vm.steps).toEqual([]);
      expect(vm.toolCalls).toEqual([]);
      expect(vm.artifacts).toEqual([]);
      expect(vm.memories).toEqual([]);
      expect(vm.suggestions).toEqual([]);
      expect(vm.feedback).toBeNull();
      expect(vm.thinking).toBeNull();
    });

    it("toolCalls 非数组按空列表处理", () => {
      const vm = toMessageVM({
        message: msg({ role: "system", content: "", toolCalls: "bad" }),
      }) as ChatToolMessageVM;
      expect(vm.toolCalls).toEqual([]);
      expect("name" in vm).toBe(false);
    });

    it("toolCalls 逐项容错取 name / arguments", () => {
      const vm = toMessageVM({
        message: msg({
          role: "system",
          content: "",
          toolCalls: [{ name: "search" }, "junk", { arguments: { a: 1 } }],
        }),
      }) as ChatToolMessageVM;
      expect(vm.toolCalls).toEqual([
        { name: "search" },
        {},
        { arguments: { a: 1 } },
      ]);
    });
  });

  describe("toThinkingVM", () => {
    it("全空返回 null", () => {
      expect(toThinkingVM({})).toBeNull();
      expect(toThinkingVM({ state: null, thoughts: [] })).toBeNull();
    });

    it("流式态优先于历史合成", () => {
      const thinking = toThinkingVM({
        state: {
          segments: [{ text: "流式", closed: false }],
          startAt: 5,
          endAt: null,
        },
        thoughts: [{ position: 1, status: 1, thought: "历史" }],
      });
      expect(thinking).toEqual({
        segments: [{ text: "流式", closed: false }],
        startAt: 5,
        endAt: null,
        streaming: true,
      });
    });

    it("无流式态时由推理步骤回退合成", () => {
      const thinking = toThinkingVM({
        thoughts: [
          { position: 1, status: 1, thought: "想" },
          { position: 2, status: 1, tool: "search", observation: "ok" },
        ],
      });
      expect(thinking).toEqual({
        segments: [{ text: "想", closed: true }],
        startAt: 0,
        endAt: null,
        streaming: false,
      });
    });

    it("空 segments 的流式态不遮挡历史合成", () => {
      const thinking = toThinkingVM({
        state: { segments: [], startAt: 5, endAt: null },
        thoughts: [{ position: 1, status: 1, thought: "历史" }],
      });
      expect(thinking).toEqual({
        segments: [{ text: "历史", closed: true }],
        startAt: 0,
        endAt: null,
        streaming: false,
      });
    });
  });

  describe("toMessageVM：steps 与思考合成", () => {
    it("steps 由 thoughts 逐条映射，思考回退合成", () => {
      const vm = asAssistant(
        toMessageVM({
          message: msg({}),
          thoughts: [
            { position: 1, status: 1, thought: "想" },
            { position: 2, status: 1, tool: "search", observation: "ok" },
          ],
        })
      );
      expect(vm.steps).toEqual([
        { position: 1, status: 1, thought: "想" },
        { position: 2, status: 1, tool: "search", observation: "ok" },
      ]);
      expect(vm.thinking).toEqual({
        segments: [{ text: "想", closed: true }],
        startAt: 0,
        endAt: null,
        streaming: false,
      });
    });
  });

  describe("toInterruptVM", () => {
    it("async_wait：同名字段直接透传", () => {
      const vm = toInterruptVM({
        type: "async_wait",
        data: { taskId: "t1", estDuration: "30s", imageCount: 3 },
      });
      expect(vm).toEqual({
        type: "async_wait",
        estDuration: "30s",
        imageCount: 3,
      });
    });

    it("confirm：推荐与备选归一", () => {
      const vm = toInterruptVM({
        type: "confirm",
        data: {
          recommendation: {
            recommendationId: 1,
            algorithmId: 7,
            algorithmName: "去雾",
            reason: "最佳",
          },
          alternatives: [
            {
              algorithmId: 8,
              algorithmName: "备选",
              matchScore: 0.5,
              reason: "次优",
            },
          ],
        },
      });
      expect(vm).toEqual({
        type: "confirm",
        recommendation: {
          algorithmId: 7,
          algorithmName: "去雾",
          reason: "最佳",
        },
        alternatives: [
          {
            algorithmId: 8,
            algorithmName: "备选",
            matchScore: 0.5,
            reason: "次优",
          },
        ],
      });
    });

    it("plan_approve：计划任务按同一 camelCase 形状透传（含依赖与状态归一）", () => {
      const vm = toInterruptVM({
        type: "plan_approve",
        data: {
          plan: {
            tasks: [
              {
                id: "t1",
                description: "步骤",
                dependsOn: ["t0"],
                status: "pending",
              },
            ],
            status: "pending",
            revisions: [],
          },
        },
      });
      expect(vm).toEqual({
        type: "plan_approve",
        plan: [
          {
            id: "t1",
            description: "步骤",
            dependsOn: ["t0"],
            status: "pending",
          },
        ],
      });
    });

    it("空 data 仅保留 type", () => {
      expect(toInterruptVM({ type: "quota", data: {} })).toEqual({
        type: "quota",
      });
    });

    it("data 缺失按空对象容错", () => {
      const interrupt = { type: "confirm" } as unknown as WireInterrupt;
      expect(toInterruptVM(interrupt)).toEqual({ type: "confirm" });
    });
  });

  describe("toArtifactVM / toMemoryVM / toFeedbackVM / toUsageVM", () => {
    it("artifact：wire isInvalid 数值归一为 invalid 布尔，缺省 summary 不落键", () => {
      expect(
        toArtifactVM({
          id: 1,
          conversationId: 2,
          type: "metric_report",
          isInvalid: 0,
        })
      ).toEqual({ id: 1, type: "metric_report", invalid: false });
      expect(
        toArtifactVM({
          id: 2,
          conversationId: 2,
          type: "file_ref",
          isInvalid: 1,
        })
      ).toEqual({ id: 2, type: "file_ref", invalid: true });
      // 对抗：缺字段容错（undefined 视为有效）
      expect(
        toArtifactVM({
          id: 3,
          conversationId: 2,
          type: "image_result",
        } as unknown as Parameters<typeof toArtifactVM>[0])
      ).toEqual({ id: 3, type: "image_result", invalid: false });
    });

    it("memory 取展示字段", () => {
      expect(
        toMemoryVM({
          id: 1,
          userId: 1,
          memoryType: "semantic",
          content: "记忆内容",
          accessCount: 0,
          source: "manual",
          status: 1,
          archived: 0,
        } as unknown as Parameters<typeof toMemoryVM>[0])
      ).toEqual({
        id: 1,
        memoryType: "semantic",
        content: "记忆内容",
        source: "manual",
      });
    });

    it("feedback：null/undefined → null，rating 语义保留", () => {
      expect(toFeedbackVM(null)).toBeNull();
      expect(toFeedbackVM(undefined)).toBeNull();
      expect(
        toFeedbackVM({
          id: 1,
          messageId: 1,
          userId: 1,
          rating: -1,
        } as unknown as Parameters<typeof toFeedbackVM>[0])
      ).toEqual({ rating: -1 });
      expect(
        toFeedbackVM({
          id: 1,
          messageId: 1,
          userId: 1,
          rating: 1,
          tags: ["accurate"],
          comment: "赞",
        } as unknown as Parameters<typeof toFeedbackVM>[0])
      ).toEqual({ rating: 1, tags: ["accurate"], comment: "赞" });
    });

    it("usage：全缺 → null，0 值保留", () => {
      expect(toUsageVM(null)).toBeNull();
      expect(toUsageVM(undefined)).toBeNull();
      expect(toUsageVM({})).toBeNull();
      expect(
        toUsageVM({
          inputTokens: 0,
          outputTokens: 0,
          cachedInputTokens: 0,
          credits: 0,
        })
      ).toEqual({
        inputTokens: 0,
        outputTokens: 0,
        cachedInputTokens: 0,
        credits: 0,
      });
    });

    it("usage：subAgents 非空落键，空数组不落键", () => {
      expect(
        toUsageVM({
          inputTokens: 1,
          outputTokens: 0,
          cachedInputTokens: 0,
          credits: 0,
          subAgents: [],
        })
      ).toEqual({
        inputTokens: 1,
        outputTokens: 0,
        cachedInputTokens: 0,
        credits: 0,
      });

      const [subAgent] = [
        {
          agentCode: "worker",
          inputTokens: 1,
          outputTokens: 2,
          cachedInputTokens: 3,
          credits: 4,
        },
      ];
      expect(
        toUsageVM({
          inputTokens: 1,
          outputTokens: 0,
          cachedInputTokens: 0,
          credits: 0,
          subAgents: [subAgent],
        })
      ).toEqual({
        inputTokens: 1,
        outputTokens: 0,
        cachedInputTokens: 0,
        credits: 0,
        subAgents: [subAgent],
      });
    });
  });

  describe("toInterruptVM：confirmKind / action 归一", () => {
    // 用例需注入契约外的对抗值（未知 confirmKind），故按 Record 构造后断言
    type InterruptDataArg = NonNullable<Parameters<typeof readConfirmKind>[0]>;
    const dataOf = (data: Record<string, unknown>): InterruptDataArg =>
      data as unknown as InterruptDataArg;

    it.each<[string, Record<string, unknown>]>([
      ["algorithm_recommend", { confirmKind: "algorithm_recommend" }],
      ["tool_permission", { confirmKind: "tool_permission" }],
      ["dangerous_op", { confirmKind: "dangerous_op" }],
      [
        "write_conflict",
        { confirmKind: "dangerous_op", action: "write_conflict" },
      ],
    ])("confirm 子类型 %s 归一为 confirmKind", (expected, raw) => {
      const vm = toInterruptVM({ type: "confirm", data: dataOf(raw) });
      expect(vm.confirmKind).toBe(expected);
    });

    it("write_conflict 同时透出 action", () => {
      const vm = toInterruptVM({
        type: "confirm",
        data: dataOf({
          confirmKind: "dangerous_op",
          action: "write_conflict",
        }),
      });
      expect(vm).toMatchObject({
        confirmKind: "write_conflict",
        action: "write_conflict",
      });
    });

    it("对抗：未知 / 缺失 confirmKind 展示映射不抛错，仅缺省 confirmKind", () => {
      expect(readConfirmKind(dataOf({ confirmKind: "bogus" }))).toBeUndefined();
      expect(readConfirmKind(dataOf({}))).toBeUndefined();
      expect(readConfirmKind(undefined)).toBeUndefined();
      const vm = toInterruptVM({
        type: "confirm",
        data: dataOf({ confirmKind: "bogus" }),
      });
      expect(vm.confirmKind).toBeUndefined();
      expect("confirmKind" in vm).toBe(false);
    });

    it("非 confirm 中断不产生 confirmKind（readConfirmKind 只读载荷，类型由调用方判定）", () => {
      const vm = toInterruptVM({
        type: "quota",
        data: dataOf({
          confirmKind: "algorithm_recommend",
          upgradeTip: "升级",
        }),
      });
      expect(vm.type).toBe("quota");
    });
  });

  describe("toPlanVM：计划归一与累积", () => {
    it("单事件：任务/状态/修订记录归一为展示态", () => {
      const vm = toPlanVM(
        {
          tasks: [
            {
              id: "t1",
              description: "步骤1",
              dependsOn: ["t0"],
              status: "executing",
              paradigm: "react",
            },
          ],
          status: "executing",
          revisions: [{ revisionNo: 1, reason: "B", changedTaskIds: ["t2"] }],
          phase: "planning",
        },
        { messageId: 7 }
      );
      expect(vm).toEqual({
        messageId: 7,
        phase: "planning",
        tasks: [
          {
            id: "t1",
            description: "步骤1",
            dependsOn: ["t0"],
            status: "running",
            paradigm: "react",
          },
        ],
        status: "running",
        revisions: [{ revisionNo: 1, reason: "B" }],
        awaitingApproval: false,
      });
    });

    it("previous 累积：revisions 去重并按 revisionNo 升序（乱序 tolerant）", () => {
      const first = toPlanVM(
        {
          tasks: [],
          status: "pending",
          revisions: [{ revisionNo: 2, reason: "后", changedTaskIds: [] }],
        },
        { messageId: 7 }
      );
      const second = toPlanVM(
        {
          tasks: [],
          status: "revised",
          revisions: [
            { revisionNo: 1, reason: "先", changedTaskIds: [] },
            { revisionNo: 2, reason: "后", changedTaskIds: [] },
          ],
        },
        { messageId: 7, previous: first }
      );
      expect(second.revisions).toEqual([
        { revisionNo: 1, reason: "先" },
        { revisionNo: 2, reason: "后" },
      ]);
      // 全量重发不重复累积
      expect(second.revisions).toHaveLength(2);
    });

    it("对抗：tasks 为空 / dependsOn 缺失均容错", () => {
      const emptyTasks = toPlanVM({ tasks: [], revisions: [] });
      expect(emptyTasks.tasks).toEqual([]);
      expect(emptyTasks.status).toBe("pending");

      const noDep = toPlanVM({
        tasks: [{ id: "t1", description: "无依赖" }],
        revisions: [],
      });
      expect(noDep.tasks).toEqual([{ id: "t1", description: "无依赖" }]);
      expect(noDep.revisions).toEqual([]);
    });

    it("awaitingApproval 可显式给定", () => {
      expect(
        toPlanVM({ tasks: [], revisions: [] }, { awaitingApproval: true })
          .awaitingApproval
      ).toBe(true);
    });
  });

  describe("宿主导出映射下沉：user 编辑标记 / tool 状态", () => {
    it("user：edited=1 与 originalContent 透传；缺失时键不落", () => {
      const edited = toMessageVM({
        message: msg({
          role: "user",
          content: "改后",
          edited: 1,
          originalContent: "改前",
        }),
      }) as ChatUserMessageVM;
      expect(edited).toEqual({
        role: "user",
        id: 1,
        content: "改后",
        edited: true,
        originalContent: "改前",
      });

      const plain = toMessageVM({
        message: msg({ role: "user", content: "原样" }),
      }) as ChatUserMessageVM;
      expect("edited" in plain).toBe(false);
      expect("originalContent" in plain).toBe(false);
    });

    it("tool：wire status 归一为徽标状态（1/2/3/4）", () => {
      const statusOf = (status: MessageStatus) =>
        (
          toMessageVM({
            message: msg({ role: "system", content: "", status }),
          }) as ChatToolMessageVM
        ).status;
      expect(statusOf(1)).toBe("streaming");
      expect(statusOf(2)).toBe("completed");
      expect(statusOf(3)).toBe("failed");
      expect(statusOf(4)).toBe("canceled");
    });
  });
});
