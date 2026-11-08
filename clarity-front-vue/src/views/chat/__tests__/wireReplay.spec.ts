// G10 真实 wire 对拍：脱敏 SSE 帧 → 真实 SDK 解析/分发 → 真实 chat store 归约 → 真实 MessageList 渲染。
//
// 与 chatFlow.spec.ts 的差别：那份用例在 SDK 边界上整体 mock 模块并手动调用 MessageStreamHandlers，
// 只验证了"回调被调用后 store 做什么"，事件名拼写、payload 字段名、null 键、SSE 分帧这类运行时形状
// 差异完全测不到。本文件只在传输层注入 fetchImpl（SSEStreamOptions），SDK 的 fetchSSE 解析器、
// createStreamDispatcher 白名单与终态闸门、store 的 stream-reducer、组件渲染全部是生产实现。
//
// 契约漂移如何被这条用例捕获：
// 1. 事件名改名（例如 block.stop → content_block.stop）：分发器按白名单丢弃该帧，工具结果与
//    thinking.durationMs 永远不落地，"契约漂移哨兵"用例会失败；"事件名逐个可达"用例也会失败。
// 2. 必填字段改名（例如 tool.status → tool.state）：归约器写入的 Block 缺字段，工具步骤回落到
//    running，"契约漂移哨兵"用例会失败。
// 3. 若后端把事件名/字段改成历史别名而 SDK 悄悄兼容，"漂移哨兵"里"改写后结果必须与基线不同"的
//    断言同样会失败，提示契约已发生未登记的变化。
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import { defineComponent, h, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AiConversationAPI, type MessageStreamHandlers } from "clarity-sdk-js";
import {
  stepsOfBlocks,
  textOfBlocks,
  thinkingOfBlocks,
} from "@/components/ai-chat/model/blocks";
import MessageList from "@/components/ai-chat/components/MessageList.vue";
import { elStubs } from "@/components/ai-chat/__tests__/stubs";
import { useChatStore } from "@/store/modules/chat";
import assistantTurnSSE from "./fixtures/sse/assistant-turn.sse?raw";
import interruptTurnSSE from "./fixtures/sse/interrupt-turn.sse?raw";
import legacyEventsSSE from "./fixtures/sse/legacy-events.sse?raw";

// MarkdownRenderer 依赖 katex/mermaid，与其它 chat 组件测试同口径用轻量桩
vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content", "streaming"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

const CONVERSATION_ID = 9001;
const ASSISTANT_ID = 9102;
const USER_ID = 9101;

// ===== 传输层注入：只替换 fetch，SDK 的 SSE 解析与分发保持真实 =====

const fetchCalls: Array<{ url: string; init: RequestInit }> = [];
/** 当前用例要吐出的 SSE 文本 */
let currentSSE = "";

/** 按固定小分片吐字节：故意把中文多字节字符与 JSON 字段切开，验证解析器跨 chunk 缓冲 */
function sseResponse(text: string): Response {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      const end = Math.min(offset + 7, bytes.length);
      controller.enqueue(bytes.slice(offset, end));
      offset = end;
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

const fakeFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  fetchCalls.push({ url: String(input), init: init ?? {} });
  return sseResponse(currentSSE);
}) as unknown as typeof fetch;

/** 捕获 spy 之前的真实实现（store 调用点不传 options，由 spy 补齐 fetchImpl 这一项） */
const realSendMessage = AiConversationAPI.sendMessage;

beforeEach(() => {
  fetchCalls.length = 0;
  currentSSE = "";
  vi.restoreAllMocks();
  vi.spyOn(AiConversationAPI, "sendMessage").mockImplementation(
    (conversationId, data, handlers) =>
      realSendMessage.call(AiConversationAPI, conversationId, data, handlers, {
        fetchImpl: fakeFetch,
      })
  );
  // 流终态会补一次已读，走 axios；此处只屏蔽这条与 wire 无关的副作用
  vi.spyOn(AiConversationAPI, "markConversationRead").mockResolvedValue(
    {} as never
  );
});

/** 走真实链路回放一段 SSE：chat.sendMessage → SDK.sendMessage → fetchSSE → dispatcher → reducer */
async function replay(sse: string) {
  const pinia = createPinia();
  setActivePinia(pinia);
  const chat = useChatStore();
  chat.currentConversationId = CONVERSATION_ID;
  currentSSE = sse;
  chat.sendMessage("帮我看下本月销售额");
  await vi.waitFor(() => expect(chat.streamStatus).not.toBe("running"), {
    timeout: 3000,
  });
  await flushPromises();
  return { chat, pinia };
}

/** 宿主只做"把 store 的实体数组喂给无状态列表"，消息本体仍由真实组件渲染 */
function mountList(chat: ReturnType<typeof useChatStore>, pinia: Pinia) {
  const Host = defineComponent({
    name: "WireReplayHost",
    setup() {
      return () =>
        h(MessageList, {
          messages: chat.messageList,
          artifacts: {},
          feedback: {},
          branches: {},
          scope: "self" as const,
          scrollFollowEnabled: true,
        });
    },
  });
  return mount(Host, { global: { plugins: [pinia], stubs: elStubs } });
}

/** SSE 帧拼装（用于"事件名逐个可达"用例） */
function frame(id: number, name: string, payload: unknown): string {
  return `id: ${id}\nevent: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
}

const EXPECTED_TEXT =
  "根据数据，本月销售额为 **123.4 万元**。\n\n建议关注华东区环比下滑。";

describe("G10 wire 对拍：真实 SSE 帧 → SDK 分发 → store 归约 → 组件渲染", () => {
  it("完整一轮：正文多段拼接、思考分段、工具状态与输出、计划、usage、终态", async () => {
    const { chat, pinia } = await replay(assistantTurnSSE);
    const assistant = chat.messages[ASSISTANT_ID];
    expect(assistant).toBeTruthy();

    // message.start 用服务端实体替换本地负数占位：id 与顺序都与服务端一致
    expect(chat.messageOrder).toEqual([USER_ID, ASSISTANT_ID]);
    expect(chat.messages[USER_ID]?.role).toBe("user");

    // 正文：两个 text 块按 seq 拼接，块间以空行衔接（不是把两段粘成一行）
    expect(textOfBlocks(assistant.blocks)).toBe(EXPECTED_TEXT);

    // 思考：两段独立 thinking 块，均已在 block.stop 收尾（durationMs 有值）
    const thinking = thinkingOfBlocks(
      assistant.blocks,
      Date.parse(assistant.createTime)
    );
    expect(thinking?.segments.map((segment) => segment.text)).toEqual([
      "先确认统计口径：销售额取已支付订单。",
      "再核对一下区域环比。",
    ]);
    expect(thinking?.segments.every((segment) => segment.closed)).toBe(true);

    // 工具步骤：block.stop 的完整 tool 覆盖 block.start 的 running 壳
    const steps = stepsOfBlocks(assistant.blocks);
    expect(steps).toHaveLength(1);
    expect(steps[0]).toMatchObject({
      position: 2,
      tool: "query_sales",
      status: "success",
      observation: "本月销售额 123.4 万元",
      latencyMs: 860,
    });

    // 计划：全量推送替换
    expect(assistant.plan?.phase).toBe("分析");
    expect(assistant.plan?.tasks).toHaveLength(2);
    expect(assistant.plan?.revisions[0]?.reason).toBe("按区域拆分查询");

    // usage：message.end 的终值覆盖此前 usage 事件（output 300 → 340）
    expect(assistant.usage).toMatchObject({
      inputTokens: 1200,
      outputTokens: 340,
      cachedInputTokens: 512,
      credits: 4,
    });
    expect(assistant.usage.subAgents?.[0]?.agentCode).toBe("sales-agent");

    // 终态：只有 message.end 能收尾
    expect(assistant.status).toBe("completed");
    expect(chat.streamStatus).toBe("idle");
    expect(chat.streamMessageId).toBeNull();
    expect(chat.streamSessionId).toBeNull();

    // 传输层：真实 SDK 发出的请求形状（URL / SSE 头 / 幂等键 / body）
    expect(fetchCalls).toHaveLength(1);
    expect(fetchCalls[0]?.url).toBe(
      `/api/v1/ai/conversations/${CONVERSATION_ID}/messages`
    );
    expect(fetchCalls[0]?.init.method).toBe("POST");
    const headers = fetchCalls[0]?.init.headers as Record<string, string>;
    expect(headers.Accept).toBe("text/event-stream");
    expect(headers["Idempotency-Key"]).toBeTruthy();
    expect(JSON.parse(String(fetchCalls[0]?.init.body))).toEqual({
      content: "帮我看下本月销售额",
    });

    // ===== 真实组件渲染 =====
    const wrapper = mountList(chat, pinia);
    await nextTick();

    expect(wrapper.find(".user-message__content").text()).toBe(
      "帮我看下本月销售额"
    );

    const answer = wrapper.find(".assistant-message__bubble .md-stub");
    expect(answer.text()).toContain("根据数据，本月销售额为 **123.4 万元**。");
    expect(answer.text()).toContain("建议关注华东区环比下滑。");

    // 思考卡：正文出现后默认收起，展开可看到两段思考
    expect(wrapper.find(".thinking-panel").exists()).toBe(true);
    expect(wrapper.find(".thinking-panel__title").text()).toContain(
      "已深度思考"
    );
    await wrapper.find(".thinking-panel__header").trigger("click");
    const thinkingBodies = wrapper.findAll(".thinking-panel__body");
    expect(thinkingBodies).toHaveLength(2);
    expect(thinkingBodies[0]?.text()).toContain("先确认统计口径");
    expect(thinkingBodies[1]?.text()).toContain("再核对一下区域环比");

    // 工具步骤：状态徽标与 observation 直接来自契约块
    expect(wrapper.findAll(".thought-step")).toHaveLength(1);
    const step = wrapper.find(".thought-step");
    expect(step.text()).toContain("query_sales");
    expect(step.text()).toContain("成功");
    expect(step.text()).toContain("本月销售额 123.4 万元");
    expect(step.text()).toContain("860ms");

    // 计划面板
    expect(wrapper.find(".plan-panel").text()).toContain("查询本月销售额");
    expect(wrapper.find(".plan-panel").text()).toContain("修订说明");

    // 用量（终态且非流式时展示）
    const meta = wrapper.find(".assistant-message__meta").text();
    expect(meta).toContain("输入 1200 / 输出 340 tokens");
    expect(meta).toContain("积分 4");

    // 推荐追问：终态后到达，正常渲染
    const suggestions = wrapper.find(".suggestion-list");
    expect(suggestions.exists()).toBe(true);
    expect(suggestions.text()).toContain("华东区下滑的主要原因是什么？");
    expect(suggestions.text()).toContain("下月销售额预测是多少？");

    wrapper.unmount();
  });

  it("message.end 之后只放行 suggestions：迟到增量/心跳/用量/错误帧全部丢弃", async () => {
    const { chat } = await replay(assistantTurnSSE);
    const assistant = chat.messages[ASSISTANT_ID];

    // fixture 在 message.end(id 21) 之后依次塞了：block.delta / ping / usage / error / suggestions
    // 前四类必须被分发器终态闸门丢弃
    const lateText = assistant.blocks.find(
      (block) => block.id === "9102-text-2"
    );
    expect(lateText?.text).toBe("建议关注华东区环比下滑。");

    expect(assistant.usage).toMatchObject({
      inputTokens: 1200,
      outputTokens: 340,
    });
    expect(assistant.error).toBeNull();

    // 唯一例外：suggestions 是终态后的内容补全，必须写入
    expect(assistant.suggestions).toEqual([
      "华东区下滑的主要原因是什么？",
      "下月销售额预测是多少？",
    ]);
    expect(assistant.status).toBe("completed");
    expect(chat.streamStatus).toBe("idle");
  });

  it("中断帧 → pendingInterrupt + awaiting_input，中断卡按契约字段渲染", async () => {
    const { chat, pinia } = await replay(interruptTurnSSE);
    const assistant = chat.messages[9202];
    expect(assistant).toBeTruthy();

    expect(assistant.status).toBe("awaiting_input");
    expect(assistant.pendingInterrupt).toMatchObject({
      id: "int-7",
      type: "confirm",
      confirmKind: "tool_permission",
      actionable: true,
    });
    expect(chat.streamStatus).toBe("interrupted");
    expect(chat.streamMessageId).toBe(9202);
    expect(chat.streamSessionId).toBeNull();

    const wrapper = mountList(chat, pinia);
    await nextTick();
    expect(wrapper.find(".interrupt-card").exists()).toBe(true);
    expect(wrapper.find(".interrupt-card").text()).toContain(
      "需要授权调用 write_file 写入报表文件"
    );
    expect(wrapper.find(".interrupt-card").text()).toContain("允许");
    wrapper.unmount();
  });

  it("事件名逐个可达：11 个契约事件名各自命中处理器，终态后仅 suggestions 例外", async () => {
    const seen: string[] = [];
    const ids: string[] = [];
    const handlers: MessageStreamHandlers = {
      onStart: () => seen.push("message.start"),
      onBlockStart: () => seen.push("block.start"),
      onBlockDelta: () => seen.push("block.delta"),
      onBlockStop: () => seen.push("block.stop"),
      onPlanUpdate: () => seen.push("plan.update"),
      onInterrupt: () => seen.push("interrupt"),
      onUsage: () => seen.push("usage"),
      onSuggestions: () => seen.push("suggestions"),
      onPing: () => seen.push("ping"),
      onError: () => seen.push("error"),
      onEnd: () => seen.push("message.end"),
      onEventId: (id) => ids.push(id),
      onNetworkError: (error) => seen.push(`network-error:${error.message}`),
      onClose: () => seen.push("close"),
    };

    currentSSE = [
      frame(1, "message.start", { streamSessionId: "s-1" }),
      frame(2, "block.start", { messageId: 1, block: {} }),
      frame(3, "block.delta", { messageId: 1, blockId: "b1", delta: "x" }),
      frame(4, "block.stop", { messageId: 1, block: {} }),
      frame(5, "plan.update", { messageId: 1, plan: {} }),
      frame(6, "interrupt", { messageId: 1, interrupt: {} }),
      frame(7, "usage", { messageId: 1, usage: {} }),
      frame(8, "suggestions", { messageId: 1, suggestions: ["a"] }),
      frame(9, "ping", {}),
      frame(10, "error", { code: "A0500", message: "x" }),
      frame(11, "message.end", { messageId: 1, status: "completed" }),
      // 终态后：非 suggestions 一律丢弃
      frame(12, "ping", {}),
      frame(13, "usage", { messageId: 1, usage: { inputTokens: 1 } }),
      // 唯一例外
      frame(14, "suggestions", { messageId: 1, suggestions: ["b"] }),
    ].join("");

    AiConversationAPI.sendMessage(CONVERSATION_ID, { content: "x" }, handlers);
    await vi.waitFor(() => expect(seen).toContain("close"), { timeout: 3000 });

    expect(seen).toEqual([
      "message.start",
      "block.start",
      "block.delta",
      "block.stop",
      "plan.update",
      "interrupt",
      "usage",
      "suggestions",
      "ping",
      "error",
      "message.end",
      "suggestions",
      "close",
    ]);
    // 事件 ID 在所有帧上都回传（含终态后被丢弃的帧），保证 Last-Event-ID 不倒退
    expect(ids).toEqual(
      Array.from({ length: 14 }, (_, index) => String(index + 1))
    );
  });

  it("未知/历史事件名与坏 JSON 帧被丢弃，不影响其后合法帧", async () => {
    const seen: string[] = [];
    const handlers: MessageStreamHandlers = {
      onStart: () => seen.push("message.start"),
      onBlockStart: () => seen.push("block.start"),
      onBlockDelta: () => seen.push("block.delta"),
      onBlockStop: () => seen.push("block.stop"),
      onPlanUpdate: () => seen.push("plan.update"),
      onInterrupt: () => seen.push("interrupt"),
      onUsage: () => seen.push("usage"),
      onSuggestions: () => seen.push("suggestions"),
      onPing: () => seen.push("ping"),
      onError: () => seen.push("error"),
      onEnd: () => seen.push("message.end"),
      onNetworkError: (error) => seen.push(`network-error:${error.message}`),
      onClose: () => seen.push("close"),
    };

    currentSSE = legacyEventsSSE;
    AiConversationAPI.sendMessage(CONVERSATION_ID, { content: "x" }, handlers);
    await vi.waitFor(() => expect(seen).toContain("close"), { timeout: 3000 });

    // content_block.start / content_block.delta / thought / message.delta 与坏 JSON 全部静默丢弃，
    // 既不能崩，也不能中断流；末尾合法 ping 与 message.end 仍按序到达
    expect(seen).toEqual(["ping", "message.end", "close"]);
  });

  it("契约漂移哨兵：事件名或必填字段被改名时，回放结果与基线不同", async () => {
    // 基线：block.stop 正常到达，工具已收尾
    const baseline = await replay(assistantTurnSSE);
    expect(
      stepsOfBlocks(baseline.chat.messages[ASSISTANT_ID].blocks)[0]?.status
    ).toBe("success");

    // 1) 事件名漂移：block.stop 改名（模拟历史别名 content_block.stop）
    const renamedEvent = assistantTurnSSE.replaceAll(
      "event: block.stop",
      "event: content_block.stop"
    );
    const byEvent = await replay(renamedEvent);
    const eventSteps = stepsOfBlocks(
      byEvent.chat.messages[ASSISTANT_ID].blocks
    );
    // 工具块永远停在 block.start 的 running 壳，thinking 也永远不收尾
    expect(eventSteps[0]?.status).toBe("running");
    expect(eventSteps[0]?.observation).toBeUndefined();
    const eventThinking = thinkingOfBlocks(
      byEvent.chat.messages[ASSISTANT_ID].blocks,
      Date.parse(byEvent.chat.messages[ASSISTANT_ID].createTime)
    );
    expect(eventThinking?.segments.some((segment) => !segment.closed)).toBe(
      true
    );

    // 2) 必填字段漂移：tool.status 改名为 tool.state
    const renamedField = assistantTurnSSE.replaceAll(
      '"status":"success"',
      '"state":"success"'
    );
    const byField = await replay(renamedField);
    const fieldStep = stepsOfBlocks(
      byField.chat.messages[ASSISTANT_ID].blocks
    )[0];
    expect(fieldStep?.status).toBe("running");
    expect(fieldStep?.observation).toBe("本月销售额 123.4 万元");
  });
});
