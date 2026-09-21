import { mount } from "@vue/test-utils";
import { nextTick } from "vue";
import { describe, expect, it, vi } from "vitest";
import MessageList from "../components/MessageList.vue";
import type {
  ChatAssistantMessageVM,
  ChatMessageVM,
  ChatUserMessageVM,
} from "../types";
import { elStubs } from "./stubs";

vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

const user = (id: number, content: string): ChatUserMessageVM => ({
  role: "user",
  id,
  content,
});

const assistant = (
  id: number,
  over: Partial<ChatAssistantMessageVM> = {}
): ChatAssistantMessageVM => ({
  role: "assistant",
  id,
  status: "completed",
  text: "",
  thinking: null,
  steps: [],
  toolCalls: [],
  artifacts: [],
  memories: [],
  feedback: null,
  suggestions: [],
  ...over,
});

const tool = (id: number): ChatMessageVM => ({
  role: "tool",
  id,
  name: "search",
  content: "",
  toolCalls: [],
});

const mountList = (over: Record<string, unknown> = {}) =>
  mount(MessageList, {
    props: {
      messages: [],
      scope: "self",
      streamingMessageId: null,
      interruptedMessageId: null,
      interrupts: [],
      scrollFollowEnabled: true,
      ...over,
    },
    global: { stubs: elStubs },
  });

function setScroll(
  el: HTMLElement,
  opts: { scrollTop: number; scrollHeight: number; clientHeight: number }
) {
  Object.defineProperty(el, "scrollHeight", {
    configurable: true,
    value: opts.scrollHeight,
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    value: opts.clientHeight,
  });
  el.scrollTop = opts.scrollTop;
}

function body(wrapper: ReturnType<typeof mountList>) {
  return wrapper.find(".chat-message-list__body");
}

describe("MessageList", () => {
  it("role 分流渲染 user / assistant / tool", () => {
    const wrapper = mountList({
      messages: [user(1, "问"), assistant(2), tool(3)],
    });
    expect(wrapper.find(".user-message").exists()).toBe(true);
    expect(wrapper.find(".assistant-message").exists()).toBe(true);
    expect(wrapper.find(".tool-message").exists()).toBe(true);
  });

  it("空 messages 展示空态", () => {
    const wrapper = mountList();
    expect(wrapper.find(".el-empty-stub").exists()).toBe(true);
    expect(wrapper.find(".user-message").exists()).toBe(false);
  });

  it("id 缺失的脏数据不崩溃（按索引兜底为 key）", () => {
    const dirty = [
      { role: "user", content: "无 id" } as unknown as ChatMessageVM,
    ];
    const wrapper = mountList({ messages: [...dirty, user(2, "正常")] });
    expect(wrapper.findAll(".user-message")).toHaveLength(2);
  });

  it("超长文本正常渲染", () => {
    const long = "x".repeat(50000);
    const wrapper = mountList({ messages: [user(1, long)] });
    expect(wrapper.find(".user-message__content").text().length).toBe(50000);
  });

  it("interrupts 为空时不渲染中断卡", () => {
    const wrapper = mountList({ messages: [assistant(2)] });
    expect(wrapper.find(".interrupt-card").exists()).toBe(false);
  });

  it("中断卡内联到被中断的助手消息", () => {
    const wrapper = mountList({
      messages: [user(1, "问"), assistant(2)],
      interruptedMessageId: 2,
      interrupts: [
        { type: "confirm", confirmKind: "dangerous_op", detail: "危险" },
      ],
    });
    expect(wrapper.find(".interrupt-card").exists()).toBe(true);
    expect(wrapper.find(".interrupt-card").text()).toContain("危险操作确认");
  });

  it("到达底部恢复跟随（上抛 scroll-follow-toggle(true)）", async () => {
    const wrapper = mountList({
      messages: [user(1, "问")],
      scrollFollowEnabled: false,
    });
    setScroll(body(wrapper).element as HTMLElement, {
      scrollTop: 950,
      scrollHeight: 1000,
      clientHeight: 100,
    });
    await body(wrapper).trigger("scroll");

    expect(wrapper.emitted("scroll-follow-toggle")).toEqual([[true]]);
  });

  it("滚动到顶且还有历史时上抛 reach-top", async () => {
    const wrapper = mountList({
      messages: [user(1, "问")],
      hasMoreHistory: true,
    });
    setScroll(body(wrapper).element as HTMLElement, {
      scrollTop: 0,
      scrollHeight: 2000,
      clientHeight: 600,
    });
    await body(wrapper).trigger("scroll");
    expect(wrapper.emitted("reach-top")).toHaveLength(1);
  });

  it("远离底部时上抛 scroll-follow-toggle(false)", async () => {
    const wrapper = mountList({
      messages: [user(1, "问")],
      scrollFollowEnabled: true,
    });
    setScroll(body(wrapper).element as HTMLElement, {
      scrollTop: 0,
      scrollHeight: 2000,
      clientHeight: 600,
    });
    await body(wrapper).trigger("scroll");
    expect(wrapper.emitted("scroll-follow-toggle")).toEqual([[false]]);
  });

  it("重复滚动到底部重复上抛恢复跟随（无内部去重）", async () => {
    const wrapper = mountList({
      messages: [user(1, "问")],
      scrollFollowEnabled: false,
    });
    const el = body(wrapper).element as HTMLElement;
    setScroll(el, { scrollTop: 950, scrollHeight: 1000, clientHeight: 100 });
    await body(wrapper).trigger("scroll");
    await body(wrapper).trigger("scroll");
    await body(wrapper).trigger("scroll");
    expect(
      wrapper.emitted("scroll-follow-toggle")?.length
    ).toBeGreaterThanOrEqual(3);
  });
});

describe("MessageList 轻量窗口化", () => {
  const EST = 96;

  const many = (n: number): ChatUserMessageVM[] =>
    Array.from({ length: n }, (_, i) => user(i + 1, `msg-${i + 1}`));

  /** 模拟布局：scrollHeight = 占位高度合计 + 已挂载条目数 × 估算高度（等价于总数 × 估算高度） */
  function renderedHeight(el: HTMLElement) {
    const spacers = Array.from(
      el.querySelectorAll<HTMLElement>(".chat-message-list__spacer")
    );
    const spacerSum = spacers.reduce(
      (sum, node) => sum + (parseFloat(node.style.height) || 0),
      0
    );
    const items = el.querySelectorAll(
      ".user-message, .assistant-message, .tool-message"
    ).length;
    return spacerSum + items * EST;
  }

  function defineHeights(el: HTMLElement, clientHeight: number) {
    Object.defineProperty(el, "scrollHeight", {
      configurable: true,
      get: () => renderedHeight(el),
    });
    Object.defineProperty(el, "clientHeight", {
      configurable: true,
      value: clientHeight,
    });
  }

  it("①仅渲染视口附近的条目（DOM 远小于总数）", () => {
    const wrapper = mountList({ messages: many(150) });
    const rendered = wrapper.findAll(".user-message").length;
    expect(rendered).toBeGreaterThan(0);
    expect(rendered).toBeLessThan(150);
    expect(rendered).toBeLessThan(100);
  });

  it("②向上加载更早历史后视口内容不跳变（scrollHeight 差值回填 scrollTop）", async () => {
    const older = many(150);
    const wrapper = mountList({
      messages: older,
      hasMoreHistory: true,
      scrollFollowEnabled: false,
    });
    const el = body(wrapper).element as HTMLElement;
    defineHeights(el, 600);
    await nextTick(); // 让 onMounted 的吸底落定
    el.scrollTop = 0; // 模拟用户上滑到顶触发 reach-top
    await nextTick();
    expect(el.scrollTop).toBe(0);

    const before = renderedHeight(el);
    // 前置插入 50 条更早消息（头部）
    const prepended = [
      ...many(50).map((m, i) => user(-i - 1, m.content)),
      ...older,
    ];
    await wrapper.setProps({ messages: prepended });
    await nextTick();
    await nextTick();

    const after = renderedHeight(el);
    // 补偿量 = 插入前后 scrollHeight 差值（50 × 96），视口内容保持原位
    expect(after - before).toBe(50 * EST);
    expect(el.scrollTop).toBe(after - before);
  });

  it("②bis管理端审计（scrollFollowEnabled 恒 true）视口不在底部，前置插入仍保位", async () => {
    const older = many(150);
    const wrapper = mountList({
      messages: older,
      hasMoreHistory: true,
      scrollFollowEnabled: true, // 管理端审计抽屉恒传 true
    });
    const el = body(wrapper).element as HTMLElement;
    defineHeights(el, 600);
    await nextTick(); // 让 onMounted 的吸底落定
    el.scrollTop = 0; // 用户上滑到顶（视口不在底部）
    await nextTick();
    expect(el.scrollTop).toBe(0);

    const before = renderedHeight(el);
    const prepended = [
      ...many(50).map((m, i) => user(-i - 1, m.content)),
      ...older,
    ];
    await wrapper.setProps({ messages: prepended });
    await nextTick();
    await nextTick();

    const after = renderedHeight(el);
    // 判定基于插入前真实位置，与管理端 scrollFollowEnabled=true 无关：视口不跳到顶部也不吸底
    expect(after - before).toBe(50 * EST);
    expect(el.scrollTop).toBe(after - before);
  });

  it("③流式追加不错位（跟随态吸底到 scrollHeight）", async () => {
    const wrapper = mountList({
      messages: many(120),
      scrollFollowEnabled: true,
    });
    const el = body(wrapper).element as HTMLElement;
    defineHeights(el, 600);
    await nextTick();

    await wrapper.setProps({ messages: [...many(120), user(121, "msg-121")] });
    await nextTick();
    await nextTick();

    expect(el.scrollTop).toBe(renderedHeight(el));
    expect(el.scrollTop).toBe(121 * EST);
  });

  it("④删除中间消息后不丢内容（周边条目仍渲染，占位正确）", async () => {
    const all = many(150);
    const wrapper = mountList({
      messages: all,
      scrollFollowEnabled: false,
    });
    const el = body(wrapper).element as HTMLElement;
    defineHeights(el, 600);
    el.scrollTop = 0;
    await nextTick();

    await wrapper.setProps({ messages: all.filter((m) => m.id !== 30) });
    await nextTick();

    const text = wrapper.text();
    expect(text).toContain("msg-29");
    expect(text).toContain("msg-31");
    expect(text).not.toContain("msg-30");
    expect(renderedHeight(el)).toBe(149 * EST);
  });
});
