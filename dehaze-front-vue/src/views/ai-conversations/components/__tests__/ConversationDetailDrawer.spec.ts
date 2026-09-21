// 管理端会话详情抽屉回归：复用 ai-chat 无状态组件（scope=admin）经宿主绑定层映射，
// 验证只读语义（无编辑/重发/删除/反馈入口）、链路下钻跳转、以及 reach-top → loadMoreDetailMessages 沿用。
import { flushPromises, mount } from "@vue/test-utils";
import { defineComponent, h, nextTick } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import { AiConversationAPI, type AiMessageVO } from "dehaze-sdk-js";
import { elStubs } from "@/components/ai-chat/__tests__/stubs";
import { useAdminAuditStore } from "@/store/modules/adminAudit";
import ConversationDetailDrawer from "../ConversationDetailDrawer.vue";

vi.mock("@/components/MarkdownRenderer.vue", () => ({
  default: {
    name: "MarkdownRenderer",
    props: ["content"],
    template: '<div class="md-stub">{{ content }}</div>',
  },
}));

vi.mock("dehaze-sdk-js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("dehaze-sdk-js")>();
  return {
    ...actual,
    AiConversationAPI: {
      ...actual.AiConversationAPI,
      getMessages: vi.fn(),
      getMessageArtifacts: vi.fn(),
      getArtifactDetail: vi.fn(),
      getFeedback: vi.fn(),
    },
  };
});

/** 抽屉桩：渲染 default 插槽（真实 el-drawer 依赖 teleport/挂载过渡） */
const DrawerStub = defineComponent({
  name: "ElDrawer",
  props: ["modelValue", "title", "size"],
  setup:
    (props, { slots }) =>
    () =>
      props.modelValue
        ? h("div", { class: "el-drawer-stub" }, slots.default?.())
        : null,
});

/** 弹窗桩：modelValue 为真时渲染 default 插槽，便于断言产物详情内容 */
const DialogStub = defineComponent({
  name: "ElDialog",
  props: ["modelValue", "title"],
  setup:
    (props, { slots }) =>
    () =>
      props.modelValue
        ? h("div", { class: "el-dialog-stub" }, slots.default?.())
        : null,
});

const message = (over: Partial<AiMessageVO>): AiMessageVO => ({
  id: 1,
  conversationId: 7,
  role: "assistant",
  status: 2,
  content: "答",
  createTime: "2026-01-01T00:00:00Z",
  ...over,
});

let router: Router;

async function mountDrawer(messages: AiMessageVO[], total: number) {
  const audit = useAdminAuditStore();
  audit.detailVisible = true;
  audit.detailConversation = {
    id: 7,
    title: "被审计会话",
    messageCount: messages.length,
  } as never;
  audit.detailMessages = messages;
  audit.detailTotal = total;
  audit.detailHasMore = messages.length < total;
  audit.detailLoading = false;
  audit.detailError = "";

  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/admin/ai-conversations", component: { render: () => null } },
      { path: "/admin/ai-observability", component: { render: () => null } },
    ],
  });
  await router.push("/admin/ai-conversations");
  const wrapper = mount(ConversationDetailDrawer, {
    global: {
      plugins: [router],
      stubs: { "el-drawer": DrawerStub, "el-dialog": DialogStub, ...elStubs },
      directives: { loading: {} },
    },
  });
  await nextTick();
  return wrapper;
}

function setScroll(el: HTMLElement, top: number) {
  Object.defineProperty(el, "scrollHeight", {
    configurable: true,
    value: 5000,
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    value: 600,
  });
  el.scrollTop = top;
}

describe("管理端会话详情抽屉（scope=admin 只读审计）", () => {
  beforeEach(() => {
    vi.mocked(AiConversationAPI.getMessages).mockResolvedValue({
      list: [],
      total: 0,
      hasMore: false,
    } as never);
  });

  it("只读渲染：无编辑/重新生成/删除/反馈操作入口", async () => {
    const wrapper = await mountDrawer(
      [message({ id: 1, role: "user", content: "问" }), message({ id: 2 })],
      2
    );

    expect(wrapper.find(".user-message").exists()).toBe(true);
    expect(wrapper.find(".assistant-message").exists()).toBe(true);
    expect(wrapper.text()).not.toContain("重新生成");
    expect(wrapper.text()).not.toContain("编辑");
    expect(wrapper.text()).not.toContain("反馈");
    // 管理端仍提供链路下钻
    expect(wrapper.text()).toContain("链路下钻");
  });

  it("产物展示：status>=2 助手消息拉取并渲染只读产物卡片（无内容操作入口）", async () => {
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValueOnce([
      {
        id: 11,
        conversationId: 7,
        messageId: 2,
        type: "metric_report",
        isInvalid: 0,
      },
    ] as never);
    const wrapper = await mountDrawer([message({ id: 2 })], 1);
    await flushPromises();
    await nextTick();

    expect(AiConversationAPI.getMessageArtifacts).toHaveBeenCalledWith(2);
    expect(wrapper.find(".artifact-card").exists()).toBe(true);
    // 只读审计：产物展示不引入反馈/编辑/重发/删除入口
    expect(wrapper.text()).not.toContain("反馈");
    expect(wrapper.text()).not.toContain("编辑");
    expect(wrapper.text()).not.toContain("重新生成");
    expect(wrapper.text()).not.toContain("删除");
  });

  it("产物详情：点击卡片调用 getArtifactDetail 并弹窗展示详情（仍无反馈/编辑/重发/删除入口）", async () => {
    vi.mocked(AiConversationAPI.getMessageArtifacts).mockResolvedValueOnce([
      {
        id: 11,
        conversationId: 7,
        messageId: 2,
        type: "metric_report",
        isInvalid: 0,
      },
    ] as never);
    vi.mocked(AiConversationAPI.getArtifactDetail).mockResolvedValueOnce({
      id: 11,
      type: "metric_report",
      payload: { metric: "psnr" },
    } as never);

    const wrapper = await mountDrawer([message({ id: 2 })], 1);
    await flushPromises();
    await nextTick();

    expect(wrapper.find(".artifact-card").exists()).toBe(true);

    await wrapper.find(".artifact-card").trigger("click");
    await flushPromises();

    expect(AiConversationAPI.getArtifactDetail).toHaveBeenCalledWith(11);
    expect(wrapper.find(".el-dialog-stub").text()).toContain("metric_report");
    // 只读审计语义不变：产物弹窗不引入反馈/编辑/重发/删除入口
    expect(wrapper.text()).not.toContain("反馈");
    expect(wrapper.text()).not.toContain("编辑");
    expect(wrapper.text()).not.toContain("重新生成");
    expect(wrapper.text()).not.toContain("删除");
  });

  it("链路下钻：点击跳转可观测中心并携 conversationId", async () => {
    const wrapper = await mountDrawer([message({ id: 2 })], 1);
    const push = vi.spyOn(router, "push").mockResolvedValue(undefined as never);

    const traceButton = wrapper
      .findAll("button")
      .find((btn) => btn.text().includes("链路下钻"))!;
    await traceButton.trigger("click");

    expect(push).toHaveBeenCalledWith({
      path: "/admin/ai-observability",
      query: { conversationId: "7" },
    });
    expect(useAdminAuditStore().detailVisible).toBe(false);
  });

  it("历史分页：reach-top 沿用 loadMoreDetailMessages（游标 before + view=admin）", async () => {
    const wrapper = await mountDrawer([message({ id: 2 })], 10);
    const body = wrapper.find(".chat-message-list__body");
    setScroll(body.element as HTMLElement, 0);
    await body.trigger("scroll");
    await flushPromises();

    // 当前最早一条 id=2 → before=2，limit=50（DETAIL_PAGE_SIZE）
    expect(AiConversationAPI.getMessages).toHaveBeenCalledWith(
      7,
      expect.objectContaining({ before: 2, limit: 50, view: "admin" })
    );
  });
});
