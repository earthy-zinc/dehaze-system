// 会话侧栏回归：验证旧对话消息组件目录整体删除后，
// ConversationList（已迁至页面组件目录）仍被自动导入解析并渲染。
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import { elStubs } from "@/components/ai-chat/__tests__/stubs";
import { useChatStore } from "@/store/modules/chat";
import ChatSidebar from "../ChatSidebar.vue";

describe("ChatSidebar", () => {
  it("渲染会话列表（ConversationList 自动导入解析成功）", async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: "/", component: { render: () => null } }],
    });
    await router.push("/");
    // 预置会话列表，避免挂载时触发会话拉取
    useChatStore().conversations = [
      {
        id: 1,
        title: "会话一",
        model: "m",
        messageCount: 2,
        pinned: 0,
        unreadCount: 0,
        createTime: "2026-01-01 00:00:00",
        updateTime: "2026-01-01 00:00:00",
      } as never,
    ];

    const wrapper = mount(ChatSidebar, {
      global: {
        plugins: [router],
        directives: { loading: {} },
        stubs: {
          ...elStubs,
          "el-select": true,
          "el-option": true,
          "el-checkbox": true,
          "el-badge": true,
          "el-icon": true,
        },
      },
    });

    expect(wrapper.find(".conversation-list").exists()).toBe(true);
    expect(wrapper.find(".conversation-item__title").text()).toContain(
      "会话一"
    );
  });
});
