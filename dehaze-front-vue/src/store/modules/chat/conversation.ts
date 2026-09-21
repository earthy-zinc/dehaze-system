// 会话 slice：会话列表与 CRUD（scope=self 本人会话 / scope=admin 全量审计）
import {
  AiConversationAPI,
  type ConversationCreateForm,
  type ConversationQuery,
  type ConversationUpdateForm,
} from "dehaze-sdk-js";
import { reactive, ref } from "vue";
import type { BatchAction, ChatCtx, ChatScope } from "./shared";

export function createConversationSlice(
  ctx: ChatCtx,
  teardownStream: () => void
) {
  const { conversations, messages, currentConversationId, scope } = ctx;

  const conversationsTotal = ref(0);
  const conversationsLoading = ref(false);
  const conversationQuery = reactive<ConversationQuery>({
    keyword: "",
    pageNum: 1,
    pageSize: 50,
  });

  /** 注入数据范围，由宿主页面在挂载时调用 */
  function initScope(next: ChatScope) {
    if (scope.value === next) return;
    teardownStream();
    scope.value = next;
    conversations.value = [];
    messages.value = [];
    currentConversationId.value = null;
  }

  async function fetchConversations() {
    conversationsLoading.value = true;
    try {
      const query: ConversationQuery = { ...conversationQuery };
      if (scope.value === "admin") {
        query.view = "admin";
      }
      const result = await AiConversationAPI.getConversations(query);
      conversations.value = result.list ?? [];
      conversationsTotal.value = result.total ?? 0;
    } finally {
      conversationsLoading.value = false;
    }
  }

  async function createConversation(form?: ConversationCreateForm) {
    const conversation = await AiConversationAPI.createConversation(form);
    await fetchConversations();
    return conversation;
  }

  async function updateConversation(id: number, form: ConversationUpdateForm) {
    const conversation = await AiConversationAPI.updateConversation(id, form);
    const local = conversations.value.find((item) => item.id === id);
    if (local) Object.assign(local, conversation);
    return conversation;
  }

  async function deleteConversation(id: number) {
    await AiConversationAPI.deleteConversation(id);
    conversations.value = conversations.value.filter((item) => item.id !== id);
    if (currentConversationId.value === id) {
      teardownStream();
      currentConversationId.value = null;
      messages.value = [];
    }
  }

  async function batchOperateConversations(action: BatchAction, ids: number[]) {
    if (ids.length === 0) return;
    // 批量删除的二次确认由宿主页面负责，此处确认后携带 confirm 执行
    await AiConversationAPI.batchConversations({
      action,
      ids,
      confirm: action === "delete",
    });
    await fetchConversations();
    if (
      action === "delete" &&
      currentConversationId.value &&
      ids.includes(currentConversationId.value)
    ) {
      teardownStream();
      currentConversationId.value = null;
      messages.value = [];
    }
  }

  async function exportConversation(
    id: number,
    format: "json" | "markdown" = "markdown"
  ) {
    const blob = await AiConversationAPI.exportConversation(id, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `conversation-${id}.${format === "json" ? "json" : "md"}`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return {
    conversationsTotal,
    conversationsLoading,
    conversationQuery,
    initScope,
    fetchConversations,
    createConversation,
    updateConversation,
    deleteConversation,
    batchOperateConversations,
    exportConversation,
  };
}
