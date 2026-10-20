// 会话 slice：会话列表与 CRUD（scope=self 本人会话 / scope=admin 全量审计）
import {
  AiConversationAPI,
  type ConversationCreateForm,
  type ConversationQuery,
  type ConversationUpdateForm,
} from "clarity-sdk-js";
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

  /** 在途请求：首屏页面与侧边栏会同时触发列表加载，joinInflight=true 时复用同一请求 */
  let conversationsInflight: Promise<void> | null = null;

  /**
   * 加载会话列表。
   *
   * `joinInflight=true`：已有同类请求在途时复用它（首屏页面与 ChatSidebar 挂载都会触发一次
   * 加载，子组件先挂载，合并后只打一次接口）；
   * 缺省 false：变更/搜索/筛选后必须拿到最新数据，始终新发请求（复用会返回过期结果）。
   */
  function fetchConversations(joinInflight = false): Promise<void> {
    if (joinInflight && conversationsInflight) return conversationsInflight;
    conversationsLoading.value = true;
    const request = (async () => {
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
    })();
    conversationsInflight = request;
    // 仅在仍指向本请求时清空在途标记；用 then 双分支而非 finally，避免产生未处理的拒绝
    void request.then(
      () => {
        if (conversationsInflight === request) conversationsInflight = null;
      },
      () => {
        if (conversationsInflight === request) conversationsInflight = null;
      }
    );
    return request;
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
