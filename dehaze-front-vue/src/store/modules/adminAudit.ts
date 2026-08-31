// 管理端会话审计 Store：审计筛选、全量会话列表、异常概览、详情抽屉与链路追踪
// 列表数据经 chatStore(scope=admin) 拉取（view=admin 返回审计字段），异常概览消费可观测性 summary
import {
  AiConversationAPI,
  AiObservabilityAPI,
  type AiMessageContextSnapshot,
  type AiMessageLlmCall,
  type AiMessageThought,
  type AiMessageVO,
  type AiObservabilitySummary,
  type ConversationStatus,
  type ConversationVO,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { reactive, ref } from "vue";
import {
  useChatStore,
  type ConversationFilterStatus,
} from "@/store/modules/chat";

export type AuditAnomalyType = "failed" | "quota" | "canceled" | "";

/** LLM 调用明细（SDK 将 toolCall/outputSnapshot 声明为 unknown，这里收敛实际结构） */
export interface AuditLlmCall extends AiMessageLlmCall {
  toolCall?: {
    has_tool_call?: boolean;
    tools?: Array<{ name: string; arguments: string }>;
  } | null;
  outputSnapshot?: { text?: string } | null;
}

/** 链路追踪结果：thought 推理步骤 + LLM 调用回放 */
export interface AuditTraceChain {
  traceId: string | null;
  contextSnapshot: AiMessageContextSnapshot | null;
  thoughts: AiMessageThought[];
  llmCalls: AuditLlmCall[];
}

const DETAIL_PAGE_SIZE = 50;

export const useAdminAuditStore = defineStore("adminAudit", () => {
  // ===== 审计筛选与列表 =====
  const auditFilter = reactive({
    userId: undefined as number | undefined,
    dateRange: null as [string, string] | null,
    status: 0 as ConversationFilterStatus,
    anomalyType: "" as AuditAnomalyType,
    keyword: "",
    pageNum: 1,
    pageSize: 10,
  });
  const auditList = ref<ConversationVO[]>([]);
  const auditTotal = ref(0);
  const auditLoading = ref(false);

  // ===== 异常概览（可观测性 summary 口径） =====
  const anomalySummary = ref<AiObservabilitySummary | null>(null);
  const summaryLoading = ref(false);

  // ===== 详情抽屉 =====
  const detailVisible = ref(false);
  const detailConversation = ref<ConversationVO | null>(null);
  const detailMessages = ref<AiMessageVO[]>([]);
  const detailTotal = ref(0);
  /** 下一次要加载的消息页码 */
  const detailPageNum = ref(1);
  const detailLoading = ref(false);
  const detailError = ref("");

  // ===== 链路追踪 =====
  const traceMessage = ref<AiMessageVO | null>(null);
  const traceChainData = ref<AuditTraceChain | null>(null);
  const traceLoading = ref(false);
  const traceError = ref("");

  /** 用户/时间/异常类型筛选（审计列表接口暂无对应查询参数，在已加载页内过滤） */
  function matchesAuditFilter(conversation: ConversationVO) {
    if (
      auditFilter.userId != null &&
      conversation.userId !== auditFilter.userId
    ) {
      return false;
    }
    if (
      auditFilter.anomalyType &&
      conversation.anomalyType !== auditFilter.anomalyType
    ) {
      return false;
    }
    if (auditFilter.dateRange) {
      const time = conversation.lastMessageAt ?? conversation.createTime;
      if (!time) return false;
      const ts = new Date(time).getTime();
      const start = new Date(`${auditFilter.dateRange[0]}T00:00:00`).getTime();
      const end = new Date(`${auditFilter.dateRange[1]}T23:59:59`).getTime();
      if (ts < start || ts > end) return false;
    }
    return true;
  }

  async function fetchAuditList() {
    const chatStore = useChatStore();
    auditLoading.value = true;
    try {
      // 关键词/状态由后端筛选（view=admin 返回全量会话与审计字段）
      chatStore.conversationQuery.keyword = auditFilter.keyword;
      chatStore.conversationQuery.status = (auditFilter.status || undefined) as
        ConversationStatus | undefined;
      chatStore.conversationQuery.pageNum = auditFilter.pageNum;
      chatStore.conversationQuery.pageSize = auditFilter.pageSize;
      await chatStore.fetchConversations();
      auditList.value = chatStore.conversations.filter(matchesAuditFilter);
      auditTotal.value = chatStore.conversationsTotal;
    } finally {
      auditLoading.value = false;
    }
  }

  /** 应用筛选并回到第一页刷新 */
  function applyAuditFilter(patch: Partial<typeof auditFilter>) {
    Object.assign(auditFilter, patch);
    auditFilter.pageNum = 1;
    return fetchAuditList();
  }

  function resetAuditFilter() {
    Object.assign(auditFilter, {
      userId: undefined,
      dateRange: null,
      status: 0,
      anomalyType: "",
      keyword: "",
      pageNum: 1,
    });
    return fetchAuditList();
  }

  async function fetchAnomalySummary() {
    summaryLoading.value = true;
    try {
      anomalySummary.value = await AiObservabilityAPI.getSummary();
    } finally {
      summaryLoading.value = false;
    }
  }

  async function fetchDetailMessages() {
    if (!detailConversation.value) return;
    detailLoading.value = true;
    try {
      // 管理端跨用户查看需显式 view=admin
      const result = await AiConversationAPI.getMessages(
        detailConversation.value.id,
        {
          pageNum: detailPageNum.value,
          pageSize: DETAIL_PAGE_SIZE,
          view: "admin",
        }
      );
      // 后端消息列表为倒序分页（pageNum=1=最新一页、页内最新在前），
      // 展示需时间正序（最早在上）：每页反转；加载更早页时插到已有列表之前
      const pageList = (result.list ?? []).slice().reverse();
      detailMessages.value =
        detailPageNum.value === 1
          ? pageList
          : [...pageList, ...detailMessages.value];
      detailTotal.value = result.total ?? 0;
      detailPageNum.value += 1;
    } catch (error) {
      detailError.value = (error as Error).message || "会话消息加载失败";
    } finally {
      detailLoading.value = false;
    }
  }

  function loadMoreDetailMessages() {
    if (
      detailLoading.value ||
      detailMessages.value.length >= detailTotal.value
    ) {
      return;
    }
    return fetchDetailMessages();
  }

  /** 打开详情抽屉：刷新审计详情（view=admin）+ 只读加载消息 */
  async function openConversationDetail(conversation: ConversationVO) {
    detailVisible.value = true;
    detailConversation.value = conversation;
    detailPageNum.value = 1;
    detailMessages.value = [];
    detailTotal.value = 0;
    detailError.value = "";
    closeChainTrace();
    try {
      detailConversation.value = await AiConversationAPI.getConversation(
        conversation.id,
        { view: "admin" }
      );
    } catch {
      // 详情刷新失败时保留审计列表行数据展示
    }
    await fetchDetailMessages();
  }

  /** 单条消息链路追踪：复用消息详情接口（含 thoughts/llmCalls，无过程链为空态） */
  async function startChainTrace(message: AiMessageVO) {
    traceMessage.value = message;
    traceChainData.value = null;
    traceError.value = "";
    traceLoading.value = true;
    try {
      // 管理端跨用户查看需显式 view=admin
      const detail = await AiConversationAPI.getMessageDetail(message.id, {
        view: "admin",
      });
      traceChainData.value = {
        traceId: detail.traceId ?? null,
        contextSnapshot: detail.contextSnapshot ?? null,
        thoughts: detail.thoughts ?? [],
        llmCalls: (detail.llmCalls ?? []) as AuditLlmCall[],
      };
    } catch (error) {
      traceError.value = (error as Error).message || "链路追踪数据加载失败";
    } finally {
      traceLoading.value = false;
    }
  }

  function closeChainTrace() {
    traceMessage.value = null;
    traceChainData.value = null;
    traceError.value = "";
  }

  return {
    auditFilter,
    auditList,
    auditTotal,
    auditLoading,
    anomalySummary,
    summaryLoading,
    detailVisible,
    detailConversation,
    detailMessages,
    detailTotal,
    detailLoading,
    detailError,
    traceMessage,
    traceChainData,
    traceLoading,
    traceError,
    applyAuditFilter,
    fetchAuditList,
    fetchAnomalySummary,
    openConversationDetail,
    fetchDetailMessages,
    loadMoreDetailMessages,
    startChainTrace,
    closeChainTrace,
    resetAuditFilter,
  };
});
