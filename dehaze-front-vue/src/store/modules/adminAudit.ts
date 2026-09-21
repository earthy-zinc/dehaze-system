// 管理端会话审计 Store：审计筛选、全量会话列表、异常概览、详情抽屉与会话时间线
// 列表数据经 chatStore(scope=admin) 拉取（view=admin 返回审计字段），异常概览消费可观测性 summary
import {
  AiConversationAPI,
  AiObservabilityAPI,
  type AiMessageVO,
  type AiObservabilitySummary,
  type AiObservabilityTimeline,
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
  /** 是否还有更早消息（游标分页 hasMore，驱动"滚动到顶加载更多"） */
  const detailHasMore = ref(false);
  const detailLoading = ref(false);
  const detailError = ref("");

  // ===== 会话审计时间线 =====
  const timelineVisible = ref(false);
  const timelineData = ref<AiObservabilityTimeline | null>(null);
  const timelineLoading = ref(false);
  const timelineError = ref("");
  /** 当前选中轮次下标（RoundNavigator 与 RoundTimeline 联动） */
  const timelineRoundIndex = ref(0);

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

  /**
   * 加载详情消息（只读游标分页）。首次缺省 before 取最新一页；加载更早历史时以当前
   * 最早一条消息的 id 作为 before（仅返回 id 更小的）。`hasMore` 直接取响应。
   */
  async function fetchDetailMessages() {
    if (!detailConversation.value) return;
    detailLoading.value = true;
    try {
      const earliest = detailMessages.value[0];
      // 管理端跨用户查看需显式 view=admin
      const result = await AiConversationAPI.getMessages(
        detailConversation.value.id,
        {
          ...(earliest ? { before: earliest.id } : {}),
          limit: DETAIL_PAGE_SIZE,
          view: "admin",
        }
      );
      // 后端按 id 倒序返回，展示需时间正序（最早在上）：本页反转；加载更早页时前置于已有列表
      const pageList = (result.list ?? []).slice().reverse();
      detailMessages.value = earliest
        ? [...pageList, ...detailMessages.value]
        : pageList;
      detailTotal.value = result.total ?? 0;
      detailHasMore.value = result.hasMore;
    } catch (error) {
      detailError.value = (error as Error).message || "会话消息加载失败";
    } finally {
      detailLoading.value = false;
    }
  }

  function loadMoreDetailMessages() {
    if (detailLoading.value || !detailHasMore.value) {
      return;
    }
    return fetchDetailMessages();
  }

  /** 打开详情抽屉：刷新审计详情（view=admin）+ 只读加载消息 */
  async function openConversationDetail(conversation: ConversationVO) {
    detailVisible.value = true;
    detailConversation.value = conversation;
    detailMessages.value = [];
    detailTotal.value = 0;
    detailHasMore.value = false;
    detailError.value = "";
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

  /** 打开会话时间线（include=raw 拉取 wire 原始报文，raw 缺失时组件显示空态） */
  async function openConversationTimeline(conversationId: number) {
    timelineVisible.value = true;
    timelineData.value = null;
    timelineError.value = "";
    timelineRoundIndex.value = 0;
    timelineLoading.value = true;
    try {
      timelineData.value = await AiObservabilityAPI.getConversationTimeline(
        conversationId,
        { include: "raw" }
      );
    } catch (error) {
      timelineError.value = (error as Error).message || "会话时间线加载失败";
    } finally {
      timelineLoading.value = false;
    }
  }

  function closeConversationTimeline() {
    timelineVisible.value = false;
    timelineData.value = null;
    timelineError.value = "";
    timelineRoundIndex.value = 0;
  }

  function selectTimelineRound(index: number) {
    timelineRoundIndex.value = index;
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
    detailHasMore,
    detailLoading,
    detailError,
    timelineVisible,
    timelineData,
    timelineLoading,
    timelineError,
    timelineRoundIndex,
    applyAuditFilter,
    fetchAuditList,
    fetchAnomalySummary,
    openConversationDetail,
    fetchDetailMessages,
    loadMoreDetailMessages,
    openConversationTimeline,
    closeConversationTimeline,
    selectTimelineRound,
    resetAuditFilter,
  };
});
