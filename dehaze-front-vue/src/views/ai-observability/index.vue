<!-- AI 可观测中心：三 Tab（会话审计为唯一审计入口 / 总览 / 检索导出），支持携 conversationId 定位时间线 -->
<template>
  <div class="app-container">
    <el-tabs v-model="activeTab">
      <el-tab-pane label="会话审计" name="audit">
        <ConversationTimelinePanel v-if="adminAuditStore.timelineVisible" />
        <template v-else>
          <audit-filter-bar class="mb-[12px]" />
          <AnomalySummaryPanel class="mb-[12px]" />
          <ConversationAuditTable />
        </template>
      </el-tab-pane>

      <el-tab-pane label="总览" name="overview" lazy>
        <anomaly-overview class="mb-[12px]" @select="handleAnomalySelect" />
        <el-row :gutter="12">
          <el-col :xs="24" :lg="12">
            <performance-trend-panel class="mb-[12px]" />
          </el-col>
          <el-col :xs="24" :lg="12">
            <cost-panel class="mb-[12px]" />
          </el-col>
        </el-row>
      </el-tab-pane>

      <el-tab-pane label="检索导出" name="search" lazy>
        <audit-search-panel />
      </el-tab-pane>
    </el-tabs>

    <trace-detail-drawer />
  </div>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import { useRoute } from "vue-router";
import type { AiObservabilityStatus } from "dehaze-sdk-js";
import AnomalyOverview from "./components/AnomalyOverview.vue";
import AuditSearchPanel from "./components/AuditSearchPanel.vue";
import ConversationTimelinePanel from "./components/ConversationTimelinePanel.vue";
import CostPanel from "./components/CostPanel.vue";
import PerformanceTrendPanel from "./components/PerformanceTrendPanel.vue";
import TraceDetailDrawer from "./components/TraceDetailDrawer.vue";
import AuditFilterBar from "./components/AuditFilterBar.vue";
import AnomalySummaryPanel from "./components/AnomalySummaryPanel.vue";
import ConversationAuditTable from "./components/ConversationAuditTable.vue";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";
import { useAdminAuditStore } from "@/store/modules/adminAudit";

defineOptions({ name: "AiObservability" });

const route = useRoute();
const observabilityStore = useAdminObservabilityStore();
const adminAuditStore = useAdminAuditStore();

type TabName = "audit" | "overview" | "search";
const activeTab = ref<TabName>("audit");

function isTabName(value: unknown): value is TabName {
  return value === "audit" || value === "overview" || value === "search";
}

onMounted(() => {
  if (isTabName(route.query.tab)) {
    activeTab.value = route.query.tab;
  }
  // 跨页跳转（会话管理页/检索导出行）携 conversationId 直达会话时间线
  const conversationId = Number(route.query.conversationId);
  if (conversationId > 0) {
    activeTab.value = "audit";
    adminAuditStore.openConversationTimeline(conversationId);
  }
  adminAuditStore.fetchAuditList();
  adminAuditStore.fetchAnomalySummary();
  observabilityStore.fetchAnomalySummary();
  observabilityStore.fetchTrends();
  observabilityStore.fetchCosts();
  observabilityStore.fetchTraces();
});

/** 总览异常卡片：切到检索导出 Tab 收敛状态筛选 */
function handleAnomalySelect(status: AiObservabilityStatus) {
  activeTab.value = "search";
  observabilityStore.filterTracesByStatus(status);
}
</script>
