<!-- AI 可观测中心页：异常总览 → 趋势/资源 → 审计检索 → 过程链下钻 -->
<template>
  <div class="app-container">
    <anomaly-overview class="mb-[12px]" @select="handleAnomalySelect" />

    <el-row :gutter="12">
      <el-col :xs="24" :lg="12">
        <trend-panel class="mb-[12px]" />
      </el-col>
      <el-col :xs="24" :lg="12">
        <cost-panel class="mb-[12px]" />
      </el-col>
    </el-row>

    <div ref="auditPanelRef">
      <audit-search-panel />
    </div>

    <trace-detail-drawer />
  </div>
</template>

<script lang="ts" setup>
import type { AiObservabilityStatus } from "dehaze-sdk-js";
import AnomalyOverview from "./components/AnomalyOverview.vue";
import AuditSearchPanel from "./components/AuditSearchPanel.vue";
import CostPanel from "./components/CostPanel.vue";
import TraceDetailDrawer from "./components/TraceDetailDrawer.vue";
import TrendPanel from "./components/TrendPanel.vue";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "AiObservability" });

const observabilityStore = useAdminObservabilityStore();

const auditPanelRef = ref<HTMLElement>();

function handleAnomalySelect(status: AiObservabilityStatus) {
  observabilityStore.filterTracesByStatus(status);
  auditPanelRef.value?.scrollIntoView({ behavior: "smooth" });
}

onMounted(() => {
  observabilityStore.fetchAnomalySummary();
  observabilityStore.fetchTrends();
  observabilityStore.fetchCosts();
  observabilityStore.fetchTraces();
});
</script>
