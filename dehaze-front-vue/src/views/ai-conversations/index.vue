<!-- AI 对话管理端会话审计页：审计筛选 → 异常概览 → 审计表格 → 详情抽屉（只读浏览 + 链路追踪） -->
<script lang="ts" setup>
import { onMounted } from "vue";
import { useChatStore } from "@/store/modules/chat";
import { useAdminAuditStore } from "@/store/modules/adminAudit";
import AuditFilterBar from "./components/AuditFilterBar.vue";
import AnomalySummaryPanel from "./components/AnomalySummaryPanel.vue";
import ConversationAuditTable from "./components/ConversationAuditTable.vue";
import ConversationDetailDrawer from "./components/ConversationDetailDrawer.vue";

// 命名须与动态路由名（component 路径推导 ai-conversations/index → AiConversations）一致，否则 keep-alive 缓存静默失效
defineOptions({ name: "AiConversations" });

const chatStore = useChatStore();
const adminAuditStore = useAdminAuditStore();

onMounted(() => {
  chatStore.initScope("admin");
  adminAuditStore.fetchAuditList();
  adminAuditStore.fetchAnomalySummary();
});
</script>

<template>
  <div class="app-container">
    <AuditFilterBar class="mb-[12px]" />
    <AnomalySummaryPanel class="mb-[12px]" />
    <ConversationAuditTable />
    <ConversationDetailDrawer />
  </div>
</template>
