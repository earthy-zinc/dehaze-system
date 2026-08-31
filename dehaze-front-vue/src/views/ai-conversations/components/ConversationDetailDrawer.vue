<!-- 会话详情抽屉：审计元信息 + 只读消息浏览（MessageViews scope=admin）+ 链路追踪面板 -->
<script lang="ts" setup>
import { storeToRefs } from "pinia";
import MessageViews from "@/components/chat/MessageViews.vue";
import { useAdminAuditStore } from "@/store/modules/adminAudit";
import ChainTracePanel from "./ChainTracePanel.vue";

defineOptions({ name: "ConversationDetailDrawer" });

const adminAuditStore = useAdminAuditStore();
const {
  detailVisible,
  detailConversation,
  detailMessages,
  detailLoading,
  detailError,
} = storeToRefs(adminAuditStore);
</script>

<template>
  <el-drawer
    v-model="detailVisible"
    :title="detailConversation?.title ?? '会话详情'"
    size="62%"
  >
    <div v-if="detailConversation" class="detail-meta">
      <span>
        用户：{{
          detailConversation.userName ??
          `用户 ${detailConversation.userId ?? "-"}`
        }}
      </span>
      <span>模型：{{ detailConversation.model ?? "-" }}</span>
      <span>消息数：{{ detailConversation.messageCount }}</span>
      <span v-if="detailConversation.tokenConsumed != null">
        Token：{{ detailConversation.tokenConsumed }}
      </span>
      <span v-if="detailConversation.creditsConsumed != null">
        积分：{{ detailConversation.creditsConsumed }}
      </span>
      <el-tag v-if="detailConversation.anomalyLabel" type="danger" size="small">
        {{ detailConversation.anomalyLabel }}
      </el-tag>
    </div>

    <el-alert
      v-if="detailError"
      type="error"
      :closable="false"
      :title="detailError"
      class="mb-2"
    />

    <div v-loading="detailLoading" class="detail-messages">
      <MessageViews
        scope="admin"
        :conversation-id="detailConversation?.id"
        :messages="detailMessages"
        @trace="adminAuditStore.startChainTrace"
        @reach-bottom="adminAuditStore.loadMoreDetailMessages"
      />
    </div>

    <ChainTracePanel v-if="adminAuditStore.traceMessage" />
  </el-drawer>
</template>

<style scoped lang="scss">
.detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding-bottom: 12px;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--el-text-color-regular);
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.detail-messages {
  height: 62vh;
}
</style>
