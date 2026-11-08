<!-- 会话详情抽屉：审计元信息 + 只读消息浏览（MessageList scope=admin，经宿主绑定层映射）+ 链路下钻跳转可观测中心 -->
<script lang="ts" setup>
import { storeToRefs } from "pinia";
import { ref } from "vue";
import { useRouter } from "vue-router";
import { AiConversationAPI, type Artifact } from "clarity-sdk-js";
import { MessageList } from "@/components/ai-chat";
import { useAdminAuditStore } from "@/store/modules/adminAudit";
import ArtifactDetailDialog from "@/components/ArtifactDetailDialog.vue";

defineOptions({ name: "ConversationDetailDrawer" });

const adminAuditStore = useAdminAuditStore();
const {
  detailVisible,
  detailConversation,
  detailMessages,
  detailHasMore,
  detailLoading,
  detailError,
} = storeToRefs(adminAuditStore);

const router = useRouter();

// 产物详情弹窗：与用户端对话页共用同一只读组件（点击产物卡片 → 页面层拉取 getArtifactDetail 展示）
const artifactDialogRef = ref<InstanceType<typeof ArtifactDetailDialog>>();

function onOpenArtifact(artifact: Artifact) {
  void artifactDialogRef.value?.open(artifact);
}

/** 链路查看统一收敛到可观测中心会话时间线（携 conversationId 定位） */
function jumpToTimeline() {
  if (!detailConversation.value) return;
  detailVisible.value = false;
  router.push({
    path: "/admin/ai-observability",
    query: { conversationId: String(detailConversation.value.id) },
  });
}

// 只读审计：消息原样透传，滚动/流式语义显式固定为审计口径（不读用户端 store 的流式态）；
// 历史分页沿用 adminAudit.loadMoreDetailMessages；产物按需拉取，不引入任何内容操作入口
const detailArtifacts = ref<Record<number, Artifact[]>>({});
const requestedArtifacts = new Set<number>();
async function loadArtifacts(messageId: number) {
  if (messageId <= 0 || requestedArtifacts.has(messageId)) return;
  requestedArtifacts.add(messageId);
  try {
    const list = await AiConversationAPI.getMessageArtifacts(messageId);
    detailArtifacts.value = { ...detailArtifacts.value, [messageId]: list };
  } catch {
    requestedArtifacts.delete(messageId);
  }
}
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
      <MessageList
        scope="admin"
        :messages="detailMessages"
        :artifacts="detailArtifacts"
        :scroll-follow-enabled="true"
        :has-more-history="detailHasMore"
        :loading-more="detailLoading"
        @trace="jumpToTimeline"
        @reach-top="adminAuditStore.loadMoreDetailMessages()"
        @load-artifacts="(id) => void loadArtifacts(id)"
        @open-artifact="onOpenArtifact"
      />
    </div>

    <ArtifactDetailDialog ref="artifactDialogRef" />
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
