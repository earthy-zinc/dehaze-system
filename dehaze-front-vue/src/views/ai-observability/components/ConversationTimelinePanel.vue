<!-- 会话审计时间线视图：会话概要 + 轮次导航 + 选中轮次时序回放 -->
<script lang="ts" setup>
import { computed } from "vue";
import { Back } from "@element-plus/icons-vue";
import { storeToRefs } from "pinia";
import { useAdminAuditStore } from "@/store/modules/adminAudit";
import RoundNavigator from "./RoundNavigator.vue";
import RoundTimeline from "./RoundTimeline.vue";

defineOptions({ name: "ConversationTimelinePanel" });

const adminAuditStore = useAdminAuditStore();
const { timelineData, timelineLoading, timelineError, timelineRoundIndex } =
  storeToRefs(adminAuditStore);

const rounds = computed(() => timelineData.value?.rounds ?? []);

const activeRound = computed(
  () => rounds.value[timelineRoundIndex.value] ?? null
);

function formatTime(time?: string) {
  if (!time) return "-";
  return time.slice(0, 16).replace("T", " ");
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <template #header>
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div class="flex flex-wrap items-center gap-3">
          <span class="font-bold">
            会话时间线{{
              timelineData?.conversation.title
                ? ` · ${timelineData.conversation.title}`
                : ""
            }}
          </span>
          <span v-if="timelineData" class="timeline-meta">
            用户 {{ timelineData.conversation.userId ?? "-" }}
            <template v-if="timelineData.conversation.agentCode">
              · 智能体 {{ timelineData.conversation.agentCode }}</template
            >
            · 创建于 {{ formatTime(timelineData.conversation.createTime) }}
          </span>
        </div>
        <el-button
          :icon="Back"
          @click="adminAuditStore.closeConversationTimeline()"
        >
          返回列表
        </el-button>
      </div>
    </template>

    <el-alert
      v-if="timelineError"
      type="error"
      :closable="false"
      :title="timelineError"
    />
    <div v-else v-loading="timelineLoading">
      <el-empty
        v-if="!timelineLoading && !timelineData"
        description="暂无时间线数据（历史会话或后端读模型未就绪）"
      />
      <template v-else-if="rounds.length">
        <div class="timeline-layout">
          <div class="timeline-layout__nav">
            <RoundNavigator
              :rounds="rounds"
              :active-index="timelineRoundIndex"
              @select="adminAuditStore.selectTimelineRound"
            />
          </div>
          <div class="timeline-layout__timeline">
            <RoundTimeline v-if="activeRound" :round="activeRound" />
          </div>
        </div>
      </template>
      <el-empty
        v-else-if="timelineData"
        description="该会话暂无可回放轮次"
        :image-size="60"
      />
    </div>
  </el-card>
</template>

<style scoped lang="scss">
.timeline-meta {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.timeline-layout {
  display: flex;
  gap: 16px;

  &__nav {
    flex-shrink: 0;
    width: 280px;
    max-height: 72vh;
    overflow: auto;
  }

  &__timeline {
    flex: 1;
    min-width: 0;
    max-height: 72vh;
    overflow: auto;
  }
}

@media (width <= 992px) {
  .timeline-layout {
    flex-direction: column;

    &__nav {
      width: 100%;
    }
  }
}
</style>
