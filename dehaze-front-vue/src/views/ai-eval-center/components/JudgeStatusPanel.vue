<!-- 判分模型状态：一致性验证 / 一致率与阈值 / 漂移门禁暂停提示 -->
<template>
  <el-card v-loading="evalStore.judgeLoading" shadow="never" class="mb-[12px]">
    <template #header>
      <div class="flex justify-between items-center">
        <span>判分模型状态</span>
        <el-button
          :loading="evalStore.judgeLoading"
          size="small"
          @click="evalStore.fetchJudgeStatus()"
        >
          刷新
        </el-button>
      </div>
    </template>

    <el-empty
      v-if="!judgeStatus"
      description="暂无判分模型状态数据"
      :image-size="60"
    />

    <template v-else>
      <el-alert
        v-if="judgeStatus.driftPaused"
        class="mb-[12px]"
        type="warning"
        :closable="false"
        title="判分模型已漂移，门禁暂停：依赖判分的发布门禁判定建议暂缓执行"
      />

      <el-descriptions :column="2" size="small" border class="mb-[12px]">
        <el-descriptions-item label="一致性状态">
          <el-tag :type="stateMeta.type" size="small">{{
            stateMeta.label
          }}</el-tag>
          <span class="ml-2 text-xs text-gray-400">{{ stateMeta.desc }}</span>
        </el-descriptions-item>
        <el-descriptions-item label="复核项">
          待复核 {{ judgeStatus.reviewStats.pending }} · 已复核
          {{ judgeStatus.reviewStats.reviewed }} · 合计
          {{ judgeStatus.reviewStats.total }}
        </el-descriptions-item>
        <el-descriptions-item label="人工判定">
          一致 {{ judgeStatus.reviewStats.agreeCount }} · 不一致
          {{ judgeStatus.reviewStats.disagreeCount }}
        </el-descriptions-item>
        <el-descriptions-item label="一致率 / 阈值">
          {{ judgeStatus.reviewStats.agreementRate }}% /
          {{ judgeStatus.consistencyThreshold }}%
        </el-descriptions-item>
      </el-descriptions>

      <el-progress
        :percentage="Math.min(judgeStatus.reviewStats.agreementRate, 100)"
        :stroke-width="14"
        :status="judgeStatus.driftPaused ? 'exception' : 'success'"
      />
    </template>
  </el-card>
</template>

<script lang="ts" setup>
import { useAdminEvalStore } from "@/store/modules/adminEval";
import { CONSISTENCY_STATE_META } from "../eval-meta";

defineOptions({ name: "JudgeStatusPanel" });

const evalStore = useAdminEvalStore();

const judgeStatus = computed(() => evalStore.judgeStatus);
const stateMeta = computed(
  () =>
    CONSISTENCY_STATE_META[
      judgeStatus.value?.consistencyState ?? "insufficient_data"
    ]
);
</script>
