<!-- 评测筛选：智能体 / 评测集 / 时间范围 -->
<template>
  <el-card shadow="never" class="mb-[12px]">
    <div class="flex items-center flex-wrap gap-2">
      <el-select
        v-model="evalStore.evalFilter.agentId"
        class="!w-[260px]"
        clearable
        filterable
        placeholder="选择智能体"
        @change="handleAgentChange"
      >
        <el-option
          v-for="item in evalStore.evalOverview"
          :key="item.agentId"
          :label="`${item.agentName}（${item.agentCode}）`"
          :value="item.agentId"
        />
      </el-select>

      <el-select
        v-model="evalStore.evalFilter.datasetId"
        class="!w-[200px]"
        clearable
        :disabled="evalStore.evalFilter.agentId == null"
        placeholder="全部评测集"
        @change="handleDatasetChange"
      >
        <el-option
          v-for="dataset in evalStore.datasets"
          :key="dataset.id"
          :label="`${dataset.name}（${datasetTypeLabel[dataset.datasetType] ?? dataset.datasetType}）`"
          :value="dataset.id"
        />
      </el-select>

      <el-date-picker
        v-model="range"
        type="daterange"
        value-format="YYYY-MM-DD"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        @change="handleRangeChange"
      />

      <el-button :loading="refreshing" @click="handleRefresh">
        <el-icon><Refresh /></el-icon>刷新
      </el-button>

      <span class="text-xs text-gray-400">
        时间范围作用于评测趋势；执行记录按智能体 + 评测集筛选
      </span>
    </div>
  </el-card>
</template>

<script lang="ts" setup>
import { Refresh } from "@element-plus/icons-vue";
import { useAdminEvalStore } from "@/store/modules/adminEval";

defineOptions({ name: "EvalAgentFilter" });

const evalStore = useAdminEvalStore();

const datasetTypeLabel: Record<string, string> = {
  dev: "开发集",
  regression: "回归集",
  heldout: "保留集",
};

const range = ref<[string, string] | null>(null);
const refreshing = ref(false);

function handleAgentChange(agentId?: number) {
  evalStore.selectAgent(agentId);
}

function handleDatasetChange() {
  evalStore.evalFilter.pageNum = 1;
  evalStore.fetchRuns();
}

function handleRangeChange(value: [string, string] | null) {
  evalStore.evalFilter.startTime = value?.[0];
  evalStore.evalFilter.endTime = value?.[1];
  evalStore.fetchTrends();
}

async function handleRefresh() {
  refreshing.value = true;
  try {
    await Promise.all([
      evalStore.fetchOverview(),
      evalStore.fetchRuns(),
      evalStore.fetchTrends(),
      evalStore.fetchJudgeStatus(),
      evalStore.fetchReviews(),
    ]);
  } finally {
    refreshing.value = false;
  }
}
</script>
