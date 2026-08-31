<!-- 评测执行记录：触发方式 / 状态 / 得分 / 耗时，分页 -->
<template>
  <el-card shadow="never" class="mb-[12px]">
    <template #header>
      <div class="flex justify-between items-center">
        <span>评测执行记录</span>
        <span class="text-xs text-gray-400"> 耗时为各样本执行耗时合计 </span>
      </div>
    </template>

    <el-table v-loading="evalStore.runsLoading" :data="rows" size="small">
      <el-table-column label="评测ID" prop="id" width="90" align="center" />
      <el-table-column label="触发方式" width="110" align="center">
        <template #default="{ row }">
          {{ TRIGGER_TYPE_META[row.triggerType] ?? row.triggerType }}
        </template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag :type="runStatusMeta(row.status).type" size="small">
            {{ runStatusMeta(row.status).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="总分" width="90" align="center">
        <template #default="{ row }">
          {{ formatScore(row.score) }}
        </template>
      </el-table-column>
      <el-table-column label="样本通过" width="100" align="center">
        <template #default="{ row }">
          {{ row.passedCount }} / {{ row.sampleCount }}
        </template>
      </el-table-column>
      <el-table-column label="通过率" width="90" align="center">
        <template #default="{ row }">
          {{ formatRate(row.passRate) }}
        </template>
      </el-table-column>
      <el-table-column label="耗时" width="90" align="center">
        <template #default="{ row }">
          {{ formatDuration(row.latencyMs) }}
        </template>
      </el-table-column>
      <el-table-column label="评测时间" width="160" align="center">
        <template #default="{ row }">
          {{ formatTime(row.createTime) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="100" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="evalStore.openDetail(row.run)"
          >
            详情
          </el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty
          :description="
            evalStore.evalFilter.agentId == null
              ? '请先选择智能体查看评测执行记录'
              : '暂无评测执行记录'
          "
          :image-size="60"
        />
      </template>
    </el-table>

    <pagination
      v-if="evalStore.evalRunsTotal > 0"
      v-model:limit="evalStore.evalFilter.pageSize"
      v-model:page="evalStore.evalFilter.pageNum"
      v-model:total="evalStore.evalRunsTotal"
      @pagination="evalStore.fetchRuns()"
    />
  </el-card>
</template>

<script lang="ts" setup>
import type { EvalRunResult } from "dehaze-sdk-js";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import {
  RUN_STATUS_META,
  TRIGGER_TYPE_META,
  averageScore,
  formatDuration,
  formatRate,
  formatScore,
  formatTime,
  parseSamples,
  parseScoreSummary,
} from "../eval-meta";

defineOptions({ name: "EvalRunTable" });

const evalStore = useAdminEvalStore();

const rows = computed(() =>
  evalStore.evalRuns.map((run: EvalRunResult) => {
    const summary = parseScoreSummary(run.scoreSummary);
    const samples = parseSamples(run.results);
    return {
      run,
      id: run.id,
      triggerType: run.triggerType,
      status: run.status,
      createTime: run.createTime,
      score: averageScore(summary.dimensions),
      sampleCount: summary.sampleCount,
      passedCount: summary.passedCount,
      passRate: summary.passRate,
      latencyMs: samples.reduce((sum, item) => sum + item.metrics.latencyMs, 0),
    };
  })
);

function runStatusMeta(status: number) {
  return (
    RUN_STATUS_META[status] ?? { label: `状态${status}`, type: "info" as const }
  );
}
</script>
