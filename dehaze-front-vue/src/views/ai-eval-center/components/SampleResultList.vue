<!-- 样本明细：通过/失败 + 四维得分 + 差异说明 -->
<template>
  <div>
    <el-radio-group v-model="onlyFailed" class="mb-[12px]" size="small">
      <el-radio-button :value="false"
        >全部 {{ samples.length }}</el-radio-button
      >
      <el-radio-button :value="true">仅失败 {{ failedCount }}</el-radio-button>
    </el-radio-group>

    <el-table :data="visibleSamples" size="small">
      <el-table-column type="expand">
        <template #default="{ row }">
          <div class="text-xs leading-6 pl-4">
            <div v-if="row.error" class="text-red-500">
              执行异常：{{ row.error }}
            </div>
            <div v-for="dimension in EVAL_DIMENSIONS" :key="dimension.key">
              <span class="text-gray-500">{{ dimension.label }}：</span>
              {{ row.notes[dimension.key] || "无说明" }}
            </div>
            <div class="text-gray-500">
              步数 {{ row.metrics.steps }} · 耗时
              {{ formatDuration(row.metrics.latencyMs) }} · Token
              {{ row.metrics.inputTokens + row.metrics.outputTokens }}
            </div>
          </div>
        </template>
      </el-table-column>
      <el-table-column
        label="样本ID"
        prop="sampleId"
        width="80"
        align="center"
      />
      <el-table-column label="任务目标" prop="taskGoal" min-width="200" />
      <el-table-column label="风险" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="riskMeta(row.riskLevel).type" size="small">
            {{ riskMeta(row.riskLevel).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        v-for="dimension in EVAL_DIMENSIONS"
        :key="dimension.key"
        :label="dimension.label"
        width="90"
        align="center"
      >
        <template #default="{ row }">
          {{ formatScore(row.scores[dimension.key]) }}
        </template>
      </el-table-column>
      <el-table-column label="总分" width="80" align="center">
        <template #default="{ row }">
          {{ formatScore(row.totalScore) }}
        </template>
      </el-table-column>
      <el-table-column label="结果" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="row.passed ? 'success' : 'danger'" size="small">
            {{ row.passed ? "通过" : "失败" }}
          </el-tag>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="暂无样本明细" :image-size="60" />
      </template>
    </el-table>
  </div>
</template>

<script lang="ts" setup>
import type { EvalRunResult } from "dehaze-sdk-js";
import {
  EVAL_DIMENSIONS,
  RISK_LEVEL_META,
  formatDuration,
  formatScore,
  parseSamples,
} from "../eval-meta";

defineOptions({ name: "SampleResultList" });

const props = defineProps<{ run: EvalRunResult }>();

const onlyFailed = ref(false);

const samples = computed(() => parseSamples(props.run.results));
const failedCount = computed(
  () => samples.value.filter((item) => !item.passed).length
);
const visibleSamples = computed(() =>
  onlyFailed.value
    ? samples.value.filter((item) => !item.passed)
    : samples.value
);

function riskMeta(riskLevel: string) {
  return (
    RISK_LEVEL_META[riskLevel] ?? {
      label: riskLevel,
      type: "info" as const,
    }
  );
}
</script>
