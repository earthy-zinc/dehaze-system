<!-- 评测报告：总分 + 四维得分 + pass@k 通过率 -->
<template>
  <div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-[12px] mb-[12px]">
      <el-card shadow="never">
        <div class="text-xs text-gray-400">总分（四维均值）</div>
        <div class="text-2xl font-semibold mt-1">
          {{ formatScore(totalScore) }}
        </div>
      </el-card>
      <el-card shadow="never">
        <div class="text-xs text-gray-400">样本数</div>
        <div class="text-2xl font-semibold mt-1">{{ summary.sampleCount }}</div>
      </el-card>
      <el-card shadow="never">
        <div class="text-xs text-gray-400">pass@k 通过率</div>
        <div class="text-2xl font-semibold mt-1">
          {{ formatRate(summary.passRate) }}
        </div>
      </el-card>
      <el-card shadow="never">
        <div class="text-xs text-gray-400">失败样本</div>
        <div
          class="text-2xl font-semibold mt-1"
          :class="summary.failedCount > 0 ? 'text-red-500' : ''"
        >
          {{ summary.failedCount }}
        </div>
      </el-card>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-[12px]">
      <div v-for="dimension in EVAL_DIMENSIONS" :key="dimension.key">
        <div class="flex justify-between text-sm mb-1">
          <span>{{ dimension.label }}</span>
          <span>{{ formatScore(summary.dimensions[dimension.key]) }}</span>
        </div>
        <el-progress
          :percentage="summary.dimensions[dimension.key] ?? 0"
          :stroke-width="12"
          :status="progressStatus(summary.dimensions[dimension.key])"
        />
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import type { EvalRunResult } from "dehaze-sdk-js";
import {
  EVAL_DIMENSIONS,
  averageScore,
  formatRate,
  formatScore,
  parseScoreSummary,
} from "../eval-meta";

defineOptions({ name: "ReportPanel" });

const props = defineProps<{ run: EvalRunResult }>();

const summary = computed(() => parseScoreSummary(props.run.scoreSummary));
const totalScore = computed(() => averageScore(summary.value.dimensions));

/** 评测器判定任一维度低于 60 分即不合格 */
function progressStatus(score?: number) {
  if (score == null) return undefined;
  if (score < 60) return "exception";
  return score >= 80 ? "success" : undefined;
}
</script>
