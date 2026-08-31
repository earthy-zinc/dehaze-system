<!-- 版本对比：两次评测 run 的四维得分差 + 样本级差异 -->
<template>
  <div>
    <div class="flex items-center gap-2 mb-[12px]">
      <el-select
        v-model="baseRunId"
        class="!w-[320px]"
        placeholder="选择基准评测"
      >
        <el-option
          v-for="item in evalStore.baseRunOptions"
          :key="item.runId"
          :label="`#${item.runId} · ${formatTime(item.createTime)} · ${formatScore(item.totalScore)} 分`"
          :value="item.runId"
        />
      </el-select>
      <el-button
        type="primary"
        :disabled="baseRunId == null"
        :loading="evalStore.compareLoading"
        @click="handleCompare"
      >
        对比
      </el-button>
    </div>

    <el-empty
      v-if="!compare"
      description="选择基准评测后查看得分对比与样本差异"
      :image-size="60"
    />

    <template v-else>
      <div class="grid grid-cols-2 gap-[12px] mb-[12px]">
        <el-card
          v-for="snapshot in snapshots"
          :key="snapshot.title"
          shadow="never"
        >
          <div class="text-xs text-gray-400">{{ snapshot.title }}</div>
          <div class="text-xl font-semibold mt-1">
            {{ formatScore(snapshot.totalScore) }}
          </div>
          <div class="text-xs text-gray-400 mt-1">
            样本 {{ snapshot.sampleCount }} · 通过率
            {{ formatRate(snapshot.passRate) }} ·
            {{ formatTime(snapshot.createTime) }}
          </div>
        </el-card>
      </div>

      <el-table :data="dimensionRows" size="small" class="mb-[12px]">
        <el-table-column label="维度" prop="label" width="120" />
        <el-table-column
          label="基准"
          prop="baseScore"
          width="90"
          align="center"
        >
          <template #default="{ row }">{{
            formatScore(row.baseScore)
          }}</template>
        </el-table-column>
        <el-table-column
          label="本次"
          prop="currentScore"
          width="90"
          align="center"
        >
          <template #default="{ row }">
            {{ formatScore(row.currentScore) }}
          </template>
        </el-table-column>
        <el-table-column label="差值" width="100" align="center">
          <template #default="{ row }">
            <span :class="deltaClass(row.delta)">{{
              formatDelta(row.delta)
            }}</span>
          </template>
        </el-table-column>
      </el-table>

      <el-descriptions :column="4" size="small" border class="mb-[12px]">
        <el-descriptions-item label="新增样本">
          {{ compare.sampleDiff.added.length }}
        </el-descriptions-item>
        <el-descriptions-item label="移除样本">
          {{ compare.sampleDiff.removed.length }}
        </el-descriptions-item>
        <el-descriptions-item label="变化样本">
          {{ compare.sampleDiff.changed.length }}
        </el-descriptions-item>
        <el-descriptions-item label="未变样本">
          {{ compare.sampleDiff.unchangedCount }}
        </el-descriptions-item>
      </el-descriptions>

      <el-table :data="compare.sampleDiff.changed" size="small">
        <el-table-column
          label="样本ID"
          prop="sampleId"
          width="80"
          align="center"
        />
        <el-table-column label="任务目标" prop="taskGoal" min-width="200" />
        <el-table-column label="基准结果" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="row.basePassed ? 'success' : 'danger'" size="small">
              {{ row.basePassed ? "通过" : "失败" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="本次结果" width="90" align="center">
          <template #default="{ row }">
            <el-tag
              :type="row.currentPassed ? 'success' : 'danger'"
              size="small"
            >
              {{ row.currentPassed ? "通过" : "失败" }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="分值变化" width="100" align="center">
          <template #default="{ row }">
            <span :class="deltaClass(row.scoreDelta)">
              {{ formatDelta(row.scoreDelta) }}
            </span>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="两次评测样本结果一致" :image-size="60" />
        </template>
      </el-table>
    </template>
  </div>
</template>

<script lang="ts" setup>
import type { EvalRunResult } from "dehaze-sdk-js";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import {
  EVAL_DIMENSIONS,
  formatRate,
  formatScore,
  formatTime,
} from "../eval-meta";

defineOptions({ name: "VersionCompare" });

const props = defineProps<{ run: EvalRunResult }>();

const evalStore = useAdminEvalStore();

const baseRunId = ref<number>();
const compare = computed(() => evalStore.evalCompare);

const snapshots = computed(() => {
  const result = compare.value;
  if (!result) return [];
  return [
    { title: `本次评测 #${result.runId}`, ...result.current },
    { title: `基准评测 #${result.baseRunId}`, ...result.base },
  ];
});

const dimensionRows = computed(() => {
  const result = compare.value;
  if (!result) return [];
  return EVAL_DIMENSIONS.map((dimension) => ({
    label: dimension.label,
    baseScore: result.base.dimensions?.[dimension.key],
    currentScore: result.current.dimensions?.[dimension.key],
    delta: result.dimensionDiff[dimension.key],
  }));
});

function handleCompare() {
  if (baseRunId.value == null) return;
  evalStore.fetchCompare(props.run.id, baseRunId.value);
}

function formatDelta(delta?: number) {
  if (delta == null) return "-";
  return `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
}

function deltaClass(delta?: number) {
  if (delta == null) return "";
  if (delta < 0) return "text-red-500";
  return delta > 0 ? "text-green-600" : "";
}
</script>
