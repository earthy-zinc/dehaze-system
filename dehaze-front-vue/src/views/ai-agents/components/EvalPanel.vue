<!-- 评测区入口：该 Agent 的评测集/执行记录概览 + 发布门禁状态；详细管理收敛至评测中心 -->
<template>
  <div>
    <div class="mb-3 flex justify-between items-center">
      <div class="flex items-center gap-3">
        <span class="text-sm text-gray-500"> 发布门禁：{{ gateSummary }} </span>
        <el-button
          v-hasPerm="['ai:agent:manage']"
          size="small"
          type="primary"
          plain
          :loading="evalRunning"
          @click="handleRunEval"
        >
          执行回归评测
        </el-button>
      </div>
      <el-button size="small" @click="router.push('/admin/ai-eval-center')">
        前往评测中心
      </el-button>
    </div>

    <el-progress
      v-if="evalRunning"
      class="mb-3"
      :percentage="evalProgress"
      :status="
        agentStore.evalTask?.status === 'failed' ? 'exception' : undefined
      "
    />

    <el-alert
      v-if="gateResult"
      class="mb-3"
      :type="gateResult.passed ? 'success' : 'error'"
      :closable="false"
      :title="gateTitle(gateResult)"
    />

    <EvalDatasetPanel :agent-id="props.agentId" />

    <el-divider content-position="left">评测执行记录</el-divider>
    <el-table
      v-loading="agentStore.evalLoading"
      :data="agentStore.evalRuns"
      size="small"
    >
      <el-table-column label="Run" prop="id" width="80" align="center" />
      <el-table-column label="触发方式" width="100" align="center">
        <template #default="{ row }">
          {{ row.triggerType === "publish" ? "发布触发" : "手动" }}
        </template>
      </el-table-column>
      <el-table-column label="评测集" width="140">
        <template #default="{ row }">
          {{ datasetName(row.datasetId) }}
        </template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag :type="runTag(row.status).type" size="small">
            {{ runTag(row.status).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="得分摘要" min-width="220" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.scoreSummary ? JSON.stringify(row.scoreSummary) : "-" }}
        </template>
      </el-table-column>
      <el-table-column label="时间" prop="createTime" width="170" />
    </el-table>
    <pagination
      v-if="agentStore.evalRunsTotal > agentStore.evalRunsQuery.pageSize"
      v-model:limit="agentStore.evalRunsQuery.pageSize"
      v-model:page="agentStore.evalRunsQuery.pageNum"
      v-model:total="agentStore.evalRunsTotal"
      layout="prev, pager, next"
      @pagination="agentStore.fetchEvalRuns(props.agentId)"
    />
  </div>
</template>

<script lang="ts" setup>
import type { EvalRunGateResult } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";
import EvalDatasetPanel from "@/views/ai-eval-center/components/EvalDatasetPanel.vue";
import { RUN_STATUS_META } from "@/views/ai-eval-center/eval-meta";

defineOptions({ name: "EvalPanel" });

const props = defineProps<{ agentId: number }>();

const router = useRouter();
const agentStore = useAdminAgentStore();

const evalRunning = ref(false);
const evalProgress = computed(() => agentStore.evalProgress);
const gateResult = ref<EvalRunGateResult | null>(null);

/** 门禁结果文案：退化阻断与样本不足时 failedSamples 为空，需分别提示避免"失败样本 0 条"误导 */
function gateTitle(result: EvalRunGateResult) {
  if (result.passed) return `回归评测通过（Run #${result.runId}），可执行发布`;
  if (result.insufficientEval)
    return "回归集样本不足，无考题可判，发布被门禁阻断";
  if (result.degraded)
    return `回归评测未通过（Run #${result.runId}）：评分较上次完成评测退化超阈值`;
  return `回归评测未通过（Run #${result.runId}），失败样本 ${result.failedSamples.length} 条，发布被门禁阻断`;
}

const regressionDatasetIds = computed(
  () =>
    new Set(
      agentStore.evalDatasets
        .filter((d) => d.datasetType === "regression")
        .map((d) => d.id)
    )
);

/** 发布门禁状态：距今最近一次回归集 Run 的结果 */
const gateSummary = computed(() => {
  const latest = agentStore.evalRuns.find((run) =>
    regressionDatasetIds.value.has(run.datasetId)
  );
  if (!latest) return "未评测";
  if (latest.status === 2) return `通过（Run #${latest.id}）`;
  if (latest.status === 3) return `未通过（Run #${latest.id}）`;
  return `执行中（Run #${latest.id}）`;
});

function datasetName(datasetId: number) {
  return (
    agentStore.evalDatasets.find((d) => d.id === datasetId)?.name ?? datasetId
  );
}

function runTag(status: number) {
  return RUN_STATUS_META[status] ?? { label: "执行中", type: "warning" };
}

onMounted(async () => {
  agentStore.evalRunsQuery.pageNum = 1;
  await agentStore.fetchEvalRuns(props.agentId);
});

async function handleRunEval() {
  evalRunning.value = true;
  try {
    const result = await agentStore.runEval(props.agentId);
    gateResult.value = result;
    if (agentStore.evalTask?.status === "failed") {
      ElMessage.error(agentStore.evalTask.error || "评测执行失败");
      return;
    }
    if (result?.passed) {
      ElMessage.success("回归评测通过");
    } else {
      ElMessage.warning("回归评测未通过，发布将被门禁阻断");
    }
  } finally {
    evalRunning.value = false;
  }
}
</script>
