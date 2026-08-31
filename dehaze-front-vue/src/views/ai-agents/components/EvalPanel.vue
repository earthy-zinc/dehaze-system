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

    <el-alert
      v-if="gateResult"
      class="mb-3"
      :type="gateResult.passed ? 'success' : 'error'"
      :closable="false"
      :title="
        gateResult.passed
          ? `回归评测通过（Run #${gateResult.runId}），可执行发布`
          : `回归评测未通过（Run #${gateResult.runId}），失败样本 ${gateResult.failedSamples?.length ?? 0} 条，发布被门禁阻断`
      "
    />

    <el-divider content-position="left">评测集</el-divider>
    <el-table :data="agentStore.evalDatasets" size="small">
      <el-table-column label="名称" prop="name" min-width="140" />
      <el-table-column label="类型" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="datasetTag(row.datasetType).type" size="small">
            {{ datasetTag(row.datasetType).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="描述"
        prop="description"
        min-width="200"
        show-overflow-tooltip
      />
      <el-table-column label="创建时间" prop="createTime" width="170" />
    </el-table>

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
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "EvalPanel" });

const props = defineProps<{ agentId: number }>();

const router = useRouter();
const agentStore = useAdminAgentStore();

const evalRunning = ref(false);
const gateResult = ref<{
  runId?: number;
  passed?: boolean;
  scoreSummary?: Record<string, unknown> | null;
  failedSamples?: Array<Record<string, unknown>> | null;
} | null>(null);

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

function datasetTag(type: string) {
  switch (type) {
    case "dev":
      return { label: "开发集", type: "primary" as const };
    case "regression":
      return { label: "回归集", type: "warning" as const };
    default:
      return { label: "保留集", type: "info" as const };
  }
}

function runTag(status: number) {
  if (status === 2) return { label: "通过", type: "success" as const };
  if (status === 3) return { label: "失败", type: "danger" as const };
  return { label: "执行中", type: "warning" as const };
}

onMounted(async () => {
  agentStore.evalRunsQuery.pageNum = 1;
  await Promise.all([
    agentStore.fetchEvalDatasets(props.agentId),
    agentStore.fetchEvalRuns(props.agentId),
  ]);
});

async function handleRunEval() {
  evalRunning.value = true;
  try {
    const result = await agentStore.runEval(props.agentId);
    gateResult.value = result as typeof gateResult.value;
    if (gateResult.value?.passed) {
      ElMessage.success("回归评测通过");
    } else {
      ElMessage.warning("回归评测未通过，发布将被门禁阻断");
    }
  } finally {
    evalRunning.value = false;
  }
}
</script>
