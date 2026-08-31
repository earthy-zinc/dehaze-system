<!-- 发布确认弹窗：回归门禁状态提示 + 变更说明 -->
<template>
  <el-dialog
    :model-value="modelValue"
    title="发布智能体"
    width="640px"
    destroy-on-close
    @update:model-value="emit('update:modelValue', $event)"
  >
    <el-alert
      class="mb-4"
      type="info"
      :closable="false"
      title="发布通过回归集门禁后转为已发布版本；发布仅对新会话生效，进行中会话锚定创建时版本。"
    />

    <el-divider content-position="left">回归门禁状态</el-divider>
    <div v-loading="gateLoading">
      <el-descriptions v-if="latestGateRun" :column="3" size="small" border>
        <el-descriptions-item label="评测 Run"
          >#{{ latestGateRun.id }}</el-descriptions-item
        >
        <el-descriptions-item label="结果">
          <el-tag :type="gateTag(latestGateRun.status).type" size="small">
            {{ gateTag(latestGateRun.status).label }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="时间">{{
          latestGateRun.createTime ?? "-"
        }}</el-descriptions-item>
      </el-descriptions>
      <el-empty
        v-else
        description="暂无回归评测记录，发布将被门禁阻断"
        :image-size="50"
      />
      <el-button
        class="mt-2"
        size="small"
        type="primary"
        plain
        :loading="evalRunning"
        @click="handleRunEval"
      >
        执行回归评测
      </el-button>
      <el-alert
        v-if="gateResult"
        class="mt-2"
        :type="gateResult.passed ? 'success' : 'error'"
        :closable="false"
        :title="
          gateResult.passed
            ? '回归评测通过，可发布'
            : `回归评测未通过（Run #${gateResult.runId}），失败样本 ${gateResult.failedSamples?.length ?? 0} 条`
        "
      />
    </div>

    <el-divider content-position="left">变更说明</el-divider>
    <el-input
      v-model="changeNote"
      type="textarea"
      :rows="3"
      placeholder="本次发布的变更说明"
    />

    <template #footer>
      <el-button type="primary" :loading="publishing" @click="handlePublish">
        发布
      </el-button>
      <el-button @click="emit('update:modelValue', false)">取 消</el-button>
    </template>
  </el-dialog>
</template>

<script lang="ts" setup>
import { EvalRunResult } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "PublishDialog" });

const props = defineProps<{ agentId: number; modelValue: boolean }>();
const emit = defineEmits<{
  "update:modelValue": [value: boolean];
  published: [];
}>();

const agentStore = useAdminAgentStore();

const changeNote = ref("");
const gateLoading = ref(false);
const evalRunning = ref(false);
const publishing = ref(false);
/** 手动触发回归评测的门禁判定结果 */
const gateResult = ref<{
  runId?: number;
  passed?: boolean;
  scoreSummary?: Record<string, unknown> | null;
  failedSamples?: Array<Record<string, unknown>> | null;
} | null>(null);

/** 距今最近一次回归集评测 Run（发布门禁依据） */
const latestGateRun = computed<EvalRunResult | null>(() => {
  const regressionIds = new Set(
    agentStore.evalDatasets
      .filter((d) => d.datasetType === "regression")
      .map((d) => d.id)
  );
  return (
    agentStore.evalRuns.find((run) => regressionIds.has(run.datasetId)) ?? null
  );
});

function gateTag(status: number) {
  if (status === 2) return { label: "通过", type: "success" as const };
  if (status === 3) return { label: "失败", type: "danger" as const };
  return { label: "执行中", type: "warning" as const };
}

watch(
  () => props.modelValue,
  async (visible) => {
    if (!visible) return;
    changeNote.value = "";
    gateResult.value = null;
    gateLoading.value = true;
    try {
      await Promise.all([
        agentStore.fetchEvalDatasets(props.agentId),
        agentStore.fetchEvalRuns(props.agentId),
      ]);
    } finally {
      gateLoading.value = false;
    }
  }
);

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

async function handlePublish() {
  publishing.value = true;
  try {
    await agentStore.publishAgent(props.agentId, changeNote.value);
    ElMessage.success("发布成功，新会话将使用已发布版本");
    emit("update:modelValue", false);
    emit("published");
  } catch (e) {
    // 门禁未通过等业务错误由请求层提示，弹窗保持打开供修正
    if (e instanceof Error) throw e;
  } finally {
    publishing.value = false;
  }
}
</script>
