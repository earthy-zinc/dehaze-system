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
      <el-progress v-if="evalRunning" class="mt-2" :percentage="evalProgress" />
      <el-alert
        v-if="gateResult"
        class="mt-2"
        :type="gateResult.passed ? 'success' : 'error'"
        :closable="false"
        :title="gateTitle(gateResult)"
      />
    </div>

    <el-divider content-position="left">发布豁免（判分漂移）</el-divider>
    <el-alert
      v-if="driftPaused"
      class="mb-3"
      type="warning"
      :closable="false"
      title="判分模型一致率低于阈值，依赖判分的门禁已暂停。确认本次变更与判分无关时，可勾选豁免并强制发布。"
    />
    <el-form label-width="140px">
      <el-form-item label="豁免强制发布">
        <el-switch v-model="force" />
        <span class="ml-2 text-xs text-gray-400">
          勾选后跳过判分漂移门禁，风险由发布人承担
        </span>
      </el-form-item>
      <el-form-item v-if="force" label="豁免原因" required>
        <el-input
          v-model="forceReason"
          type="textarea"
          :rows="2"
          maxlength="200"
          show-word-limit
          placeholder="必填，随变更说明一并留档（≤200 字）"
        />
      </el-form-item>
    </el-form>

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
import { EvalRunGateResult, EvalRunResult } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import { RUN_STATUS_META } from "@/views/ai-eval-center/eval-meta";

defineOptions({ name: "PublishDialog" });

const props = defineProps<{ agentId: number; modelValue: boolean }>();
const emit = defineEmits<{
  "update:modelValue": [value: boolean];
  published: [];
}>();

const agentStore = useAdminAgentStore();
const evalStore = useAdminEvalStore();

const changeNote = ref("");
/** 判分漂移豁免：勾选后需填写原因，随变更说明留档 */
const force = ref(false);
const forceReason = ref("");
const driftPaused = computed(() => evalStore.judgeStatus?.driftPaused ?? false);
const evalProgress = computed(() => agentStore.evalProgress);
const gateLoading = ref(false);
const evalRunning = ref(false);
const publishing = ref(false);
/** 手动触发回归评测的门禁判定结果 */
const gateResult = ref<EvalRunGateResult | null>(null);

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
  return RUN_STATUS_META[status] ?? { label: "执行中", type: "warning" };
}

/** 门禁结果文案：退化阻断与样本不足时 failedSamples 为空，需分别提示避免"失败样本 0 条"误导 */
function gateTitle(result: EvalRunGateResult): string {
  if (result.passed) return "回归评测通过，可发布";
  if (result.insufficientEval)
    return "回归集样本不足，无考题可判，发布被门禁阻断";
  if (result.degraded)
    return `回归评测未通过（Run #${result.runId}）：评分较上次完成评测退化超阈值，请检查本次变更`;
  return `回归评测未通过（Run #${result.runId}），失败样本 ${result.failedSamples.length} 条`;
}

watch(
  () => props.modelValue,
  async (visible) => {
    if (!visible) return;
    changeNote.value = "";
    force.value = false;
    forceReason.value = "";
    gateResult.value = null;
    gateLoading.value = true;
    try {
      await Promise.all([
        agentStore.fetchEvalDatasets(props.agentId),
        agentStore.fetchEvalRuns(props.agentId),
        evalStore.fetchJudgeStatus(),
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
    gateResult.value = result;
    if (agentStore.evalTask?.status === "failed") {
      ElMessage.error(agentStore.evalTask.error || "评测执行失败");
      return;
    }
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
  if (force.value && !forceReason.value.trim()) {
    ElMessage.warning("勾选豁免强制发布后，豁免原因必填");
    return;
  }
  if (force.value) {
    try {
      await ElMessageBox.confirm(
        "强制发布将跳过判分漂移门禁：本次变更未经过可信的回归判分，可能将质量退化直接带到线上。确认继续？",
        "豁免确认",
        { type: "warning" }
      );
    } catch {
      return;
    }
  }
  // 豁免原因随变更说明留档，后端仅存单一 change_note 字段
  const note = force.value
    ? `${changeNote.value}\n【判分漂移豁免】${forceReason.value.trim()}`
    : changeNote.value;

  publishing.value = true;
  try {
    await agentStore.publishAgent(props.agentId, note, force.value);
    ElMessage.success("发布成功，新会话将使用已发布版本");
    emit("update:modelValue", false);
    emit("published");
  } finally {
    // 门禁未通过等业务错误由请求层按后端 msg 提示（见 utils/request.ts onBizError），弹窗保持打开供修正
    publishing.value = false;
  }
}
</script>
