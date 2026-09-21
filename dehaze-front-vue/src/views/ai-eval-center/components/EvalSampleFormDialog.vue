<!-- 评测样本表单：新建/编辑；传入 preset 可从评测失败样本预填后回流进评测集 -->
<template>
  <el-dialog
    :model-value="modelValue"
    :title="sample ? '编辑评测样本' : '新建评测样本'"
    width="720px"
    destroy-on-close
    @update:model-value="emit('update:modelValue', $event)"
  >
    <el-form ref="formRef" :model="form" :rules="rules" label-width="110px">
      <el-form-item
        v-if="props.datasetId == null"
        label="所属评测集"
        prop="datasetId"
      >
        <el-select
          v-model="form.datasetId"
          class="!w-full"
          placeholder="选择要纳入的评测集"
        >
          <el-option
            v-for="dataset in agentStore.evalDatasets"
            :key="dataset.id"
            :label="dataset.name"
            :value="dataset.id"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="任务目标" prop="taskGoal">
        <el-input
          v-model="form.taskGoal"
          type="textarea"
          :rows="2"
          maxlength="500"
          show-word-limit
          placeholder="该样本要验证的智能体任务"
        />
      </el-form-item>
      <el-form-item label="允许输入">
        <el-input
          v-model="form.allowedInput"
          placeholder="如 纯文本 / 图片URL"
        />
      </el-form-item>
      <el-form-item label="可用工具">
        <el-input
          v-model="form.tools"
          placeholder="逗号分隔，如 file_read,web_search"
        />
      </el-form-item>
      <el-form-item label="期望结果">
        <el-input
          v-model="form.expectedResult"
          type="textarea"
          :rows="3"
          placeholder="判分依据：期望的最终产出"
        />
      </el-form-item>
      <el-form-item label="期望过程">
        <el-input
          v-model="form.expectedProcess"
          type="textarea"
          :rows="2"
          placeholder="选填，期望的工具调用或推理过程"
        />
      </el-form-item>
      <el-form-item label="禁止行为">
        <el-input
          v-model="form.forbiddenBehavior"
          type="textarea"
          :rows="2"
          placeholder="选填，出现即判定不通过的行为"
        />
      </el-form-item>
      <el-form-item label="风险等级">
        <el-select v-model="form.riskLevel" class="!w-[160px]">
          <el-option
            v-for="option in RISK_LEVEL_OPTIONS"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
      </el-form-item>
    </el-form>

    <template #footer>
      <el-button type="primary" :loading="submitting" @click="handleSubmit">
        确 定
      </el-button>
      <el-button @click="emit('update:modelValue', false)">取 消</el-button>
    </template>
  </el-dialog>
</template>

<script lang="ts" setup>
import type { EvalRiskLevel, EvalSampleResult } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";
import { RISK_LEVEL_OPTIONS } from "../eval-meta";

defineOptions({ name: "EvalSampleFormDialog" });

const props = defineProps<{
  modelValue: boolean;
  agentId: number;
  /** 目标评测集，为空时由用户在表单内选择 */
  datasetId?: number | null;
  /** 编辑态样本，为空为新建 */
  sample?: EvalSampleResult | null;
  /** 新建预填（评测失败样本回流） */
  preset?: { taskGoal?: string; riskLevel?: string } | null;
}>();
const emit = defineEmits<{
  "update:modelValue": [value: boolean];
  saved: [];
}>();

const agentStore = useAdminAgentStore();

const formRef = ref(ElForm);
const submitting = ref(false);

const emptyForm = () => ({
  datasetId: undefined as number | undefined,
  taskGoal: "",
  allowedInput: "",
  tools: "",
  expectedResult: "",
  expectedProcess: "",
  forbiddenBehavior: "",
  riskLevel: "low" as EvalRiskLevel,
});

const form = reactive(emptyForm());

const rules = {
  datasetId: [
    { required: true, message: "请选择所属评测集", trigger: "change" },
  ],
  taskGoal: [{ required: true, message: "任务目标不能为空", trigger: "blur" }],
};

watch(
  () => props.modelValue,
  (visible) => {
    if (!visible) return;
    Object.assign(form, emptyForm());
    form.datasetId = props.datasetId ?? undefined;
    form.riskLevel = (props.sample?.riskLevel ?? "low") as EvalRiskLevel;
    if (props.sample) {
      Object.assign(form, {
        datasetId: props.sample.datasetId,
        taskGoal: props.sample.taskGoal,
        allowedInput: props.sample.allowedInput ?? "",
        tools: (props.sample.tools ?? []).join(","),
        expectedResult: props.sample.expectedResult ?? "",
        expectedProcess: props.sample.expectedProcess ?? "",
        forbiddenBehavior: props.sample.forbiddenBehavior ?? "",
      });
    } else if (props.preset) {
      form.taskGoal = props.preset.taskGoal ?? "";
      form.riskLevel = (props.preset.riskLevel ?? "low") as EvalRiskLevel;
    }
  }
);

async function handleSubmit() {
  await formRef.value.validate();
  const datasetId = (props.datasetId ?? form.datasetId)!;
  const tools = form.tools
    .split(/[,，]/)
    .map((tool) => tool.trim())
    .filter(Boolean);

  submitting.value = true;
  try {
    if (props.sample) {
      await agentStore.updateEvalSample(
        props.agentId,
        datasetId,
        props.sample.id,
        {
          taskGoal: form.taskGoal,
          allowedInput: form.allowedInput || null,
          tools,
          expectedProcess: form.expectedProcess || null,
          expectedResult: form.expectedResult || null,
          forbiddenBehavior: form.forbiddenBehavior || null,
          riskLevel: form.riskLevel,
        }
      );
      ElMessage.success("样本已更新");
    } else {
      await agentStore.createEvalSample(props.agentId, datasetId, {
        datasetId,
        taskGoal: form.taskGoal,
        allowedInput: form.allowedInput || null,
        tools,
        expectedProcess: form.expectedProcess || null,
        expectedResult: form.expectedResult || null,
        forbiddenBehavior: form.forbiddenBehavior || null,
        riskLevel: form.riskLevel,
      });
      ElMessage.success("样本已创建");
    }
    emit("saved");
    emit("update:modelValue", false);
  } finally {
    submitting.value = false;
  }
}
</script>
