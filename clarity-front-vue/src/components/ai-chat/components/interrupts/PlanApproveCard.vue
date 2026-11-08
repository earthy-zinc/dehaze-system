<!-- 计划确认：可编辑任务列表（移除 / 上移下移 / 新增单条）。
     内容签名变化才重建编辑列表，避免父级重渲染把用户正在改的任务清空；
     批准时按实际改动构造 planEdit，无改动只回 confirm=true。 -->
<script lang="ts" setup>
import { computed, ref, watch } from "vue";
import type { Interrupt } from "clarity-sdk-js";
import type { ChatResumeFormVM } from "../../types";
import {
  appendPlanTask,
  buildPlanEdit,
  canApprovePlan,
  editablePlanOf,
  hasNewPlanTask,
  isPlanEditEmpty,
  movePlanTask,
  planIdsOf,
  planSignatureOf,
  removePlanTaskAt,
  type EditablePlanTask,
} from "../../model/interrupt";

defineOptions({ name: "PlanApproveCard" });

const props = defineProps<{
  interrupt: Interrupt;
  unavailable: boolean;
}>();

const emit = defineEmits<{
  resume: [payload: ChatResumeFormVM];
}>();

const editable = ref<EditablePlanTask[]>([]);
const originalIds = ref<string[]>([]);

const signature = computed(() => planSignatureOf(props.interrupt));

watch(
  signature,
  () => {
    editable.value = editablePlanOf(props.interrupt.plan);
    originalIds.value = planIdsOf(props.interrupt.plan);
  },
  { immediate: true }
);

const hasNew = computed(() => hasNewPlanTask(editable.value));
const canApprove = computed(() => canApprovePlan(editable.value));

function move(index: number, delta: number) {
  editable.value = movePlanTask(editable.value, index, delta);
}

function remove(index: number) {
  editable.value = removePlanTaskAt(editable.value, index);
}

function add() {
  editable.value = appendPlanTask(editable.value);
}

function approve() {
  const planEdit = buildPlanEdit(editable.value, originalIds.value);
  emit(
    "resume",
    isPlanEditEmpty(planEdit) ? { confirm: true } : { confirm: true, planEdit }
  );
}

function reject() {
  emit("resume", { confirm: false });
}
</script>

<template>
  <div class="interrupt-card__body">
    <div class="interrupt-card__desc">
      执行计划共 {{ editable.length }} 个任务：
    </div>
    <ol class="interrupt-card__plan">
      <li
        v-for="(task, index) in editable"
        :key="task.id ?? `new-${index}`"
        class="interrupt-card__task"
      >
        <el-input
          v-model="task.description"
          size="small"
          class="interrupt-card__task-input"
          placeholder="任务描述"
          :disabled="unavailable"
        />
        <el-button
          link
          size="small"
          :disabled="unavailable || index === 0"
          @click="move(index, -1)"
          >上移</el-button
        >
        <el-button
          link
          size="small"
          :disabled="unavailable || index === editable.length - 1"
          @click="move(index, 1)"
          >下移</el-button
        >
        <el-button
          link
          size="small"
          :disabled="unavailable"
          @click="remove(index)"
          >移除</el-button
        >
      </li>
    </ol>
    <el-button link size="small" :disabled="unavailable || hasNew" @click="add"
      >+ 添加任务</el-button
    >
  </div>
  <div class="interrupt-card__actions">
    <el-button
      type="primary"
      size="small"
      :disabled="unavailable || !canApprove"
      @click="approve"
    >
      批准执行
    </el-button>
    <el-button size="small" :disabled="unavailable" @click="reject"
      >取消</el-button
    >
  </div>
</template>
