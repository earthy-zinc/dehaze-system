<!-- 计划面板：阶段/状态 + 任务清单（状态徽标）+ 修订说明 + 待批准提示（纯展示，props 进） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatPlanTaskVM, ChatPlanVM, ChatPlanStatusVM } from "../types";

defineOptions({ name: "PlanPanel" });

const props = defineProps<{
  plan: ChatPlanVM;
}>();

const statusMeta: Record<ChatPlanStatusVM, { label: string; type: string }> = {
  pending: { label: "待执行", type: "info" },
  running: { label: "执行中", type: "warning" },
  completed: { label: "已完成", type: "success" },
  failed: { label: "失败", type: "danger" },
};

const planStatus = computed(
  () => statusMeta[props.plan.status] ?? statusMeta.pending
);

function taskStatusMeta(task: ChatPlanTaskVM) {
  return task.status ? statusMeta[task.status] : null;
}
</script>

<template>
  <div class="plan-panel">
    <div class="plan-panel__header">
      <span class="plan-panel__title">执行计划</span>
      <el-tag size="small" :type="planStatus.type as any">{{
        planStatus.label
      }}</el-tag>
      <span v-if="plan.phase" class="plan-panel__phase">{{ plan.phase }}</span>
      <el-tag v-if="plan.awaitingApproval" size="small" type="warning"
        >待批准</el-tag
      >
    </div>

    <ol class="plan-panel__tasks">
      <li
        v-for="(task, index) in plan.tasks"
        :key="task.id ?? index"
        class="plan-panel__task"
      >
        <span class="plan-panel__index">{{ index + 1 }}.</span>
        <span class="plan-panel__desc">{{
          task.description ?? "（未命名任务）"
        }}</span>
        <el-tag
          v-if="taskStatusMeta(task)"
          size="small"
          :type="taskStatusMeta(task)!.type as any"
        >
          {{ taskStatusMeta(task)!.label }}
        </el-tag>
        <span v-if="task.dependsOn?.length" class="plan-panel__deps">
          依赖：{{ task.dependsOn.join("、") }}
        </span>
      </li>
    </ol>

    <div v-if="plan.revisions.length" class="plan-panel__revisions">
      <div class="plan-panel__revisions-title">修订说明</div>
      <div
        v-for="(revision, index) in plan.revisions"
        :key="index"
        class="plan-panel__revision"
      >
        <span class="plan-panel__rev-no">#{{ revision.revisionNo }}</span>
        <span>{{ revision.reason }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.plan-panel {
  padding: 10px 12px;
  margin-bottom: 8px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;

  &__header {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
  }

  &__title {
    font-size: 14px;
    font-weight: 600;
  }

  &__phase {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__tasks {
    padding-left: 0;
    margin: 0;
    list-style: none;
  }

  &__task {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 4px 0;
    font-size: 13px;
  }

  &__index {
    color: var(--el-text-color-secondary);
  }

  &__desc {
    overflow-wrap: anywhere;
  }

  &__deps {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__revisions {
    padding-top: 6px;
    margin-top: 6px;
    border-top: 1px solid var(--el-border-color-lighter);
  }

  &__revisions-title {
    margin-bottom: 4px;
    font-size: 12px;
    font-weight: 600;
    color: var(--el-text-color-secondary);
  }

  &__revision {
    display: flex;
    gap: 6px;
    font-size: 12px;
    color: var(--el-text-color-regular);
  }

  &__rev-no {
    color: var(--el-text-color-secondary);
  }
}
</style>
