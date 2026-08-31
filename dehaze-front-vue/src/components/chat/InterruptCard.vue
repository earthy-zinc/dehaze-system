<!-- 中断交互卡片：confirm 算法确认 / quota 配额不足 / async_wait 异步等待 / plan_approve 计划确认 -->
<script lang="ts" setup>
import type { InterruptEvent, MessageResumeForm } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "InterruptCard" });

const props = defineProps<{
  interrupt: InterruptEvent;
}>();

const emit = defineEmits<{
  confirm: [data: MessageResumeForm];
  reject: [];
  resume: [data: MessageResumeForm];
}>();

const recommendation = computed(() => props.interrupt.data.recommendation);
const plan = computed(() => props.interrupt.data.plan);

const cardMeta = computed(() => {
  switch (props.interrupt.type) {
    case "confirm":
      return { title: "需要你的确认" };
    case "quota":
      return { title: "配额不足" };
    case "async_wait":
      return { title: "异步任务处理中" };
    case "plan_approve":
      return { title: "计划待确认" };
    default:
      return { title: "推理已暂停" };
  }
});
</script>

<template>
  <div class="interrupt-card">
    <div class="interrupt-card__header">
      <span class="interrupt-card__title">{{ cardMeta.title }}</span>
    </div>

    <!-- 算法推荐确认 -->
    <template v-if="interrupt.type === 'confirm' && recommendation">
      <div class="interrupt-card__body">
        <div>
          推荐算法：<strong>{{ recommendation.algorithm.name }}</strong>
          （匹配度 {{ Math.round(recommendation.matchScore * 100) }}%）
        </div>
        <div class="interrupt-card__desc">{{ recommendation.reason }}</div>
        <div
          v-if="recommendation.alternatives?.length"
          class="interrupt-card__desc"
        >
          备选：
          {{
            recommendation.alternatives
              .map((alt) => alt.algorithm.name)
              .join("、")
          }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button
          type="primary"
          size="small"
          @click="
            emit('confirm', {
              confirm: true,
              params: { algorithmId: recommendation.algorithm.id },
            })
          "
        >
          采纳推荐
        </el-button>
        <el-button size="small" @click="emit('reject')">拒绝</el-button>
      </div>
    </template>

    <!-- 配额不足 -->
    <template v-else-if="interrupt.type === 'quota'">
      <div class="interrupt-card__body">
        <div>{{ interrupt.data.message ?? "积分配额已用尽，推理已暂停" }}</div>
        <div v-if="interrupt.data.used != null" class="interrupt-card__desc">
          本{{ interrupt.data.period === "monthly" ? "月" : "日" }}已用
          {{ interrupt.data.used }} / {{ interrupt.data.limit }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button size="small" type="primary" @click="emit('resume', {})"
          >重试</el-button
        >
      </div>
    </template>

    <!-- 异步任务等待 -->
    <template v-else-if="interrupt.type === 'async_wait'">
      <div class="interrupt-card__body">
        <div>后台任务执行中，完成后将自动继续推理</div>
        <div
          v-if="interrupt.data.estimatedDuration"
          class="interrupt-card__desc"
        >
          预计耗时 {{ Math.round(interrupt.data.estimatedDuration / 1000) }} 秒
        </div>
      </div>
    </template>

    <!-- 计划确认 -->
    <template v-else-if="interrupt.type === 'plan_approve' && plan">
      <div class="interrupt-card__body">
        <div class="interrupt-card__desc">
          执行计划共 {{ plan.tasks.length }} 个任务：
        </div>
        <ol class="interrupt-card__plan">
          <li v-for="task in plan.tasks" :key="task.id">
            {{ task.description }}
            <span v-if="task.dependsOn?.length" class="interrupt-card__deps">
              （依赖：{{ task.dependsOn.join("、") }}）
            </span>
          </li>
        </ol>
      </div>
      <div class="interrupt-card__actions">
        <el-button
          type="primary"
          size="small"
          @click="emit('confirm', { confirm: true })"
        >
          批准执行
        </el-button>
        <el-button size="small" @click="emit('reject')">取消</el-button>
      </div>
    </template>

    <!-- 其他中断类型 -->
    <template v-else>
      <div class="interrupt-card__body">
        {{ interrupt.data.message ?? "推理已暂停" }}
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" @click="emit('resume', {})"
          >继续</el-button
        >
      </div>
    </template>
  </div>
</template>

<style scoped lang="scss">
.interrupt-card {
  max-width: 92%;
  padding: 12px 14px;
  margin: 0 auto 16px;
  background-color: var(--el-color-warning-light-9);
  border: 1px solid var(--el-color-warning-light-5);
  border-radius: 8px;

  &__header {
    margin-bottom: 6px;
  }

  &__title {
    font-size: 14px;
    font-weight: 600;
  }

  &__body {
    font-size: 13px;
    line-height: 1.6;
  }

  &__desc {
    margin-top: 4px;
    color: var(--el-text-color-secondary);
  }

  &__plan {
    padding-left: 20px;
    margin: 4px 0 0;
    font-size: 13px;
  }

  &__deps {
    color: var(--el-text-color-secondary);
  }

  &__actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
}
</style>
