<!-- 中断交互卡壳层：只做路由与统一不可用判定。
     confirm 按 confirmKind 路由（算法推荐/工具授权/危险操作/写入冲突 → ConfirmCard）/
     plan_approve → PlanApproveCard（可编辑）/ quota / async_wait / 其余类型在此小模板内。
     所有决策经单一 emit resume(payload)；不可用（actionable=false、已过期、confirmKind 缺失或非法）
     时一律禁用操作且不发起 resume。 -->
<script lang="ts" setup>
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { Interrupt } from "clarity-sdk-js";
import type { ChatResumeFormVM } from "../types";
import {
  isConfirmKindInvalid,
  resolveConfirmKind,
} from "@/utils/interruptKind";
import {
  interruptTitle,
  isExpiredAt,
  parseExpiresAt,
} from "../model/interrupt";
import ConfirmCard from "./interrupts/ConfirmCard.vue";
import PlanApproveCard from "./interrupts/PlanApproveCard.vue";

defineOptions({ name: "InterruptCard" });

const props = defineProps<{
  interrupt: Interrupt;
}>();

const emit = defineEmits<{
  resume: [payload: ChatResumeFormVM];
}>();

// ===== 统一不可用判定：后端 actionable=false 为准，expiresAt 本地兜底，confirmKind 缺失/非法 =====
const now = ref(Date.now());
let expiryTimer: number | null = null;

const expiresAtMs = computed(() => parseExpiresAt(props.interrupt.expiresAt));
const kind = computed(() => resolveConfirmKind(props.interrupt));
const invalidKind = computed(() => isConfirmKindInvalid(props.interrupt));
const expired = computed(() => isExpiredAt(expiresAtMs.value, now.value));
const unavailable = computed(
  () =>
    props.interrupt.actionable === false || invalidKind.value || expired.value
);

const title = computed(() => interruptTitle(props.interrupt, kind.value));
const unavailableText = computed(() =>
  invalidKind.value ? "确认类型缺失，无法继续" : "已失效，不可继续"
);

watch(
  expiresAtMs,
  (at) => {
    if (expiryTimer !== null) {
      window.clearInterval(expiryTimer);
      expiryTimer = null;
    }
    if (at === null) return;
    expiryTimer = window.setInterval(() => {
      now.value = Date.now();
    }, 30_000);
  },
  { immediate: true }
);

onBeforeUnmount(() => {
  if (expiryTimer !== null) window.clearInterval(expiryTimer);
});

/** 统一出口：不可用时丢弃，子组件的 emit 也经此处收口 */
function emitResume(payload: ChatResumeFormVM) {
  if (unavailable.value) return;
  emit("resume", payload);
}
</script>

<template>
  <div class="interrupt-card">
    <div class="interrupt-card__header">
      <span class="interrupt-card__title">{{ title }}</span>
      <el-tag v-if="unavailable" type="info" size="small">
        {{ unavailableText }}
      </el-tag>
    </div>

    <!-- plan_approve：可编辑计划 -->
    <PlanApproveCard
      v-if="interrupt.type === 'plan_approve'"
      :interrupt="interrupt"
      :unavailable="unavailable"
      @resume="emitResume"
    />

    <!-- confirm：已识别的四种内置子类型 -->
    <ConfirmCard
      v-else-if="interrupt.type === 'confirm' && kind"
      :interrupt="interrupt"
      :kind="kind"
      :unavailable="unavailable"
      @resume="emitResume"
    />

    <!-- confirm：confirmKind 缺失或契约外取值，与缺失同样处理 -->
    <template v-else-if="interrupt.type === 'confirm'">
      <div class="interrupt-card__body">
        {{ interrupt.detail ?? interrupt.reason ?? "需要你的确认" }}
      </div>
      <div class="interrupt-card__actions">
        <el-button type="primary" size="small" disabled>继续</el-button>
        <el-button size="small" disabled>拒绝</el-button>
      </div>
    </template>

    <!-- quota：配额不足，重试续流 -->
    <template v-else-if="interrupt.type === 'quota'">
      <div class="interrupt-card__body">
        <div>
          {{ interrupt.upgradeTip ?? "积分配额已用尽，推理已暂停" }}
        </div>
        <div
          v-if="interrupt.usedDaily != null && interrupt.dailyLimit"
          class="interrupt-card__desc"
        >
          今日已用 {{ interrupt.usedDaily }} / {{ interrupt.dailyLimit }}
        </div>
        <div
          v-if="interrupt.usedMonthly != null && interrupt.monthlyLimit"
          class="interrupt-card__desc"
        >
          本月已用 {{ interrupt.usedMonthly }} / {{ interrupt.monthlyLimit }}
        </div>
      </div>
      <div class="interrupt-card__actions">
        <el-button
          type="primary"
          size="small"
          :disabled="unavailable"
          @click="emitResume({})"
          >重试</el-button
        >
      </div>
    </template>

    <!-- async_wait：纯信息展示，无动作 -->
    <template v-else-if="interrupt.type === 'async_wait'">
      <div class="interrupt-card__body">
        <div>后台任务执行中，完成后将自动继续推理</div>
        <div v-if="interrupt.estDuration" class="interrupt-card__desc">
          预计耗时 {{ interrupt.estDuration }}
        </div>
        <div v-if="interrupt.imageCount" class="interrupt-card__desc">
          共 {{ interrupt.imageCount }} 张图片
        </div>
      </div>
    </template>

    <!-- 契约外中断类型：给一个兜底续流入口 -->
    <template v-else>
      <div class="interrupt-card__body">
        {{ interrupt.reason ?? interrupt.detail ?? "推理已暂停" }}
      </div>
      <div class="interrupt-card__actions">
        <el-button
          type="primary"
          size="small"
          :disabled="unavailable"
          @click="emitResume({})"
          >继续</el-button
        >
      </div>
    </template>
  </div>
</template>

<style lang="scss" src="../styles/interrupt-card.scss"></style>
