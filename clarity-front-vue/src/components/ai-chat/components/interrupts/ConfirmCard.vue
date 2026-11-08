<!-- confirm 的四种内置子类型：按钮文案/类型与正文兜底由描述表驱动。
     只有 algorithm_recommend 需要专门排版（推荐 + 备选），其余共用"detail/reason + 兜底"正文。 -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ConfirmKind, Interrupt } from "clarity-sdk-js";
import type { ChatResumeFormVM } from "../../types";
import { confirmActionOf } from "../../model/interrupt";

defineOptions({ name: "ConfirmCard" });

const props = defineProps<{
  interrupt: Interrupt;
  kind: ConfirmKind;
  unavailable: boolean;
}>();

const emit = defineEmits<{
  resume: [payload: ChatResumeFormVM];
}>();

const action = computed(() => confirmActionOf(props.kind));
const recommendation = computed(() => props.interrupt.recommendation ?? null);
const alternatives = computed(() => props.interrupt.alternatives ?? []);
const detailText = computed(
  () =>
    props.interrupt.detail ?? props.interrupt.reason ?? action.value.fallback
);
// 仅 tool_permission 会把推荐模型名作为附注展示（与原实现一致）
const hasRecommendation = computed(
  () =>
    !!props.interrupt.recommendation || !!props.interrupt.alternatives?.length
);

/** 接受：算法推荐需带所选 modelId，其余子类型只需 confirm=true */
function accept() {
  const modelId =
    props.kind === "algorithm_recommend"
      ? props.interrupt.recommendation?.modelId
      : undefined;
  emit(
    "resume",
    modelId === undefined
      ? { confirm: true }
      : { confirm: true, params: { modelId } }
  );
}

function acceptAlternative(modelId: string) {
  emit("resume", { confirm: true, params: { modelId } });
}

function reject() {
  emit("resume", { confirm: false });
}
</script>

<template>
  <div class="interrupt-card__body">
    <template v-if="kind === 'algorithm_recommend'">
      <template v-if="recommendation">
        <div>
          推荐模型：<strong>{{ recommendation.modelName }}</strong>
        </div>
        <div class="interrupt-card__desc">
          {{ recommendation.reason }}
        </div>
      </template>
      <div v-if="alternatives.length" class="interrupt-card__desc">
        备选模型：
        <el-button
          v-for="alt in alternatives"
          :key="alt.modelId"
          link
          size="small"
          :disabled="unavailable"
          @click="acceptAlternative(alt.modelId)"
        >
          {{ alt.modelName }}
        </el-button>
      </div>
    </template>

    <template v-else>
      <div>{{ detailText }}</div>
      <div
        v-if="kind === 'tool_permission' && hasRecommendation"
        class="interrupt-card__desc"
      >
        {{ recommendation?.modelName }}
      </div>
    </template>
  </div>

  <div class="interrupt-card__actions">
    <el-button
      v-if="kind !== 'algorithm_recommend' || recommendation"
      :type="action.acceptType"
      size="small"
      :disabled="unavailable"
      @click="accept"
    >
      {{ action.accept }}
    </el-button>
    <el-button size="small" :disabled="unavailable" @click="reject">
      {{ action.reject }}
    </el-button>
  </div>
</template>
