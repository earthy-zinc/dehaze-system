<!-- 计费类型筛选 -->
<script lang="ts">
import type { BillingType } from "dehaze-sdk-js";

/** 计费类型选项，明细表类型列复用同一份中文映射 */
export const BILL_TYPE_OPTIONS: { value: BillingType | ""; label: string }[] = [
  { value: "", label: "全部" },
  { value: "chat", label: "对话" },
  { value: "kb_inject", label: "知识库注入" },
  { value: "asr", label: "语音识别" },
  { value: "tts", label: "语音合成" },
  { value: "tool_llm", label: "工具" },
];
</script>

<script lang="ts" setup>
import { computed } from "vue";

defineOptions({ name: "BillTypeFilter" });

const props = defineProps<{
  modelValue: BillingType | "";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: BillingType | ""): void;
}>();

const selected = computed<BillingType | "">({
  get: () => props.modelValue,
  set: (value) => emit("update:modelValue", value),
});
</script>

<template>
  <el-radio-group v-model="selected">
    <el-radio-button
      v-for="option in BILL_TYPE_OPTIONS"
      :key="option.value"
      :value="option.value"
    >
      {{ option.label }}
    </el-radio-button>
  </el-radio-group>
</template>
