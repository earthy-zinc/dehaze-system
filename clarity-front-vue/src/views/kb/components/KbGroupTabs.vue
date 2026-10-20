<!-- 分组切换（我的/公共知识库） -->
<script lang="ts" setup>
import { computed } from "vue";

defineOptions({ name: "KbGroupTabs" });

const props = defineProps<{
  modelValue: "mine" | "public";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: "mine" | "public"): void;
}>();

const group = computed({
  get: () => props.modelValue,
  set: (value) => emit("update:modelValue", value as "mine" | "public"),
});
</script>

<template>
  <el-radio-group v-model="group">
    <el-radio-button value="mine">我的知识库</el-radio-button>
    <el-radio-button value="public">公共知识库</el-radio-button>
  </el-radio-group>
</template>
