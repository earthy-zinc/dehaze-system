<!-- 管理端列表区 Tab：公共知识库 / 私有库监控 -->
<script lang="ts" setup>
defineOptions({ name: "KbAdminTabs" });

const props = defineProps<{
  modelValue: "public" | "private";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: "public" | "private"): void;
}>();

const tabOptions = [
  { label: "公共知识库", value: "public" },
  { label: "私有库监控", value: "private" },
];
</script>

<template>
  <el-tabs
    :model-value="props.modelValue"
    @update:model-value="
      emit('update:modelValue', $event as 'public' | 'private')
    "
  >
    <el-tab-pane
      v-for="tab in tabOptions"
      :key="tab.value"
      :label="tab.label"
      :name="tab.value"
    />
  </el-tabs>
</template>
