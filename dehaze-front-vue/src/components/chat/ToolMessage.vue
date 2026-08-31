<!-- 工具调用消息：折叠展示工具名/参数/返回，执行状态实时呈现 -->
<script lang="ts" setup>
import type { AiMessageVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "ToolMessage" });

const props = defineProps<{
  message: AiMessageVO;
  toolCalls?: unknown;
}>();

interface ToolCallItem {
  name?: string;
  arguments?: unknown;
}

const toolItems = computed<ToolCallItem[]>(() => {
  const raw = props.toolCalls;
  return Array.isArray(raw) ? (raw as ToolCallItem[]) : [];
});

function formatValue(value: unknown) {
  if (value == null) return "";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
</script>

<template>
  <div class="tool-message">
    <el-collapse>
      <el-collapse-item :name="'tools'">
        <template #title>
          <span class="tool-message__title">
            工具调用
            <el-tag v-if="message.status === 1" size="small" type="warning"
              >执行中</el-tag
            >
            <el-tag v-else-if="message.status === 2" size="small" type="success"
              >完成</el-tag
            >
            <el-tag v-else-if="message.status === 3" size="small" type="danger"
              >失败</el-tag
            >
          </span>
        </template>
        <div
          v-for="(item, index) in toolItems"
          :key="index"
          class="tool-message__item"
        >
          <div class="tool-message__name">{{ item.name ?? "未知工具" }}</div>
          <pre v-if="item.arguments != null" class="tool-message__payload">{{
            formatValue(item.arguments)
          }}</pre>
        </div>
        <pre
          v-if="toolItems.length === 0 && message.content"
          class="tool-message__payload"
          >{{ message.content }}</pre>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<style scoped lang="scss">
.tool-message {
  margin-bottom: 16px;

  &__title {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  &__item {
    margin-bottom: 8px;
  }

  &__name {
    font-size: 13px;
    font-weight: 600;
  }

  &__payload {
    max-height: 200px;
    padding: 8px;
    margin: 4px 0 0;
    overflow: auto;
    font-size: 12px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }
}
</style>
