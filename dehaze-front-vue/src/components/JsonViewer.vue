<!-- JSON 原始报文查看器：按节点折叠、语法高亮、复制、下载；data 为空时降级提示（审计时间线 raw 报文回放） -->
<script lang="ts" setup>
import {
  computed,
  defineComponent,
  h,
  ref,
  type PropType,
  type Ref,
  type VNode,
} from "vue";
import { CopyDocument, Download } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";

defineOptions({ name: "JsonViewer" });

const props = withDefaults(
  defineProps<{
    /** 任意 JSON 值；null/undefined 显示降级提示 */
    data?: unknown;
    /** 空态提示文案 */
    emptyText?: string;
    /** 下载文件名 */
    filename?: string;
    /** 默认展开层级（超过则折叠，0 表示全部折叠） */
    expandDepth?: number;
  }>(),
  { emptyText: "无原始报文记录", filename: "raw.json", expandDepth: 1 }
);

const copied = ref(false);

const isEmpty = computed(() => props.data === null || props.data === undefined);

const prettyText = computed(() =>
  isEmpty.value ? "" : JSON.stringify(props.data, null, 2)
);

async function handleCopy() {
  try {
    await navigator.clipboard.writeText(prettyText.value);
    copied.value = true;
    ElMessage.success("已复制");
    setTimeout(() => (copied.value = false), 1500);
  } catch {
    ElMessage.error("复制失败");
  }
}

function handleDownload() {
  const blob = new Blob([prettyText.value], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = props.filename;
  link.click();
  URL.revokeObjectURL(url);
}

// ==================== 递归节点渲染 ====================

const VALUE_COLOR: Record<string, string> = {
  string: "var(--el-color-success)",
  number: "var(--el-color-primary)",
  boolean: "var(--el-color-danger)",
  null: "var(--el-text-color-secondary)",
};

interface JsonNodeProps {
  name: string;
  value: unknown;
  depth: number;
  expandDepth: number;
  isLast?: boolean;
}

function formatPrimitive(value: unknown): string {
  if (typeof value === "string") return JSON.stringify(value);
  if (value === null || value === undefined) return "null";
  return String(value);
}

/** 对象/数组按 key 折叠，基础值按类型着色（函数声明提升，JsonNode 渲染时已初始化，避免递归类型循环） */
function renderJsonNode(
  nodeProps: JsonNodeProps,
  expanded: Ref<boolean>,
  toggle: () => void
): VNode {
  const { name, value, depth } = nodeProps;
  const keyLabel = h("span", { class: "json-node__key" }, name);

  const isLeaf = value === null || typeof value !== "object";
  if (isLeaf) {
    const type = value === null || value === undefined ? "null" : typeof value;
    return h("div", { class: "json-node json-node--leaf" }, [
      keyLabel,
      h("span", { class: "json-node__colon" }, ": "),
      h(
        "span",
        { class: "json-node__value", style: { color: VALUE_COLOR[type] } },
        formatPrimitive(value)
      ),
      nodeProps.isLast ? null : h("span", { class: "json-node__colon" }, ","),
    ]);
  }

  const entries = Array.isArray(value)
    ? value.map((item, index) => ({
        key: String(index),
        value: item,
        isLast: index === value.length - 1,
      }))
    : Object.entries(value as Record<string, unknown>).map(
        ([key, entryValue], index, allEntries) => ({
          key,
          value: entryValue,
          isLast: index === allEntries.length - 1,
        })
      );

  const openBracket = Array.isArray(value) ? "[" : "{";
  const closeBracket = Array.isArray(value) ? "]" : "}";
  const children = expanded.value
    ? entries.map((entry) =>
        h(JsonNode, {
          key: entry.key,
          name: entry.key,
          value: entry.value,
          depth: depth + 1,
          expandDepth: nodeProps.expandDepth,
          isLast: entry.isLast,
        })
      )
    : [];

  return h("div", { class: "json-node json-node--container" }, [
    h(
      "span",
      { class: "json-node__toggle", onClick: toggle },
      `${expanded.value ? "▾" : "▸"} `
    ),
    keyLabel,
    h("span", { class: "json-node__colon" }, `: ${openBracket}`),
    expanded.value
      ? h("div", { class: "json-node__children" }, [
          ...children,
          h("div", { class: "json-node__close" }, closeBracket),
        ])
      : h(
          "span",
          { class: "json-node__collapsed", onClick: toggle },
          `…${closeBracket}`
        ),
  ]);
}

const JsonNode = defineComponent({
  name: "JsonNode",
  props: {
    name: { type: String, required: true },
    value: { type: null as unknown as PropType<unknown>, required: true },
    depth: { type: Number, required: true },
    expandDepth: { type: Number, required: true },
    isLast: { type: Boolean, default: false },
  },
  setup(nodeProps) {
    const expanded = ref(nodeProps.depth < nodeProps.expandDepth);
    const toggle = () => {
      expanded.value = !expanded.value;
    };
    return (): VNode => renderJsonNode(nodeProps, expanded, toggle);
  },
});
</script>

<template>
  <div v-if="isEmpty" class="json-viewer json-viewer--empty">
    <el-alert :title="emptyText" type="info" :closable="false" />
  </div>
  <div v-else class="json-viewer">
    <div class="json-viewer__toolbar">
      <el-button link size="small" :icon="CopyDocument" @click="handleCopy">
        {{ copied ? "已复制" : "复制" }}
      </el-button>
      <el-button link size="small" :icon="Download" @click="handleDownload">
        下载
      </el-button>
    </div>
    <div class="json-viewer__body">
      <JsonNode
        :value="data"
        :name="'root'"
        :depth="0"
        :expand-depth="expandDepth"
      />
    </div>
  </div>
</template>

<style scoped lang="scss">
.json-viewer {
  background-color: var(--el-fill-color-lighter);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;

  &__toolbar {
    display: flex;
    gap: 4px;
    justify-content: flex-end;
    padding: 2px 8px;
    border-bottom: 1px solid var(--el-border-color-lighter);
  }

  &__body {
    max-height: 320px;
    padding: 8px 12px;
    overflow: auto;
    font-family: var(--el-font-family, monospace);
    font-size: 12px;
    line-height: 1.6;
  }

  &--empty {
    padding: 8px;
    background: transparent;
    border: none;
  }
}

.json-node {
  word-break: break-all;
  white-space: pre-wrap;

  &__key {
    color: var(--el-color-danger);
  }

  &__toggle,
  &__collapsed {
    color: var(--el-text-color-secondary);
    cursor: pointer;
    user-select: none;

    &:hover {
      color: var(--el-color-primary);
    }
  }

  &__children {
    padding-left: 16px;
    margin-left: 4px;
    border-left: 1px dashed var(--el-border-color);
  }
}
</style>
