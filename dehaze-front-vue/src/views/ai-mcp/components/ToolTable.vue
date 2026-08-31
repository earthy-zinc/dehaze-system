<!-- Server 工具清单：工具名/描述/参数 schema 概要 + 试运行（验证连通性与参数） -->
<template>
  <el-table v-loading="loading" :data="tools" size="small" max-height="320">
    <el-table-column label="工具名" prop="name" min-width="160" />
    <el-table-column label="描述" min-width="200" show-overflow-tooltip>
      <template #default="{ row }">{{ row.description ?? "-" }}</template>
    </el-table-column>
    <el-table-column label="参数 schema" min-width="200">
      <template #default="{ row }">
        <span class="text-xs">{{ schemaSummary(row.inputSchema) }}</span>
      </template>
    </el-table-column>
    <el-table-column label="操作" width="90" align="center">
      <template #default="{ row }">
        <el-button
          v-hasPerm="['ai:mcp:manage']"
          link
          type="primary"
          size="small"
          @click="openTest(row)"
        >
          试运行
        </el-button>
      </template>
    </el-table-column>
    <template #empty>
      <el-empty
        description="未拉取到工具，请确认 Server 可连通后重试"
        :image-size="60"
      />
    </template>
  </el-table>

  <el-dialog v-model="testVisible" title="试运行 MCP 工具" width="640px" append-to-body>
    <el-form label-width="80px">
      <el-form-item label="工具名">
        <el-input v-model="form.toolName" readonly />
      </el-form-item>
      <el-form-item label="参数 JSON">
        <el-input
          v-model="form.argumentsText"
          class="args-input"
          type="textarea"
          :rows="8"
          placeholder='{"key": "value"}（无参数可留空）'
        />
      </el-form-item>
    </el-form>
    <el-alert
      v-if="result"
      class="mb-1"
      :type="result.success ? 'success' : 'error'"
      :closable="false"
      :title="
        result.success
          ? '调用成功，耗时 ' + (result.latencyMs ?? '-') + 'ms'
          : '调用失败'
      "
    >
      <pre class="result-pre">{{ result.success ? result.result : result.error }}</pre>
    </el-alert>
    <template #footer>
      <el-button @click="testVisible = false">关闭</el-button>
      <el-button type="primary" :loading="testing" @click="run">调用</el-button>
    </template>
  </el-dialog>
</template>

<script lang="ts" setup>
defineOptions({ name: "ToolTable" });

import { McpToolTestResult, McpToolVO } from "dehaze-sdk-js";
import { useAdminMcpStore } from "@/store/modules/adminMcp";

const props = defineProps<{ tools: McpToolVO[]; loading?: boolean; serverId: number }>();

const mcpStore = useAdminMcpStore();

const testVisible = ref(false);
const testing = ref(false);
const form = reactive<{ toolName: string; argumentsText: string }>({
  toolName: "",
  argumentsText: "",
});
const result = ref<McpToolTestResult | null>(null);

function openTest(tool: McpToolVO) {
  form.toolName = tool.name;
  form.argumentsText = "";
  result.value = null;
  testVisible.value = true;
}

async function run() {
  let args: Record<string, unknown> = {};
  const text = form.argumentsText.trim();
  if (text) {
    try {
      args = JSON.parse(text);
    } catch {
      ElMessage.error("参数不是合法 JSON");
      return;
    }
  }
  testing.value = true;
  try {
    result.value = await mcpStore.testTool(props.serverId, form.toolName, args);
  } finally {
    testing.value = false;
  }
}

/** 参数 schema 概要：字段名:类型（required 加 *） */
function schemaSummary(schema?: Record<string, unknown>) {
  if (!schema) return "-";
  const properties =
    (schema.properties as Record<string, { type?: string }> | undefined) ?? {};
  const required = (schema.required as string[] | undefined) ?? [];
  const parts = Object.entries(properties).map(
    ([key, value]) =>
      `${key}${required.includes(key) ? "*" : ""}: ${value?.type ?? "any"}`
  );
  return parts.length > 0 ? parts.join("，") : "-";
}
</script>

<style lang="scss" scoped>
.args-input :deep(textarea) {
  font-family: Menlo, Consolas, monospace;
}

.result-pre {
  margin: 0;
  max-height: 200px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 12px;
}
</style>
