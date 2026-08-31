<!-- 工具与命名空间面板：工具清单预览 + 命名空间分组（供 Agent 关联，最小权限） -->
<template>
  <div>
    <div class="mb-2 flex justify-between items-center">
      <span class="font-bold">工具清单（{{ mcpStore.tools.length }}）</span>
      <el-button size="small" @click="refresh">
        <el-icon><Refresh /></el-icon>重新拉取
      </el-button>
    </div>
    <ToolTable
      :tools="mcpStore.tools"
      :loading="mcpStore.toolsLoading"
      :server-id="props.serverId"
    />

    <el-divider content-position="left">命名空间（工具分组）</el-divider>
    <NamespaceConfigPanel
      :tools="mcpStore.tools"
      :namespaces="mcpStore.namespaces"
      :saving="mcpStore.namespacesLoading"
      @save="handleSave"
    />
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "ToolNamespacePanel" });

import { Refresh } from "@element-plus/icons-vue";
import { McpNamespaceVO } from "dehaze-sdk-js";
import NamespaceConfigPanel from "./NamespaceConfigPanel.vue";
import ToolTable from "./ToolTable.vue";
import { useAdminMcpStore } from "@/store/modules/adminMcp";

const props = defineProps<{ serverId: number }>();

const mcpStore = useAdminMcpStore();

function refresh() {
  mcpStore.fetchTools(props.serverId);
}

async function handleSave(namespaces: McpNamespaceVO[]) {
  await mcpStore.configureNamespaces(props.serverId, namespaces);
}
</script>
