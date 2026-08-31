<!-- 管理端 MCP 管理页：市场（一键接入）→ Server 管理（注册表/工具/凭据/健康）→ 调用审计 -->
<template>
  <div class="app-container">
    <el-tabs v-model="mcpStore.activeTab" @tab-change="handleTabChange">
      <el-tab-pane label="MCP 市场" name="market" lazy>
        <el-card shadow="never">
          <template #header>
            <span class="font-bold">内置 MCP Server 预设</span>
          </template>
          <MarketPresetList />
        </el-card>
      </el-tab-pane>

      <el-tab-pane label="Server 管理" name="servers">
        <el-card shadow="never" class="mb-[12px]">
          <ServerTable />
        </el-card>
        <HealthPanel />
      </el-tab-pane>

      <el-tab-pane label="调用审计" name="calls" lazy>
        <el-card shadow="never">
          <McpCallAuditTable />
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <ServerFormDrawer />
  </div>
</template>

<script lang="ts" setup>
// name 需与动态路由名（由 component 路径 ai-mcp/index 推导为 AiMcp）一致，否则 keep-alive include 匹配不到
defineOptions({ name: "AiMcp" });

import HealthPanel from "./components/HealthPanel.vue";
import MarketPresetList from "./components/MarketPresetList.vue";
import McpCallAuditTable from "./components/McpCallAuditTable.vue";
import ServerFormDrawer from "./components/ServerFormDrawer.vue";
import ServerTable from "./components/ServerTable.vue";
import { useAdminMcpStore } from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

function handleTabChange(name: string | number) {
  mcpStore.switchTab(name as "market" | "servers" | "calls");
}

onMounted(() => {
  mcpStore.fetchServers();
});
</script>
