<!-- 外部 MCP 工具调用审计：谁/何时/调用什么/结果/耗时（分页） -->
<template>
  <div>
    <div class="mb-4 flex items-center gap-2">
      <el-select
        v-model="mcpStore.mcpCallQuery.serverId"
        clearable
        placeholder="按 Server 筛选"
        class="!w-[200px]"
        @change="handleQuery"
      >
        <el-option
          v-for="server in mcpStore.servers"
          :key="server.id"
          :label="server.name"
          :value="server.id"
        />
      </el-select>
      <el-input
        v-model="mcpStore.mcpCallQuery.toolName"
        clearable
        placeholder="按工具名筛选"
        style="width: 180px"
        @keyup.enter="handleQuery"
        @clear="handleQuery"
      />
      <el-button @click="handleQuery">查询</el-button>
    </div>

    <el-table
      v-loading="mcpStore.mcpCallLoading"
      :data="mcpStore.mcpCalls"
      size="small"
    >
      <el-table-column label="调用者" width="110" align="center">
        <template #default="{ row }">{{ row.userId ?? "-" }}</template>
      </el-table-column>
      <el-table-column label="调用时间" prop="createTime" width="170" />
      <el-table-column label="Server" min-width="160">
        <template #default="{ row }">
          {{ row.serverName ?? `#${row.serverId}` }}
        </template>
      </el-table-column>
      <el-table-column
        label="工具"
        prop="toolName"
        min-width="180"
        show-overflow-tooltip
      />
      <el-table-column label="结果" width="90" align="center">
        <template #default="{ row }">
          <el-tag
            :type="row.result === 'success' ? 'success' : 'danger'"
            size="small"
          >
            {{ row.result === "success" ? "成功" : "失败" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="耗时" width="100" align="center">
        <template #default="{ row }">
          {{ row.latencyMs != null ? `${row.latencyMs} ms` : "-" }}
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-if="mcpStore.mcpCallTotal > 0"
      v-model:limit="mcpStore.mcpCallQuery.pageSize"
      v-model:page="mcpStore.mcpCallQuery.pageNum"
      v-model:total="mcpStore.mcpCallTotal"
      @pagination="mcpStore.fetchMcpCalls()"
    />
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "McpCallAuditTable" });

import { useAdminMcpStore } from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

function handleQuery() {
  mcpStore.mcpCallQuery.pageNum = 1;
  mcpStore.fetchMcpCalls();
}

onMounted(() => {
  if (mcpStore.servers.length === 0) {
    mcpStore.fetchServers();
  }
  mcpStore.fetchMcpCalls();
});
</script>
