<!-- 已接入 MCP Server 列表：协议/端点/鉴权/启用/健康/工具数 + 注册与行操作 -->
<template>
  <div>
    <div class="mb-4 flex justify-between items-center">
      <div class="flex items-center gap-2">
        <el-input
          v-model="mcpStore.serverQuery.keyword"
          clearable
          placeholder="名称/描述"
          style="width: 180px"
          @keyup.enter="handleQuery"
          @clear="handleQuery"
        />
        <el-select
          v-model="mcpStore.serverQuery.status"
          clearable
          class="!w-[120px]"
          placeholder="状态"
          @change="handleQuery"
        >
          <el-option label="启用" :value="1" />
          <el-option label="禁用" :value="0" />
        </el-select>
      </div>
      <div class="flex items-center gap-2">
        <el-button @click="mcpStore.fetchServers()">
          <el-icon><Refresh /></el-icon>刷新
        </el-button>
        <el-button
          v-hasPerm="['ai:mcp:manage']"
          type="success"
          @click="mcpStore.openCreateDrawer()"
        >
          <el-icon><Plus /></el-icon>注册 Server
        </el-button>
      </div>
    </div>

    <el-table v-loading="mcpStore.serverLoading" :data="mcpStore.servers">
      <el-table-column label="Server" min-width="180">
        <template #default="{ row }">
          <div class="font-bold">{{ row.name }}</div>
          <div class="text-xs text-gray-400">{{ row.description ?? "-" }}</div>
        </template>
      </el-table-column>
      <el-table-column label="传输协议" width="150">
        <template #default="{ row }">
          <el-tag size="small">
            {{ MCP_PROTOCOL_LABELS[row.protocolType] ?? row.protocolType }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="端点"
        prop="endpoint"
        min-width="200"
        show-overflow-tooltip
      >
        <template #default="{ row }">{{ row.endpoint ?? "-" }}</template>
      </el-table-column>
      <el-table-column label="鉴权方式" width="110" align="center">
        <template #default="{ row }">
          {{ MCP_AUTH_LABELS[row.authType ?? "none"] ?? row.authType ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="启用状态" width="100" align="center">
        <template #default="{ row }">
          <el-switch
            v-hasPerm="['ai:mcp:manage']"
            :model-value="row.status"
            :active-value="1"
            :inactive-value="0"
            :loading="statusSwitchingId === row.id"
            @change="handleStatusChange(row as McpServerVO, $event as 0 | 1)"
          />
        </template>
      </el-table-column>
      <el-table-column label="健康状态" width="120" align="center">
        <template #default="{ row }">
          <el-tag
            :type="healthTag(row as McpServerVO).type"
            size="small"
            effect="dark"
          >
            {{ healthTag(row as McpServerVO).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="工具数"
        prop="toolCount"
        width="80"
        align="center"
      />
      <el-table-column label="操作" width="300" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="mcpStore.openServerDrawer(row as McpServerVO, 'config')"
          >
            编辑
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            @click="mcpStore.openServerDrawer(row as McpServerVO, 'tools')"
          >
            工具与命名空间
          </el-button>
          <el-button
            link
            type="primary"
            size="small"
            @click="
              mcpStore.openServerDrawer(row as McpServerVO, 'credentials')
            "
          >
            凭据
          </el-button>
          <el-button
            v-hasPerm="['ai:mcp:manage']"
            link
            type="primary"
            size="small"
            :loading="
              mcpStore.healthLoading && mcpStore.healthServerId === row.id
            "
            @click="mcpStore.probeHealth(row as McpServerVO)"
          >
            健康探测
          </el-button>
          <el-button
            v-hasPerm="['ai:mcp:manage']"
            link
            type="danger"
            size="small"
            @click="handleDelete(row as McpServerVO)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-if="mcpStore.serverTotal > 0"
      v-model:limit="mcpStore.serverQuery.pageSize"
      v-model:page="mcpStore.serverQuery.pageNum"
      v-model:total="mcpStore.serverTotal"
      @pagination="mcpStore.fetchServers()"
    />
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "ServerTable" });

import { Plus, Refresh } from "@element-plus/icons-vue";
import { McpServerVO } from "dehaze-sdk-js";
import {
  MCP_AUTH_LABELS,
  MCP_PROTOCOL_LABELS,
  useAdminMcpStore,
} from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

const statusSwitchingId = ref<number | null>(null);

function handleQuery() {
  mcpStore.serverQuery.pageNum = 1;
  mcpStore.fetchServers();
}

function healthTag(row: McpServerVO) {
  if (row.health === "online") {
    return { label: "在线", type: "success" as const };
  }
  if (row.health === "offline") {
    return { label: "异常", type: "danger" as const };
  }
  return { label: "未探测", type: "info" as const };
}

async function handleStatusChange(row: McpServerVO, status: 0 | 1) {
  try {
    await ElMessageBox.confirm(
      `确认${status === 1 ? "启用" : "禁用"} Server「${row.name}」？禁用后不再参与命名空间预筛选，Agent 无法调用其工具。`,
      "状态变更",
      { type: "warning" }
    );
  } catch {
    // 未使用 v-model，取消时不回滚即可保持原状态
    return;
  }
  statusSwitchingId.value = row.id;
  try {
    await mcpStore.switchServerStatus(row, status);
    ElMessage.success(status === 1 ? "已启用" : "已禁用");
  } finally {
    statusSwitchingId.value = null;
  }
}

async function handleDelete(server: McpServerVO) {
  try {
    await ElMessageBox.confirm(
      `确认删除 Server「${server.name}」？若已被 Agent 关联需先解绑，否则删除会被拒绝。`,
      "删除确认",
      { type: "warning", confirmButtonText: "确定", cancelButtonText: "取消" }
    );
  } catch {
    return;
  }
  await mcpStore.deleteServer(server);
  ElMessage.success("Server 已删除");
}
</script>
