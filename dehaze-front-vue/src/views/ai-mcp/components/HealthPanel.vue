<!-- Server 健康探测结果：连通性与延迟，异常显著提示 -->
<template>
  <el-card v-if="result" shadow="never">
    <template #header>
      <div class="flex justify-between items-center">
        <span>健康探测 · {{ serverName }}</span>
        <el-button
          v-hasPerm="['ai:mcp:manage']"
          size="small"
          :loading="mcpStore.healthLoading"
          @click="reprobe"
        >
          <el-icon><Refresh /></el-icon>重新探测
        </el-button>
      </div>
    </template>

    <el-alert
      v-if="result.status === 'offline'"
      class="mb-3"
      type="error"
      :closable="false"
      title="Server 不可连通，其工具不会被装载到任何 Agent。请检查端点地址、凭据与网络可达性。"
    />
    <el-descriptions :column="2" size="small" border>
      <el-descriptions-item label="连通状态">
        <el-tag
          :type="result.status === 'online' ? 'success' : 'danger'"
          size="small"
          effect="dark"
        >
          {{ result.status === "online" ? "在线" : "异常" }}
        </el-tag>
      </el-descriptions-item>
      <el-descriptions-item label="延迟">
        {{ result.latencyMs != null ? `${result.latencyMs} ms` : "-" }}
      </el-descriptions-item>
    </el-descriptions>
  </el-card>
  <el-empty
    v-else
    description="在 Server 列表执行「健康探测」后，在此查看连通性与延迟"
    :image-size="60"
  />
</template>

<script lang="ts" setup>
defineOptions({ name: "HealthPanel" });

import { Refresh } from "@element-plus/icons-vue";
import { McpServerVO } from "dehaze-sdk-js";
import { useAdminMcpStore } from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

const result = computed(() => {
  const id = mcpStore.healthServerId;
  return id == null ? null : (mcpStore.health[id] ?? null);
});

const serverName = computed(() => {
  const id = mcpStore.healthServerId;
  return mcpStore.servers.find((server) => server.id === id)?.name ?? "-";
});

function reprobe() {
  const server = mcpStore.servers.find(
    (item: McpServerVO) => item.id === mcpStore.healthServerId
  );
  if (server) {
    mcpStore.probeHealth(server);
  }
}
</script>
