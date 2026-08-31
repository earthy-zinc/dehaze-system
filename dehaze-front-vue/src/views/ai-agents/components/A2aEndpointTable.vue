<!-- 外部 A2A 端点列表：端点地址/认证方式/状态/Agent Card 操作 -->
<template>
  <div>
    <el-table v-loading="agentStore.a2aLoading" :data="agentStore.a2aEndpoints">
      <el-table-column label="名称" prop="name" min-width="130" />
      <el-table-column
        label="端点地址"
        prop="baseUrl"
        min-width="220"
        show-overflow-tooltip
      />
      <el-table-column
        label="Agent Card"
        prop="agentCardUrl"
        min-width="180"
        show-overflow-tooltip
      >
        <template #default="{ row }">{{ row.agentCardUrl || "-" }}</template>
      </el-table-column>
      <el-table-column label="认证方式" width="130" align="center">
        <template #default="{ row }">{{
          authLabel[row.authType] ?? row.authType
        }}</template>
      </el-table-column>
      <el-table-column label="状态" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="row.status === 1 ? 'success' : 'info'" size="small">
            {{ row.status === 1 ? "启用" : "禁用" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="190" align="center">
        <template #default="{ row }">
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="emit('edit', row as EndpointResult)"
          >
            编辑
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="handleRefresh(row as EndpointResult)"
          >
            刷新 Card
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="danger"
            size="small"
            @click="handleDelete(row as EndpointResult)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>
    <pagination
      v-if="agentStore.a2aTotal > agentStore.a2aQuery.pageSize"
      v-model:limit="agentStore.a2aQuery.pageSize"
      v-model:page="agentStore.a2aQuery.pageNum"
      v-model:total="agentStore.a2aTotal"
      layout="prev, pager, next"
      @pagination="agentStore.fetchA2aEndpoints()"
    />
  </div>
</template>

<script lang="ts" setup>
import { EndpointResult } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "A2aEndpointTable" });

const emit = defineEmits<{ edit: [row: EndpointResult] }>();

const agentStore = useAdminAgentStore();

const authLabel: Record<string, string> = {
  apiKey: "API Key",
  http: "HTTP 认证",
  oauth2: "OAuth2",
  openIdConnect: "OpenID Connect",
  mutualTLS: "双向 TLS",
};

async function handleRefresh(row: EndpointResult) {
  await agentStore.manageA2aEndpoints("refresh", { id: row.id });
  ElMessage.success("Agent Card 已刷新");
}

async function handleDelete(row: EndpointResult) {
  await ElMessageBox.confirm(
    `确认删除外部端点「${row.name}」？引用该端点的远程子 Agent 关联将失效。`,
    "删除确认",
    { type: "warning" }
  );
  await agentStore.manageA2aEndpoints("delete", { id: row.id });
  ElMessage.success("已删除");
}

onMounted(() => {
  agentStore.a2aQuery.pageNum = 1;
  agentStore.fetchA2aEndpoints();
});
</script>
