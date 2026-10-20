<!-- 版本历史列表：版本号/状态/变更说明/操作（快照/回滚） -->
<template>
  <div>
    <el-table
      v-loading="agentStore.versionsLoading"
      :data="agentStore.versions"
    >
      <el-table-column label="版本号" width="90" align="center">
        <template #default="{ row }">v{{ row.versionNo }}</template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag :type="row.status === 2 ? 'success' : 'info'" size="small">
            {{ row.status === 2 ? "已发布" : "草稿" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="变更说明"
        prop="changeNote"
        min-width="180"
        show-overflow-tooltip
      >
        <template #default="{ row }">{{ row.changeNote || "-" }}</template>
      </el-table-column>
      <el-table-column
        label="变更人"
        prop="operatorId"
        width="90"
        align="center"
      />
      <el-table-column label="时间" prop="createTime" width="170" />
      <el-table-column label="操作" width="150" align="center">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="emit('view-snapshot', row.versionNo)"
          >
            快照
          </el-button>
          <el-button
            v-if="row.status === 2"
            v-hasPerm="['ai:agent:manage']"
            link
            type="warning"
            size="small"
            @click="handleRollback(row.versionNo)"
          >
            回滚
          </el-button>
        </template>
      </el-table-column>
    </el-table>
    <pagination
      v-if="agentStore.versionsTotal > agentStore.versionsQuery.pageSize"
      v-model:limit="agentStore.versionsQuery.pageSize"
      v-model:page="agentStore.versionsQuery.pageNum"
      v-model:total="agentStore.versionsTotal"
      layout="prev, pager, next"
      @pagination="agentStore.fetchVersions(props.agentId)"
    />
  </div>
</template>

<script lang="ts" setup>
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "VersionList" });

const props = defineProps<{ agentId: number }>();
const emit = defineEmits<{ "view-snapshot": [versionNo: number] }>();

const agentStore = useAdminAgentStore();

async function handleRollback(versionNo: number) {
  await ElMessageBox.confirm(
    `确认回滚到 v${versionNo}？将生成新版本号，完整历史保留；仅对新会话生效，进行中会话锚定创建时版本。`,
    "回滚确认",
    { type: "warning" }
  );
  await agentStore.rollbackVersion(props.agentId, versionNo);
  ElMessage.success(`已回滚，生成新草稿版本`);
}
</script>
