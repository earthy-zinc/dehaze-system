<template>
  <div class="app-container">
    <el-card shadow="never" class="mb-[12px]">
      <template #header>
        <div class="flex justify-between items-center">
          <span>供应商健康看板</span>
          <el-button
            :loading="providerStore.healthLoading"
            @click="providerStore.fetchHealthBoard()"
          >
            <el-icon><Refresh /></el-icon>刷新
          </el-button>
        </div>
      </template>
      <el-table
        v-loading="providerStore.healthLoading"
        :data="providerStore.healthBoard"
        size="small"
      >
        <el-table-column label="供应商" prop="providerName" min-width="120" />
        <el-table-column label="健康状态" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="healthTag(row.health).type" size="small">
              {{ healthTag(row.health).label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="调用量"
          prop="callCount"
          width="90"
          align="center"
        />
        <el-table-column label="成功率" width="90" align="center">
          <template #default="{ row }">{{ row.successRate }}%</template>
        </el-table-column>
        <el-table-column
          label="限流(429)"
          prop="rate429"
          width="100"
          align="center"
        />
        <el-table-column label="P95延迟" width="100" align="center">
          <template #default="{ row }">
            {{ row.p95LatencyMs != null ? row.p95LatencyMs + "ms" : "-" }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="110" align="center">
          <template #default="{ row }">
            <el-button
              v-if="row.circuitOpen"
              v-hasPerm="['ai:model:manage']"
              link
              type="danger"
              size="small"
              @click="handleCloseCircuit(row as ProviderVO)"
            >
              解除熔断
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <el-input
            v-model="providerStore.query.keyword"
            clearable
            placeholder="名称/编码"
            style="width: 200px"
            @keyup.enter="handleQuery"
            @input="debouncedQuery"
          />
          <el-button
            v-hasPerm="['ai:model:manage']"
            type="success"
            @click="providerStore.openDrawer(null)"
          >
            <el-icon><Plus /></el-icon>新增供应商
          </el-button>
        </div>
      </template>

      <el-table
        v-loading="providerStore.loading"
        :data="providerStore.providers"
      >
        <el-table-column label="编码" prop="providerCode" min-width="110" />
        <el-table-column label="名称" prop="displayName" min-width="120" />
        <el-table-column
          label="协议类型"
          prop="protocolType"
          width="130"
          align="center"
        />
        <el-table-column label="健康状态" width="90" align="center">
          <template #default="{ row }">
            <el-tag
              v-if="row.health"
              :type="healthTag(row.health).type"
              size="small"
            >
              {{ healthTag(row.health).label }}
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-switch
              v-model="row.status"
              :active-value="1"
              :inactive-value="0"
              @change="handleStatusChange(row as ProviderVO)"
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" width="240" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="primary"
              size="small"
              @click="providerStore.openDrawer(row as ProviderVO)"
            >
              配置
            </el-button>
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="primary"
              size="small"
              :loading="testingId === row.id"
              @click="handleTestConnection(row as ProviderVO)"
            >
              连通测试
            </el-button>
            <el-button
              v-if="row.health === 'open'"
              v-hasPerm="['ai:model:manage']"
              link
              type="danger"
              size="small"
              @click="handleCloseCircuit(row as ProviderVO)"
            >
              解除熔断
            </el-button>
            <el-button
              v-hasPerm="['ai:model:manage']"
              link
              type="danger"
              size="small"
              @click="handleDelete(row as ProviderVO)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="providerStore.total > 0"
        v-model:limit="providerStore.query.pageSize"
        v-model:page="providerStore.query.pageNum"
        v-model:total="providerStore.total"
        @pagination="providerStore.fetchProviders()"
      />
    </el-card>

    <ProviderDrawer
      v-model="providerStore.drawer.visible"
      :provider="providerStore.drawer.provider"
    />
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "AiModelsProviders" });

import { Plus, Refresh } from "@element-plus/icons-vue";
import { useDebounceFn } from "@vueuse/core";
import { AiProviderAPI, ProviderHealth, ProviderVO } from "dehaze-sdk-js";
import ProviderDrawer from "./components/ProviderDrawer.vue";
import { useAdminProviderStore } from "@/store/modules/adminProvider";

const providerStore = useAdminProviderStore();
const testingId = ref<number | null>(null);

function healthTag(health?: string) {
  return (
    providerStore.healthTagMap[health as ProviderHealth] ?? {
      label: health ?? "-",
      type: "info" as const,
    }
  );
}

function handleQuery() {
  providerStore.query.pageNum = 1;
  providerStore.fetchProviders();
}
const debouncedQuery = useDebounceFn(handleQuery, 300);

async function handleTestConnection(row: ProviderVO) {
  testingId.value = row.id;
  try {
    const result = await providerStore.testConnection(row.id);
    ElMessageBox.alert(
      `<pre style="max-height:300px;overflow:auto">${JSON.stringify(result, null, 2)}</pre>`,
      `连通性测试结果 - ${row.displayName}`,
      { dangerouslyUseHTMLString: true }
    );
  } finally {
    testingId.value = null;
  }
}

async function handleCloseCircuit(row: ProviderVO) {
  await ElMessageBox.confirm(
    `确认手动解除供应商「${row.displayName}」的熔断状态？外部故障可能尚未恢复，解除后调用仍可能失败。`,
    "解除熔断",
    { type: "warning" }
  );
  await providerStore.closeCircuit(row.id);
  ElMessage.success("熔断已解除");
}

async function handleStatusChange(row: ProviderVO) {
  try {
    // 状态切换不触发连通性测试，直接更新
    await AiProviderAPI.updateProvider(row.id, { status: row.status });
    ElMessage.success(row.status === 1 ? "已启用" : "已禁用");
  } catch {
    row.status = row.status === 1 ? 0 : 1;
  }
}

async function handleDelete(row: ProviderVO) {
  await ElMessageBox.confirm(
    `确认删除供应商「${row.displayName}」？其关联的 API Key 将一并失效。`,
    "删除确认",
    { type: "warning" }
  );
  await providerStore.deleteProvider(row.id);
  ElMessage.success("删除成功");
}

onMounted(() => {
  providerStore.fetchProviders();
  providerStore.fetchHealthBoard();
});
</script>
