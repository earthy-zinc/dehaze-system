<!-- 智能体详情页：配置/版本/评测/A2A 四区 -->
<template>
  <div class="app-container">
    <el-card shadow="never" class="mb-3">
      <div class="flex items-center gap-3">
        <el-button @click="router.push('/admin/ai-agents')">
          <el-icon><Back /></el-icon>返回列表
        </el-button>
        <span class="text-base font-medium">{{
          agentStore.detail?.name ?? "-"
        }}</span>
        <span class="text-sm text-gray-400">{{
          agentStore.detail?.agentCode
        }}</span>
        <el-tag
          v-for="tag in agentStore.detail?.tags ?? []"
          :key="tag"
          size="small"
          type="info"
        >
          {{ tag }}
        </el-tag>
        <el-tag
          v-if="agentStore.detail"
          :type="agentStore.detail.status === 1 ? 'success' : 'info'"
          size="small"
        >
          {{ agentStore.detail.status === 1 ? "启用" : "禁用" }}
        </el-tag>
      </div>
    </el-card>

    <el-card shadow="never">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="配置" name="config">
          <AgentConfigForm :agent-id="agentId" />
        </el-tab-pane>
        <el-tab-pane label="版本" name="versions" lazy>
          <VersionPanel :agent-id="agentId" />
        </el-tab-pane>
        <el-tab-pane label="评测" name="eval" lazy>
          <EvalPanel :agent-id="agentId" />
        </el-tab-pane>
        <el-tab-pane label="A2A" name="a2a" lazy>
          <A2aPanel :agent-id="agentId" />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "AgentDetailPage" });

import { Back } from "@element-plus/icons-vue";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

const route = useRoute();
const router = useRouter();
const agentStore = useAdminAgentStore();

const agentId = computed(() => Number(route.params.id));
const activeTab = ref("config");

onMounted(() => {
  agentStore.fetchAgentDetail(agentId.value);
});
</script>
