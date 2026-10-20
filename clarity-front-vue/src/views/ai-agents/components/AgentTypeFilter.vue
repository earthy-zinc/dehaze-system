<!-- Agent 类型筛选：类型透传 AgentPageQuery.type 服务端筛选 -->
<template>
  <el-select
    v-model="agentStore.agentTypeFilter"
    class="!w-[130px]"
    @change="handleFilterChange"
  >
    <el-option label="全部类型" value="all" />
    <el-option label="普通 Agent" value="agent" />
    <el-option label="子 Agent" value="subagent" />
    <el-option label="Team 团队" value="team" />
  </el-select>
</template>

<script lang="ts" setup>
import { useAdminAgentStore } from "@/store/modules/adminAgent";

const agentStore = useAdminAgentStore();

function handleFilterChange() {
  agentStore.query.pageNum = 1;
  agentStore.fetchAgents();
}
</script>
