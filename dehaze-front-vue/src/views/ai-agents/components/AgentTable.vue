<!-- Agent 列表表格：类型标签/推理范式/模型/状态/操作 -->
<template>
  <div>
    <el-table v-loading="agentStore.loading" :data="agentStore.agents">
      <el-table-column label="名称" min-width="150">
        <template #default="{ row }">
          <div>{{ row.name }}</div>
          <div class="text-xs text-gray-400">{{ row.agentCode }}</div>
          <div v-if="(row as AgentListItem).tags?.length" class="mt-1">
            <el-tag
              v-for="tag in (row as AgentListItem).tags"
              :key="tag"
              size="small"
              type="info"
              class="mr-1"
            >
              {{ tag }}
            </el-tag>
          </div>
        </template>
      </el-table-column>
      <el-table-column label="类型" width="160" align="center">
        <template #default="{ row }">
          <el-tag
            v-for="tag in typeTags(row as AgentListItem)"
            :key="tag.label"
            :type="tag.type"
            size="small"
            class="mr-1"
          >
            {{ tag.label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="推理范式" width="110" align="center">
        <template #default="{ row }">
          {{ reasoningLabel[row.reasoningMode] ?? row.reasoningMode }}
        </template>
      </el-table-column>
      <el-table-column
        label="模型"
        prop="modelId"
        min-width="140"
        show-overflow-tooltip
      />
      <el-table-column width="110" align="center">
        <template #header>
          <span>Skills / MCP / 子 Agent</span>
        </template>
        <template #default="{ row }">
          {{
            `${row.skillCount ?? 0} / ${row.mcpCount ?? 0} / ${row.subAgentCount ?? 0}`
          }}
        </template>
      </el-table-column>
      <el-table-column
        label="排序"
        prop="sortOrder"
        width="70"
        align="center"
      />
      <el-table-column label="状态" width="80" align="center">
        <template #default="{ row }">
          <el-switch
            v-model="row.status"
            :active-value="1"
            :inactive-value="0"
            v-hasPerm="['ai:agent:manage']"
            @change="handleStatusChange(row as AgentListItem)"
          />
        </template>
      </el-table-column>
      <el-table-column label="创建时间" prop="createTime" width="170" />
      <el-table-column label="操作" width="240" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="emit('detail', row as AgentListItem)"
          >
            详情
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="emit('edit', row as AgentListItem)"
          >
            编辑
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="emit('copy', row as AgentListItem)"
          >
            复制
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="emit('test', row as AgentListItem)"
          >
            测试
          </el-button>
          <el-button
            v-hasPerm="['ai:agent:manage']"
            link
            type="danger"
            size="small"
            @click="handleDelete(row as AgentListItem)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>
    <pagination
      v-if="agentStore.total > 0"
      v-model:limit="agentStore.query.pageSize"
      v-model:page="agentStore.query.pageNum"
      v-model:total="agentStore.total"
      @pagination="agentStore.fetchAgents()"
    />
  </div>
</template>

<script lang="ts" setup>
import { AgentListItem } from "dehaze-sdk-js";
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "AgentTable" });

const emit = defineEmits<{
  detail: [row: AgentListItem];
  edit: [row: AgentListItem];
  copy: [row: AgentListItem];
  test: [row: AgentListItem];
}>();

const agentStore = useAdminAgentStore();

const reasoningLabel: Record<string, string> = {
  auto: "自动",
  direct: "直接回复",
  react: "ReAct",
  plan_execute: "计划执行",
  reflexion: "反思",
};

function typeTags(row: AgentListItem) {
  if (row.isTeam === 1) {
    return [{ label: "Team", type: "warning" as const }];
  }
  if (row.isSubagent === 1) {
    return [{ label: "子 Agent", type: "info" as const }];
  }
  return [{ label: "普通", type: "primary" as const }];
}

async function handleStatusChange(row: AgentListItem) {
  const target = row.status === 1 ? "启用" : "禁用";
  try {
    await ElMessageBox.confirm(
      `确认${target}智能体「${row.name}」？禁用后不可被会话选择，进行中会话不受影响。`,
      "状态变更",
      { type: "warning" }
    );
    await agentStore.switchAgentStatus(row.id, row.status as 0 | 1);
    ElMessage.success(`已${target}`);
  } catch (e) {
    // 取消确认或请求失败回滚开关
    row.status = row.status === 1 ? 0 : 1;
    if (e instanceof Error) throw e;
  }
}

async function handleDelete(row: AgentListItem) {
  await ElMessageBox.confirm(
    `确认删除智能体「${row.name}」？若存在会话引用或被其他 Agent 作为子 Agent 引用，需先解绑后删除。`,
    "删除确认",
    { type: "warning" }
  );
  await agentStore.deleteAgent(row.id);
  ElMessage.success("已删除");
}
</script>
