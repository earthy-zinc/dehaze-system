<!-- SKILL 市场：已共享 Skill 目录，一键启用后用户端会话可按需加载 -->
<script lang="ts" setup>
import { SkillMarketVO } from "dehaze-sdk-js";
import { useAdminSkillStore } from "@/store/modules/adminSkill";

defineOptions({ name: "MarketSkillList" });

const skillStore = useAdminSkillStore();
const enablingId = ref<number | null>(null);

async function handleEnable(row: SkillMarketVO) {
  enablingId.value = row.skillId;
  try {
    await skillStore.installMarketSkill(row.skillId);
  } finally {
    enablingId.value = null;
  }
}
</script>

<template>
  <div>
    <div class="mb-3 text-xs text-gray-400">
      市场目录为管理员共享的
      Skill，启用后用户端会话可按需加载；会话启动仅加载名称与描述，命中后才加载完整指令
    </div>

    <el-table
      v-loading="skillStore.marketLoading"
      :data="skillStore.marketSkills"
      border
      size="small"
    >
      <el-table-column label="名称" prop="name" min-width="150" />
      <el-table-column
        label="描述"
        prop="description"
        min-width="200"
        show-overflow-tooltip
      />
      <el-table-column
        label="适用场景"
        prop="scene"
        min-width="140"
        show-overflow-tooltip
      >
        <template #default="{ row }">
          {{ (row as SkillMarketVO).scene ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="关联 Agent 数" width="120" align="center">
        <template #default="{ row }">
          {{ (row as SkillMarketVO).agentCount ?? 0 }}
        </template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag
            :type="(row as SkillMarketVO).enabled ? 'success' : 'info'"
            size="small"
          >
            {{ (row as SkillMarketVO).enabled ? "已启用" : "未启用" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="100" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            v-hasPerm="['ai:skill:manage']"
            link
            type="primary"
            size="small"
            :disabled="(row as SkillMarketVO).enabled"
            :loading="enablingId === (row as SkillMarketVO).skillId"
            @click="handleEnable(row as SkillMarketVO)"
          >
            {{ (row as SkillMarketVO).enabled ? "已启用" : "一键启用" }}
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-empty
      v-if="!skillStore.marketLoading && skillStore.marketSkills.length === 0"
      description="暂无共享至市场的 Skill"
      :image-size="60"
    />
  </div>
</template>
