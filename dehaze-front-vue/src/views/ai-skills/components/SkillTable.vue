<!-- SKILL 管理区：全量 Skill 列表（含停用）+ 编辑/启停/试运行/市场共享/删除 -->
<script lang="ts" setup>
import { SkillVO } from "dehaze-sdk-js";
import { useDebounceFn } from "@vueuse/core";
import { useAdminSkillStore } from "@/store/modules/adminSkill";

defineOptions({ name: "SkillTable" });

const emit = defineEmits<{
  edit: [skill: SkillVO];
  test: [skill: SkillVO];
}>();

const skillStore = useAdminSkillStore();

function handleQuery() {
  skillStore.query.pageNum = 1;
  skillStore.fetchSkills();
}
const debouncedQuery = useDebounceFn(handleQuery, 300);

/** 指令中的 Markdown 有序步骤数（列表不返回指令全文时无法统计） */
function stepCount(instruction?: string) {
  if (!instruction) return "-";
  return instruction.split("\n").filter((line) => /^\s*\d+[.、)]/.test(line))
    .length;
}

async function handleStatusChange(row: SkillVO, status: 0 | 1) {
  await skillStore.switchSkillStatus(row, status);
  ElMessage.success(status === 1 ? "已启用" : "已禁用，LLM 不再自动选择");
}

async function handleShare(row: SkillVO) {
  // 后端共享要求 Skill 已启用（否则 A0400），在入口拦截避免无效请求
  if (row.status !== 1) {
    ElMessageBox.alert(
      `Skill「${row.name}」当前为禁用状态，请先启用后再共享至市场。`,
      "无法共享",
      { type: "warning" }
    );
    return;
  }
  await skillStore.shareSkillToMarket(row);
}

async function handleDelete(row: SkillVO) {
  // 后端删除同样强校验关联关系，前端先引导解绑
  if (row.agentCount) {
    ElMessageBox.alert(
      `Skill「${row.name}」已被 ${row.agentCount} 个 Agent 关联，请先在智能体管理中解绑后再删除。`,
      "无法删除",
      { type: "warning" }
    );
    return;
  }
  try {
    await ElMessageBox.confirm(
      `确认删除 Skill「${row.name}」？删除后不可恢复。`,
      "删除确认",
      { type: "warning", confirmButtonText: "确定", cancelButtonText: "取消" }
    );
  } catch {
    return;
  }
  await skillStore.deleteSkill(row);
  ElMessage.success("Skill 已删除");
}
</script>

<template>
  <div>
    <div class="mb-3 flex items-center gap-2">
      <el-input
        v-model="skillStore.query.keyword"
        clearable
        placeholder="Skill 名称"
        style="width: 200px"
        @keyup.enter="handleQuery"
        @input="debouncedQuery"
      />
      <el-select
        v-model="skillStore.query.status"
        class="!w-[120px]"
        clearable
        placeholder="全部状态"
        @change="handleQuery"
      >
        <el-option label="已启用" :value="1" />
        <el-option label="已禁用" :value="0" />
      </el-select>
    </div>

    <el-table v-loading="skillStore.loading" :data="skillStore.skills" border>
      <el-table-column label="名称" min-width="160">
        <template #default="{ row }">
          <div class="font-bold">{{ (row as SkillVO).name }}</div>
          <div class="text-xs text-gray-400">
            {{ (row as SkillVO).description ?? "-" }}
          </div>
        </template>
      </el-table-column>
      <el-table-column
        label="适用场景"
        prop="scene"
        min-width="140"
        show-overflow-tooltip
      >
        <template #default="{ row }">{{
          (row as SkillVO).scene ?? "-"
        }}</template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-switch
            :model-value="(row as SkillVO).status"
            :active-value="1"
            :inactive-value="0"
            @change="handleStatusChange(row as SkillVO, $event as 0 | 1)"
          />
        </template>
      </el-table-column>
      <el-table-column label="被 Agent 关联数" width="130" align="center">
        <template #default="{ row }">{{
          (row as SkillVO).agentCount ?? 0
        }}</template>
      </el-table-column>
      <el-table-column label="步骤数" width="90" align="center">
        <template #default="{ row }">
          {{ stepCount((row as SkillVO).instruction) }}
        </template>
      </el-table-column>
      <el-table-column
        label="更新时间"
        prop="updateTime"
        width="170"
        align="center"
      />
      <el-table-column label="操作" width="240" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            v-hasPerm="['ai:skill:manage']"
            link
            type="primary"
            size="small"
            @click="emit('edit', row as SkillVO)"
          >
            编辑
          </el-button>
          <el-button
            v-hasPerm="['ai:skill:manage']"
            link
            type="primary"
            size="small"
            @click="emit('test', row as SkillVO)"
          >
            试运行
          </el-button>
          <el-button
            v-hasPerm="['ai:skill:manage']"
            link
            type="primary"
            size="small"
            :disabled="(row as SkillVO).marketShared === 1"
            @click="handleShare(row as SkillVO)"
          >
            {{ (row as SkillVO).marketShared === 1 ? "已共享" : "共享至市场" }}
          </el-button>
          <el-button
            v-hasPerm="['ai:skill:manage']"
            link
            type="danger"
            size="small"
            @click="handleDelete(row as SkillVO)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-if="skillStore.total > 0"
      v-model:limit="skillStore.query.pageSize"
      v-model:page="skillStore.query.pageNum"
      v-model:total="skillStore.total"
      @pagination="skillStore.fetchSkills()"
    />
  </div>
</template>
