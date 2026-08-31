<!-- 命名空间配置：工具分组（覆盖式更新，保存即整体替换服务端配置） -->
<template>
  <div>
    <el-alert
      class="mb-2"
      type="warning"
      :closable="false"
      title="命名空间覆盖式更新：保存后服务端配置将被本次提交的分组整体替换；Agent 仅可访问其关联的命名空间（最小权限）。"
    />
    <el-table :data="rows" size="small">
      <el-table-column label="命名空间" min-width="180">
        <template #default="{ row }">
          <el-input v-model="row.name" size="small" placeholder="分组标识" />
        </template>
      </el-table-column>
      <el-table-column label="组内工具" min-width="280">
        <template #default="{ row }">
          <el-select
            v-model="row.toolNames"
            multiple
            filterable
            collapse-tags
            collapse-tags-tooltip
            size="small"
            class="w-full"
            placeholder="选择归入该命名空间的工具"
          >
            <el-option
              v-for="tool in tools"
              :key="tool.name"
              :label="tool.name"
              :value="tool.name"
            />
          </el-select>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="80" align="center">
        <template #default="{ $index }">
          <el-button
            link
            type="danger"
            size="small"
            @click="rows.splice($index, 1)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <div class="mt-2 flex justify-between">
      <el-button size="small" type="primary" plain @click="addNamespace">
        <el-icon><Plus /></el-icon>添加命名空间
      </el-button>
      <el-button
        v-hasPerm="['ai:mcp:manage']"
        size="small"
        type="primary"
        :loading="saving"
        @click="handleSave()"
      >
        保存命名空间
      </el-button>
    </div>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "NamespaceConfigPanel" });

import { Plus } from "@element-plus/icons-vue";
import { McpNamespaceVO, McpToolVO } from "dehaze-sdk-js";

const props = defineProps<{
  tools: McpToolVO[];
  namespaces: McpNamespaceVO[];
  saving?: boolean;
}>();
const emit = defineEmits<{ save: [namespaces: McpNamespaceVO[]] }>();

const rows = ref<McpNamespaceVO[]>([]);

watch(
  () => props.namespaces,
  (value) => {
    rows.value = value.map((item) => ({
      name: item.name,
      toolNames: [...item.toolNames],
    }));
  },
  { immediate: true, deep: true }
);

function addNamespace() {
  rows.value.push({ name: "", toolNames: [] });
}

function handleSave() {
  const invalid = rows.value.find((row) => !row.name.trim());
  if (invalid) {
    ElMessage.warning("命名空间标识不能为空");
    return;
  }
  const names = rows.value.map((row) => row.name.trim());
  if (new Set(names).size !== names.length) {
    ElMessage.warning("命名空间标识不可重复");
    return;
  }
  emit(
    "save",
    rows.value.map((row) => ({
      name: row.name.trim(),
      toolNames: row.toolNames,
    }))
  );
}
</script>
