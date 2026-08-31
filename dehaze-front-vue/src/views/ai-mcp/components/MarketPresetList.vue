<!-- MCP 市场预设列表：内置 Server 目录，一键接入后转注册表管理 -->
<template>
  <div v-loading="mcpStore.marketLoading">
    <el-empty
      v-if="!mcpStore.marketLoading && mcpStore.marketPresets.length === 0"
      description="暂无可用 MCP Server 预设"
      :image-size="80"
    />
    <div v-else class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      <el-card
        v-for="preset in mcpStore.marketPresets"
        :key="preset.presetId"
        shadow="never"
      >
        <template #header>
          <div class="flex justify-between items-center">
            <span class="font-bold">{{ preset.name }}</span>
            <el-tag v-if="preset.installed" type="success" size="small">
              已接入
            </el-tag>
          </div>
        </template>
        <div class="min-h-[40px] text-xs text-gray-500">
          {{ preset.description ?? "暂无描述" }}
        </div>
        <div class="mt-2 flex flex-wrap gap-1">
          <el-tag
            v-for="tag in preset.capabilityTags ?? []"
            :key="tag"
            size="small"
            type="info"
          >
            {{ tag }}
          </el-tag>
        </div>
        <div class="mt-3 flex justify-end">
          <el-button
            v-if="preset.installed"
            size="small"
            @click="mcpStore.switchTab('servers')"
          >
            去管理
          </el-button>
          <el-button
            v-else
            v-hasPerm="['ai:mcp:manage']"
            size="small"
            type="primary"
            :loading="mcpStore.installingPresetId === preset.presetId"
            @click="mcpStore.installPreset(preset)"
          >
            一键接入
          </el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script lang="ts" setup>
defineOptions({ name: "MarketPresetList" });

import { useAdminMcpStore } from "@/store/modules/adminMcp";

const mcpStore = useAdminMcpStore();

onMounted(() => {
  if (mcpStore.marketPresets.length === 0) {
    mcpStore.fetchMarketPresets();
  }
});
</script>
