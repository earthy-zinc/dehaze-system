<!-- 检索质量区：检索测试 / 召回测试 / 低质量片段三个 Tab -->
<script lang="ts" setup>
import { useAdminKbStore } from "@/store/modules/adminKb";

defineOptions({ name: "QualitySection" });

defineProps<{
  kbId: number;
}>();

const adminKbStore = useAdminKbStore();

function handleTabChange(tab: string | number) {
  adminKbStore.qualityTab = tab as "retrieve" | "recall" | "low-quality";
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <template #header>
      <span>检索质量</span>
    </template>
    <el-tabs
      :model-value="adminKbStore.qualityTab"
      @tab-change="handleTabChange"
    >
      <el-tab-pane label="检索测试" name="retrieve">
        <RetrieveTestPanel :knowledge-base-id="kbId" />
      </el-tab-pane>
      <el-tab-pane label="召回测试" name="recall" lazy>
        <RecallTestPanel :kb-id="kbId" />
      </el-tab-pane>
      <el-tab-pane label="低质量片段" name="low-quality" lazy>
        <LowQualityPanel :kb-id="kbId" />
      </el-tab-pane>
    </el-tabs>
  </el-card>
</template>
