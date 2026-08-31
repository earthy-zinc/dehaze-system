<!-- 版本管理区：版本列表 + 发布弹窗 + 版本差异对比 + 快照详情 -->
<template>
  <div>
    <div class="mb-3 flex justify-between items-center">
      <div class="flex items-center gap-2">
        <span class="text-sm text-gray-500">版本差异对比</span>
        <el-select v-model="diffBase" class="!w-[110px]" placeholder="基准版本">
          <el-option
            v-for="v in agentStore.versions"
            :key="v.id"
            :label="`v${v.versionNo}`"
            :value="v.versionNo"
          />
        </el-select>
        <el-select
          v-model="diffTarget"
          class="!w-[110px]"
          placeholder="目标版本"
        >
          <el-option
            v-for="v in agentStore.versions"
            :key="v.id"
            :label="`v${v.versionNo}`"
            :value="v.versionNo"
          />
        </el-select>
        <el-button
          :disabled="diffBase == null || diffTarget == null"
          :loading="agentStore.diffLoading"
          @click="handleCompare"
        >
          对比
        </el-button>
      </div>
      <el-button
        v-hasPerm="['ai:agent:manage']"
        type="primary"
        @click="publishVisible = true"
      >
        发布
      </el-button>
    </div>

    <el-collapse
      v-if="agentStore.versionDiff.length"
      v-model="diffCollapse"
      class="mb-3"
    >
      <el-collapse-item
        :title="`v${diffBase} → v${diffTarget} 差异（${agentStore.versionDiff.length} 项）`"
        name="diff"
      >
        <el-table :data="agentStore.versionDiff" size="small">
          <el-table-column label="字段" min-width="140">
            <template #default="{ row }">{{ diffField(row) }}</template>
          </el-table-column>
          <el-table-column label="基准值" min-width="200">
            <template #default="{ row }">
              <span class="text-xs">{{ diffValue(row, "base") }}</span>
            </template>
          </el-table-column>
          <el-table-column label="目标值" min-width="200">
            <template #default="{ row }">
              <span class="text-xs">{{ diffValue(row, "target") }}</span>
            </template>
          </el-table-column>
        </el-table>
      </el-collapse-item>
    </el-collapse>

    <VersionList
      :agent-id="props.agentId"
      @view-snapshot="handleViewSnapshot"
    />

    <PublishDialog
      v-model="publishVisible"
      :agent-id="props.agentId"
      @published="handlePublished"
    />

    <el-dialog
      v-model="snapshotVisible"
      :title="`版本快照 v${snapshotVersion}`"
      width="720px"
    >
      <pre v-if="agentStore.versionDetail" class="snapshot-pre">{{
        JSON.stringify(agentStore.versionDetail.snapshot, null, 2)
      }}</pre>
      <el-empty v-else description="暂无快照数据" :image-size="60" />
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { useAdminAgentStore } from "@/store/modules/adminAgent";

defineOptions({ name: "VersionPanel" });

const props = defineProps<{ agentId: number }>();

const agentStore = useAdminAgentStore();

const diffBase = ref<number>();
const diffTarget = ref<number>();
const diffCollapse = ref<string[]>([]);
const publishVisible = ref(false);
const snapshotVisible = ref(false);
const snapshotVersion = ref<number>();

onMounted(() => {
  agentStore.versionsQuery.pageNum = 1;
  agentStore.fetchVersions(props.agentId);
});

async function handleCompare() {
  await agentStore.compareVersions(
    props.agentId,
    diffBase.value!,
    diffTarget.value!
  );
  if (agentStore.versionDiff.length) {
    diffCollapse.value = ["diff"];
  } else {
    ElMessage.info("两个版本无差异");
  }
}

async function handleViewSnapshot(versionNo: number) {
  await agentStore.fetchVersionDetail(props.agentId, versionNo);
  snapshotVersion.value = versionNo;
  snapshotVisible.value = true;
}

/** 发布成功后回滚/发布均可能产生新草稿，重置对比选择避免指向已失效语义 */
function handlePublished() {
  diffBase.value = undefined;
  diffTarget.value = undefined;
  agentStore.versionDiff = [];
}

function diffField(row: Record<string, unknown>) {
  return String(row.field ?? row.key ?? JSON.stringify(row));
}

function diffValue(row: Record<string, unknown>, key: string) {
  const value = row[key] ?? row[key === "base" ? "baseValue" : "targetValue"];
  return typeof value === "string" ? value : JSON.stringify(value);
}
</script>

<style lang="scss" scoped>
.snapshot-pre {
  max-height: 60vh;
  margin: 0;
  overflow: auto;
  font-size: 12px;
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
}
</style>
