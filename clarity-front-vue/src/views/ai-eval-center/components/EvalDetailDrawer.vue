<!-- 评测详情抽屉：报告 + 样本明细 + 版本对比 -->
<template>
  <el-drawer
    v-model="evalStore.detailVisible"
    size="60%"
    destroy-on-close
    :title="`#${run?.id} 评测详情`"
  >
    <template v-if="run">
      <el-descriptions :column="4" size="small" border class="mb-[12px]">
        <el-descriptions-item label="触发方式">
          {{ TRIGGER_TYPE_META[run.triggerType] ?? run.triggerType }}
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="runStatusMeta(run.status).type" size="small">
            {{ runStatusMeta(run.status).label }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="评测集"
          >#{{ run.datasetId }}</el-descriptions-item
        >
        <el-descriptions-item label="评测时间">
          {{ formatTime(run.createTime) }}
        </el-descriptions-item>
      </el-descriptions>

      <el-tabs model-value="report">
        <el-tab-pane label="评测报告" name="report" lazy>
          <ReportPanel :run="run" />
        </el-tab-pane>
        <el-tab-pane label="样本明细" name="samples" lazy>
          <SampleResultList :run="run" />
        </el-tab-pane>
        <el-tab-pane label="版本对比" name="compare" lazy>
          <VersionCompare :run="run" />
        </el-tab-pane>
      </el-tabs>
    </template>
  </el-drawer>
</template>

<script lang="ts" setup>
import { useAdminEvalStore } from "@/store/modules/adminEval";
import ReportPanel from "./ReportPanel.vue";
import SampleResultList from "./SampleResultList.vue";
import VersionCompare from "./VersionCompare.vue";
import { RUN_STATUS_META, TRIGGER_TYPE_META, formatTime } from "../eval-meta";

defineOptions({ name: "EvalDetailDrawer" });

const evalStore = useAdminEvalStore();

const run = computed(() => evalStore.evalDetail);

function runStatusMeta(status: number) {
  return (
    RUN_STATUS_META[status] ?? { label: `状态${status}`, type: "info" as const }
  );
}
</script>
