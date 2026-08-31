<!-- 评测中心：筛选 → 总览 → 执行记录 → 趋势与判分状态 → 人工复核 -->
<template>
  <div class="app-container">
    <EvalAgentFilter />
    <EvalOverview />
    <EvalRunTable />

    <el-row :gutter="12">
      <el-col :xs="24" :md="14">
        <TrendPanel />
      </el-col>
      <el-col :xs="24" :md="10">
        <JudgeStatusPanel />
      </el-col>
    </el-row>

    <ReviewPanel />
    <EvalDetailDrawer />
  </div>
</template>

<script lang="ts" setup>
import { useAdminEvalStore } from "@/store/modules/adminEval";
import EvalAgentFilter from "./components/EvalAgentFilter.vue";
import EvalDetailDrawer from "./components/EvalDetailDrawer.vue";
import EvalOverview from "./components/EvalOverview.vue";
import EvalRunTable from "./components/EvalRunTable.vue";
import JudgeStatusPanel from "./components/JudgeStatusPanel.vue";
import ReviewPanel from "./components/ReviewPanel.vue";
import TrendPanel from "./components/TrendPanel.vue";

defineOptions({ name: "AiEvalCenter" });

const evalStore = useAdminEvalStore();

onMounted(() => {
  evalStore.refreshAll();
});
</script>
