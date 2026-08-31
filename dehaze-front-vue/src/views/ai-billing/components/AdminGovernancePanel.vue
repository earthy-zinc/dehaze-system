<!-- 治理与对账：成本单价 / 账单对账 / 调价测算 / 异常监控 / 积分调整 -->
<template>
  <el-tabs v-model="activeTab">
    <el-tab-pane label="成本单价管理" name="cost" lazy>
      <cost-unit-price-manage />
    </el-tab-pane>
    <el-tab-pane label="账单对账" name="reconcile" lazy>
      <reconcile-panel />
    </el-tab-pane>
    <el-tab-pane label="调价影响测算" name="simulator" lazy>
      <price-impact-simulator />
    </el-tab-pane>
    <el-tab-pane label="异常监控" name="anomaly" lazy>
      <anomaly-monitor />
    </el-tab-pane>
    <el-tab-pane label="积分调整" name="adjust" lazy>
      <credit-adjust-panel />
    </el-tab-pane>
  </el-tabs>
</template>

<script lang="ts" setup>
defineOptions({ name: "AdminGovernancePanel" });

import CostUnitPriceManage from "./CostUnitPriceManage.vue";
import ReconcilePanel from "./ReconcilePanel.vue";
import PriceImpactSimulator from "./PriceImpactSimulator.vue";
import AnomalyMonitor from "./AnomalyMonitor.vue";
import CreditAdjustPanel from "./CreditAdjustPanel.vue";

const activeTab = ref("cost");
</script>
