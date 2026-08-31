<!-- 评测总览：各智能体最近得分 / 门禁状态 / 退化与高风险告警 -->
<template>
  <el-card
    v-loading="evalStore.overviewLoading"
    shadow="never"
    class="mb-[12px]"
  >
    <template #header>
      <div class="flex justify-between items-center">
        <span>评测总览</span>
        <span class="text-xs text-gray-400">
          共 {{ evalStore.evalOverview.length }} 个智能体
        </span>
      </div>
    </template>

    <el-alert
      v-if="warningCount > 0"
      class="mb-[12px]"
      type="warning"
      :closable="false"
      :title="`${warningCount} 个智能体存在退化或高风险样本失败，建议优先复核`"
    />

    <el-table :data="evalStore.evalOverview" size="small">
      <el-table-column label="智能体" min-width="180">
        <template #default="{ row }">
          <div>{{ row.agentName }}</div>
          <div class="text-xs text-gray-400">{{ row.agentCode }}</div>
        </template>
      </el-table-column>
      <el-table-column label="最近得分" width="110" align="center">
        <template #default="{ row }">
          {{ formatScore(row.totalScore) }}
        </template>
      </el-table-column>
      <el-table-column label="门禁状态" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="gateStatusMeta(row.gateStatus).type" size="small">
            {{ gateStatusMeta(row.gateStatus).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="告警" width="180" align="center">
        <template #default="{ row }">
          <el-tag v-if="row.degraded" type="warning" size="small" class="mr-1">
            得分退化
          </el-tag>
          <el-tag v-if="row.highRiskFailed" type="danger" size="small">
            高风险失败
          </el-tag>
          <span v-if="!row.degraded && !row.highRiskFailed">-</span>
        </template>
      </el-table-column>
      <el-table-column label="触发方式" width="100" align="center">
        <template #default="{ row }">
          {{ row.triggerType ? TRIGGER_TYPE_META[row.triggerType] : "-" }}
        </template>
      </el-table-column>
      <el-table-column label="最近评测" width="160" align="center">
        <template #default="{ row }">
          {{ formatTime(row.runTime) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="100" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click="evalStore.selectAgent(row.agentId)"
          >
            执行记录
          </el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="暂无智能体评测数据" :image-size="60" />
      </template>
    </el-table>
  </el-card>
</template>

<script lang="ts" setup>
import type { AiEvalGateStatus } from "dehaze-sdk-js";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import {
  GATE_STATUS_META,
  TRIGGER_TYPE_META,
  formatScore,
  formatTime,
} from "../eval-meta";

defineOptions({ name: "EvalOverview" });

const evalStore = useAdminEvalStore();

function gateStatusMeta(status: AiEvalGateStatus) {
  return GATE_STATUS_META[status];
}

const warningCount = computed(
  () =>
    evalStore.evalOverview.filter(
      (item) => item.degraded || item.highRiskFailed
    ).length
);
</script>
