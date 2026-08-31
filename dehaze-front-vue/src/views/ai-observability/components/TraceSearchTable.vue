<!-- 过程链检索结果表：异常状态标注，点击行下钻详情 -->
<template>
  <el-table
    v-loading="store.tracesLoading"
    :data="store.traceList"
    highlight-current-row
    @row-click="handleRowClick"
  >
    <el-table-column label="Trace ID" min-width="180" show-overflow-tooltip>
      <template #default="{ row }">
        <span class="font-mono text-xs">{{ row.traceId }}</span>
        <el-tag
          v-if="traceTypeMeta(row.traceType)"
          :type="traceTypeMeta(row.traceType)?.tag"
          size="small"
          class="trace-type-tag"
        >
          {{ traceTypeMeta(row.traceType)?.label }}
        </el-tag>
      </template>
    </el-table-column>
    <el-table-column
      label="会话ID"
      prop="conversationId"
      width="90"
      align="center"
    />
    <el-table-column
      label="模型"
      prop="model"
      min-width="140"
      show-overflow-tooltip
    />
    <el-table-column
      label="智能体"
      prop="agentCode"
      width="130"
      show-overflow-tooltip
    >
      <template #default="{ row }">{{ row.agentCode ?? "-" }}</template>
    </el-table-column>
    <el-table-column label="状态" width="80" align="center">
      <template #default="{ row }">
        <el-tag :type="statusMeta(row.status).tag" size="small">
          {{ statusMeta(row.status).label }}
        </el-tag>
      </template>
    </el-table-column>
    <el-table-column label="首Token" width="90" align="center">
      <template #default="{ row }">{{
        fmtDuration(row.firstTokenMs)
      }}</template>
    </el-table-column>
    <el-table-column label="总耗时" width="90" align="center">
      <template #default="{ row }">{{ fmtDuration(row.durationMs) }}</template>
    </el-table-column>
    <el-table-column
      label="LLM调用"
      prop="llmCallCount"
      width="90"
      align="center"
    />
    <el-table-column label="步数" prop="stepCount" width="70" align="center" />
    <el-table-column
      label="Token(总/输入/输出/缓存)"
      min-width="180"
      align="center"
    >
      <template #default="{ row }">
        {{ fmtTokens(row.totalTokens) }} / {{ fmtTokens(row.promptTokens) }} /
        {{ fmtTokens(row.completionTokens) }} /
        {{ fmtTokens(row.cachedTokens) }}
      </template>
    </el-table-column>
    <el-table-column label="创建时间" prop="createTime" width="170" />
  </el-table>

  <pagination
    v-if="store.traceTotal > 0"
    v-model:limit="store.auditPageSize"
    v-model:page="store.auditPageNum"
    v-model:total="store.traceTotal"
    @pagination="store.fetchTraces()"
  />
</template>

<script lang="ts" setup>
import type {
  AiObservabilityStatus,
  AiObservabilityTraceItem,
} from "dehaze-sdk-js";
import { TRACE_STATUS_META, fmtDuration, fmtTokens, traceTypeMeta } from "../format";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "TraceSearchTable" });

const store = useAdminObservabilityStore();

// 表格插槽 row 无类型，经此收敛到状态枚举索引
function statusMeta(status: number) {
  return TRACE_STATUS_META[status as AiObservabilityStatus];
}

function handleRowClick(row: AiObservabilityTraceItem) {
  store.fetchTraceDetail(row.traceId);
}
</script>

<style lang="scss" scoped>
.trace-type-tag {
  margin-left: 6px;
}
</style>
