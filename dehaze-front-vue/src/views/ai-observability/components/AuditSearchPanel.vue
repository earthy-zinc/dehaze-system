<!-- 审计检索：全量过程链检索表单（用户/会话/时间/状态/智能体/模型）+ 导出留痕 -->
<template>
  <el-card shadow="never">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>审计检索</span>
        <div class="flex items-center gap-2">
          <el-button
            :loading="store.exportLoading"
            @click="store.exportTraces()"
          >
            <el-icon><Download /></el-icon>导出CSV
          </el-button>
        </div>
      </div>
    </template>

    <el-form inline class="mb-[12px]" @submit.prevent="store.searchTraces()">
      <el-form-item label="用户ID">
        <el-input-number
          v-model="store.auditFilter.userId"
          :min="1"
          :controls="false"
          placeholder="用户ID"
          class="!w-[120px]"
          @keyup.enter="store.searchTraces()"
        />
      </el-form-item>
      <el-form-item label="会话ID">
        <el-input-number
          v-model="store.auditFilter.conversationId"
          :min="1"
          :controls="false"
          placeholder="会话ID"
          class="!w-[120px]"
          @keyup.enter="store.searchTraces()"
        />
      </el-form-item>
      <el-form-item label="状态">
        <el-select
          v-model="store.auditFilter.status"
          placeholder="全部状态"
          clearable
          class="!w-[110px]"
          @change="store.searchTraces()"
        >
          <el-option
            v-for="(meta, status) in TRACE_STATUS_META"
            :key="status"
            :label="meta.label"
            :value="Number(status)"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="智能体">
        <el-input
          v-model="store.auditFilter.agentCode"
          clearable
          placeholder="agent_code"
          class="!w-[150px]"
          @keyup.enter="store.searchTraces()"
        />
      </el-form-item>
      <el-form-item label="模型">
        <el-input
          v-model="store.auditFilter.model"
          clearable
          placeholder="模型标识"
          class="!w-[150px]"
          @keyup.enter="store.searchTraces()"
        />
      </el-form-item>
      <el-form-item label="时间范围">
        <el-date-picker
          v-model="store.auditFilter.timeRange"
          type="daterange"
          value-format="YYYY-MM-DD"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          class="!w-[240px]"
        />
      </el-form-item>
      <el-form-item>
        <el-button
          type="primary"
          :loading="store.tracesLoading"
          @click="store.searchTraces()"
        >
          <el-icon><Search /></el-icon>检索
        </el-button>
        <el-button @click="store.resetAuditFilter()">重置</el-button>
      </el-form-item>
    </el-form>

    <trace-search-table />
  </el-card>
</template>

<script lang="ts" setup>
import { Download, Search } from "@element-plus/icons-vue";
import TraceSearchTable from "./TraceSearchTable.vue";
import { TRACE_STATUS_META } from "../format";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "AuditSearchPanel" });

const store = useAdminObservabilityStore();
</script>
