<!-- 审计筛选栏：用户/时间范围/状态/异常类型/关键词，驱动审计列表查询 -->
<script lang="ts" setup>
import { Refresh, Search } from "@element-plus/icons-vue";
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useAdminAuditStore } from "@/store/modules/adminAudit";

defineOptions({ name: "AuditFilterBar" });

const adminAuditStore = useAdminAuditStore();
const { auditFilter } = storeToRefs(adminAuditStore);

const anomalyOptions = [
  { label: "失败", value: "failed" },
  { label: "配额拒绝", value: "quota" },
  { label: "已取消", value: "canceled" },
];

// 后端审计列表暂不支持用户/时间/异常类型查询参数，这三项在已加载页内过滤
const clientFilterActive = computed(
  () =>
    auditFilter.value.userId != null ||
    !!auditFilter.value.anomalyType ||
    !!auditFilter.value.dateRange
);

function handleSearch() {
  adminAuditStore.applyAuditFilter({});
}

function handleReset() {
  adminAuditStore.resetAuditFilter();
}
</script>

<template>
  <el-card shadow="never" class="!border-none">
    <el-form inline @submit.prevent>
      <el-form-item label="用户ID">
        <el-input-number
          v-model="auditFilter.userId"
          :min="1"
          :controls="false"
          placeholder="用户ID"
          class="!w-[120px]"
        />
      </el-form-item>
      <el-form-item label="时间范围">
        <el-date-picker
          v-model="auditFilter.dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          value-format="YYYY-MM-DD"
          class="!w-[240px]"
        />
      </el-form-item>
      <el-form-item label="状态">
        <el-select v-model="auditFilter.status" class="!w-[100px]">
          <el-option label="全部" :value="0" />
          <el-option label="活跃" :value="1" />
          <el-option label="已归档" :value="2" />
        </el-select>
      </el-form-item>
      <el-form-item label="异常类型">
        <el-select
          v-model="auditFilter.anomalyType"
          clearable
          placeholder="全部"
          class="!w-[120px]"
        >
          <el-option
            v-for="option in anomalyOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="关键词">
        <el-input
          v-model="auditFilter.keyword"
          placeholder="会话标题/消息内容"
          clearable
          class="!w-[200px]"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" :icon="Search" @click="handleSearch">
          查询
        </el-button>
        <el-button :icon="Refresh" @click="handleReset">重置</el-button>
      </el-form-item>
    </el-form>
    <el-alert
      v-if="clientFilterActive"
      type="info"
      :closable="false"
      title="用户/时间/异常类型筛选当前在已加载页内生效（审计列表接口暂不支持对应查询参数）"
    />
  </el-card>
</template>
