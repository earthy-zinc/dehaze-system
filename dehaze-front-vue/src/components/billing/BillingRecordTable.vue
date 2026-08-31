<!-- 计费明细表：用户端(scope=self)与管理端(scope=userId)共用 -->
<script lang="ts" setup>
import type { BillingRecordVO, BillingType } from "dehaze-sdk-js";
import { computed, ref, watch } from "vue";
import BillTypeFilter, { BILL_TYPE_OPTIONS } from "./BillTypeFilter.vue";
import RecordStatusTags from "./RecordStatusTags.vue";
import type { BillingDataScope } from "@/store/modules/billingData";
import { useBillingDataStore } from "@/store/modules/billingData";

defineOptions({ name: "BillingRecordTable" });

const props = withDefaults(
  defineProps<{
    scope?: BillingDataScope;
    showRefundAction?: boolean;
  }>(),
  { scope: "self", showRefundAction: false }
);

const emit = defineEmits<{
  (e: "refund", record: BillingRecordVO): void;
}>();

const store = useBillingDataStore();

const BILL_TYPE_LABELS = new Map(
  BILL_TYPE_OPTIONS.map((option) => [option.value, option.label])
);

const dateRange = ref<[string, string] | null>(null);

const billType = computed<BillingType | "">({
  get: () => store.recordQuery.billType ?? "",
  set: (value) => {
    store.recordQuery.billType = value || undefined;
  },
});

/** 实扣相对预估：偏高说明实际消耗超出预估值，偏低为预扣多退少补的正常结果 */
function deductTag(record: BillingRecordVO) {
  if (record.credits > record.preDeduct) {
    return { label: "偏高", type: "danger" as const };
  }
  if (record.credits < record.preDeduct) {
    return { label: "偏低", type: "info" as const };
  }
  return { label: "正常", type: "success" as const };
}

function canRefund(record: BillingRecordVO) {
  return (
    record.refundStatus == null ||
    record.refundStatus === 0 ||
    record.refundStatus === 3
  );
}

function handleFilterChange() {
  store.recordQuery.pageNum = 1;
  store.fetchRecords();
}

function handleDateRangeChange(value: [string, string] | null) {
  store.recordQuery.dateStart = value?.[0];
  store.recordQuery.dateEnd = value?.[1];
  handleFilterChange();
}

function handleSizeChange(size: number) {
  store.recordQuery.pageSize = size;
  store.recordQuery.pageNum = 1;
  store.fetchRecords();
}

function handlePageChange(page: number) {
  store.recordQuery.pageNum = page;
  store.fetchRecords();
}

watch(
  () => props.scope,
  (next) => {
    store.initScope(next);
    store.recordQuery.pageNum = 1;
    store.fetchRecords();
  },
  { immediate: true }
);
</script>

<template>
  <div>
    <div class="mb-3 flex flex-wrap items-center gap-3">
      <BillTypeFilter
        v-model="billType"
        @update:model-value="handleFilterChange"
      />
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        value-format="YYYY-MM-DD"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        unlink-panels
        @change="handleDateRangeChange"
      />
    </div>

    <el-table v-loading="store.recordsLoading" :data="store.records" border>
      <el-table-column
        label="时间"
        prop="createTime"
        width="170"
        align="center"
      />
      <el-table-column label="类型" width="110" align="center">
        <template #default="{ row }">
          <el-tag size="small">
            {{
              BILL_TYPE_LABELS.get((row as BillingRecordVO).billType) ??
              (row as BillingRecordVO).billType
            }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="会话" min-width="120" align="center">
        <template #default="{ row }">
          {{ (row as BillingRecordVO).conversationId ?? "-" }}
        </template>
      </el-table-column>
      <el-table-column label="模型" min-width="160" show-overflow-tooltip>
        <template #default="{ row }">
          <div>{{ (row as BillingRecordVO).model }}</div>
          <div
            v-if="(row as BillingRecordVO).actualModel"
            class="text-xs text-[#e6a23c]"
          >
            实际：{{ (row as BillingRecordVO).actualModel }}
          </div>
        </template>
      </el-table-column>
      <el-table-column label="Token(输入/输出/缓存)" width="170" align="center">
        <template #default="{ row }">
          {{ (row as BillingRecordVO).inputTokens }} /
          {{ (row as BillingRecordVO).outputTokens }} /
          {{ (row as BillingRecordVO).cachedInputTokens }}
        </template>
      </el-table-column>
      <el-table-column label="实扣 / 预估" width="140" align="center">
        <template #default="{ row }">
          <div>
            {{ (row as BillingRecordVO).credits }} /
            {{ (row as BillingRecordVO).preDeduct }}
          </div>
          <el-tag :type="deductTag(row as BillingRecordVO).type" size="small">
            {{ deductTag(row as BillingRecordVO).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="缓存节省"
        prop="creditsSaved"
        width="100"
        align="center"
      />
      <el-table-column label="状态" min-width="180">
        <template #default="{ row }">
          <RecordStatusTags :record="row as BillingRecordVO" />
        </template>
      </el-table-column>
      <el-table-column
        v-if="showRefundAction"
        label="操作"
        width="110"
        align="center"
        fixed="right"
      >
        <template #default="{ row }">
          <el-button
            v-if="canRefund(row as BillingRecordVO)"
            link
            type="warning"
            size="small"
            @click="emit('refund', row as BillingRecordVO)"
          >
            误扣申诉
          </el-button>
          <span v-else>-</span>
        </template>
      </el-table-column>
    </el-table>

    <div class="mt-4 flex justify-end">
      <el-pagination
        :current-page="store.recordQuery.pageNum"
        :page-size="store.recordQuery.pageSize"
        :total="store.recordsTotal"
        :page-sizes="[10, 20, 50, 100]"
        background
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handlePageChange"
      />
    </div>
  </div>
</template>
