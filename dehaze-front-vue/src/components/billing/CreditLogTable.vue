<!-- 积分流水表：用户端(scope=self)与管理端(scope=userId)共用 -->
<script lang="ts" setup>
import type { CreditLogSource } from "dehaze-sdk-js";
import { computed, watch } from "vue";
import type { BillingDataScope } from "@/store/modules/billingData";
import { useBillingDataStore } from "@/store/modules/billingData";

defineOptions({ name: "CreditLogTable" });

const props = withDefaults(
  defineProps<{
    scope?: BillingDataScope;
  }>(),
  { scope: "self" }
);

const store = useBillingDataStore();

const SOURCE_OPTIONS: { value: CreditLogSource | ""; label: string }[] = [
  { value: "", label: "全部来源" },
  { value: "recharge", label: "到账" },
  { value: "vip_gift", label: "赠送" },
  { value: "trial", label: "试用" },
  { value: "consume", label: "消费" },
  { value: "refund", label: "回退" },
  { value: "admin_adjust", label: "管理员调整" },
  { value: "vip_gift_expire", label: "月末清零" },
];

const source = computed<CreditLogSource | "">({
  get: () => store.creditLogQuery.source ?? "",
  set: (value) => {
    store.creditLogQuery.source = value || undefined;
  },
});

function handleFilterChange() {
  store.creditLogQuery.pageNum = 1;
  store.fetchCreditLogs();
}

function handleSizeChange(size: number) {
  store.creditLogQuery.pageSize = size;
  store.creditLogQuery.pageNum = 1;
  store.fetchCreditLogs();
}

function handlePageChange(page: number) {
  store.creditLogQuery.pageNum = page;
  store.fetchCreditLogs();
}

watch(
  () => props.scope,
  (next) => {
    store.initScope(next);
    store.creditLogQuery.pageNum = 1;
    store.fetchCreditLogs();
  },
  { immediate: true }
);
</script>

<template>
  <div>
    <div class="mb-3">
      <el-select
        v-model="source"
        class="!w-[160px]"
        @change="handleFilterChange"
      >
        <el-option
          v-for="option in SOURCE_OPTIONS"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
    </div>

    <el-table
      v-loading="store.creditLogsLoading"
      :data="store.creditLogs"
      border
    >
      <el-table-column
        label="时间"
        prop="createTime"
        width="170"
        align="center"
      />
      <el-table-column label="来源" width="120" align="center">
        <template #default="{ row }">
          {{
            SOURCE_OPTIONS.find((option) => option.value === row.source)
              ?.label ?? row.source
          }}
        </template>
      </el-table-column>
      <el-table-column label="变动金额" width="110" align="center">
        <template #default="{ row }">
          <span :class="row.amount >= 0 ? 'text-[#67c23a]' : 'text-[#f56c6c]'">
            {{ row.amount >= 0 ? "+" : "" }}{{ row.amount }}
          </span>
        </template>
      </el-table-column>
      <el-table-column
        label="变动后余额"
        prop="balanceAfter"
        width="120"
        align="center"
      />
      <el-table-column
        label="原因"
        prop="reason"
        min-width="180"
        show-overflow-tooltip
      />
      <el-table-column label="关联业务ID" width="120" align="center">
        <template #default="{ row }">{{ row.relatedId ?? "-" }}</template>
      </el-table-column>
    </el-table>

    <div class="mt-4 flex justify-end">
      <el-pagination
        :current-page="store.creditLogQuery.pageNum"
        :page-size="store.creditLogQuery.pageSize"
        :total="store.creditLogsTotal"
        :page-sizes="[10, 20, 50, 100]"
        background
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handlePageChange"
      />
    </div>
  </div>
</template>
