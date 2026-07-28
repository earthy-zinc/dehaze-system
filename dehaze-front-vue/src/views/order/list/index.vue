<template>
  <div class="app-container">
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="订单号" prop="orderNo">
          <el-input
            v-model="queryParams.orderNo"
            clearable
            placeholder="订单号"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="用户名/套餐"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in statusOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="支付方式" prop="payMethod">
          <el-select
            v-model="queryParams.payMethod"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in payMethodOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="金额区间">
          <el-input-number
            v-model="queryParams.amountMin"
            :min="0"
            :precision="2"
            controls-position="right"
            style="width: 120px"
          />
          <span style="margin: 0 6px">-</span>
          <el-input-number
            v-model="queryParams.amountMax"
            :min="0"
            :precision="2"
            controls-position="right"
            style="width: 120px"
          />
        </el-form-item>
        <el-form-item label="支付时间">
          <el-date-picker
            v-model="paidTimeRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            value-format="YYYY-MM-DD HH:mm:ss"
            style="width: 360px"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleQuery"
            ><el-icon><Search /></el-icon>搜索</el-button
          >
          <el-button @click="resetQuery"
            ><el-icon><Refresh /></el-icon>重置</el-button
          >
        </el-form-item>
      </el-form>
    </div>

    <el-card class="table-container" shadow="never">
      <template #header>
        <div class="flex justify-between items-center">
          <el-button
            v-hasPerm="['order:stats']"
            type="primary"
            plain
            @click="openStats"
            ><el-icon><DataLine /></el-icon>订单统计</el-button
          >
        </div>
      </template>

      <el-table
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
      >
        <el-table-column label="订单号" prop="orderNo" width="200" />
        <el-table-column label="用户" prop="username" width="120" />
        <el-table-column label="套餐" min-width="160">
          <template #default="scope">
            <span>{{ (scope.row as OrderPageVO).packageName }}</span>
            <el-tag size="small" type="info" style="margin-left: 6px">{{
              (scope.row as OrderPageVO).packageLevel
            }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="实付"
          prop="payableAmount"
          width="100"
          align="right"
        >
          <template #default="scope">
            ¥{{ ((scope.row as OrderPageVO).payableAmount ?? 0).toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column label="优惠" width="120" align="right">
          <template #default="scope">
            <span
              v-if="
                (scope.row as OrderPageVO).discountAmount > 0 ||
                (scope.row as OrderPageVO).couponAmount > 0
              "
            >
              -¥{{
                (
                  ((scope.row as OrderPageVO).discountAmount ?? 0) +
                  ((scope.row as OrderPageVO).couponAmount ?? 0)
                ).toFixed(2)
              }}
            </span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="支付方式" width="110" align="center">
          <template #default="scope">
            <el-tag v-if="(scope.row as OrderPageVO).payMethod" size="small">{{
              payMethodLabel((scope.row as OrderPageVO).payMethod!)
            }}</el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="scope">
            <el-tag
              :type="statusTagType((scope.row as OrderPageVO).status)"
              size="small"
              >{{ statusLabel((scope.row as OrderPageVO).status) }}</el-tag
            >
          </template>
        </el-table-column>
        <el-table-column label="创建时间" prop="createTime" width="170" />
        <el-table-column label="支付时间" prop="paidTime" width="170">
          <template #default="scope">
            <span>{{ (scope.row as OrderPageVO).paidTime || "-" }}</span>
          </template>
        </el-table-column>
        <el-table-column fixed="right" label="操作" width="110" align="center">
          <template #default="scope">
            <el-button
              link
              size="small"
              type="primary"
              @click="handleDetail(scope.row as OrderPageVO)"
            >
              <el-icon><View /></el-icon>查看详情
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="total > 0"
        v-model:limit="queryParams.pageSize"
        v-model:page="queryParams.pageNum"
        v-model:total="total"
        @pagination="handleQuery"
      />
    </el-card>

    <!-- 订单详情弹窗 -->
    <el-dialog
      v-model="detailDialog.visible"
      title="订单详情"
      width="720px"
      @close="detailDialog.data = null"
    >
      <div v-loading="detailDialog.loading">
        <el-descriptions v-if="detailDialog.data" :column="2" border>
          <el-descriptions-item label="订单号">{{
            detailDialog.data.orderNo
          }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag
              :type="statusTagType(detailDialog.data.status)"
              size="small"
              >{{ statusLabel(detailDialog.data.status) }}</el-tag
            >
          </el-descriptions-item>
          <el-descriptions-item label="用户">{{
            detailDialog.data.username
          }}</el-descriptions-item>
          <el-descriptions-item label="套餐"
            >{{ detailDialog.data.packageName }} ({{
              detailDialog.data.packageLevel
            }})</el-descriptions-item
          >
          <el-descriptions-item label="原价"
            >¥{{
              detailDialog.data.originalPrice.toFixed(2)
            }}</el-descriptions-item
          >
          <el-descriptions-item label="折扣优惠"
            >¥{{
              detailDialog.data.discountAmount.toFixed(2)
            }}</el-descriptions-item
          >
          <el-descriptions-item label="优惠券抵扣"
            >¥{{
              detailDialog.data.couponAmount.toFixed(2)
            }}</el-descriptions-item
          >
          <el-descriptions-item label="实付"
            >¥{{
              detailDialog.data.payableAmount.toFixed(2)
            }}</el-descriptions-item
          >
          <el-descriptions-item label="支付方式">{{
            detailDialog.data.payMethod
              ? payMethodLabel(detailDialog.data.payMethod)
              : "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="已支付金额"
            >¥{{
              detailDialog.data.paidAmount.toFixed(2)
            }}</el-descriptions-item
          >
          <el-descriptions-item label="创建时间">{{
            detailDialog.data.createTime
          }}</el-descriptions-item>
          <el-descriptions-item label="支付时间">{{
            detailDialog.data.paidTime || "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="生效时间">{{
            detailDialog.data.effectiveTime || "-"
          }}</el-descriptions-item>
          <el-descriptions-item label="到期时间">{{
            detailDialog.data.expireTime
          }}</el-descriptions-item>
          <el-descriptions-item label="自动续费">{{
            detailDialog.data.isAutoRenew ? "是" : "否"
          }}</el-descriptions-item>
          <el-descriptions-item
            v-if="detailDialog.data.cancelReason"
            label="取消原因"
            >{{ detailDialog.data.cancelReason }}</el-descriptions-item
          >
        </el-descriptions>

        <el-tabs v-if="detailDialog.data" style="margin-top: 16px">
          <el-tab-pane label="支付流水">
            <el-table
              :data="detailDialog.data.paymentRecords || []"
              border
              size="small"
            >
              <el-table-column label="流水号" prop="paymentNo" />
              <el-table-column label="渠道" width="120">
                <template #default="scope">{{
                  payMethodLabel((scope.row as PaymentRecordVO).channel)
                }}</template>
              </el-table-column>
              <el-table-column
                label="金额"
                prop="amount"
                align="right"
                width="100"
              >
                <template #default="scope"
                  >¥{{
                    (scope.row as PaymentRecordVO).amount.toFixed(2)
                  }}</template
                >
              </el-table-column>
              <el-table-column
                label="状态"
                prop="status"
                align="center"
                width="80"
              />
              <el-table-column label="回调时间" prop="callbackTime" width="170">
                <template #default="scope">{{
                  (scope.row as PaymentRecordVO).callbackTime || "-"
                }}</template>
              </el-table-column>
              <el-table-column label="创建时间" prop="createTime" width="170" />
            </el-table>
            <el-empty
              v-if="
                !detailDialog.data.paymentRecords ||
                detailDialog.data.paymentRecords.length === 0
              "
              description="无支付流水"
              :image-size="60"
            />
          </el-tab-pane>
          <el-tab-pane label="退款信息">
            <template v-if="detailDialog.data.refundRecord">
              <el-descriptions :column="2" border size="small">
                <el-descriptions-item label="退款单号">{{
                  detailDialog.data.refundRecord.refundNo
                }}</el-descriptions-item>
                <el-descriptions-item label="退款金额"
                  >¥{{
                    detailDialog.data.refundRecord.refundAmount.toFixed(2)
                  }}</el-descriptions-item
                >
                <el-descriptions-item label="退款原因">{{
                  detailDialog.data.refundRecord.reason
                }}</el-descriptions-item>
                <el-descriptions-item label="已用配额">{{
                  detailDialog.data.refundRecord.usedQuota
                }}</el-descriptions-item>
                <el-descriptions-item label="状态">{{
                  refundStatusLabel(detailDialog.data.refundRecord.status)
                }}</el-descriptions-item>
                <el-descriptions-item label="申请时间">{{
                  detailDialog.data.refundRecord.applyTime
                }}</el-descriptions-item>
                <el-descriptions-item label="审核时间">{{
                  detailDialog.data.refundRecord.auditTime || "-"
                }}</el-descriptions-item>
                <el-descriptions-item label="退款时间">{{
                  detailDialog.data.refundRecord.refundTime || "-"
                }}</el-descriptions-item>
                <el-descriptions-item
                  v-if="detailDialog.data.refundRecord.auditRemark"
                  label="审核备注"
                  >{{
                    detailDialog.data.refundRecord.auditRemark
                  }}</el-descriptions-item
                >
                <el-descriptions-item
                  v-if="detailDialog.data.refundRecord.errorMessage"
                  label="错误信息"
                  >{{
                    detailDialog.data.refundRecord.errorMessage
                  }}</el-descriptions-item
                >
              </el-descriptions>
            </template>
            <el-empty v-else description="无退款记录" :image-size="60" />
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-dialog>

    <!-- 订单统计抽屉 -->
    <el-drawer
      v-model="statsDrawer.visible"
      title="订单统计"
      size="540px"
      @opened="loadStats"
      @close="disposeCharts"
    >
      <div v-loading="statsDrawer.loading">
        <template v-if="statsDrawer.data">
          <el-row :gutter="12">
            <el-col :span="12">
              <el-card shadow="never">
                <div class="stats-label">总订单数</div>
                <div class="stats-value">
                  {{ statsDrawer.data.totalOrders }}
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never">
                <div class="stats-label">总收入</div>
                <div class="stats-value">
                  ¥{{ statsDrawer.data.totalRevenue.toFixed(2) }}
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never">
                <div class="stats-label">总退款</div>
                <div class="stats-value">
                  ¥{{ statsDrawer.data.totalRefund.toFixed(2) }}
                </div>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never">
                <div class="stats-label">退款率</div>
                <div class="stats-value">
                  {{ (statsDrawer.data.refundRate * 100).toFixed(2) }}%
                </div>
              </el-card>
            </el-col>
          </el-row>

          <div class="stats-section-title">状态分布</div>
          <div id="orderStatusChart" style="width: 100%; height: 260px"></div>

          <div class="stats-section-title">支付方式分布</div>
          <div id="payMethodChart" style="width: 100%; height: 260px"></div>

          <div class="stats-section-title">套餐分布</div>
          <div id="packageDistChart" style="width: 100%; height: 260px"></div>

          <div class="stats-section-title">每日趋势</div>
          <div id="dailyTrendChart" style="width: 100%; height: 260px"></div>
        </template>
      </div>
    </el-drawer>
  </div>
</template>

<script lang="ts" setup>
import {
  OrderAPI,
  OrderQuery,
  OrderPageVO,
  OrderDetailVO,
  OrderStatsVO,
  PaymentRecordVO,
  OrderStatus,
  PayMethod,
  RefundStatus,
} from "dehaze-sdk-js";
import { Search, Refresh, View, DataLine } from "@element-plus/icons-vue";
import * as echarts from "echarts";

defineOptions({ name: "OrderList" });

const queryFormRef = ref(ElForm);
const loading = ref(false);
const pageData = ref<OrderPageVO[]>([]);
const total = ref(0);
const paidTimeRange = ref<[string, string] | null>(null);

const queryParams = reactive<OrderQuery>({
  pageNum: 1,
  pageSize: 10,
});

const statusOptions: { label: string; value: OrderStatus }[] = [
  { label: "待支付", value: "pending" },
  { label: "已支付", value: "paid" },
  { label: "已完成", value: "completed" },
  { label: "已取消", value: "cancelled" },
  { label: "退款中", value: "refunding" },
  { label: "已退款", value: "refunded" },
];

const payMethodOptions: { label: string; value: PayMethod }[] = [
  { label: "微信支付", value: "wechat" },
  { label: "支付宝", value: "alipay" },
  { label: "余额支付", value: "balance" },
  { label: "组合支付", value: "combined" },
];

function statusLabel(status: OrderStatus): string {
  const map: Record<OrderStatus, string> = {
    pending: "待支付",
    paid: "已支付",
    completed: "已完成",
    cancelled: "已取消",
    refunding: "退款中",
    refunded: "已退款",
  };
  return map[status] || status;
}

function statusTagType(
  status: OrderStatus
): "success" | "warning" | "info" | "primary" | "danger" {
  const map: Record<
    OrderStatus,
    "success" | "warning" | "info" | "primary" | "danger"
  > = {
    pending: "warning",
    paid: "primary",
    completed: "info",
    cancelled: "info",
    refunding: "warning",
    refunded: "info",
  };
  return map[status];
}

function payMethodLabel(method: PayMethod): string {
  const map: Record<PayMethod, string> = {
    wechat: "微信支付",
    alipay: "支付宝",
    balance: "余额支付",
    combined: "组合支付",
  };
  return map[method] || method;
}

function refundStatusLabel(status: RefundStatus): string {
  const map: Record<RefundStatus, string> = {
    refunding: "退款中",
    refunded: "退款成功",
    refund_failed: "退款失败",
  };
  return map[status] || status;
}

function handleQuery() {
  loading.value = true;
  if (paidTimeRange.value && paidTimeRange.value.length === 2) {
    queryParams.paidTimeStart = paidTimeRange.value[0];
    queryParams.paidTimeEnd = paidTimeRange.value[1];
  } else {
    queryParams.paidTimeStart = undefined;
    queryParams.paidTimeEnd = undefined;
  }
  OrderAPI.getPage(queryParams)
    .then((data) => {
      pageData.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  paidTimeRange.value = null;
  queryParams.orderNo = undefined;
  queryParams.keywords = undefined;
  queryParams.status = undefined;
  queryParams.payMethod = undefined;
  queryParams.amountMin = undefined;
  queryParams.amountMax = undefined;
  queryParams.paidTimeStart = undefined;
  queryParams.paidTimeEnd = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

const detailDialog = reactive<{
  visible: boolean;
  loading: boolean;
  data: OrderDetailVO | null;
}>({
  visible: false,
  loading: false,
  data: null,
});

function handleDetail(row: OrderPageVO) {
  detailDialog.visible = true;
  detailDialog.loading = true;
  detailDialog.data = null;
  OrderAPI.getDetail(row.orderNo)
    .then((data) => {
      detailDialog.data = data;
    })
    .finally(() => {
      detailDialog.loading = false;
    });
}

const statsDrawer = reactive<{
  visible: boolean;
  loading: boolean;
  data: OrderStatsVO | null;
}>({
  visible: false,
  loading: false,
  data: null,
});

function openStats() {
  statsDrawer.visible = true;
}

function loadStats() {
  statsDrawer.loading = true;
  statsDrawer.data = null;
  OrderAPI.getStats()
    .then((data) => {
      statsDrawer.data = data;
      nextTick(() => initCharts(data));
    })
    .finally(() => {
      statsDrawer.loading = false;
    });
}

const statusColorMap: Record<OrderStatus, string> = {
  pending: "#e6a23c",
  paid: "#409eff",
  completed: "#909399",
  cancelled: "#909399",
  refunding: "#e6a23c",
  refunded: "#909399",
};

const statusChart = ref<echarts.ECharts | null>(null);
const payMethodChart = ref<echarts.ECharts | null>(null);
const packageDistChart = ref<echarts.ECharts | null>(null);
const dailyTrendChart = ref<echarts.ECharts | null>(null);

function initCharts(data: OrderStatsVO) {
  disposeCharts();

  const statusEl = document.getElementById("orderStatusChart");
  if (statusEl) {
    statusChart.value = markRaw(echarts.init(statusEl));
    statusChart.value.setOption({
      tooltip: { trigger: "item", formatter: "{a} <br/>{b}: {c} ({d}%)" },
      legend: { bottom: 0 },
      series: [
        {
          name: "订单数",
          type: "pie",
          radius: ["40%", "70%"],
          data: Object.entries(data.statusDistribution).map(
            ([status, value]) => ({
              name: statusLabel(status as OrderStatus),
              value,
              itemStyle: { color: statusColorMap[status as OrderStatus] },
            })
          ),
        },
      ],
    });
  }

  const payEl = document.getElementById("payMethodChart");
  if (payEl) {
    payMethodChart.value = markRaw(echarts.init(payEl));
    payMethodChart.value.setOption({
      tooltip: { trigger: "item", formatter: "{a} <br/>{b}: {c} ({d}%)" },
      legend: { bottom: 0 },
      series: [
        {
          name: "订单数",
          type: "pie",
          radius: ["40%", "70%"],
          data: Object.entries(data.payMethodDistribution).map(
            ([method, value]) => ({
              name: payMethodLabel(method as PayMethod),
              value,
            })
          ),
        },
      ],
    });
  }

  const pkgEl = document.getElementById("packageDistChart");
  if (pkgEl) {
    packageDistChart.value = markRaw(echarts.init(pkgEl));
    packageDistChart.value.setOption({
      tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
      legend: { bottom: 0, data: ["收入", "订单数"] },
      grid: { left: "2%", right: "5%", bottom: "15%", containLabel: true },
      xAxis: {
        type: "category",
        data: data.packageDistribution.map((p) => p.packageName),
      },
      yAxis: [
        { type: "value", name: "收入", axisLabel: { formatter: "¥{value}" } },
        { type: "value", name: "订单数" },
      ],
      series: [
        {
          name: "收入",
          type: "bar",
          data: data.packageDistribution.map((p) => p.revenue),
        },
        {
          name: "订单数",
          type: "line",
          yAxisIndex: 1,
          data: data.packageDistribution.map((p) => p.count),
        },
      ],
    });
  }

  const dailyEl = document.getElementById("dailyTrendChart");
  if (dailyEl) {
    dailyTrendChart.value = markRaw(echarts.init(dailyEl));
    dailyTrendChart.value.setOption({
      tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
      legend: { bottom: 0, data: ["收入", "订单数"] },
      grid: { left: "2%", right: "5%", bottom: "15%", containLabel: true },
      xAxis: {
        type: "category",
        data: data.dailyStats.map((d) => d.date),
      },
      yAxis: [
        { type: "value", name: "收入", axisLabel: { formatter: "¥{value}" } },
        { type: "value", name: "订单数" },
      ],
      series: [
        {
          name: "收入",
          type: "line",
          yAxisIndex: 0,
          data: data.dailyStats.map((d) => d.revenue),
        },
        {
          name: "订单数",
          type: "bar",
          yAxisIndex: 1,
          data: data.dailyStats.map((d) => d.count),
        },
      ],
    });
  }
}

function disposeCharts() {
  statusChart.value?.dispose();
  statusChart.value = null;
  payMethodChart.value?.dispose();
  payMethodChart.value = null;
  packageDistChart.value?.dispose();
  packageDistChart.value = null;
  dailyTrendChart.value?.dispose();
  dailyTrendChart.value = null;
}

function handleResize() {
  statusChart.value?.resize();
  payMethodChart.value?.resize();
  packageDistChart.value?.resize();
  dailyTrendChart.value?.resize();
}

onMounted(() => {
  handleQuery();
  window.addEventListener("resize", handleResize);
});

onBeforeUnmount(() => {
  disposeCharts();
  window.removeEventListener("resize", handleResize);
});
</script>

<style lang="scss" scoped>
.stats-label {
  margin-bottom: 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.stats-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.stats-section-title {
  margin: 20px 0 10px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}
</style>
