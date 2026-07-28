<template>
  <div class="app-container my-order-center">
    <div class="page-header">
      <span class="title-text">我的订单</span>
    </div>

    <div class="status-tabs">
      <button
        v-for="tab in statusTabs"
        :key="tab.value ?? 'all'"
        :class="['status-tab', { active: activeStatus === tab.value }]"
        @click="handleTabChange(tab.value)"
      >
        {{ tab.label }}
      </button>
    </div>

    <div v-loading="loading" class="order-list">
      <template v-if="orderList.length > 0">
        <div
          v-for="order in orderList"
          :key="order.id"
          :class="['order-card', `status-${order.status}`]"
        >
          <div class="card-header">
            <div class="header-left">
              <span class="order-no">订单号：{{ order.orderNo }}</span>
              <span class="create-time">{{ order.createTime }}</span>
            </div>
            <el-tag
              :type="statusTagType(order.status)"
              size="small"
              effect="light"
            >
              {{ statusLabel(order.status) }}
            </el-tag>
          </div>

          <div class="card-body">
            <div class="package-info">
              <el-icon class="info-icon"><CreditCard /></el-icon>
              <div class="info-content">
                <div class="package-name">
                  {{ order.packageName }}
                  <el-tag size="small" type="info" effect="plain">{{
                    order.packageLevel
                  }}</el-tag>
                </div>
                <div v-if="order.packageExpireTime" class="package-expire">
                  <el-icon><Clock /></el-icon>
                  到期时间：{{ order.packageExpireTime }}
                </div>
              </div>
            </div>
          </div>

          <div class="card-footer">
            <div class="amount-area">
              <span class="payable-amount"
                >¥{{ order.payableAmount.toFixed(2) }}</span
              >
              <span
                v-if="
                  order.payableAmount < order.paidAmount || hasDiscount(order)
                "
                class="original-amount"
                >¥{{ order.paidAmount.toFixed(2) }}</span
              >
              <span v-if="order.payMethod" class="pay-method-text">
                {{ payMethodLabel(order.payMethod) }}
              </span>
            </div>
            <div class="action-area">
              <template v-if="order.status === 'pending'">
                <el-button size="small" type="primary" @click="goDetail(order)"
                  >去支付</el-button
                >
                <el-button size="small" @click="handleCancel(order)"
                  >取消订单</el-button
                >
              </template>
              <template v-else-if="order.status === 'paid'">
                <el-button size="small" @click="openRefundDialog(order)">
                  <el-icon><RefreshLeft /></el-icon>申请退款
                </el-button>
                <el-button size="small" @click="goDetail(order)"
                  >查看详情</el-button
                >
              </template>
              <template v-else>
                <el-button size="small" @click="goDetail(order)"
                  >查看详情</el-button
                >
              </template>
            </div>
          </div>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="暂无订单" :image-size="120" />
    </div>

    <pagination
      v-if="total > 0"
      v-model:limit="queryParams.pageSize"
      v-model:page="queryParams.pageNum"
      v-model:total="total"
      @pagination="handleQuery"
    />

    <!-- 退款申请弹窗 -->
    <el-dialog
      v-model="refundDialog.visible"
      title="申请退款"
      width="500px"
      @close="closeRefundDialog"
    >
      <div v-if="refundDialog.row" class="refund-order-info">
        <div class="refund-info-row">
          <span class="label">订单号：</span>
          <span>{{ refundDialog.row.orderNo }}</span>
        </div>
        <div class="refund-info-row">
          <span class="label">套餐：</span>
          <span
            >{{ refundDialog.row.packageName }} ({{
              refundDialog.row.packageLevel
            }})</span
          >
        </div>
        <div class="refund-info-row">
          <span class="label">退款金额：</span>
          <span class="refund-amount"
            >¥{{ refundDialog.row.paidAmount.toFixed(2) }}</span
          >
        </div>
      </div>

      <el-alert
        type="warning"
        :closable="false"
        show-icon
        style="margin-bottom: 16px"
      >
        <template #title>退款规则</template>
        <div class="refund-rules">
          <p>· 购买 7 天内可申请退款</p>
          <p>· 已使用配额超过 50% 不可退款</p>
          <p>· 退款金额按未使用配额比例退还</p>
          <p>· 退款审核通过后将原路退回</p>
        </div>
      </el-alert>

      <el-form
        ref="refundFormRef"
        :model="refundForm"
        :rules="refundRules"
        label-width="100px"
      >
        <el-form-item label="退款原因" prop="reason">
          <el-select
            v-model="refundForm.reason"
            placeholder="请选择退款原因"
            style="width: 100%"
          >
            <el-option
              v-for="opt in refundReasonOptions"
              :key="opt"
              :label="opt"
              :value="opt"
            />
          </el-select>
        </el-form-item>
        <el-form-item
          v-if="refundForm.reason === '其他原因'"
          label="具体原因"
          prop="customReason"
        >
          <el-input
            v-model="refundForm.customReason"
            type="textarea"
            :rows="3"
            placeholder="请描述具体退款原因"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleRefundSubmit"
            >提交申请</el-button
          >
          <el-button @click="closeRefundDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  OrderAPI,
  MyOrderQuery,
  MyOrderVO,
  OrderStatus,
  PayMethod,
  RefundApplyForm,
} from "dehaze-sdk-js";
import { CreditCard, Clock, RefreshLeft } from "@element-plus/icons-vue";

defineOptions({ name: "OrderMy" });

const router = useRouter();
const refundFormRef = ref(ElForm);
const loading = ref(false);
const orderList = ref<MyOrderVO[]>([]);
const total = ref(0);
const activeStatus = ref<OrderStatus | undefined>(undefined);

const queryParams = reactive<MyOrderQuery>({
  pageNum: 1,
  pageSize: 10,
});

const statusTabs: { label: string; value: OrderStatus | undefined }[] = [
  { label: "全部", value: undefined },
  { label: "待支付", value: "pending" },
  { label: "已支付", value: "paid" },
  { label: "已完成", value: "completed" },
  { label: "已取消", value: "cancelled" },
  { label: "退款中", value: "refunding" },
  { label: "已退款", value: "refunded" },
];

const refundReasonOptions = [
  "功能不满足需求",
  "使用体验不佳",
  "重复购买",
  "暂不需要",
  "其他原因",
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

function hasDiscount(order: MyOrderVO): boolean {
  return order.paidAmount > order.payableAmount && order.payableAmount > 0;
}

function handleTabChange(value: OrderStatus | undefined) {
  activeStatus.value = value;
  queryParams.status = value;
  queryParams.pageNum = 1;
  handleQuery();
}

function handleQuery() {
  loading.value = true;
  OrderAPI.listMy(queryParams)
    .then((data) => {
      orderList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function goDetail(order: MyOrderVO) {
  router.push({ path: "/order/detail", query: { orderNo: order.orderNo } });
}

function handleCancel(order: MyOrderVO) {
  ElMessageBox.prompt("请输入取消订单的原因", "取消订单", {
    confirmButtonText: "确定取消",
    cancelButtonText: "返回",
    inputType: "textarea",
    inputPlaceholder: "请输入取消原因",
    inputValidator: (val: string) => {
      if (!val || !val.trim()) return "取消原因不能为空";
      return true;
    },
    lockScroll: false,
  })
    .then(({ value }) => {
      return OrderAPI.cancel(order.orderNo, value.trim());
    })
    .then(() => {
      ElMessage.success("订单已取消");
      handleQuery();
    })
    .catch(() => {});
}

const refundDialog = reactive<{
  visible: boolean;
  loading: boolean;
  row: MyOrderVO | null;
}>({
  visible: false,
  loading: false,
  row: null,
});

const refundForm = reactive<RefundApplyForm>({
  reason: "",
  customReason: "",
});

const refundRules = {
  reason: [{ required: true, message: "请选择退款原因", trigger: "change" }],
  customReason: [
    {
      validator: (_rule: any, value: string, callback: any) => {
        if (refundForm.reason === "其他原因" && !value?.trim()) {
          callback(new Error("请描述具体退款原因"));
        } else {
          callback();
        }
      },
      trigger: "blur",
    },
  ],
};

function openRefundDialog(order: MyOrderVO) {
  refundDialog.row = order;
  refundForm.reason = "";
  refundForm.customReason = "";
  refundDialog.visible = true;
}

function closeRefundDialog() {
  refundDialog.visible = false;
  refundDialog.row = null;
  refundForm.reason = "";
  refundForm.customReason = "";
  refundFormRef.value?.resetFields();
}

function handleRefundSubmit() {
  const row = refundDialog.row;
  if (!row) return;
  refundFormRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    refundDialog.loading = true;
    const payload: RefundApplyForm = {
      reason: refundForm.reason,
      customReason:
        refundForm.reason === "其他原因" ? refundForm.customReason : undefined,
    };
    OrderAPI.applyRefund(row.orderNo, payload)
      .then(() => {
        ElMessage.success("退款申请已提交");
        closeRefundDialog();
        handleQuery();
      })
      .finally(() => {
        refundDialog.loading = false;
      });
  });
}

onMounted(() => {
  handleQuery();
});

onActivated(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.my-order-center {
  max-width: 960px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 20px;

  .title-text {
    font-size: 22px;
    font-weight: 600;
    color: var(--el-text-color-primary);
    letter-spacing: 0.5px;
  }
}

.status-tabs {
  display: flex;
  gap: 4px;
  align-items: center;
  margin-bottom: 16px;
  overflow-x: auto;
  scrollbar-width: none;

  &::-webkit-scrollbar {
    display: none;
  }
}

.status-tab {
  display: inline-flex;
  align-items: center;
  padding: 6px 14px;
  font-size: 13px;
  font-weight: 500;
  color: var(--el-text-color-regular);
  white-space: nowrap;
  cursor: pointer;
  background: transparent;
  border: none;
  border-radius: 18px;
  transition: all 0.2s ease;

  &:hover {
    color: var(--el-color-primary);
    background: var(--el-color-primary-light-9);
  }

  &.active {
    color: #fff;
    background: var(--el-color-primary);
  }
}

.order-list {
  min-height: 240px;
}

.order-card {
  margin-bottom: 12px;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-left: 4px solid var(--el-border-color);
  border-radius: 10px;
  transition: all 0.25s ease;

  &:hover {
    border-color: var(--el-color-primary-light-5);
    box-shadow: 0 4px 16px rgb(0 0 0 / 6%);
  }

  &.status-pending {
    border-left-color: #e6a23c;
  }

  &.status-paid {
    border-left-color: var(--el-color-primary);
  }

  &.status-completed,
  &.status-cancelled,
  &.status-refunded {
    border-left-color: var(--el-text-color-disabled);
  }

  &.status-refunding {
    border-left-color: #e6a23c;
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background: var(--el-fill-color-light);
    border-bottom: 1px solid var(--el-border-color-lighter);

    .header-left {
      display: flex;
      gap: 14px;
      align-items: center;

      .order-no {
        font-size: 13px;
        font-weight: 600;
        color: var(--el-text-color-primary);
      }

      .create-time {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }
  }

  .card-body {
    padding: 16px 18px;
  }

  .package-info {
    display: flex;
    gap: 10px;
    align-items: flex-start;

    .info-icon {
      margin-top: 2px;
      font-size: 18px;
      color: var(--el-color-primary);
    }

    .info-content {
      flex: 1;
    }

    .package-name {
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .package-expire {
      display: flex;
      gap: 4px;
      align-items: center;
      margin-top: 6px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-top: 1px solid var(--el-border-color-lighter);

    .amount-area {
      display: flex;
      gap: 8px;
      align-items: baseline;

      .payable-amount {
        font-size: 18px;
        font-weight: 600;
        color: var(--el-color-danger);
      }

      .original-amount {
        font-size: 13px;
        color: var(--el-text-color-secondary);
        text-decoration: line-through;
      }

      .pay-method-text {
        margin-left: 8px;
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }

    .action-area {
      display: flex;
      gap: 8px;
      align-items: center;
    }
  }
}

.refund-order-info {
  padding: 12px 14px;
  margin-bottom: 16px;
  background: var(--el-fill-color-light);
  border-radius: 8px;

  .refund-info-row {
    display: flex;
    align-items: center;
    font-size: 13px;
    line-height: 1.8;
    color: var(--el-text-color-regular);

    .label {
      width: 80px;
      color: var(--el-text-color-secondary);
    }

    .refund-amount {
      font-weight: 600;
      color: var(--el-color-danger);
    }
  }
}

.refund-rules {
  margin: 0;
  font-size: 12px;
  line-height: 1.8;
  color: var(--el-text-color-regular);

  p {
    margin: 0;
  }
}
</style>
