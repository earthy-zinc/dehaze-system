<template>
  <div class="app-container order-detail-page">
    <div class="detail-wrapper" v-loading="loading">
      <template v-if="order">
        <div class="detail-header">
          <el-button link @click="goBack">
            <el-icon><ArrowLeft /></el-icon>
            返回我的订单
          </el-button>
        </div>

        <div :class="['status-card', `status-${order.status}`]">
          <div class="status-card-stripe"></div>
          <div class="status-card-content">
            <div class="status-row">
              <span class="status-text">{{ statusLabel(order.status) }}</span>
              <el-tag
                :type="statusTagType(order.status)"
                size="default"
                effect="light"
                >{{ statusLabel(order.status) }}</el-tag
              >
            </div>
            <div class="meta-row">
              <span class="order-no">订单号：{{ order.orderNo }}</span>
              <span class="create-time">创建时间：{{ order.createTime }}</span>
            </div>
            <div v-if="order.cancelReason" class="cancel-reason">
              取消原因：{{ order.cancelReason }}
            </div>
          </div>
        </div>

        <div class="info-card">
          <div class="info-card-title">套餐信息</div>
          <div class="info-card-body">
            <div class="package-name">
              <el-icon><CreditCard /></el-icon>
              <span>{{ order.packageName }}</span>
              <el-tag size="small" type="info" effect="plain">{{
                order.packageLevel
              }}</el-tag>
            </div>
            <div
              v-if="order.effectiveTime || order.expireTime"
              class="validity"
            >
              <el-icon><Clock /></el-icon>
              <span v-if="order.effectiveTime"
                >生效：{{ order.effectiveTime }}</span
              >
              <span v-if="order.expireTime">到期：{{ order.expireTime }}</span>
            </div>
          </div>
        </div>

        <div class="info-card">
          <div class="info-card-title">金额明细</div>
          <div class="info-card-body">
            <div class="amount-row">
              <span class="amount-label">订单原价</span>
              <span class="amount-value"
                >¥{{ order.originalPrice.toFixed(2) }}</span
              >
            </div>
            <div v-if="order.discountAmount > 0" class="amount-row discount">
              <span class="amount-label">折扣优惠</span>
              <span class="amount-value"
                >-¥{{ order.discountAmount.toFixed(2) }}</span
              >
            </div>
            <div v-if="order.couponAmount > 0" class="amount-row discount">
              <span class="amount-label">优惠券抵扣</span>
              <span class="amount-value"
                >-¥{{ order.couponAmount.toFixed(2) }}</span
              >
            </div>
            <el-divider />
            <div class="amount-row total">
              <span class="amount-label">实付金额</span>
              <span class="amount-value payable"
                >¥{{ order.payableAmount.toFixed(2) }}</span
              >
            </div>
            <div v-if="order.paidAmount > 0" class="amount-row">
              <span class="amount-label">已支付金额</span>
              <span class="amount-value"
                >¥{{ order.paidAmount.toFixed(2) }}</span
              >
            </div>
          </div>
        </div>

        <div class="info-card">
          <div class="info-card-title">支付信息</div>
          <div class="info-card-body">
            <div class="amount-row">
              <span class="amount-label">支付方式</span>
              <span class="amount-value">{{
                order.payMethod ? payMethodLabel(order.payMethod) : "-"
              }}</span>
            </div>
            <div class="amount-row">
              <span class="amount-label">支付时间</span>
              <span class="amount-value">{{ order.paidTime || "-" }}</span>
            </div>
            <div v-if="order.isAutoRenew !== undefined" class="amount-row">
              <span class="amount-label">自动续费</span>
              <span class="amount-value">{{
                order.isAutoRenew ? "已开启" : "未开启"
              }}</span>
            </div>
            <div
              v-if="order.paymentRecords && order.paymentRecords.length > 0"
              class="payment-records"
            >
              <div class="payment-records-title">支付流水</div>
              <el-table
                :data="order.paymentRecords"
                border
                size="small"
                style="margin-top: 8px"
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
                <el-table-column
                  label="回调时间"
                  prop="callbackTime"
                  width="170"
                />
              </el-table>
            </div>
          </div>
        </div>

        <div v-if="order.refundRecord" class="info-card refund-card">
          <div class="info-card-title">退款信息</div>
          <div class="info-card-body">
            <div class="amount-row">
              <span class="amount-label">退款单号</span>
              <span class="amount-value">{{
                order.refundRecord.refundNo
              }}</span>
            </div>
            <div class="amount-row">
              <span class="amount-label">退款金额</span>
              <span class="amount-value refund-amount"
                >¥{{ order.refundRecord.refundAmount.toFixed(2) }}</span
              >
            </div>
            <div class="amount-row">
              <span class="amount-label">退款原因</span>
              <span class="amount-value">{{ order.refundRecord.reason }}</span>
            </div>
            <div class="amount-row">
              <span class="amount-label">已用配额</span>
              <span class="amount-value">{{
                order.refundRecord.usedQuota
              }}</span>
            </div>
            <div class="amount-row">
              <span class="amount-label">退款状态</span>
              <el-tag
                :type="refundStatusTagType(order.refundRecord.status)"
                size="small"
                >{{ refundStatusLabel(order.refundRecord.status) }}</el-tag
              >
            </div>
            <div class="amount-row">
              <span class="amount-label">申请时间</span>
              <span class="amount-value">{{
                order.refundRecord.applyTime
              }}</span>
            </div>
            <div v-if="order.refundRecord.auditTime" class="amount-row">
              <span class="amount-label">审核时间</span>
              <span class="amount-value">{{
                order.refundRecord.auditTime
              }}</span>
            </div>
            <div v-if="order.refundRecord.refundTime" class="amount-row">
              <span class="amount-label">退款时间</span>
              <span class="amount-value">{{
                order.refundRecord.refundTime
              }}</span>
            </div>
            <div v-if="order.refundRecord.auditRemark" class="amount-row">
              <span class="amount-label">审核备注</span>
              <span class="amount-value">{{
                order.refundRecord.auditRemark
              }}</span>
            </div>
            <div v-if="order.refundRecord.errorMessage" class="amount-row">
              <span class="amount-label">错误信息</span>
              <span class="amount-value error-text">{{
                order.refundRecord.errorMessage
              }}</span>
            </div>
          </div>
        </div>

        <div v-if="showFooterActions" class="footer-action-bar">
          <template v-if="order.status === 'pending'">
            <el-button type="primary" @click="openPaymentDialog"
              >立即支付</el-button
            >
            <el-button @click="handleCancel">取消订单</el-button>
          </template>
          <template v-else-if="order.status === 'paid'">
            <el-button @click="goApplyRefund">
              <el-icon><RefreshLeft /></el-icon>申请退款
            </el-button>
          </template>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="订单不存在或已被删除">
        <el-button type="primary" @click="goBack">返回我的订单</el-button>
      </el-empty>
    </div>

    <!-- 支付弹窗 -->
    <el-dialog
      v-model="paymentDialog.visible"
      title="订单支付"
      width="420px"
      @close="closePaymentDialog"
    >
      <div v-if="order" class="payment-summary">
        <div class="payment-row">
          <span class="label">订单号：</span>
          <span>{{ order.orderNo }}</span>
        </div>
        <div class="payment-row">
          <span class="label">应付金额：</span>
          <span class="pay-amount">¥{{ order.payableAmount.toFixed(2) }}</span>
        </div>
      </div>

      <el-form label-width="80px" style="margin-top: 16px">
        <el-form-item label="支付方式">
          <el-radio-group v-model="paymentForm.payMethod">
            <el-radio-button value="wechat">
              <el-icon><ChatDotRound /></el-icon> 微信支付
            </el-radio-button>
            <el-radio-button value="alipay">
              <el-icon><Wallet /></el-icon> 支付宝
            </el-radio-button>
            <el-radio-button value="balance">
              <el-icon><CreditCard /></el-icon> 余额支付
            </el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>

      <div v-if="paymentDialog.payResult" class="pay-result-box">
        <template v-if="paymentDialog.payResult.paid">
          <el-result
            icon="success"
            title="支付成功"
            sub-title="订单已支付完成"
          />
        </template>
        <template v-else>
          <el-result icon="info" title="请扫码支付" :sub-title="payHint" />
          <div v-if="qrCodeDataUrl" class="pay-qr-code">
            <img :src="qrCodeDataUrl" alt="支付二维码" class="qr-img" />
          </div>
          <div v-if="paymentDialog.payResult.payUrl" class="pay-link-row">
            <a
              :href="paymentDialog.payResult.payUrl"
              target="_blank"
              rel="noopener"
              >点击打开支付页面 →</a
            >
          </div>
        </template>
      </div>

      <template #footer>
        <div class="dialog-footer" v-if="!paymentDialog.payResult">
          <el-button
            type="primary"
            :loading="paymentDialog.loading"
            @click="handlePay"
            >确认支付</el-button
          >
          <el-button @click="closePaymentDialog">取 消</el-button>
        </div>
        <div v-else class="dialog-footer">
          <el-button @click="closePaymentDialog">关 闭</el-button>
          <el-button
            v-if="paymentDialog.payResult.paid"
            type="primary"
            @click="handlePaySuccess"
            >完成</el-button
          >
          <el-button v-else type="primary" @click="handlePay"
            >重新支付</el-button
          >
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  OrderAPI,
  OrderDetailVO,
  PaymentRecordVO,
  PayRequest,
  PayResult,
  PayMethod,
  OrderStatus,
  RefundStatus,
} from "dehaze-sdk-js";
import QRCode from "qrcode";
import {
  ArrowLeft,
  CreditCard,
  Clock,
  ChatDotRound,
  Wallet,
  RefreshLeft,
} from "@element-plus/icons-vue";

defineOptions({ name: "OrderDetail" });

const route = useRoute();
const router = useRouter();
const loading = ref(false);
const order = ref<OrderDetailVO | null>(null);

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

function refundStatusTagType(
  status: RefundStatus
): "success" | "warning" | "info" | "primary" | "danger" {
  const map: Record<
    RefundStatus,
    "success" | "warning" | "info" | "primary" | "danger"
  > = {
    refunding: "warning",
    refunded: "info",
    refund_failed: "danger",
  };
  return map[status];
}

const showFooterActions = computed(() => {
  if (!order.value) return false;
  return order.value.status === "pending" || order.value.status === "paid";
});

function loadDetail() {
  const orderNo = route.query.orderNo as string;
  if (!orderNo) {
    order.value = null;
    return;
  }
  loading.value = true;
  OrderAPI.getDetail(orderNo)
    .then((data) => {
      order.value = data;
    })
    .catch(() => {
      order.value = null;
    })
    .finally(() => {
      loading.value = false;
    });
}

function goBack() {
  router.push("/order/my");
}

function goApplyRefund() {
  router.push("/order/my");
}

function handleCancel() {
  if (!order.value) return;
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
      return OrderAPI.cancel(order.value!.orderNo, value.trim());
    })
    .then(() => {
      ElMessage.success("订单已取消");
      loadDetail();
    })
    .catch(() => {});
}

const paymentDialog = reactive<{
  visible: boolean;
  loading: boolean;
  payResult: PayResult | null;
}>({
  visible: false,
  loading: false,
  payResult: null,
});

const paymentForm = reactive<PayRequest>({
  payMethod: "wechat",
});

const qrCodeDataUrl = ref("");

const payHint = computed(() => {
  if (!paymentDialog.payResult) return "";
  if (paymentDialog.payResult.payMethod === "wechat")
    return "请使用微信扫码完成支付";
  if (paymentDialog.payResult.payMethod === "alipay")
    return "请使用支付宝扫码完成支付";
  return "请完成支付";
});

function openPaymentDialog() {
  paymentDialog.payResult = null;
  qrCodeDataUrl.value = "";
  paymentDialog.visible = true;
}

function closePaymentDialog() {
  paymentDialog.visible = false;
  paymentDialog.payResult = null;
  paymentDialog.loading = false;
  qrCodeDataUrl.value = "";
}

function handlePay() {
  if (!order.value) return;
  paymentDialog.loading = true;
  qrCodeDataUrl.value = "";
  OrderAPI.pay(order.value.orderNo, { payMethod: paymentForm.payMethod })
    .then((data: PayResult) => {
      paymentDialog.payResult = data;
      if (data.paid) {
        ElMessage.success("支付成功");
      } else {
        const qrContent = data.qrCode || data.payUrl || "";
        if (qrContent) {
          QRCode.toDataURL(qrContent, {
            width: 240,
            margin: 2,
            color: { dark: "#000000", light: "#ffffff" },
          }).then((url) => {
            qrCodeDataUrl.value = url;
          });
        }
      }
    })
    .finally(() => {
      paymentDialog.loading = false;
    });
}

function handlePaySuccess() {
  closePaymentDialog();
  loadDetail();
}

onMounted(() => {
  loadDetail();
});

watch(
  () => route.query.orderNo,
  () => {
    loadDetail();
  }
);
</script>

<style lang="scss" scoped>
.order-detail-page {
  max-width: 800px;
  padding: 24px 20px 80px;
  margin: 0 auto;
}

.detail-header {
  margin-bottom: 16px;
}

.status-card {
  display: flex;
  margin-bottom: 16px;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;

  .status-card-stripe {
    flex-shrink: 0;
    width: 5px;
  }

  &.status-pending .status-card-stripe {
    background: linear-gradient(180deg, #e6a23c, #f3d19e);
  }

  &.status-paid .status-card-stripe {
    background: linear-gradient(180deg, var(--el-color-primary), #79bbff);
  }

  &.status-completed .status-card-stripe,
  &.status-cancelled .status-card-stripe,
  &.status-refunded .status-card-stripe {
    background: linear-gradient(180deg, #909399, #c8c9c4);
  }

  &.status-refunding .status-card-stripe {
    background: linear-gradient(180deg, #e6a23c, #f3d19e);
  }

  .status-card-content {
    flex: 1;
    padding: 18px 22px;
  }

  .status-row {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 8px;

    .status-text {
      font-size: 22px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }
  }

  .meta-row {
    display: flex;
    gap: 20px;
    align-items: center;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }

  .cancel-reason {
    margin-top: 8px;
    font-size: 13px;
    color: var(--el-color-danger);
  }
}

.info-card {
  margin-bottom: 16px;
  overflow: hidden;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;

  .info-card-title {
    padding: 12px 20px;
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);
    background: var(--el-fill-color-light);
    border-bottom: 1px solid var(--el-border-color-lighter);
  }

  .info-card-body {
    padding: 16px 20px;
  }

  .package-name {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);

    .el-icon {
      color: var(--el-color-primary);
    }
  }

  .validity {
    display: flex;
    gap: 16px;
    align-items: center;
    margin-top: 10px;
    font-size: 13px;
    color: var(--el-text-color-secondary);

    .el-icon {
      margin-right: 2px;
    }
  }

  .amount-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 14px;
    color: var(--el-text-color-regular);

    .amount-label {
      color: var(--el-text-color-secondary);
    }

    &.discount .amount-value {
      color: var(--el-color-danger);
    }

    &.total {
      font-size: 16px;
      font-weight: 600;

      .amount-value.payable {
        font-size: 18px;
        color: var(--el-color-danger);
      }
    }
  }

  .payment-records {
    margin-top: 12px;

    .payment-records-title {
      font-size: 13px;
      font-weight: 500;
      color: var(--el-text-color-secondary);
    }
  }

  &.refund-card .refund-amount {
    font-weight: 600;
    color: var(--el-color-danger);
  }

  .error-text {
    color: var(--el-color-danger);
  }
}

.footer-action-bar {
  display: flex;
  gap: 12px;
  justify-content: center;
  padding: 20px 0;
}

.payment-summary {
  padding: 12px 14px;
  background: var(--el-fill-color-light);
  border-radius: 8px;

  .payment-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 13px;
    line-height: 1.8;
    color: var(--el-text-color-regular);

    .label {
      color: var(--el-text-color-secondary);
    }

    .pay-amount {
      font-size: 16px;
      font-weight: 600;
      color: var(--el-color-danger);
    }
  }
}

.pay-result-box {
  padding: 8px 0;
  margin-top: 8px;
  border-top: 1px dashed var(--el-border-color);

  .pay-link-row {
    margin-top: 8px;
    text-align: center;

    a {
      color: var(--el-color-primary);
      text-decoration: none;
    }
  }

  .pay-qr-code {
    display: flex;
    justify-content: center;
    margin-top: 12px;

    .qr-img {
      width: 200px;
      height: 200px;
      border: 1px solid var(--el-border-color);
      border-radius: 8px;
    }
  }
}
</style>
