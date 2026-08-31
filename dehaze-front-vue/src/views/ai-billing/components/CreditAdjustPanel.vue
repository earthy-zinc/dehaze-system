<!-- 积分调整：补扣/补偿表单 + 误扣申诉审核 + 目标用户余额/流水 -->
<template>
  <div>
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="90px"
      style="max-width: 480px"
    >
      <el-form-item label="用户ID" prop="userId">
        <el-input-number
          v-model="form.userId"
          :min="1"
          controls-position="right"
          @change="handleUserChange"
        />
      </el-form-item>
      <el-form-item label="调整金额" prop="amount">
        <el-input-number
          v-model="form.amount"
          :precision="0"
          controls-position="right"
        />
        <span class="ml-2 text-xs text-gray-400"
          >正为补偿增加，负为误扣补回</span
        >
      </el-form-item>
      <el-form-item label="原因" prop="reason">
        <el-input
          v-model="form.reason"
          type="textarea"
          :rows="3"
          placeholder="调整原因（记录到流水，用户可见）"
        />
      </el-form-item>
      <el-form-item>
        <el-button
          v-hasPerm="['ai:billing:adjust']"
          type="primary"
          :loading="submitting"
          @click="submit"
        >
          提交调整
        </el-button>
      </el-form-item>
    </el-form>

    <!-- 误扣申诉审核（ai:billing:refund） -->
    <el-divider content-position="left">误扣申诉审核</el-divider>
    <div v-hasPerm="['ai:billing:refund']">
      <div class="mb-2">
        <el-select
          v-model="billingStore.refundFilter.status"
          clearable
          placeholder="全部状态"
          style="width: 140px"
          @change="
            billingStore.refundPageNum = 1;
            billingStore.fetchRefunds();
          "
        >
          <el-option label="待审核" :value="1" />
          <el-option label="已通过" :value="2" />
          <el-option label="已驳回" :value="3" />
        </el-select>
      </div>
      <el-table
        v-loading="billingStore.refundLoading"
        :data="billingStore.refunds"
        size="small"
      >
        <el-table-column prop="id" label="申请ID" width="80" align="center" />
        <el-table-column
          prop="userId"
          label="用户ID"
          width="80"
          align="center"
        />
        <el-table-column
          prop="billingId"
          label="计费记录"
          width="90"
          align="center"
        />
        <el-table-column prop="amount" label="金额" width="80" align="center" />
        <el-table-column
          prop="reason"
          label="原因"
          min-width="140"
          show-overflow-tooltip
        />
        <el-table-column label="状态" width="90" align="center">
          <template #default="{ row }">
            <el-tag :type="refundStatusTag(row.status).type" size="small">
              {{ refundStatusTag(row.status).label }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createTime" label="申请时间" width="160" />
        <el-table-column
          v-if="billingStore.refunds.some((r) => r.status === 1)"
          label="操作"
          width="140"
          align="center"
        >
          <template #default="{ row }">
            <template v-if="row.status === 1">
              <el-button
                link
                type="success"
                size="small"
                @click="openAudit(row as BillingRefundVO, true)"
              >
                通过
              </el-button>
              <el-button
                link
                type="danger"
                size="small"
                @click="openAudit(row as BillingRefundVO, false)"
              >
                驳回
              </el-button>
            </template>
          </template>
        </el-table-column>
      </el-table>
      <div class="mt-2 flex justify-end">
        <el-pagination
          v-model:current-page="billingStore.refundPageNum"
          v-model:page-size="billingStore.refundPageSize"
          :total="billingStore.refundTotal"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @current-change="billingStore.fetchRefunds()"
          @size-change="
            billingStore.refundPageNum = 1;
            billingStore.fetchRefunds();
          "
        />
      </div>
    </div>

    <!-- 审核弹窗 -->
    <el-dialog
      v-model="auditDialog.visible"
      :title="auditDialog.approved ? '通过审核' : '驳回申请'"
      width="420px"
    >
      <el-form label-width="90px">
        <el-form-item label="用户ID">
          <span>{{ auditDialog.row?.userId }}</span>
        </el-form-item>
        <el-form-item label="退款原因">
          <span>{{ auditDialog.row?.reason }}</span>
        </el-form-item>
        <el-form-item label="审核备注">
          <el-input
            v-model="auditDialog.remark"
            type="textarea"
            :rows="3"
            :placeholder="
              auditDialog.approved ? '通过可留空' : '驳回建议说明原因'
            "
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="auditDialog.visible = false">取消</el-button>
        <el-button type="primary" :loading="auditing" @click="submitAudit">
          确认{{ auditDialog.approved ? "通过" : "驳回" }}
        </el-button>
      </template>
    </el-dialog>

    <!-- 目标用户余额与流水 -->
    <template v-if="form.userId">
      <div class="mb-3 max-w-[420px]">
        <balance-quota-card :scope="form.userId" />
      </div>
      <el-divider content-position="left"
        >用户 {{ form.userId }} 积分流水</el-divider
      >
      <credit-log-table :scope="form.userId" />
    </template>
  </div>
</template>

<script lang="ts" setup>
import { BillingRefundVO } from "dehaze-sdk-js";
import { useAdminBillingStore } from "@/store/modules/adminBilling";
import BalanceQuotaCard from "@/components/billing/BalanceQuotaCard.vue";
import CreditLogTable from "@/components/billing/CreditLogTable.vue";

defineOptions({ name: "CreditAdjustPanel" });

const billingStore = useAdminBillingStore();

const formRef = ref(ElForm);
const submitting = ref(false);

const form = reactive({
  userId: undefined as number | undefined,
  amount: 0,
  reason: "",
});

const rules = {
  userId: [{ required: true, message: "用户ID不能为空", trigger: "change" }],
  amount: [
    { required: true, message: "调整金额不能为空", trigger: "change" },
    {
      validator: (
        _rule: unknown,
        value: number,
        callback: (err?: Error) => void
      ) => {
        if (value === 0) callback(new Error("调整金额不能为 0"));
        else callback();
      },
      trigger: "change",
    },
  ],
  reason: [{ required: true, message: "调整原因不能为空", trigger: "blur" }],
};

const auditDialog = reactive<{
  visible: boolean;
  row: BillingRefundVO | null;
  approved: boolean;
  remark: string;
}>({ visible: false, row: null, approved: true, remark: "" });
const auditing = ref(false);

function refundStatusTag(status: number) {
  switch (status) {
    case 1:
      return { label: "待审核", type: "warning" as const };
    case 2:
      return { label: "已通过", type: "success" as const };
    default:
      return { label: "已驳回", type: "info" as const };
  }
}

function openAudit(row: BillingRefundVO, approved: boolean) {
  auditDialog.row = row;
  auditDialog.approved = approved;
  auditDialog.remark = "";
  auditDialog.visible = true;
}

async function submitAudit() {
  if (!auditDialog.row) return;
  auditing.value = true;
  try {
    await billingStore.auditRefund(
      auditDialog.row.id,
      auditDialog.approved,
      auditDialog.remark || undefined
    );
    ElMessage.success(auditDialog.approved ? "已通过" : "已驳回");
    auditDialog.visible = false;
  } finally {
    auditing.value = false;
  }
}

function handleUserChange() {
  // userId 变化时由内嵌公共组件按 scope 自行拉取，无需额外处理
}

async function submit() {
  await formRef.value.validate();
  submitting.value = true;
  try {
    await billingStore.submitCreditAdjust({
      userId: form.userId!,
      amount: form.amount,
      reason: form.reason,
    });
    ElMessage.success("积分调整成功");
    form.reason = "";
    form.amount = 0;
  } finally {
    submitting.value = false;
  }
}

onMounted(() => {
  billingStore.fetchRefunds();
});
</script>
