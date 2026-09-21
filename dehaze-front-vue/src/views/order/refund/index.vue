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
            placeholder="退款单号/用户名"
            @keyup.enter="handleQuery"
          />
        </el-form-item>
        <el-form-item label="退款状态" prop="status">
          <el-select
            v-model="queryParams.status"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in refundStatusOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="原因类型" prop="reasonType">
          <el-select
            v-model="queryParams.reasonType"
            clearable
            placeholder="全部"
            style="width: 140px"
          >
            <el-option
              v-for="opt in refundReasonOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="申请时间">
          <el-date-picker
            v-model="applyTimeRange"
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
      <el-table
        v-loading="loading"
        :data="pageData"
        border
        highlight-current-row
      >
        <el-table-column label="退款单号" prop="refundNo" width="200" />
        <el-table-column label="订单号" prop="orderNo" width="200" />
        <el-table-column label="用户" prop="username" width="120" />
        <el-table-column
          label="退款金额"
          prop="refundAmount"
          width="110"
          align="right"
        >
          <template #default="scope"
            >¥{{ yuan((scope.row as RefundRecordVO).refundAmount) }}</template
          >
        </el-table-column>
        <el-table-column
          label="退款原因"
          prop="reason"
          min-width="160"
          show-overflow-tooltip
        />
        <el-table-column label="已用配额" width="120" align="center">
          <template #default="scope">
            <span v-if="(scope.row as RefundRecordVO).usedDays != null"
              >{{ (scope.row as RefundRecordVO).usedDays }} 天</span
            >
            <span v-else-if="(scope.row as RefundRecordVO).usedCredits != null"
              >{{ (scope.row as RefundRecordVO).usedCredits }} 积分</span
            >
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="110" align="center">
          <template #default="scope">
            <el-tag
              :type="refundStatusTagType((scope.row as RefundRecordVO).status)"
              size="small"
              >{{
                refundStatusLabel((scope.row as RefundRecordVO).status)
              }}</el-tag
            >
          </template>
        </el-table-column>
        <el-table-column label="申请时间" prop="applyTime" width="170" />
        <el-table-column label="审核时间" prop="auditTime" width="170">
          <template #default="scope">{{
            (scope.row as RefundRecordVO).auditTime || "-"
          }}</template>
        </el-table-column>
        <el-table-column fixed="right" label="操作" width="160" align="center">
          <template #default="scope">
            <template
              v-if="(scope.row as RefundRecordVO).status === 'refunding'"
            >
              <el-button
                v-hasPerm="['order:refund:approve']"
                link
                size="small"
                type="success"
                @click="openAuditDialog(scope.row as RefundRecordVO, true)"
              >
                <el-icon><Check /></el-icon>通过
              </el-button>
              <el-button
                v-hasPerm="['order:refund:approve']"
                link
                size="small"
                type="danger"
                @click="openAuditDialog(scope.row as RefundRecordVO, false)"
              >
                <el-icon><Close /></el-icon>驳回
              </el-button>
            </template>
            <span v-else>-</span>
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

    <!-- 退款审核弹窗 -->
    <el-dialog
      v-model="auditDialog.visible"
      :title="auditDialog.approved ? '退款审核通过' : '退款审核驳回'"
      width="500px"
      @close="closeAuditDialog"
    >
      <el-form
        ref="auditFormRef"
        :model="auditForm"
        :rules="auditRules"
        label-width="100px"
      >
        <el-form-item label="退款单号">
          <span>{{ auditDialog.row?.refundNo }}</span>
        </el-form-item>
        <el-form-item label="订单号">
          <span>{{ auditDialog.row?.orderNo }}</span>
        </el-form-item>
        <el-form-item label="用户">
          <span>{{ auditDialog.row?.username }}</span>
        </el-form-item>
        <el-form-item label="退款金额">
          <span>¥{{ yuan(auditDialog.row?.refundAmount) }}</span>
        </el-form-item>
        <el-form-item label="审核结果">
          <el-tag :type="auditDialog.approved ? 'success' : 'danger'">{{
            auditDialog.approved ? "通过" : "驳回"
          }}</el-tag>
        </el-form-item>
        <el-form-item label="审核备注" prop="remark">
          <el-input
            v-model="auditForm.remark"
            type="textarea"
            :rows="3"
            placeholder="请输入审核备注"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button type="primary" @click="handleAuditSubmit">确 定</el-button>
          <el-button @click="closeAuditDialog">取 消</el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  OrderAPI,
  RefundQuery,
  RefundRecordVO,
  RefundAuditForm,
  RefundStatus,
} from "dehaze-sdk-js";
import { Search, Refresh, Check, Close } from "@element-plus/icons-vue";
import { refundStatusOptions, refundReasonOptions } from "../constants";
import type { TagType } from "@/enums/TagType";

defineOptions({ name: "OrderRefund" });

const queryFormRef = ref(ElForm);
const auditFormRef = ref(ElForm);
const loading = ref(false);
const pageData = ref<RefundRecordVO[]>([]);
const total = ref(0);
const applyTimeRange = ref<[string, string] | null>(null);

const queryParams = reactive<RefundQuery>({
  pageNum: 1,
  pageSize: 10,
});

/** 后端金额单位为分，前端展示用元 */
function yuan(cents?: number) {
  return ((cents ?? 0) / 100).toFixed(2);
}

function refundStatusLabel(status: RefundStatus): string {
  return refundStatusOptions.find((o) => o.value === status)?.label ?? status;
}

function refundStatusTagType(status: RefundStatus): TagType {
  return refundStatusOptions.find((o) => o.value === status)!.tag;
}

function handleQuery() {
  loading.value = true;
  if (applyTimeRange.value && applyTimeRange.value.length === 2) {
    queryParams.applyTimeStart = applyTimeRange.value[0];
    queryParams.applyTimeEnd = applyTimeRange.value[1];
  } else {
    queryParams.applyTimeStart = undefined;
    queryParams.applyTimeEnd = undefined;
  }
  OrderAPI.listRefunds(queryParams)
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
  applyTimeRange.value = null;
  queryParams.orderNo = undefined;
  queryParams.keywords = undefined;
  queryParams.status = undefined;
  queryParams.reasonType = undefined;
  queryParams.applyTimeStart = undefined;
  queryParams.applyTimeEnd = undefined;
  queryParams.pageNum = 1;
  handleQuery();
}

const auditDialog = reactive<{
  visible: boolean;
  loading: boolean;
  approved: boolean;
  row: RefundRecordVO | null;
}>({
  visible: false,
  loading: false,
  approved: true,
  row: null,
});

const auditForm = reactive<RefundAuditForm>({
  approved: true,
  remark: "",
});

const auditRules = {
  remark: [{ required: true, message: "请输入审核备注", trigger: "blur" }],
};

function openAuditDialog(row: RefundRecordVO, approved: boolean) {
  auditDialog.row = row;
  auditDialog.approved = approved;
  auditForm.approved = approved;
  auditForm.remark = "";
  auditDialog.visible = true;
}

function closeAuditDialog() {
  auditDialog.visible = false;
  auditDialog.row = null;
  auditForm.remark = "";
  auditFormRef.value?.resetFields();
}

function handleAuditSubmit() {
  if (!auditDialog.row) return;
  auditFormRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    auditDialog.loading = true;
    const refundId = auditDialog.row!.id;
    const payload: RefundAuditForm = {
      approved: auditDialog.approved,
      remark: auditForm.remark,
    };
    const action = auditDialog.approved
      ? OrderAPI.approveRefund(refundId, payload)
      : OrderAPI.rejectRefund(refundId, payload);
    action
      .then(() => {
        ElMessage.success(auditDialog.approved ? "审核通过" : "已驳回");
        closeAuditDialog();
        handleQuery();
      })
      .finally(() => {
        auditDialog.loading = false;
      });
  });
}

onMounted(() => {
  handleQuery();
});
</script>
