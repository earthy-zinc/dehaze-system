<!-- 误扣申诉弹窗：原因类型 + 补充说明，提交后对应明细行状态置"待审核" -->
<script lang="ts" setup>
import { BillingRecordVO } from "dehaze-sdk-js";
import { ElMessage } from "element-plus";
import { computed, ref, watch } from "vue";
import { useBillingStore } from "@/store/modules/billing";

defineOptions({ name: "RefundApplyDialog" });

const props = defineProps<{ record: BillingRecordVO | null }>();

const billingStore = useBillingStore();

const REFUND_REASONS = [
  "预扣过高",
  "重复扣费",
  "未使用却扣费",
  "其他",
] as const;

const reason = ref("");
const remark = ref("");
const submitting = ref(false);

const refundAmount = computed(() => props.record?.credits ?? 0);

watch(
  () => billingStore.refundDialog.visible,
  (visible) => {
    if (!visible) return;
    reason.value = "";
    remark.value = "";
  }
);

async function handleSubmit() {
  if (!props.record) return;
  if (!reason.value) {
    ElMessage.warning("请选择申诉原因");
    return;
  }
  const detail = remark.value.trim();
  submitting.value = true;
  try {
    await billingStore.submitRefund(props.record.id, {
      amount: refundAmount.value,
      reason: detail ? `${reason.value}：${detail}` : reason.value,
    });
    ElMessage.success("申诉已提交，请等待审核");
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <el-dialog
    v-model="billingStore.refundDialog.visible"
    title="误扣申诉"
    width="520px"
    :close-on-click-modal="false"
  >
    <el-descriptions v-if="record" :column="1" size="small" border>
      <el-descriptions-item label="计费时间">
        {{ record.createTime }}
      </el-descriptions-item>
      <el-descriptions-item label="使用模型">
        {{ record.actualModel || record.model }}
      </el-descriptions-item>
      <el-descriptions-item label="实扣积分">
        {{ record.credits }}
      </el-descriptions-item>
    </el-descriptions>

    <el-form class="refund-form" label-position="top">
      <el-form-item label="申诉原因" required>
        <el-select v-model="reason" placeholder="请选择申诉原因" class="w-full">
          <el-option
            v-for="item in REFUND_REASONS"
            :key="item"
            :label="item"
            :value="item"
          />
        </el-select>
      </el-form-item>
      <el-form-item label="补充说明">
        <el-input
          v-model="remark"
          type="textarea"
          :rows="3"
          maxlength="200"
          show-word-limit
          placeholder="可补充具体的会话内容或扣费疑点，便于审核"
        />
      </el-form-item>
    </el-form>

    <template #footer>
      <el-button @click="billingStore.closeRefund()">取消</el-button>
      <el-button type="primary" :loading="submitting" @click="handleSubmit">
        提交申诉
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.refund-form {
  margin-top: 16px;
}

.w-full {
  width: 100%;
}
</style>
