<!-- 账单对账：导入 → 核对 → 校准 分步引导 -->
<template>
  <div>
    <el-steps :active="step" align-center class="mb-4">
      <el-step title="导入账单" description="上传或粘贴供应商账单" />
      <el-step title="核对差异" description="估算 vs 实际差异说明" />
      <el-step title="价格校准" description="差异反哺成本单价配置" />
    </el-steps>

    <!-- 步骤① 导入 -->
    <div v-if="step === 0">
      <el-upload
        class="mb-2"
        :auto-upload="false"
        :limit="1"
        accept=".csv,.xlsx,.xls"
        :on-change="handleFileChange"
        :on-remove="() => (billContent = '')"
      >
        <el-button type="primary" plain>
          <el-icon><Upload /></el-icon>上传 CSV / Excel 账单
        </el-button>
      </el-upload>
      <el-divider>或直接粘贴账单文本</el-divider>
      <el-input
        v-model="billContent"
        type="textarea"
        :rows="6"
        placeholder="粘贴账单内容（CSV/TSV 文本，含 request_id、模型、token 明细、金额等列）"
      />
      <el-date-picker
        v-model="periodRange"
        class="mt-2"
        type="daterange"
        value-format="YYYY-MM-DD"
        start-placeholder="对账周期起"
        end-placeholder="对账周期止"
      />
      <div class="mt-3">
        <el-button
          type="primary"
          :loading="billingStore.reconcileImporting"
          @click="submitImport"
        >
          导入并核对
        </el-button>
      </div>
    </div>

    <!-- 步骤② 核对 -->
    <div v-else-if="step === 1">
      <el-result
        icon="success"
        title="账单导入完成"
        :sub-title="`共导入 ${importedCount} 条账单记录`"
      >
        <template #extra>
          <el-alert
            class="w-full"
            type="info"
            :closable="false"
            title="差异核对说明"
            description="系统以 request_id / 模型 / token 明细与导入账单逐笔比对，差异来源通常包括：价格配置过期（按旧版本核算）、供应商赠送或折扣、计量口径差异（如向上取整规则）。请结合差异明细归因后进入校准步骤。"
          />
          <el-button class="mt-3" type="primary" @click="step = 2"
            >下一步：价格校准</el-button
          >
        </template>
      </el-result>
    </div>

    <!-- 步骤③ 校准 -->
    <div v-else>
      <el-alert
        type="warning"
        :closable="false"
        title="价格校准引导"
        description="对账发现的成本偏差应反哺成本单价配置：若差异源于价格配置过期，请生成新的成本价格版本（可在「调价影响测算」中先评估毛利影响）；若为供应商折扣，可按折后价修正档位单价。"
      />
      <div class="mt-3">
        <el-button type="primary" @click="step = 0">开始新一轮对账</el-button>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { Upload } from "@element-plus/icons-vue";
import type { UploadFile } from "element-plus";
import * as XLSX from "xlsx";
import { useAdminBillingStore } from "@/store/modules/adminBilling";

defineOptions({ name: "ReconcilePanel" });

const billingStore = useAdminBillingStore();

const step = ref(0);
const billContent = ref("");
const periodRange = ref<[string, string]>(["", ""]);
const importedCount = ref(0);

/** Excel 解析为 CSV 文本统一提交；CSV 直接读文本 */
async function handleFileChange(file: UploadFile) {
  const raw = file.raw;
  if (!raw) return;
  if (raw.name.endsWith(".csv")) {
    billContent.value = await raw.text();
    return;
  }
  const buffer = await raw.arrayBuffer();
  const workbook = XLSX.read(buffer, { type: "array" });
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  billContent.value = XLSX.utils.sheet_to_csv(sheet);
}

async function submitImport() {
  if (!billContent.value.trim()) {
    ElMessage.warning("请先上传或粘贴账单内容");
    return;
  }
  if (!periodRange.value?.[0]) {
    ElMessage.warning("请选择对账周期");
    return;
  }
  importedCount.value = await billingStore.importReconcileBill(
    billContent.value,
    periodRange.value[0],
    periodRange.value[1]
  );
  step.value = 1;
}
</script>
