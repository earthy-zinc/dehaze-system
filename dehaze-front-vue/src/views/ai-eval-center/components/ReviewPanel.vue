<!-- 人工复核队列：判分抽样待确认项 + 复核回填（status 1→2 不可逆） -->
<template>
  <el-card shadow="never" class="mb-[12px]">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>人工复核</span>
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400">
            待复核 {{ reviewQueue?.pending ?? 0 }} · 已复核
            {{ reviewQueue?.reviewed ?? 0 }}
          </span>
          <el-radio-group
            v-model="evalStore.reviewStatus"
            size="small"
            @change="evalStore.fetchReviews()"
          >
            <el-radio-button value="all">全部</el-radio-button>
            <el-radio-button :value="1">待复核</el-radio-button>
            <el-radio-button :value="2">已复核</el-radio-button>
          </el-radio-group>
        </div>
      </div>
    </template>

    <el-table v-loading="evalStore.reviewLoading" :data="items" size="small">
      <el-table-column label="智能体" min-width="140">
        <template #default="{ row }">
          {{ row.agentName ?? `#${row.agentId}` }}
        </template>
      </el-table-column>
      <el-table-column label="评测ID" prop="runId" width="90" align="center" />
      <el-table-column
        label="样本ID"
        prop="sampleId"
        width="90"
        align="center"
      />
      <el-table-column label="判分结果" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="row.judgePassed ? 'success' : 'danger'" size="small">
            {{ row.judgePassed ? "通过" : "失败" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="风险" width="80" align="center">
        <template #default="{ row }">
          <el-tag :type="riskMeta(row.riskLevel).type" size="small">
            {{ riskMeta(row.riskLevel).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="状态" width="90" align="center">
        <template #default="{ row }">
          <el-tag :type="row.status === 1 ? 'warning' : 'info'" size="small">
            {{ row.status === 1 ? "待复核" : "已复核" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="人工判定" width="100" align="center">
        <template #default="{ row }">
          <span v-if="row.agree == null">-</span>
          <el-tag v-else :type="row.agree ? 'success' : 'danger'" size="small">
            {{ row.agree ? "一致" : "不一致" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="备注" prop="remark" min-width="160">
        <template #default="{ row }">{{ row.remark || "-" }}</template>
      </el-table-column>
      <el-table-column label="操作" width="90" align="center" fixed="right">
        <template #default="{ row }">
          <el-button
            v-if="row.status === 1"
            v-hasPerm="['ai:agent:manage']"
            link
            type="primary"
            size="small"
            @click="openReviewDialog(row.id)"
          >
            复核
          </el-button>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="暂无人工复核项" :image-size="60" />
      </template>
    </el-table>

    <el-dialog
      v-model="dialogVisible"
      title="复核回填"
      width="480px"
      destroy-on-close
    >
      <el-descriptions :column="2" size="small" border class="mb-[12px]">
        <el-descriptions-item label="智能体">
          {{ currentReview?.agentName ?? `#${currentReview?.agentId}` }}
        </el-descriptions-item>
        <el-descriptions-item label="样本ID">
          {{ currentReview?.sampleId }}
        </el-descriptions-item>
        <el-descriptions-item label="判分结果">
          {{ currentReview?.judgePassed ? "通过" : "失败" }}
        </el-descriptions-item>
        <el-descriptions-item label="评测ID">
          {{ currentReview?.runId }}
        </el-descriptions-item>
      </el-descriptions>

      <el-form label-width="80px">
        <el-form-item label="人工判定">
          <el-radio-group v-model="agree">
            <el-radio :value="true">与判分一致</el-radio>
            <el-radio :value="false">与判分不一致</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="备注">
          <el-input
            v-model="remark"
            type="textarea"
            :rows="3"
            maxlength="500"
            show-word-limit
            placeholder="选填，用于判分校准（≤500 字）"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button
          type="primary"
          :loading="evalStore.reviewSubmitting"
          @click="handleSubmit"
        >
          提交复核
        </el-button>
        <el-button @click="dialogVisible = false">取 消</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script lang="ts" setup>
import type { AiEvalReviewItem } from "dehaze-sdk-js";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import { RISK_LEVEL_META } from "../eval-meta";

defineOptions({ name: "ReviewPanel" });

const evalStore = useAdminEvalStore();

const reviewQueue = computed(() => evalStore.reviewQueue);
const items = computed(() => reviewQueue.value?.items ?? []);

const dialogVisible = ref(false);
const currentReview = ref<AiEvalReviewItem | null>(null);
const agree = ref(true);
const remark = ref("");

function openReviewDialog(reviewId: number) {
  currentReview.value =
    items.value.find((item) => item.id === reviewId) ?? null;
  agree.value = true;
  remark.value = "";
  dialogVisible.value = true;
}

async function handleSubmit() {
  const review = currentReview.value;
  if (!review) return;
  await evalStore.submitReview(
    review.id,
    agree.value,
    remark.value || undefined
  );
  ElMessage.success("复核结果已回填");
  dialogVisible.value = false;
}

function riskMeta(riskLevel: string) {
  return (
    RISK_LEVEL_META[riskLevel] ?? { label: riskLevel, type: "info" as const }
  );
}
</script>
