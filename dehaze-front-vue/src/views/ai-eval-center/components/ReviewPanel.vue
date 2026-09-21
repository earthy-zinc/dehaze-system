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
      width="760px"
      destroy-on-close
    >
      <div v-loading="evalStore.reviewDetailLoading">
        <template v-if="detail">
          <el-descriptions :column="3" size="small" border class="mb-[12px]">
            <el-descriptions-item label="智能体">
              {{ detail.agentName ?? `#${detail.agentId}` }}
            </el-descriptions-item>
            <el-descriptions-item label="评测ID">
              {{ detail.runId }}
            </el-descriptions-item>
            <el-descriptions-item label="样本ID">
              {{ detail.sampleId }}
            </el-descriptions-item>
            <el-descriptions-item label="风险等级">
              <el-tag :type="riskMeta(detail.riskLevel).type" size="small">
                {{ riskMeta(detail.riskLevel).label }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="判分结果">
              <el-tag
                :type="detail.judgePassed ? 'success' : 'danger'"
                size="small"
              >
                {{ detail.judgePassed ? "判分通过" : "判分失败" }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="可用工具">
              {{ detail.tools?.length ? detail.tools.join("、") : "-" }}
            </el-descriptions-item>
          </el-descriptions>

          <el-divider content-position="left">样本定义</el-divider>
          <div class="review-field">
            <div class="review-field__label">任务目标</div>
            <div class="review-field__value">{{ detail.taskGoal || "-" }}</div>
          </div>
          <div class="review-field">
            <div class="review-field__label">允许输入</div>
            <div class="review-field__value">
              {{ detail.allowedInput || "-" }}
            </div>
          </div>
          <div class="review-field">
            <div class="review-field__label">期望结果</div>
            <div class="review-field__value">
              {{ detail.expectedResult || "-" }}
            </div>
          </div>
          <div class="review-field">
            <div class="review-field__label">期望过程</div>
            <div class="review-field__value">
              {{ detail.expectedProcess || "-" }}
            </div>
          </div>
          <div class="review-field">
            <div class="review-field__label">禁止行为</div>
            <div class="review-field__value">
              {{ detail.forbiddenBehavior || "-" }}
            </div>
          </div>

          <el-divider content-position="left">实际输出</el-divider>
          <pre class="review-output">{{
            detail.actualOutput || "本次评测未记录实际输出"
          }}</pre>
          <el-alert
            v-if="detail.error"
            class="mb-[12px]"
            type="error"
            :closable="false"
            :title="`样本执行异常：${detail.error}`"
          />

          <el-divider content-position="left">判分明细</el-divider>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-[8px] mb-[12px]">
            <el-card
              v-for="dimension in EVAL_DIMENSIONS"
              :key="dimension.key"
              shadow="never"
            >
              <div class="text-xs text-gray-400">{{ dimension.label }}</div>
              <div class="text-lg font-semibold mt-1">
                {{ formatScore(detail.scores?.[dimension.key]) }}
              </div>
            </el-card>
          </div>
          <div
            v-for="dimension in EVAL_DIMENSIONS"
            :key="dimension.key"
            class="text-xs leading-6"
          >
            <span class="text-gray-500">{{ dimension.label }}：</span>
            {{ detail.notes?.[dimension.key] || "无说明" }}
          </div>
        </template>
        <el-empty v-else description="复核详情加载失败" :image-size="60" />
      </div>

      <el-form label-width="80px" class="mt-4">
        <el-form-item label="复核结论">
          <el-radio-group v-model="agree">
            <el-radio :value="true">通过（认同判分结果）</el-radio>
            <el-radio :value="false">驳回（判分结果有误）</el-radio>
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
import { EVAL_DIMENSIONS, RISK_LEVEL_META, formatScore } from "../eval-meta";

defineOptions({ name: "ReviewPanel" });

const evalStore = useAdminEvalStore();

const reviewQueue = computed(() => evalStore.reviewQueue);
const items = computed(() => reviewQueue.value?.items ?? []);

const dialogVisible = ref(false);
const currentReview = ref<AiEvalReviewItem | null>(null);
const detail = computed(() => evalStore.reviewDetail);
const agree = ref(true);
const remark = ref("");

async function openReviewDialog(reviewId: number) {
  const review = items.value.find((item) => item.id === reviewId) ?? null;
  currentReview.value = review;
  agree.value = true;
  remark.value = "";
  dialogVisible.value = true;
  if (review) {
    await evalStore.fetchReviewDetail(review.runId, review.sampleId);
  }
}

async function handleSubmit() {
  const review = currentReview.value;
  if (!review) return;
  // 驳回判分必须写明理由，否则判分模型无法据此校准
  if (!agree.value && !remark.value.trim()) {
    ElMessage.warning("驳回判分需填写理由");
    return;
  }
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

<style lang="scss" scoped>
.review-field {
  display: flex;
  margin-bottom: 6px;
  font-size: 13px;
  line-height: 1.8;

  &__label {
    flex-shrink: 0;
    width: 80px;
    color: var(--el-text-color-secondary);
  }

  &__value {
    flex: 1;
    word-break: break-all;
    white-space: pre-wrap;
  }
}

.review-output {
  max-height: 240px;
  padding: 8px 12px;
  margin: 0 0 12px;
  overflow: auto;
  font-size: 12px;
  line-height: 1.7;
  word-break: break-all;
  white-space: pre-wrap;
  background-color: var(--el-fill-color-light);
  border-radius: 4px;
}
</style>
