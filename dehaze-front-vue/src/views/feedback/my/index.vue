<template>
  <div class="app-container my-feedback">
    <div class="page-header">
      <span class="title-text">我的反馈</span>
      <el-button type="primary" @click="openCreateDialog">
        <el-icon><Plus /></el-icon>
        <span>新建反馈</span>
      </el-button>
    </div>

    <div v-loading="loading" class="feedback-list">
      <template v-if="feedbackList.length > 0">
        <div
          v-for="feedback in feedbackList"
          :key="feedback.id"
          class="feedback-card"
          :class="`status-${feedback.status}`"
          @click="goDetail(feedback)"
        >
          <div :class="['status-stripe', `stripe-${feedback.status}`]"></div>
          <div class="card-body">
            <div class="card-meta">
              <span :class="['status-tag', `tag-${feedback.status}`]">
                {{ statusLabel(feedback.status) }}
              </span>
              <span class="type-tag">{{ typeLabel(feedback.feedbackType) }}</span>
              <span v-if="feedback.relatedModule" class="module-tag">
                {{ moduleLabel(feedback.relatedModule) }}
              </span>
              <span class="time-text">{{ feedback.createTime }}</span>
            </div>

            <div class="card-title">{{ feedback.title }}</div>
            <div class="card-summary">{{ feedback.content }}</div>

            <div class="card-footer">
              <span v-if="feedback.assigneeName" class="assignee-text">
                处理人：{{ feedback.assigneeName }}
              </span>
              <span v-else class="assignee-text unassigned">暂未分配</span>

              <el-button
                class="detail-btn"
                link
                type="primary"
                @click.stop="goDetail(feedback)"
              >
                查看详情
                <el-icon><ArrowRight /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </template>

      <el-empty v-else-if="!loading" description="暂无反馈" :image-size="120">
        <template #description>
          <p class="empty-text">有任何建议或问题，欢迎提交反馈</p>
        </template>
      </el-empty>
    </div>

    <pagination
      v-if="total > 0"
      v-model:limit="queryParams.pageSize"
      v-model:page="queryParams.pageNum"
      v-model:total="total"
      @pagination="handleQuery"
    />

    <el-dialog
      v-model="formDialog.visible"
      title="新建反馈"
      width="560px"
      @close="resetForm"
    >
      <el-form
        ref="formRef"
        :model="formData"
        :rules="formRules"
        label-width="90px"
      >
        <el-form-item label="反馈类型" prop="feedbackType">
          <el-select
            v-model="formData.feedbackType"
            placeholder="请选择反馈类型"
            style="width: 100%"
          >
            <el-option
              v-for="opt in typeOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="标题" prop="title">
          <el-input
            v-model="formData.title"
            maxlength="50"
            show-word-limit
            placeholder="请输入标题（5-50 字符）"
          />
        </el-form-item>

        <el-form-item label="内容" prop="content">
          <el-input
            v-model="formData.content"
            type="textarea"
            :rows="6"
            maxlength="1000"
            show-word-limit
            placeholder="请详细描述您的问题或建议（10-1000 字符）"
          />
        </el-form-item>

        <el-form-item label="相关模块" prop="relatedModule">
          <el-select
            v-model="formData.relatedModule"
            clearable
            placeholder="请选择相关模块"
            style="width: 100%"
          >
            <el-option
              v-for="opt in moduleOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="联系方式" prop="contact">
          <el-input
            v-model="formData.contact"
            placeholder="手机/邮箱（仅管理员可见）"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <div class="dialog-footer">
          <el-button @click="formDialog.visible = false">取 消</el-button>
          <el-button
            type="primary"
            :loading="formDialog.loading"
            @click="handleSubmit"
          >
            提 交
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import {
  FeedbackAPI,
  FeedbackPageVO,
  FeedbackCreateForm,
  FeedbackType,
  FeedbackStatus,
} from "dehaze-sdk-js";
import { ArrowRight, Plus } from "@element-plus/icons-vue";

defineOptions({ name: "FeedbackMy" });

const router = useRouter();
const formRef = ref(ElForm);
const loading = ref(false);
const total = ref(0);
const feedbackList = ref<FeedbackPageVO[]>([]);
const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
});

const typeOptions: { label: string; value: FeedbackType }[] = [
  { label: "功能建议", value: "suggestion" },
  { label: "问题报告", value: "bug" },
  { label: "体验反馈", value: "experience" },
  { label: "投诉", value: "complaint" },
];

const moduleOptions = [
  { label: "去雾处理", value: "dehaze" },
  { label: "指标评估", value: "evaluate" },
  { label: "数据集", value: "dataset" },
  { label: "会员", value: "member" },
  { label: "套餐", value: "package" },
  { label: "订单", value: "order" },
  { label: "其他", value: "other" },
];

const MODULE_LABEL_MAP: Record<string, string> = moduleOptions.reduce(
  (acc, item) => {
    acc[item.value] = item.label;
    return acc;
  },
  {} as Record<string, string>
);

const formDialog = reactive<{
  visible: boolean;
  loading: boolean;
}>({
  visible: false,
  loading: false,
});

const formData = reactive<FeedbackCreateForm>({
  feedbackType: "suggestion",
  title: "",
  content: "",
  relatedModule: undefined,
  contact: "",
});

const formRules = reactive({
  feedbackType: [
    { required: true, message: "请选择反馈类型", trigger: "change" },
  ],
  title: [
    { required: true, message: "请输入标题", trigger: "blur" },
    {
      min: 5,
      max: 50,
      message: "标题长度为 5-50 字符",
      trigger: "blur",
    },
  ],
  content: [
    { required: true, message: "请输入内容", trigger: "blur" },
    {
      min: 10,
      max: 1000,
      message: "内容长度为 10-1000 字符",
      trigger: "blur",
    },
  ],
});

function statusLabel(status: FeedbackStatus): string {
  const map: Record<FeedbackStatus, string> = {
    pending: "待处理",
    processing: "处理中",
    replied: "已回复",
    closed: "已关闭",
  };
  return map[status] || status;
}

function typeLabel(type: FeedbackType): string {
  const map: Record<FeedbackType, string> = {
    suggestion: "功能建议",
    bug: "问题报告",
    experience: "体验反馈",
    complaint: "投诉",
  };
  return map[type] || type;
}

function moduleLabel(module: string): string {
  return MODULE_LABEL_MAP[module] || module;
}

function handleQuery() {
  loading.value = true;
  FeedbackAPI.listMyFeedback(queryParams)
    .then((data) => {
      feedbackList.value = data.list;
      total.value = data.total;
    })
    .finally(() => {
      loading.value = false;
    });
}

function goDetail(feedback: FeedbackPageVO) {
  router.push(`/feedback/detail?id=${feedback.id}`);
}

function openCreateDialog() {
  resetForm();
  formDialog.visible = true;
}

function resetForm() {
  formData.feedbackType = "suggestion";
  formData.title = "";
  formData.content = "";
  formData.relatedModule = undefined;
  formData.contact = "";
  formRef.value?.resetFields();
}

const handleSubmit = useThrottleFn(() => {
  formRef.value?.validate((valid: boolean) => {
    if (!valid) return;
    formDialog.loading = true;
    FeedbackAPI.createFeedback(formData)
      .then(() => {
        ElMessage.success("反馈提交成功");
        formDialog.visible = false;
        queryParams.pageNum = 1;
        handleQuery();
      })
      .finally(() => {
        formDialog.loading = false;
      });
  });
}, 3000);

onMounted(() => {
  handleQuery();
});
</script>

<style lang="scss" scoped>
.my-feedback {
  max-width: 960px;
  padding: 24px 20px 40px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;

  .title-text {
    font-size: 22px;
    font-weight: 600;
    color: var(--el-text-color-primary);
    letter-spacing: 0.5px;
  }
}

.feedback-list {
  min-height: 240px;
}

.feedback-card {
  position: relative;
  display: flex;
  margin-bottom: 12px;
  overflow: hidden;
  cursor: pointer;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  transition: all 0.25s ease;

  &:hover {
    border-color: var(--el-color-primary-light-5);
    box-shadow: 0 4px 16px rgb(0 0 0 / 6%);
    transform: translateY(-1px);

    .detail-btn {
      opacity: 1;
    }
  }

  .status-stripe {
    flex-shrink: 0;
    width: 4px;

    &.stripe-pending {
      background: linear-gradient(180deg, #fa8c16, #ffc069);
    }

    &.stripe-processing {
      background: linear-gradient(180deg, #409eff, #79bbff);
    }

    &.stripe-replied {
      background: linear-gradient(180deg, #67c23a, #95d475);
    }

    &.stripe-closed {
      background: linear-gradient(180deg, #909399, #b1b3b8);
    }
  }

  .card-body {
    flex: 1;
    padding: 14px 18px;
  }

  .card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;

    .status-tag {
      display: inline-block;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 500;
      border-radius: 4px;

      &.tag-pending {
        color: #fa8c16;
        background: #fff7e6;
      }

      &.tag-processing {
        color: #409eff;
        background: #ecf5ff;
      }

      &.tag-replied {
        color: #67c23a;
        background: #f0f9eb;
      }

      &.tag-closed {
        color: #909399;
        background: #f4f4f5;
      }
    }

    .type-tag,
    .module-tag {
      display: inline-block;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--el-text-color-regular);
      background: var(--el-fill-color-light);
      border-radius: 4px;
    }

    .time-text {
      margin-left: auto;
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .card-title {
    margin-bottom: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 15px;
    font-weight: 600;
    color: var(--el-text-color-primary);
    white-space: nowrap;
  }

  .card-summary {
    display: -webkit-box;
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-line-clamp: 2;
    font-size: 13px;
    line-height: 1.5;
    color: var(--el-text-color-secondary);
    -webkit-box-orient: vertical;
  }

  .card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 10px;

    .assignee-text {
      font-size: 12px;
      color: var(--el-text-color-secondary);

      &.unassigned {
        color: var(--el-text-color-placeholder);
      }
    }

    .detail-btn {
      display: inline-flex;
      gap: 2px;
      align-items: center;
      font-size: 12px;
      opacity: 0.8;
      transition: opacity 0.2s ease;
    }
  }
}

.empty-text {
  margin: 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

@media (width <= 768px) {
  .feedback-card {
    .card-footer .detail-btn {
      opacity: 1;
    }
  }
}
</style>
