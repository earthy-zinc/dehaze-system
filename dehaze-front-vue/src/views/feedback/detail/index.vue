<template>
  <div class="app-container feedback-detail">
    <div class="detail-wrapper" v-loading="loading">
      <template v-if="detail">
        <div class="detail-header">
          <el-button link @click="goBack">
            <el-icon><ArrowLeft /></el-icon>
            返回列表
          </el-button>
        </div>

        <div :class="['status-card', `status-${detail.status}`]">
          <div class="card-stripe"></div>
          <div class="card-content">
            <div class="meta-row">
              <span :class="['status-tag', `tag-${detail.status}`]">
                {{ statusLabel(detail.status) }}
              </span>
              <span class="type-tag">{{ typeLabel(detail.feedbackType) }}</span>
              <span v-if="detail.relatedModule" class="module-tag">
                {{ moduleLabel(detail.relatedModule) }}
              </span>
              <span class="time-text">{{ detail.createTime }}</span>
            </div>

            <h1 class="detail-title">{{ detail.title }}</h1>

            <div v-if="detail.assigneeName" class="assignee-info">
              <el-icon><ChatLineRound /></el-icon>
              处理人：{{ detail.assigneeName }}
              <span v-if="detail.assignedTime" class="assigned-time">
                分配于 {{ detail.assignedTime }}
              </span>
            </div>
          </div>
        </div>

        <el-card class="content-card" shadow="never">
          <template #header>
            <div class="card-header-title">
              <el-icon><Picture /></el-icon>
              <span>反馈内容</span>
            </div>
          </template>
          <div class="content-text">{{ detail.content }}</div>

          <div v-if="detail.images?.length" class="content-images">
            <el-image
              v-for="(url, idx) in detail.images"
              :key="idx"
              :src="url"
              :preview-src-list="detail.images"
              :initial-index="idx"
              fit="cover"
              class="content-thumb"
            />
          </div>

          <div v-if="detail.contact" class="content-contact">
            <el-tag type="info" size="small">
              联系方式：{{ detail.contact }}
            </el-tag>
          </div>

          <div v-if="detail.tags?.length" class="content-tags">
            <el-tag
              v-for="tag in detail.tags"
              :key="tag"
              size="small"
              effect="light"
            >
              {{ tag }}
            </el-tag>
          </div>
        </el-card>

        <el-card
          v-if="detail.replies?.length"
          class="timeline-card"
          shadow="never"
        >
          <template #header>
            <div class="card-header-title">
              <el-icon><ChatLineRound /></el-icon>
              <span>处理时间线</span>
            </div>
          </template>
          <el-timeline>
            <el-timeline-item
              v-for="reply in detail.replies"
              :key="reply.id"
              :timestamp="reply.createTime"
              placement="top"
              :color="reply.replierType === 2 ? '#409eff' : '#67c23a'"
            >
              <div class="reply-meta">
                <span class="replier-name">{{ reply.replierName }}</span>
                <el-tag
                  v-if="reply.replyType"
                  :type="replyTypeTagType(reply.replyType)"
                  size="small"
                >
                  {{ replyTypeLabel(reply.replyType) }}
                </el-tag>
                <span class="replier-type-text">
                  {{ reply.replierType === 2 ? "管理员" : "用户" }}
                </span>
              </div>
              <div class="reply-content">{{ reply.content }}</div>
              <div v-if="reply.attachments?.length" class="reply-attachments">
                <el-image
                  v-for="(att, idx) in reply.attachments"
                  :key="idx"
                  :src="att"
                  :preview-src-list="reply.attachments"
                  :initial-index="idx"
                  fit="cover"
                  class="attachment-thumb"
                />
              </div>
            </el-timeline-item>
          </el-timeline>
        </el-card>

        <el-card v-if="detail.closeReason" class="close-card" shadow="never">
          <template #header>
            <div class="card-header-title close-header">
              <el-icon><CircleClose /></el-icon>
              <span>关闭原因</span>
            </div>
          </template>
          <div class="close-content">{{ detail.closeReason }}</div>
        </el-card>

        <el-card
          v-if="detail.status !== 'closed'"
          class="supplement-card"
          shadow="never"
        >
          <template #header>
            <div class="card-header-title">
              <el-icon><ChatLineRound /></el-icon>
              <span>补充说明</span>
            </div>
          </template>
          <el-input
            v-model="supplementContent"
            type="textarea"
            :rows="4"
            maxlength="1000"
            show-word-limit
            placeholder="请输入补充说明内容"
          />
          <div class="supplement-footer">
            <el-button
              type="primary"
              :loading="supplementLoading"
              :disabled="!supplementContent.trim()"
              @click="handleSupplement"
            >
              提交补充
            </el-button>
          </div>
        </el-card>
      </template>

      <el-empty v-else-if="!loading" description="反馈不存在或已被删除">
        <el-button type="primary" @click="goBack">返回反馈列表</el-button>
      </el-empty>
    </div>
  </div>
</template>

<script lang="ts" setup>
import {
  FeedbackAPI,
  FeedbackDetailVO,
  FeedbackType,
  FeedbackStatus,
  FeedbackReplyType,
} from "dehaze-sdk-js";
import {
  ArrowLeft,
  ChatLineRound,
  CircleClose,
  Picture,
} from "@element-plus/icons-vue";

defineOptions({ name: "FeedbackDetail" });

const route = useRoute();
const router = useRouter();
const loading = ref(false);
const detail = ref<FeedbackDetailVO | null>(null);
const supplementContent = ref("");
const supplementLoading = ref(false);

const MODULE_LABEL_MAP: Record<string, string> = {
  dehaze: "去雾处理",
  evaluate: "指标评估",
  dataset: "数据集",
  member: "会员",
  package: "套餐",
  order: "订单",
  other: "其他",
};

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

function replyTypeLabel(type: FeedbackReplyType): string {
  const map: Record<FeedbackReplyType, string> = {
    info: "通知",
    resolved: "已解决",
    unsupported: "不支持",
    dev_transfer: "转开发",
  };
  return map[type] || type;
}

function replyTypeTagType(
  type: FeedbackReplyType
): "info" | "success" | "warning" {
  const map: Record<FeedbackReplyType, "info" | "success" | "warning"> = {
    info: "info",
    resolved: "success",
    unsupported: "info",
    dev_transfer: "warning",
  };
  return map[type];
}

function loadDetail() {
  const id = Number(route.query.id);
  if (!id) {
    detail.value = null;
    return;
  }
  loading.value = true;
  FeedbackAPI.getFeedbackDetail(id)
    .then((data) => {
      detail.value = data;
    })
    .catch(() => {
      detail.value = null;
    })
    .finally(() => {
      loading.value = false;
    });
}

function goBack() {
  router.push("/feedback/my");
}

function handleSupplement() {
  if (!detail.value) return;
  const content = supplementContent.value.trim();
  if (!content) return;
  supplementLoading.value = true;
  FeedbackAPI.supplementFeedback(detail.value.id, { content })
    .then(() => {
      ElMessage.success("补充说明已提交");
      supplementContent.value = "";
      loadDetail();
    })
    .finally(() => {
      supplementLoading.value = false;
    });
}

onMounted(() => {
  loadDetail();
});

watch(
  () => route.query.id,
  () => {
    loadDetail();
  }
);
</script>

<style lang="scss" scoped>
.feedback-detail {
  max-width: 900px;
  padding: 24px 20px 40px;
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

  .card-stripe {
    flex-shrink: 0;
    width: 5px;
  }

  &.status-pending .card-stripe {
    background: linear-gradient(180deg, #fa8c16, #ffc069);
  }

  &.status-processing .card-stripe {
    background: linear-gradient(180deg, #409eff, #79bbff);
  }

  &.status-replied .card-stripe {
    background: linear-gradient(180deg, #67c23a, #95d475);
  }

  &.status-closed .card-stripe {
    background: linear-gradient(180deg, #909399, #b1b3b8);
  }

  .card-content {
    flex: 1;
    padding: 20px 24px;
  }
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;

  .status-tag {
    display: inline-block;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
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

.detail-title {
  margin: 0 0 10px;
  font-size: 22px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--el-text-color-primary);
}

.assignee-info {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  font-size: 13px;
  color: var(--el-text-color-secondary);

  .assigned-time {
    margin-left: 6px;
    font-size: 12px;
    color: var(--el-text-color-placeholder);
  }
}

.content-card,
.timeline-card,
.close-card,
.supplement-card {
  margin-bottom: 16px;
  border-radius: 10px;

  .card-header-title {
    display: flex;
    gap: 6px;
    align-items: center;
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);

    &.close-header {
      color: var(--el-color-info);
    }
  }
}

.content-text {
  font-size: 14px;
  line-height: 1.8;
  color: var(--el-text-color-primary);
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.content-images {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;

  .content-thumb {
    width: 100px;
    height: 100px;
    border-radius: 6px;
  }
}

.content-contact {
  margin-top: 12px;
}

.content-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.reply-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 6px;

  .replier-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .replier-type-text {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.reply-content {
  font-size: 13px;
  line-height: 1.6;
  color: var(--el-text-color-regular);
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.reply-attachments {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;

  .attachment-thumb {
    width: 80px;
    height: 80px;
    border-radius: 6px;
  }
}

.close-card {
  background: var(--el-fill-color-light);

  .close-content {
    font-size: 13px;
    line-height: 1.6;
    color: var(--el-text-color-regular);
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }
}

.supplement-footer {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
}
</style>
