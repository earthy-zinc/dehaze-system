<template>
  <PageLayout level="L2" title="反馈详情">
    <view class="page-body">
      <view v-if="feedback.id" class="detail-card">
        <view class="info-row">
          <text class="label">类型</text>
          <text>{{
            feedback.feedbackType
              ? typeMap[feedback.feedbackType] || feedback.feedbackType
              : ""
          }}</text>
        </view>
        <view class="info-row">
          <text class="label">状态</text>
          <text :class="['status-tag', 's-' + feedback.status]">
            {{
              feedback.status
                ? statusMap[feedback.status] || feedback.status
                : ""
            }}
          </text>
        </view>
        <view class="info-row">
          <text class="label">标题</text>
          <text>{{ feedback.title }}</text>
        </view>
        <view class="info-row">
          <text class="label">用户</text>
          <text>{{ feedback.username }} (ID:{{ feedback.userId }})</text>
        </view>
        <view class="info-row">
          <text class="label">内容</text>
          <text class="content-text">{{ feedback.content }}</text>
        </view>
        <view v-if="feedback.contact" class="info-row">
          <text class="label">联系方式</text>
          <text>{{ feedback.contact }}</text>
        </view>
        <view v-if="feedback.assigneeName" class="info-row">
          <text class="label">处理人</text>
          <text>{{ feedback.assigneeName }}</text>
        </view>
        <view class="info-row">
          <text class="label">优先级</text>
          <text>{{ feedback.priority }}</text>
        </view>
        <view class="info-row">
          <text class="label">时间</text>
          <text>{{ feedback.createTime }}</text>
        </view>
        <view v-if="feedback.closeReason" class="info-row">
          <text class="label">关闭原因</text>
          <text>{{ feedback.closeReason }}</text>
        </view>
      </view>

      <view
        v-if="feedback.replies && feedback.replies.length > 0"
        class="replies-section"
      >
        <text class="section-title">回复记录</text>
        <view
          v-for="replyItem in feedback.replies"
          :key="replyItem.id"
          class="reply-item"
        >
          <view class="reply-header">
            <text class="reply-author">{{ replyItem.replierName }}</text>
            <text class="reply-time">{{ replyItem.createTime }}</text>
          </view>
          <text class="reply-content">{{ replyItem.content }}</text>
        </view>
      </view>

      <view
        v-if="feedback.id && feedback.status !== 'closed'"
        class="action-section"
      >
        <text class="section-title">操作</text>
        <textarea
          class="form-textarea"
          v-model="reply"
          placeholder="请输入回复内容"
        />
        <view class="btn-row">
          <button
            class="btn btn-primary"
            :disabled="replying"
            @click="handleReply"
          >
            回复
          </button>
          <button class="btn btn-warning" @click="openClose">关闭反馈</button>
        </view>
      </view>
    </view>

    <!-- 关闭弹窗 -->
    <Popup
      :show="closeVisible"
      mode="bottom"
      round
      @close="closeVisible = false"
    >
      <view class="popup-content">
        <text class="popup-title">关闭反馈</text>
        <textarea
          class="form-textarea"
          v-model="closeReason"
          placeholder="请填写关闭原因（必填）"
        />
        <view class="popup-footer">
          <button class="btn btn-default" @click="closeVisible = false">
            取消
          </button>
          <button
            class="btn btn-danger"
            :disabled="closing"
            @click="handleClose"
          >
            确认关闭
          </button>
        </view>
      </view>
    </Popup>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref, reactive } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import { FeedbackAPI } from "dehaze-sdk-js";
import type {
  FeedbackDetailVO,
  FeedbackStatus,
  FeedbackType,
} from "dehaze-sdk-js";

const statusMap: Record<string, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};

const typeMap: Record<string, string> = {
  suggestion: "建议",
  bug: "缺陷",
  experience: "体验",
  complaint: "投诉",
};

const feedback = reactive<Partial<FeedbackDetailVO>>({});
const reply = ref("");
const replying = ref(false);
const closeVisible = ref(false);
const closeReason = ref("");
const closing = ref(false);

onLoad((options: any) => {
  fetchDetail(+(options?.id || 0));
});

async function fetchDetail(id: number) {
  try {
    const res = await FeedbackAPI.getFeedbackDetail(id);
    Object.assign(feedback, res);
  } catch {
    uni.showToast({ title: "获取详情失败", icon: "error" });
  }
}

async function handleReply() {
  if (!reply.value.trim()) {
    uni.showToast({ title: "请输入回复内容", icon: "none" });
    return;
  }
  if (!feedback.id) return;
  replying.value = true;
  try {
    await FeedbackAPI.replyFeedback(feedback.id, { content: reply.value });
    uni.showToast({ title: "回复成功", icon: "success" });
    reply.value = "";
    fetchDetail(feedback.id);
  } catch {
    uni.showToast({ title: "回复失败", icon: "error" });
  } finally {
    replying.value = false;
  }
}

function openClose() {
  closeReason.value = "";
  closeVisible.value = true;
}

async function handleClose() {
  if (!closeReason.value.trim()) {
    uni.showToast({ title: "请输入关闭原因", icon: "none" });
    return;
  }
  if (!feedback.id) return;
  closing.value = true;
  try {
    await FeedbackAPI.closeFeedback(feedback.id, {
      closeReason: closeReason.value,
    });
    uni.showToast({ title: "已关闭", icon: "success" });
    closeVisible.value = false;
    fetchDetail(feedback.id);
  } catch {
    uni.showToast({ title: "关闭失败", icon: "error" });
  } finally {
    closing.value = false;
  }
}
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
}
.detail-card {
  background: $color-white;
  border-radius: 16rpx;
  padding: 20rpx;
  margin-bottom: 30rpx;
}
.info-row {
  display: flex;
  padding: 16rpx 0;
  border-bottom: 1rpx solid $color-border;
  align-items: flex-start;
}
.info-row:last-child {
  border-bottom: none;
}
.label {
  color: $color-text-secondary;
  width: 140rpx;
  flex-shrink: 0;
  font-size: $font-sm;
}
.content-text {
  line-height: 1.6;
}
.status-tag {
  font-size: $font-xs;
  font-weight: 500;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}
.s-pending {
  color: $color-warning;
  background: $color-warning-bg;
}
.s-processing {
  color: $color-primary;
  background: $color-primary-bg;
}
.s-replied {
  color: $color-success;
  background: $color-success-bg;
}
.s-closed {
  color: $color-text-placeholder;
  background: $color-bg-secondary;
}

.replies-section {
  margin-bottom: 30rpx;
}
.section-title {
  font-size: $font-lg;
  font-weight: bold;
  padding: 20rpx 0;
  display: block;
}
.reply-item {
  background: $color-bg-primary;
  border-radius: $radius-md;
  padding: 16rpx;
  margin-bottom: 12rpx;
}
.reply-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8rpx;
}
.reply-author {
  font-size: $font-sm;
  font-weight: 500;
  color: $color-primary;
}
.reply-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.reply-content {
  font-size: $font-sm;
  color: $color-text-primary;
  line-height: 1.5;
}

.action-section {
  margin-bottom: 30rpx;
}
.form-textarea {
  width: 100%;
  box-sizing: border-box;
  min-height: 160rpx;
  padding: 16rpx 20rpx;
  font-size: $font-md;
  border: 1rpx solid $color-border;
  border-radius: $radius-md;
  margin-bottom: 16rpx;
}
.btn-row {
  display: flex;
  gap: 20rpx;
  margin-top: 20rpx;
}
.btn {
  flex: 1;
  padding: 12rpx 20rpx;
  border-radius: $radius-sm;
  font-size: $font-sm;
  line-height: 1.6;
  &::after {
    border: none;
  }
}
.btn-primary {
  color: $color-white;
  background: $color-primary;
}
.btn-warning {
  color: $color-white;
  background: $color-warning;
}
.btn-danger {
  color: $color-white;
  background: $color-danger;
}
.btn-default {
  color: $color-text-primary;
  background: $color-bg-secondary;
}

.popup-content {
  padding: 32rpx;
}
.popup-title {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
  display: block;
  margin-bottom: 24rpx;
}
.popup-footer {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
</style>
