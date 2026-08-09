<template>
  <PageLayout level="L2" title="反馈评价" class="page">
    <view class="main-content">
      <!-- Tab 切换 -->
      <view class="tabs">
        <view
          :class="['tab', { active: activeTab === 'feedback' }]"
          @click="activeTab = 'feedback'"
        >
          <text>我的反馈</text>
        </view>
        <view
          :class="['tab', { active: activeTab === 'rating' }]"
          @click="activeTab = 'rating'"
        >
          <text>我的评价</text>
        </view>
      </view>

      <!-- 反馈列表 -->
      <view v-if="activeTab === 'feedback'">
        <view v-if="loading" class="loading-container">
          <view class="loading-spinner" />
          <text class="loading-text">加载中...</text>
        </view>

        <view v-else-if="feedbacks.length > 0" class="list">
          <view v-for="fb in feedbacks" :key="fb.id" class="card">
            <view class="card-header">
              <text class="card-title">{{ fb.title }}</text>
              <text :class="['fb-status', statusClass(fb.status)]">{{
                statusText(fb.status)
              }}</text>
            </view>
            <text class="card-content">{{ fb.content }}</text>
            <text class="card-time">{{
              formatRelativeTime(fb.createTime)
            }}</text>
          </view>
          <view v-if="!hasMoreFb" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreFeedback"
            >加载更多</view
          >
        </view>

        <view v-else class="empty-state">
          <view class="empty-tip">暂无反馈</view>
        </view>

        <view class="submit-btn" @click="showForm = true">
          <SvgIcon name="plus" size="20" color="#fff" />
          <text>提交反馈</text>
        </view>
      </view>

      <!-- 评价列表 -->
      <view v-else>
        <view v-if="loading" class="loading-container">
          <view class="loading-spinner" />
          <text class="loading-text">加载中...</text>
        </view>

        <view v-else-if="ratings.length > 0" class="list">
          <view v-for="r in ratings" :key="r.id" class="card">
            <view class="card-header">
              <text class="card-title">{{ r.algorithmName }}</text>
              <view class="stars">
                <text v-for="i in 5" :key="i" :class="{ active: i <= r.rating }"
                  >★</text
                >
              </view>
            </view>
            <text v-if="r.comment" class="card-content">{{ r.comment }}</text>
            <view v-if="r.adminReply" class="admin-reply">
              <text>管理员回复: {{ r.adminReply }}</text>
            </view>
            <text class="card-time">{{
              formatRelativeTime(r.createTime)
            }}</text>
          </view>
          <view v-if="!hasMoreR" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreRatings"
            >加载更多</view
          >
        </view>

        <view v-else class="empty-state">
          <view class="empty-tip">暂无评价</view>
        </view>
      </view>

      <!-- 提交反馈弹窗 -->
      <Popup :show="showForm" mode="bottom" round @close="showForm = false">
        <view class="form-container">
          <text class="form-title">提交反馈</text>
          <view class="type-selector">
            <view
              v-for="t in feedbackTypes"
              :key="t.value"
              :class="[
                'type-option',
                { active: form.feedbackType === t.value },
              ]"
              @click="form.feedbackType = t.value"
            >
              {{ t.label }}
            </view>
          </view>
          <input v-model="form.title" placeholder="标题" class="form-item" />
          <textarea
            v-model="form.content"
            placeholder="内容"
            class="form-item"
            :rows="4"
          />
          <input
            v-model="form.contact"
            placeholder="联系方式（选填）"
            class="form-item"
          />
          <view class="form-footer">
            <button class="btn" @click="showForm = false">取消</button>
            <button
              class="btn btn-primary"
              :disabled="submitting"
              @click="submitFeedback"
            >
              提交
            </button>
          </view>
        </view>
      </Popup>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted, watch } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import { FeedbackAPI } from "dehaze-sdk-js";
import type {
  FeedbackPageVO,
  FeedbackStatus,
  MyRatingVO,
  FeedbackType,
} from "dehaze-sdk-js";
import { formatRelativeTime } from "@/utils/format";

const activeTab = ref("feedback");
const loading = ref(false);
const showForm = ref(false);
const submitting = ref(false);

// 反馈
const feedbacks = ref<FeedbackPageVO[]>([]);
const fbPage = ref(1);
const hasMoreFb = ref(true);

// 评价
const ratings = ref<MyRatingVO[]>([]);
const rPage = ref(1);
const hasMoreR = ref(true);

const feedbackTypes: { value: FeedbackType; label: string }[] = [
  { value: "suggestion", label: "建议" },
  { value: "bug", label: "缺陷" },
  { value: "experience", label: "体验" },
  { value: "complaint", label: "投诉" },
];

// 表单
const form = ref({
  feedbackType: "suggestion" as FeedbackType,
  title: "",
  content: "",
  contact: "",
});

function statusText(status: FeedbackStatus): string {
  const map: Record<string, string> = {
    pending: "待处理",
    processing: "处理中",
    replied: "已回复",
    closed: "已关闭",
  };
  return map[status] || status;
}

function statusClass(status: FeedbackStatus): string {
  const map: Record<string, string> = {
    pending: "s-pending",
    processing: "s-processing",
    replied: "s-replied",
    closed: "s-closed",
  };
  return map[status] || "";
}

async function loadFeedbacks(page = 1) {
  if (loading.value) return;
  loading.value = true;
  try {
    const result = await FeedbackAPI.listMyFeedback({
      pageNum: page,
      pageSize: 20,
    });
    if (page === 1) feedbacks.value = result.list;
    else feedbacks.value = [...feedbacks.value, ...result.list];
    hasMoreFb.value = feedbacks.value.length < result.total;
    fbPage.value = page;
  } catch {
    uni.showToast({ title: "加载失败", icon: "none" });
  } finally {
    loading.value = false;
  }
}

async function loadRatings(page = 1) {
  if (loading.value) return;
  loading.value = true;
  try {
    const result = await FeedbackAPI.listMyRatings({
      pageNum: page,
      pageSize: 20,
    });
    if (page === 1) ratings.value = result.list;
    else ratings.value = [...ratings.value, ...result.list];
    hasMoreR.value = ratings.value.length < result.total;
    rPage.value = page;
  } catch {
    uni.showToast({ title: "加载失败", icon: "none" });
  } finally {
    loading.value = false;
  }
}

function loadMoreFeedback() {
  if (hasMoreFb.value) loadFeedbacks(fbPage.value + 1);
}

function loadMoreRatings() {
  if (hasMoreR.value) loadRatings(rPage.value + 1);
}

async function submitFeedback() {
  if (!form.value.title || !form.value.content) {
    uni.showToast({ title: "请填写标题和内容", icon: "none" });
    return;
  }
  submitting.value = true;
  try {
    await FeedbackAPI.createFeedback({
      feedbackType: form.value.feedbackType,
      title: form.value.title,
      content: form.value.content,
      contact: form.value.contact || undefined,
    });
    uni.showToast({ title: "提交成功", icon: "success" });
    showForm.value = false;
    form.value = {
      feedbackType: "suggestion",
      title: "",
      content: "",
      contact: "",
    };
    loadFeedbacks(1);
  } catch {
    uni.showToast({ title: "提交失败", icon: "none" });
  } finally {
    submitting.value = false;
  }
}

watch(activeTab, (tab) => {
  if (tab === "feedback") loadFeedbacks(1);
  else loadRatings(1);
});

onMounted(() => loadFeedbacks(1));
</script>

<style lang="scss" scoped>
@import "@/styles/mixins.scss";

.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
  @include safe-area-bottom(120rpx);
}

.tabs {
  display: flex;
  background: $color-white;
  border-radius: $radius-lg;
  margin-bottom: $spacing-md;
  overflow: hidden;
}
.tab {
  flex: 1;
  text-align: center;
  padding: 24rpx;
  font-size: $font-md;
  color: $color-text-secondary;
  &.active {
    color: $color-primary;
    font-weight: 600;
    background: $color-primary-bg;
  }
}

.list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.card {
  background: $color-white;
  border-radius: $radius-lg;
  padding: 24rpx;
  box-shadow: $shadow-sm;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12rpx;
}
.card-title {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.card-content {
  font-size: $font-sm;
  color: $color-text-secondary;
  display: block;
  margin-bottom: 12rpx;
  line-height: 1.6;
}
.card-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.fb-status {
  font-size: $font-xs;
  font-weight: 500;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
  flex-shrink: 0;
}
.s-pending {
  color: $color-warning;
  background: #fef3c7;
}
.s-processing {
  color: $color-primary;
  background: #dbeafe;
}
.s-replied {
  color: $color-success;
  background: #ecfdf5;
}
.s-closed {
  color: $color-text-placeholder;
  background: $color-bg-secondary;
}

.stars {
  font-size: $font-md;
  color: #d1d5db;
}
.stars .active {
  color: $color-warning;
}

.end-text {
  text-align: center;
  font-size: $font-sm;
  color: $color-text-disabled;
  padding: 32rpx 0;
}
.load-more {
  text-align: center;
  font-size: $font-sm;
  color: $color-secondary;
  padding: 24rpx 0;
}
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.loading-text {
  margin-top: 24rpx;
  font-size: $font-md;
  color: $color-text-placeholder;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}
.empty-tip {
  font-size: $font-md;
}

.submit-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8rpx;
  background: $color-primary;
  color: $color-white;
  font-size: $font-md;
  padding: 24rpx;
  border-radius: $radius-lg;
  margin-top: 32rpx;
  &:active {
    opacity: 0.8;
  }
}

.form-container {
  padding: 32rpx;
}
.form-title {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-text-primary;
  display: block;
  margin-bottom: 24rpx;
}
.type-selector {
  display: flex;
  gap: 16rpx;
  margin-bottom: 24rpx;
}
.type-option {
  padding: 12rpx 24rpx;
  font-size: $font-sm;
  color: $color-text-secondary;
  background: $color-bg-primary;
  border-radius: $radius-md;
  &.active {
    color: $color-white;
    background: $color-primary;
  }
}
.form-item {
  display: block;
  width: 100%;
  box-sizing: border-box;
  padding: 20rpx 24rpx;
  margin-bottom: 16rpx;
  font-size: $font-md;
  color: $color-text-primary;
  background: $color-bg-primary;
  border: 2rpx solid $color-border;
  border-radius: $radius-md;
}
.form-footer {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
.btn {
  flex: 1;
  padding: 20rpx;
  border-radius: $radius-md;
  font-size: $font-md;
  background: $color-bg-secondary;
  color: $color-text-secondary;
  &::after {
    border: none;
  }
}
.btn-primary {
  background: $color-primary;
  color: $color-white;
  &:disabled {
    opacity: 0.5;
  }
}

.admin-reply {
  margin-top: 12rpx;
  padding: 12rpx 16rpx;
  background: #eff6ff;
  border-radius: 8rpx;
  font-size: $font-sm;
  color: $color-primary;
}
</style>
