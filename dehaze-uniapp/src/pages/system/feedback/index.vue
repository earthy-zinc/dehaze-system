<template>
  <PageLayout level="L2" title="反馈评价管理">
    <view class="page-body">
      <view class="tabs">
        <view :class="['tab', { active: tab === 0 }]" @click="tab = 0">
          <text>反馈列表</text>
        </view>
        <view :class="['tab', { active: tab === 1 }]" @click="tab = 1">
          <text>评价列表</text>
        </view>
      </view>

      <!-- 反馈列表 -->
      <view v-if="tab === 0">
        <view class="search-bar">
          <u-search
            v-model="keyword"
            placeholder="搜索反馈内容"
            @search="handleSearch"
            @clear="handleSearch"
          />
          <view class="filter-row">
            <view
              v-for="s in statusOptions"
              :key="s.value"
              :class="['filter-tag', { active: statusFilter === s.value }]"
              @click="applyStatusFilter(s.value)"
            >
              {{ s.label }}
            </view>
          </view>
          <view class="filter-row">
            <view
              v-for="t in typeOptions"
              :key="t.value"
              :class="['filter-tag', { active: typeFilter === t.value }]"
              @click="applyTypeFilter(t.value)"
            >
              {{ t.label }}
            </view>
          </view>
        </view>

        <view v-if="loading" class="loading-container">
          <up-loading-icon mode="circle" size="40" />
          <text class="loading-text">加载中...</text>
        </view>

        <view v-else-if="feedbacks.length > 0" class="list">
          <view v-for="f in feedbacks" :key="f.id" class="card">
            <view class="card-header">
              <text class="card-title">{{ f.title }}</text>
              <view class="header-right">
                <text :class="['status-tag', 's-' + f.status]">
                  {{ statusMap[f.status] || f.status }}
                </text>
              </view>
            </view>
            <view class="card-meta">
              <text>用户: {{ f.username }} (ID:{{ f.userId }})</text>
              <text>类型: {{ typeMap[f.feedbackType] || f.feedbackType }}</text>
              <text>优先级: {{ f.priority }}</text>
            </view>
            <text class="card-content">{{ f.content }}</text>
            <view class="card-footer">
              <text class="card-time">{{ f.createTime }}</text>
              <view v-if="f.status !== 'closed'" class="card-actions">
                <text class="action-btn" @click="openReply(f.id, 'feedback')">回复</text>
                <text class="action-btn danger" @click="openClose(f.id)">关闭</text>
              </view>
            </view>
          </view>
          <view v-if="!hasMoreFb" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreFeedback">加载更多</view>
        </view>

        <view v-else class="empty-state">
          <up-empty mode="list" text="暂无反馈数据" />
        </view>
      </view>

      <!-- 评价列表 -->
      <view v-else>
        <view v-if="ratingLoading" class="loading-container">
          <up-loading-icon mode="circle" size="40" />
          <text class="loading-text">加载中...</text>
        </view>

        <view v-else-if="ratings.length > 0" class="list">
          <view v-for="r in ratings" :key="r.id" class="card">
            <view class="card-header">
              <text class="card-title">{{ r.algorithmName }}</text>
              <view class="stars">
                <text v-for="i in 5" :key="i" :class="{ active: i <= r.rating }">★</text>
              </view>
            </view>
            <text v-if="r.comment" class="card-content">{{ r.comment }}</text>
            <view class="card-meta">
              <text>用户: {{ r.username || 'ID:' + r.userId }}</text>
            </view>
            <view v-if="r.adminReply" class="admin-reply">
              <text>管理员回复: {{ r.adminReply }}</text>
            </view>
            <view class="card-footer">
              <text class="card-time">{{ r.createTime }}</text>
              <view class="card-actions">
                <text class="action-btn" @click="openReply(r.id, 'rating')">回复</text>
                <text v-if="r.isHidden !== 1" class="action-btn danger" @click="handleHideRating(r.id)">隐藏</text>
              </view>
            </view>
          </view>
          <view v-if="!hasMoreR" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreRatings">加载更多</view>
        </view>

        <view v-else class="empty-state">
          <up-empty mode="list" text="暂无评价数据" />
        </view>
      </view>

      <!-- 回复弹窗 -->
      <u-popup :show="replyVisible" mode="bottom" round="24" @close="replyVisible = false">
        <view class="popup-content">
          <text class="popup-title">回复</text>
          <u-input
            v-model="replyContent"
            type="textarea"
            placeholder="请输入回复内容"
            border="surround"
            :rows="4"
          />
          <view class="popup-footer">
            <u-button text="取消" @click="replyVisible = false" />
            <u-button text="提交" type="primary" @click="handleReply" :loading="replying" />
          </view>
        </view>
      </u-popup>

      <!-- 关闭弹窗 -->
      <u-popup :show="closeVisible" mode="bottom" round="24" @close="closeVisible = false">
        <view class="popup-content">
          <text class="popup-title">关闭反馈</text>
          <u-input
            v-model="closeReason"
            type="textarea"
            placeholder="请填写关闭原因（必填）"
            border="surround"
            :rows="3"
          />
          <view class="popup-footer">
            <u-button text="取消" @click="closeVisible = false" />
            <u-button text="确认关闭" type="error" @click="handleClose" :loading="closing" />
          </view>
        </view>
      </u-popup>
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import { FeedbackAPI } from "dehaze-sdk-js";
import type { FeedbackPageVO, RatingPageVO, FeedbackStatus, FeedbackType } from "dehaze-sdk-js";

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

const statusOptions = [
  { value: "", label: "全部" },
  { value: "pending", label: "待处理" },
  { value: "processing", label: "处理中" },
  { value: "replied", label: "已回复" },
  { value: "closed", label: "已关闭" },
];

const typeOptions = [
  { value: "", label: "全部类型" },
  { value: "suggestion", label: "建议" },
  { value: "bug", label: "缺陷" },
  { value: "experience", label: "体验" },
  { value: "complaint", label: "投诉" },
];

const tab = ref(0);

// 反馈
const feedbacks = ref<FeedbackPageVO[]>([]);
const loading = ref(false);
const fbPage = ref(1);
const hasMoreFb = ref(true);
const keyword = ref("");
const statusFilter = ref("");
const typeFilter = ref("");

// 评价
const ratings = ref<RatingPageVO[]>([]);
const ratingLoading = ref(false);
const rPage = ref(1);
const hasMoreR = ref(true);

// 回复
const replyVisible = ref(false);
const replyTarget = ref<{ id: number; type: "feedback" | "rating" } | null>(null);
const replyContent = ref("");
const replying = ref(false);

// 关闭
const closeVisible = ref(false);
const closeTargetId = ref(0);
const closeReason = ref("");
const closing = ref(false);

async function fetchFeedbacks(page = 1) {
  loading.value = true;
  try {
    const params: any = { pageNum: page, pageSize: 15 };
    if (keyword.value) params.keywords = keyword.value;
    if (statusFilter.value) params.status = statusFilter.value;
    if (typeFilter.value) params.feedbackType = typeFilter.value;
    const res = await FeedbackAPI.listFeedback(params);
    if (page === 1) feedbacks.value = res.list;
    else feedbacks.value = [...feedbacks.value, ...res.list];
    hasMoreFb.value = feedbacks.value.length < res.total;
    fbPage.value = page;
  } catch {
    uni.showToast({ title: "加载失败", icon: "none" });
  } finally {
    loading.value = false;
  }
}

async function fetchRatings(page = 1) {
  ratingLoading.value = true;
  try {
    const res = await FeedbackAPI.listRatings({ pageNum: page, pageSize: 15 });
    if (page === 1) ratings.value = res.list;
    else ratings.value = [...ratings.value, ...res.list];
    hasMoreR.value = ratings.value.length < res.total;
    rPage.value = page;
  } catch {
    uni.showToast({ title: "加载评价失败", icon: "none" });
  } finally {
    ratingLoading.value = false;
  }
}

function handleSearch() {
  fetchFeedbacks(1);
}

function applyStatusFilter(val: string) {
  statusFilter.value = val;
  fetchFeedbacks(1);
}

function applyTypeFilter(val: string) {
  typeFilter.value = val;
  fetchFeedbacks(1);
}

function loadMoreFeedback() {
  if (hasMoreFb.value) fetchFeedbacks(fbPage.value + 1);
}

function loadMoreRatings() {
  if (hasMoreR.value) fetchRatings(rPage.value + 1);
}

function openReply(id: number, type: "feedback" | "rating") {
  replyTarget.value = { id, type };
  replyContent.value = "";
  replyVisible.value = true;
}

async function handleReply() {
  if (!replyTarget.value || !replyContent.value.trim()) {
    uni.showToast({ title: "请输入回复内容", icon: "none" });
    return;
  }
  replying.value = true;
  try {
    if (replyTarget.value.type === "feedback") {
      await FeedbackAPI.replyFeedback(replyTarget.value.id, { content: replyContent.value });
    } else {
      await FeedbackAPI.replyRating(replyTarget.value.id, replyContent.value);
    }
    uni.showToast({ title: "回复成功", icon: "success" });
    replyVisible.value = false;
    if (replyTarget.value.type === "feedback") fetchFeedbacks(fbPage.value);
    else fetchRatings(rPage.value);
  } catch {
    uni.showToast({ title: "回复失败", icon: "error" });
  } finally {
    replying.value = false;
  }
}

function openClose(id: number) {
  closeTargetId.value = id;
  closeReason.value = "";
  closeVisible.value = true;
}

async function handleClose() {
  if (!closeReason.value.trim()) {
    uni.showToast({ title: "请输入关闭原因", icon: "none" });
    return;
  }
  closing.value = true;
  try {
    await FeedbackAPI.closeFeedback(closeTargetId.value, { closeReason: closeReason.value });
    uni.showToast({ title: "已关闭", icon: "success" });
    closeVisible.value = false;
    fetchFeedbacks(fbPage.value);
  } catch {
    uni.showToast({ title: "关闭失败", icon: "error" });
  } finally {
    closing.value = false;
  }
}

async function handleHideRating(id: number) {
  const res = await uni.showModal({ title: "确认隐藏", content: "确定要隐藏这条评价吗？" });
  if (!res.confirm) return;
  try {
    await FeedbackAPI.hideRating(id);
    uni.showToast({ title: "已隐藏", icon: "success" });
    fetchRatings(rPage.value);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
}

fetchFeedbacks(1);
fetchRatings(1);
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
  padding-bottom: 60rpx;
}

.tabs {
  display: flex;
  background: #fff;
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

.search-bar {
  background: #fff;
  border-radius: $radius-lg;
  padding: 16rpx;
  margin-bottom: $spacing-md;
}
.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
  margin-top: 12rpx;
}
.filter-tag {
  padding: 8rpx 16rpx;
  font-size: $font-xs;
  color: $color-text-secondary;
  background: $color-bg-primary;
  border-radius: 8rpx;
  &.active {
    color: #fff;
    background: $color-primary;
  }
}

.list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.card {
  background: #fff;
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
.header-right {
  flex-shrink: 0;
}
.status-tag {
  font-size: $font-xs;
  font-weight: 500;
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}
.s-pending { color: #f59e0b; background: #fef3c7; }
.s-processing { color: #3b82f6; background: #dbeafe; }
.s-replied { color: #10b981; background: #ecfdf5; }
.s-closed { color: #9ca3af; background: #f3f4f6; }

.card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8rpx 24rpx;
  margin-bottom: 12rpx;
  font-size: $font-xs;
  color: $color-text-secondary;
}
.card-content {
  font-size: $font-sm;
  color: $color-text-primary;
  display: block;
  margin-bottom: 12rpx;
  line-height: 1.6;
}
.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.card-time {
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.card-actions {
  display: flex;
  gap: 16rpx;
}
.action-btn {
  font-size: $font-sm;
  color: $color-primary;
  &.danger {
    color: #ef4444;
  }
}

.admin-reply {
  margin: 12rpx 0;
  padding: 12rpx 16rpx;
  background: #eff6ff;
  border-radius: 8rpx;
  font-size: $font-sm;
  color: #3b82f6;
}

.stars {
  font-size: $font-md;
  color: #d1d5db;
  .active {
    color: #f59e0b;
  }
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
