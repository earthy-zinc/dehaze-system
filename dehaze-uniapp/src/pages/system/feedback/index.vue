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
          <input
            class="search-input"
            v-model="keyword"
            placeholder="搜索反馈内容"
            confirm-type="search"
            @confirm="handleSearch"
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
          <view class="loading-spinner" />
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
                <text class="action-btn" @click="openReply(f.id, 'feedback')"
                  >回复</text
                >
                <text class="action-btn danger" @click="openClose(f.id)"
                  >关闭</text
                >
              </view>
            </view>
          </view>
          <view v-if="!hasMoreFb" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreFeedback"
            >加载更多</view
          >
        </view>

        <view v-else class="empty-tip">暂无反馈数据</view>
      </view>

      <!-- 评价列表 -->
      <view v-else>
        <view v-if="ratingLoading" class="loading-container">
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
            <view class="card-meta">
              <text>用户: {{ r.username || "ID:" + r.userId }}</text>
            </view>
            <view v-if="r.adminReply" class="admin-reply">
              <text>管理员回复: {{ r.adminReply }}</text>
            </view>
            <view class="card-footer">
              <text class="card-time">{{ r.createTime }}</text>
              <view class="card-actions">
                <text class="action-btn" @click="openReply(r.id, 'rating')"
                  >回复</text
                >
                <text
                  v-if="r.isHidden !== 1"
                  class="action-btn danger"
                  @click="handleHideRating(r.id)"
                  >隐藏</text
                >
              </view>
            </view>
          </view>
          <view v-if="!hasMoreR" class="end-text">— 没有更多了 —</view>
          <view v-else class="load-more" @click="loadMoreRatings"
            >加载更多</view
          >
        </view>

        <view v-else class="empty-tip">暂无评价数据</view>
      </view>

      <!-- 回复弹窗 -->
      <Popup
        :show="replyVisible"
        mode="bottom"
        round
        @close="replyVisible = false"
      >
        <view class="popup-content">
          <text class="popup-title">回复</text>
          <textarea
            class="form-textarea"
            v-model="replyContent"
            placeholder="请输入回复内容"
          />
          <view class="popup-footer">
            <button class="btn btn-default" @click="replyVisible = false">
              取消
            </button>
            <button
              class="btn btn-primary"
              :disabled="replying"
              @click="handleReply"
            >
              提交
            </button>
          </view>
        </view>
      </Popup>

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
    </view>
  </PageLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import Popup from "@/components/common/Popup.vue";
import { usePagedList } from "@/composables/usePagedList";
import { FeedbackAPI } from "dehaze-sdk-js";
import type { FeedbackPageVO, RatingPageVO } from "dehaze-sdk-js";

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

const statusFilter = ref("");
const typeFilter = ref("");

const {
  list: feedbacks,
  keyword,
  hasMore: hasMoreFb,
  loading,
  fetchList: fetchFeedbacks,
  handleSearch,
  loadMore: loadMoreFeedback,
} = usePagedList<FeedbackPageVO>({
  fetcher: (p) => {
    const params: any = { pageNum: p.pageNum, pageSize: 15 };
    if (p.keyword) params.keywords = p.keyword;
    if (statusFilter.value) params.status = statusFilter.value;
    if (typeFilter.value) params.feedbackType = typeFilter.value;
    return FeedbackAPI.listFeedback(params).then((r) => r.list || []);
  },
});

const {
  list: ratings,
  hasMore: hasMoreR,
  loading: ratingLoading,
  fetchList: fetchRatings,
  loadMore: loadMoreRatings,
} = usePagedList<RatingPageVO>({
  fetcher: (p) =>
    FeedbackAPI.listRatings({ pageNum: p.pageNum, pageSize: 15 }).then(
      (r) => r.list || []
    ),
});

// 回复
const replyVisible = ref(false);
const replyTarget = ref<{ id: number; type: "feedback" | "rating" } | null>(
  null
);
const replyContent = ref("");
const replying = ref(false);

// 关闭
const closeVisible = ref(false);
const closeTargetId = ref(0);
const closeReason = ref("");
const closing = ref(false);

function applyStatusFilter(val: string) {
  statusFilter.value = val;
  fetchFeedbacks(true);
}

function applyTypeFilter(val: string) {
  typeFilter.value = val;
  fetchFeedbacks(true);
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
      await FeedbackAPI.replyFeedback(replyTarget.value.id, {
        content: replyContent.value,
      });
    } else {
      await FeedbackAPI.replyRating(replyTarget.value.id, replyContent.value);
    }
    uni.showToast({ title: "回复成功", icon: "success" });
    replyVisible.value = false;
    if (replyTarget.value.type === "feedback") fetchFeedbacks(true);
    else fetchRatings(true);
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
    await FeedbackAPI.closeFeedback(closeTargetId.value, {
      closeReason: closeReason.value,
    });
    uni.showToast({ title: "已关闭", icon: "success" });
    closeVisible.value = false;
    fetchFeedbacks(true);
  } catch {
    uni.showToast({ title: "关闭失败", icon: "error" });
  } finally {
    closing.value = false;
  }
}

async function handleHideRating(id: number) {
  const res = await uni.showModal({
    title: "确认隐藏",
    content: "确定要隐藏这条评价吗？",
  });
  if (!res.confirm) return;
  try {
    await FeedbackAPI.hideRating(id);
    uni.showToast({ title: "已隐藏", icon: "success" });
    fetchRatings(true);
  } catch {
    uni.showToast({ title: "操作失败", icon: "error" });
  }
}

fetchFeedbacks(true);
fetchRatings(true);
</script>

<style lang="scss" scoped>
.page-body {
  padding: 20rpx;
  padding-bottom: 60rpx;
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

.search-bar {
  background: $color-white;
  border-radius: $radius-lg;
  padding: 16rpx;
  margin-bottom: $spacing-md;

  .search-input {
    width: 100%;
    box-sizing: border-box;
    padding: 14rpx 20rpx;
    font-size: 28rpx;
    background: $color-bg-secondary;
    border-radius: $radius-md;
  }
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
    color: $color-white;
    background: $color-primary;
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
.header-right {
  flex-shrink: 0;
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
    color: $color-danger;
  }
}

.admin-reply {
  margin: 12rpx 0;
  padding: 12rpx 16rpx;
  background: $color-primary-bg;
  border-radius: 8rpx;
  font-size: $font-sm;
  color: $color-primary;
}

.stars {
  font-size: $font-md;
  color: $color-text-disabled;
  .active {
    color: $color-warning;
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
.form-textarea {
  width: 100%;
  box-sizing: border-box;
  min-height: 160rpx;
  padding: 16rpx 20rpx;
  font-size: 28rpx;
  border: 1rpx solid $color-border;
  border-radius: $radius-md;
}
.popup-footer {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
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
.btn-danger {
  color: $color-white;
  background: $color-danger;
}
.btn-default {
  color: $color-text-primary;
  background: $color-bg-secondary;
}
</style>
