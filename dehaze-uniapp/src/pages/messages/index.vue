<template>
  <PageLayout level="L1" title="消息">
    <view class="messages-page">
      <!-- 搜索栏 -->
      <view v-if="showSearch" class="messages-search-bar">
        <view class="messages-search-input-wrap">
          <input
            class="messages-search-input"
            placeholder="搜索消息"
            v-model="keyword"
            @confirm="handleSearchConfirm"
            focus
          />
          <view class="messages-search-cancel" @click="closeSearch">
            <text>取消</text>
          </view>
        </view>
        <view
          v-if="!keyword && searchHistory.length > 0"
          class="messages-search-history"
        >
          <text class="search-history-title">搜索历史</text>
          <view class="search-history-tags">
            <view
              v-for="h in searchHistory"
              :key="h"
              class="search-history-tag"
              @click="
                () => {
                  keyword = h;
                  doSearch(1, h, false);
                }
              "
            >
              <text>{{ h }}</text>
            </view>
          </view>
        </view>
      </view>

      <!-- 顶部操作区 -->
      <view class="messages-header">
        <view class="messages-tabs-scroll">
          <scroll-view scroll-x class="messages-tabs">
            <view class="messages-tabs-row">
              <view
                v-for="tab in tabs"
                :key="tab.key"
                class="messages-tab"
                :class="{ active: activeTab === tab.key }"
                @click="handleTabChange(tab.key)"
              >
                <text>{{ tab.label }}</text>
              </view>
            </view>
          </scroll-view>
        </view>
        <view class="messages-header-actions">
          <view class="messages-header-icon" @click="showSearch = true">
            <text class="icon-text">🔍</text>
          </view>
          <template v-if="deleteMode">
            <view class="messages-action-btn" @click="exitDeleteMode">
              <text>取消</text>
            </view>
            <view
              v-if="selectedIds.size > 0"
              class="messages-action-btn danger"
              @click="handleBatchDelete"
            >
              <text>删除({{ selectedIds.size }})</text>
            </view>
          </template>
          <template v-else>
            <view
              v-if="unreadCount > 0"
              class="messages-mark-all"
              @click="handleMarkAllRead"
            >
              <text>全部已读</text>
            </view>
            <view class="messages-header-icon" @click="deleteMode = true">
              <text class="icon-text">🗑</text>
            </view>
            <view class="messages-header-icon" @click="handleGoSettings">
              <text class="icon-text">⚙</text>
            </view>
          </template>
        </view>
      </view>

      <!-- 消息列表 -->
      <scroll-view
        class="messages-list"
        scroll-y
        @scrolltolower="handleLoadMore"
      >
        <view v-if="loading && messages.length === 0" class="messages-status">
          <text>加载中...</text>
        </view>
        <view v-else-if="messages.length === 0" class="messages-empty">
          <SvgIcon name="bell" size="56" color="#d1d5db" />
          <text class="messages-empty-text">暂无消息</text>
          <text class="messages-empty-sub"
            >处理完成、系统通知等将在这里展示</text
          >
        </view>
        <view v-else class="messages-items">
          <view
            v-for="msg in messages"
            :key="msg.id"
            class="message-item"
            :class="{
              unread: msg.readStatus === 0,
              selected: deleteMode && selectedIds.has(msg.id),
            }"
            @click="handleMessageClick(msg)"
            @longpress="handleDeleteSingle(msg.id)"
          >
            <view class="message-item-left">
              <view class="message-item-header">
                <text class="message-item-type">
                  {{ msg.typeLabel || getTypeLabel(msg.type) }}
                </text>
                <view v-if="msg.readStatus === 0" class="message-unread-dot" />
                <view
                  v-if="deleteMode"
                  class="message-checkbox"
                  :class="{ checked: selectedIds.has(msg.id) }"
                >
                  <text v-if="selectedIds.has(msg.id)">✓</text>
                </view>
              </view>
              <text class="message-item-title">{{ msg.title }}</text>
              <text class="message-item-summary">{{ msg.summary || "" }}</text>
            </view>
            <text class="message-item-time">{{
              formatTime(msg.createTime)
            }}</text>
          </view>
        </view>
        <view v-if="loading && messages.length > 0" class="messages-status">
          <text>加载更多...</text>
        </view>
        <view v-if="!hasMore && messages.length > 0" class="messages-status">
          <text>没有更多了</text>
        </view>
      </scroll-view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import PageLayout from "@/layout/index.vue";
import { MessageAPI } from "dehaze-sdk-js";
import type { MessageVO } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

const tabs = [
  { key: "", label: "全部" },
  { key: "announcement", label: "系统公告" },
  { key: "business", label: "业务通知" },
  { key: "member", label: "会员通知" },
  { key: "alert", label: "告警" },
];

const activeTab = ref("");
const messages = ref<MessageVO[]>([]);
const loading = ref(true);
const unreadCount = ref(0);
const pageNum = ref(1);
const hasMore = ref(true);
const showSearch = ref(false);
const keyword = ref("");
const searchHistory = ref<string[]>([]);
const deleteMode = ref(false);
const selectedIds = ref<Set<number>>(new Set());

function formatTime(dateStr: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - d.getTime()) / 86400000);
  const pad = (n: number) => String(n).padStart(2, "0");
  const hhmm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (diffDays === 0) return hhmm;
  if (diffDays === 1) return "昨天";
  if (diffDays === 2) return "前天";
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const fetchUnreadCount = async () => {
  try {
    const res = await MessageAPI.getUnreadCount();
    unreadCount.value = res.count || 0;
    if (res.count > 0) {
      uni.setTabBarBadge({
        index: 3,
        text: String(res.count > 99 ? "99+" : res.count),
      });
    } else {
      uni.removeTabBarBadge({ index: 3 });
    }
  } catch {
    // ignore
  }
};

const fetchMessages = async (page: number, type: string, append = false) => {
  try {
    loading.value = true;
    const queryParams: Record<string, unknown> = {
      pageNum: page,
      pageSize: 20,
    };
    if (type) queryParams.type = type;
    const res = await MessageAPI.getPage(queryParams);
    const list = (res.list as unknown as MessageVO[]) || [];
    if (append) {
      messages.value = [...messages.value, ...list];
    } else {
      messages.value = list;
    }
    hasMore.value = list.length >= 20;
  } catch (error) {
    uni.showToast({
      title: getErrorMessage(error, "加载消息失败"),
      icon: "none",
    });
  } finally {
    loading.value = false;
  }
};

const doSearch = async (page: number, kw: string, append = false) => {
  if (!kw.trim()) return;
  try {
    loading.value = true;
    const res = await MessageAPI.search({
      keyword: kw.trim(),
      pageNum: page,
      pageSize: 20,
    });
    const list = (res.list as unknown as MessageVO[]) || [];
    if (append) {
      messages.value = [...messages.value, ...list];
    } else {
      messages.value = list;
    }
    hasMore.value = list.length >= 20;
    searchHistory.value = [
      kw.trim(),
      ...searchHistory.value.filter((h) => h !== kw.trim()),
    ].slice(0, 5);
  } catch (error) {
    uni.showToast({ title: getErrorMessage(error, "搜索失败"), icon: "none" });
  } finally {
    loading.value = false;
  }
};

const handleSearchConfirm = () => {
  if (!keyword.value.trim()) {
    showSearch.value = false;
    pageNum.value = 1;
    fetchMessages(1, activeTab.value, false);
    return;
  }
  messages.value = [];
  pageNum.value = 1;
  hasMore.value = true;
  doSearch(1, keyword.value, false);
};

const closeSearch = () => {
  showSearch.value = false;
  keyword.value = "";
  pageNum.value = 1;
  fetchMessages(1, activeTab.value, false);
};

const exitDeleteMode = () => {
  deleteMode.value = false;
  selectedIds.value = new Set();
};

onMounted(() => {
  fetchUnreadCount();
  fetchMessages(1, activeTab.value, false);
});

const handleTabChange = (key: string) => {
  activeTab.value = key;
  messages.value = [];
  pageNum.value = 1;
  hasMore.value = true;
  showSearch.value = false;
  keyword.value = "";
  deleteMode.value = false;
  selectedIds.value = new Set();
  fetchMessages(1, key, false);
};

const handleMessageClick = async (message: MessageVO) => {
  if (deleteMode.value) {
    const next = new Set(selectedIds.value);
    if (next.has(message.id)) next.delete(message.id);
    else next.add(message.id);
    selectedIds.value = next;
    return;
  }
  if (message.readStatus === 0) {
    try {
      await MessageAPI.markRead(message.id);
      messages.value = messages.value.map((m) =>
        m.id === message.id ? { ...m, readStatus: 1 } : m
      );
      unreadCount.value = Math.max(0, unreadCount.value - 1);
    } catch {
      // continue
    }
  }
  uni.navigateTo({ url: `/pages/messages/detail/index?id=${message.id}` });
};

const handleMarkAllRead = async () => {
  try {
    await MessageAPI.markAllRead(activeTab.value || undefined);
    messages.value = messages.value.map((m) => ({ ...m, readStatus: 1 }));
    unreadCount.value = 0;
    uni.removeTabBarBadge({ index: 3 });
    uni.showToast({ title: "已全部标记为已读", icon: "success" });
  } catch (error) {
    uni.showToast({ title: getErrorMessage(error, "操作失败"), icon: "none" });
  }
};

const handleLoadMore = () => {
  if (loading.value || !hasMore.value) return;
  const nextPage = pageNum.value + 1;
  pageNum.value = nextPage;
  if (showSearch.value && keyword.value.trim()) {
    doSearch(nextPage, keyword.value, true);
  } else {
    fetchMessages(nextPage, activeTab.value, true);
  }
};

const handleGoSettings = () => {
  uni.navigateTo({ url: "/pages/notify/index" });
};

const handleDeleteSingle = async (id: number) => {
  if (deleteMode.value) return;
  const res = await uni.showModal({
    title: "确认删除",
    content: "确定删除这条消息吗？",
  });
  if (!res.confirm) return;
  try {
    await MessageAPI.deleteByIds(String(id));
    messages.value = messages.value.filter((m) => m.id !== id);
    uni.showToast({ title: "已删除", icon: "success" });
    fetchUnreadCount();
  } catch (error) {
    uni.showToast({ title: getErrorMessage(error, "删除失败"), icon: "none" });
  }
};

const handleBatchDelete = async () => {
  if (selectedIds.value.size === 0) return;
  const res = await uni.showModal({
    title: "批量删除",
    content: `确定删除选中的 ${selectedIds.value.size} 条消息吗？`,
  });
  if (!res.confirm) return;
  try {
    await MessageAPI.deleteByIds(Array.from(selectedIds.value).join(","));
    messages.value = messages.value.filter((m) => !selectedIds.value.has(m.id));
    selectedIds.value = new Set();
    deleteMode.value = false;
    uni.showToast({ title: `已删除`, icon: "success" });
    fetchUnreadCount();
  } catch (error) {
    uni.showToast({
      title: getErrorMessage(error, "批量删除失败"),
      icon: "none",
    });
  }
};

const getTypeLabel = (type: string): string => {
  const tab = tabs.find((t) => t.key === type);
  return tab ? tab.label : type;
};
</script>

<style lang="scss" scoped>
.messages-page {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: $color-bg-primary;
}

/* 搜索栏 */
.messages-search-bar {
  padding: 16rpx 24rpx;
  background: $color-white;
  border-bottom: 1rpx solid $color-border-light;
}
.messages-search-input-wrap {
  display: flex;
  gap: 16rpx;
  align-items: center;
}
.messages-search-input {
  flex: 1;
  padding: 14rpx 20rpx;
  font-size: $font-sm;
  background: $color-bg-secondary;
  border-radius: 999rpx;
}
.messages-search-cancel {
  padding: 6rpx 12rpx;
  font-size: $font-sm;
  color: $color-primary;
}
.messages-search-history {
  margin-top: 16rpx;
}
.search-history-title {
  display: block;
  margin-bottom: 12rpx;
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.search-history-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
}
.search-history-tag {
  padding: 8rpx 20rpx;
  font-size: $font-xs;
  color: $color-text-secondary;
  background: $color-bg-secondary;
  border-radius: 999rpx;
}

/* 顶部操作区 */
.messages-header {
  display: flex;
  gap: 16rpx;
  align-items: center;
  padding: 20rpx 24rpx 12rpx;
  background: $color-white;
  border-bottom: 1rpx solid $color-border-light;
}
.messages-tabs-scroll {
  flex: 1;
  overflow: hidden;
}
.messages-tabs {
  white-space: nowrap;
}
.messages-tabs-row {
  display: inline-flex;
  gap: 12rpx;
}
.messages-tab {
  padding: 10rpx 24rpx;
  font-size: $font-sm;
  color: $color-text-secondary;
  white-space: nowrap;
  background: $color-bg-secondary;
  border-radius: 999rpx;
  &.active {
    color: $color-white;
    background: $color-primary;
  }
}
.messages-header-actions {
  display: flex;
  flex-shrink: 0;
  gap: 8rpx;
  align-items: center;
}
.messages-header-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 56rpx;
  height: 56rpx;
  border-radius: 50%;
  .icon-text {
    font-size: 32rpx;
  }
}
.messages-mark-all {
  padding: 6rpx 16rpx;
  font-size: $font-xs;
  color: $color-primary;
  border-radius: 999rpx;
}
.messages-action-btn {
  padding: 6rpx 16rpx;
  font-size: $font-xs;
  color: $color-primary;
  border-radius: 999rpx;
  &.danger {
    color: $color-danger;
  }
}

/* 消息列表 */
.messages-list {
  flex: 1;
}
.messages-items {
  padding: 12rpx 24rpx;
}
.message-item {
  display: flex;
  gap: 16rpx;
  padding: 24rpx;
  margin-bottom: 12rpx;
  background: $color-white;
  border-radius: $radius-md;
  box-shadow: $shadow-sm;
  &.unread {
    border-left: 4rpx solid $color-primary;
  }
  &.selected {
    background: $color-primary-bg;
  }
}
.message-item-left {
  flex: 1;
  min-width: 0;
}
.message-item-header {
  display: flex;
  gap: 10rpx;
  align-items: center;
  margin-bottom: 6rpx;
}
.message-item-type {
  padding: 2rpx 10rpx;
  font-size: $font-xs;
  color: $color-primary;
  background: $color-primary-bg;
  border-radius: $radius-sm;
}
.message-unread-dot {
  width: 12rpx;
  height: 12rpx;
  background: $color-danger;
  border-radius: 50%;
}
.message-checkbox {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32rpx;
  height: 32rpx;
  margin-left: auto;
  font-size: 22rpx;
  color: #fff;
  border: 2rpx solid $color-border-light;
  border-radius: 50%;
  &.checked {
    background: $color-primary;
    border-color: $color-primary;
  }
}
.message-item-title {
  display: block;
  margin-bottom: 6rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: $font-md;
  font-weight: 500;
  color: $color-text-primary;
  white-space: nowrap;
}
.message-item-summary {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: $font-xs;
  color: $color-text-placeholder;
  white-space: nowrap;
}
.message-item-time {
  flex-shrink: 0;
  margin-top: 6rpx;
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.messages-status {
  display: flex;
  justify-content: center;
  padding: 32rpx;
  font-size: $font-xs;
  color: $color-text-placeholder;
}
.messages-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  padding: 96rpx 24rpx;
  .messages-empty-text {
    font-size: $font-lg;
    font-weight: 600;
    color: $color-text-primary;
  }
  .messages-empty-sub {
    font-size: $font-sm;
    color: $color-text-placeholder;
  }
}
</style>
