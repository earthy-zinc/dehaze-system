<template>
  <view class="history-list">
    <!-- 头部操作栏 -->
    <view class="list-header">
      <text class="header-title">最近处理的图片</text>
      <view
        v-if="historyRecords.length > 0"
        class="clear-btn"
        @click="handleClearAll"
      >
        <u-icon name="trash" size="14" color="#ef4444" />
        <text class="clear-text">清空</text>
      </view>
    </view>

    <!-- 历史记录列表 -->
    <view v-if="groupedHistory.length > 0" class="history-groups">
      <view
        v-for="group in groupedHistory"
        :key="group.title"
        class="history-group"
      >
        <text class="group-title">{{ group.title }}</text>
        <view class="group-list">
          <HistoryCard
            v-for="record in group.records"
            :key="record.id"
            :record="record"
            @load="handleLoad"
            @delete="handleDelete"
          />
        </view>
      </view>
    </view>

    <!-- 空状态 -->
    <view v-else class="empty-state">
      <view class="empty-icon">
        <u-icon name="clock" size="48" color="#d1d5db" />
      </view>
      <text class="empty-text">暂无历史记录</text>
      <text class="empty-hint">处理过的图片会显示在这里</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import HistoryCard from "./HistoryCard.vue";
import type { HistoryRecord, ImageData } from "../data/imageInputData";
import {
  getHistoryRecords,
  deleteHistoryRecord,
  clearHistoryRecords,
  groupHistoryByTime,
} from "../data/imageInputData";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const historyRecords = ref<HistoryRecord[]>([]);

/** 按时间分组的历史记录 */
const groupedHistory = computed(() => {
  return groupHistoryByTime(historyRecords.value);
});

/** 加载历史记录 */
const loadHistory = () => {
  historyRecords.value = getHistoryRecords();
};

/** 加载历史记录图片 */
const handleLoad = (record: HistoryRecord) => {
  // 从历史记录加载图片
  uni.showToast({
    title: "正在加载...",
    icon: "loading",
  });

  // 模拟加载过程
  setTimeout(() => {
    const imageData: ImageData = {
      url: record.thumbnail,
      width: 800,
      height: 600,
      size: 0,
      name: record.fileName,
    };

    emit("select", imageData);
    uni.hideToast();
  }, 500);
};

/** 删除单条记录 */
const handleDelete = (id: number) => {
  uni.showModal({
    title: "确认删除",
    content: "确定要删除这条历史记录吗？",
    success: (res) => {
      if (res.confirm) {
        deleteHistoryRecord(id);
        loadHistory();
        uni.showToast({
          title: "已删除",
          icon: "success",
        });
      }
    },
  });
};

/** 清空所有记录 */
const handleClearAll = () => {
  uni.showModal({
    title: "确认清空",
    content: "确定要清空所有历史记录吗？此操作不可恢复。",
    confirmColor: "#ef4444",
    success: (res) => {
      if (res.confirm) {
        clearHistoryRecords();
        loadHistory();
        uni.showToast({
          title: "已清空",
          icon: "success",
        });
      }
    },
  });
};

onMounted(() => {
  loadHistory();
});

// 暴露刷新方法
defineExpose({
  refresh: loadHistory,
});
</script>

<style lang="scss" scoped>
.history-list {
  min-height: 300rpx;
}

.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24rpx;
}

.header-title {
  font-size: 28rpx;
  color: #6b7280;
}

.clear-btn {
  display: flex;
  align-items: center;
  gap: 8rpx;
  padding: 8rpx 16rpx;
  background: #fef2f2;
  border-radius: 12rpx;

  &:active {
    opacity: 0.8;
  }
}

.clear-text {
  font-size: 24rpx;
  color: #ef4444;
}

.history-groups {
  display: flex;
  flex-direction: column;
  gap: 32rpx;
}

.history-group {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.group-title {
  font-size: 26rpx;
  font-weight: 600;
  color: #4b5563;
  padding-left: 8rpx;
}

.group-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80rpx 0;
}

.empty-icon {
  width: 120rpx;
  height: 120rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 50%;
  margin-bottom: 24rpx;
}

.empty-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 8rpx;
}

.empty-hint {
  font-size: 26rpx;
  color: #9ca3af;
}
</style>
