<template>
  <view class="history-list">
    <!-- 头部操作栏 -->
    <view class="list-header">
      <text class="header-title">最近处理的图片</text>
      <view v-if="records.length > 0" class="clear-btn" @click="handleClear">
        <SvgIcon name="delete" size="14" color="#ef4444" />
        <text class="clear-text">清空</text>
      </view>
    </view>

    <!-- 加载状态 -->
    <view v-if="loading" class="loading-container">
      <view
        class="loading-spinner"
        style="border-top-color: $color-text-placeholder"
      />
      <text class="loading-text">加载中...</text>
    </view>

    <!-- 分组列表 -->
    <view v-else-if="groupedRecords.length > 0" class="history-list-content">
      <view
        v-for="group in groupedRecords"
        :key="group.title"
        class="history-group"
      >
        <text class="group-title">{{ group.title }}</text>
        <view class="group-list">
          <view
            v-for="record in group.records"
            :key="record.id"
            class="history-item"
          >
            <view class="item-main" @click="handleSelect(record)">
              <view class="item-thumbnail">
                <image
                  v-if="record.originalThumbnailUrl || record.originalImageUrl"
                  :src="
                    record.originalThumbnailUrl || record.originalImageUrl || ''
                  "
                  mode="aspectFill"
                  :lazy-load="true"
                />
                <view v-else class="thumbnail-placeholder">
                  <SvgIcon name="photo" size="28" color="#d1d5db" />
                </view>
                <view
                  v-if="record.status === 1 && record.resultImageUrl"
                  class="result-badge"
                >
                  <text>已处理</text>
                </view>
                <view v-if="record.status === 2" class="result-badge failed">
                  <text>失败</text>
                </view>
              </view>
              <view class="item-info">
                <text class="item-name">{{
                  extractFilename(record.originalImageUrl || "") || "未命名图片"
                }}</text>
                <text class="item-time">{{
                  formatTimestamp(record.createTime)
                }}</text>
                <text v-if="record.algorithmName" class="item-algorithm">{{
                  record.algorithmName
                }}</text>
              </view>
              <view class="item-arrow">
                <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
              </view>
            </view>
            <view class="item-actions">
              <view
                class="action-btn reprocess"
                @click="handleReprocess(record)"
              >
                <text>重新处理</text>
              </view>
              <view class="action-btn delete" @click="handleDelete(record.id)">
                <text>删除</text>
              </view>
            </view>
          </view>
        </view>
      </view>
      <view class="history-footer">
        <text class="footer-text">云端同步保存历史记录</text>
      </view>
    </view>

    <!-- 空状态 -->
    <view v-else class="empty-state">
      <view class="empty-icon">
        <SvgIcon name="clock" size="48" color="#d1d5db" />
      </view>
      <text class="empty-text">暂无历史记录</text>
      <text class="empty-hint">处理过的图片会显示在这里</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import type { InputHistoryVO } from "dehaze-sdk-js";
import type { ImageData } from "../data/imageInputData";
import {
  getHistoryPage,
  deleteHistoryRecord,
  clearAllHistory,
  groupHistoryByDate,
  formatTimestamp,
} from "../services/historyService";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const loading = ref(false);
const records = ref<InputHistoryVO[]>([]);

const groupedRecords = computed(() => groupHistoryByDate(records.value));

const loadHistory = async () => {
  loading.value = true;
  try {
    const { list } = await getHistoryPage();
    records.value = list;
  } catch {
    records.value = [];
  } finally {
    loading.value = false;
  }
};

const extractFilename = (url: string): string => {
  if (!url) return "";
  const path = url.split("?")[0] ?? "";
  const segments = path.split("/");
  return segments[segments.length - 1] || "";
};

const handleSelect = (record: InputHistoryVO) => {
  const url = record.originalImageUrl || "";
  if (!url) {
    uni.showToast({ title: "原图地址缺失", icon: "none" });
    return;
  }
  const imageData: ImageData = {
    url,
    name: extractFilename(url),
  };
  emit("select", imageData);
};

const handleReprocess = (record: InputHistoryVO) => {
  const url = record.originalImageUrl || "";
  if (!url) {
    uni.showToast({ title: "原图地址缺失", icon: "none" });
    return;
  }
  uni.setStorageSync(
    "current_image",
    JSON.stringify({ url, path: url, name: extractFilename(url) })
  );
  uni.navigateTo({ url: "/pages/algorithm-select/index" });
};

const handleDelete = (id: number) => {
  uni.showModal({
    title: "确认删除",
    content: "确定要删除这条历史记录吗？",
    confirmColor: "#ef4444",
    success: async (res) => {
      if (res.confirm) {
        try {
          await deleteHistoryRecord(id);
          records.value = records.value.filter((r) => r.id !== id);
          uni.showToast({ title: "已删除", icon: "success" });
        } catch {
          uni.showToast({ title: "删除失败", icon: "none" });
        }
      }
    },
  });
};

const handleClear = () => {
  uni.showModal({
    title: "确认清空",
    content: "确定要清空所有历史记录吗？",
    confirmColor: "#ef4444",
    success: async (res) => {
      if (res.confirm) {
        try {
          await clearAllHistory(true);
          records.value = [];
          uni.showToast({ title: "已清空", icon: "success" });
        } catch {
          uni.showToast({ title: "清空失败", icon: "none" });
        }
      }
    },
  });
};

onMounted(() => {
  loadHistory();
});

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
  gap: 4rpx;
  padding: 8rpx 16rpx;
  border-radius: 8rpx;

  &:active {
    background: $color-danger-bg;
  }
}

.clear-text {
  font-size: 24rpx;
  color: $color-danger;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.loading-text {
  margin-top: 16rpx;
  font-size: 26rpx;
  color: $color-text-placeholder;
}

.history-list-content {
  display: flex;
  flex-direction: column;
}

.history-group {
  margin-bottom: 24rpx;
}

.group-title {
  display: block;
  font-size: 26rpx;
  font-weight: 600;
  color: #374151;
  margin-bottom: 16rpx;
}

.group-list {
  display: flex;
  flex-direction: column;
  gap: 12rpx;
}

.history-item {
  background: $color-white;
  border-radius: 16rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.item-main {
  display: flex;
  align-items: center;
  gap: 20rpx;
  padding: 20rpx;

  &:active {
    background: $color-bg-primary;
  }
}

.item-thumbnail {
  position: relative;
  width: 120rpx;
  height: 120rpx;
  border-radius: 12rpx;
  overflow: hidden;
  flex-shrink: 0;
  background: $color-bg-secondary;

  image {
    width: 100%;
    height: 100%;
  }
}

.thumbnail-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.result-badge {
  position: absolute;
  bottom: 6rpx;
  left: 6rpx;
  padding: 4rpx 10rpx;
  background: $color-success;
  border-radius: 6rpx;

  text {
    font-size: 18rpx;
    color: $color-white;
  }

  &.failed {
    background: #ef4444;
  }
}

.item-info {
  flex: 1;
  min-width: 0;
}

.item-name {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: $color-text-primary;
  margin-bottom: 6rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-time {
  display: block;
  font-size: 24rpx;
  color: $color-text-placeholder;
  margin-bottom: 4rpx;
}

.item-algorithm {
  font-size: 22rpx;
  color: $color-primary;
}

.item-arrow {
  flex-shrink: 0;
}

.item-actions {
  display: flex;
  border-top: 2rpx solid $color-bg-secondary;
}

.action-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18rpx 0;

  text {
    font-size: 26rpx;
  }

  &:active {
    opacity: 0.7;
  }

  &.reprocess {
    text {
      color: $color-primary;
    }
  }

  &.delete {
    border-left: 2rpx solid $color-bg-secondary;

    text {
      color: $color-danger;
    }
  }
}

.history-footer {
  display: flex;
  justify-content: center;
  padding: 24rpx 0;
}

.footer-text {
  font-size: 24rpx;
  color: $color-text-placeholder;
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
  background: $color-bg-secondary;
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
  color: $color-text-placeholder;
}
</style>
