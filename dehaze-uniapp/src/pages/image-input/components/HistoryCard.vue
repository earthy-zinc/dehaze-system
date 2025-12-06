<template>
  <view class="history-card">
    <view class="card-thumbnail">
      <up-image
        :src="record.thumbnail"
        mode="aspectFill"
        width="120rpx"
        height="120rpx"
        :lazy-load="true"
        :fade="true"
      />
      <view v-if="record.status === 'failed'" class="status-badge failed">
        <u-icon name="close" size="12" color="#ffffff" />
      </view>
    </view>

    <view class="card-content" @click="handleLoad">
      <text class="card-filename">{{ record.fileName }}</text>
      <text class="card-time">{{ formatTime(record.timestamp) }}</text>
      <view v-if="record.algorithm" class="card-algorithm">
        <u-icon name="bulb" size="14" color="#3b82f6" />
        <text class="algorithm-text">{{ record.algorithm }}</text>
      </view>
    </view>

    <view class="card-actions">
      <view class="action-btn load-btn" @click="handleLoad">
        <u-icon name="reload" size="18" color="#3b82f6" />
      </view>
      <view class="action-btn delete-btn" @click="handleDelete">
        <u-icon name="trash" size="18" color="#ef4444" />
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import type { HistoryRecord } from "../data/imageInputData";
import { formatTime } from "../data/imageInputData";

interface Props {
  record: HistoryRecord;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "load", record: HistoryRecord): void;
  (e: "delete", id: number): void;
}>();

const handleLoad = () => {
  emit("load", props.record);
};

const handleDelete = () => {
  emit("delete", props.record.id);
};
</script>

<style lang="scss" scoped>
.history-card {
  display: flex;
  align-items: center;
  gap: 20rpx;
  padding: 20rpx;
  background: #ffffff;
  border-radius: 16rpx;
  box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.06);
}

.card-thumbnail {
  position: relative;
  width: 120rpx;
  height: 120rpx;
  border-radius: 12rpx;
  overflow: hidden;
  flex-shrink: 0;
}

.status-badge {
  position: absolute;
  top: 8rpx;
  right: 8rpx;
  width: 32rpx;
  height: 32rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;

  &.failed {
    background: #ef4444;
  }
}

.card-content {
  flex: 1;
  min-width: 0;
}

.card-filename {
  display: block;
  font-size: 28rpx;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 8rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-time {
  display: block;
  font-size: 24rpx;
  color: #9ca3af;
  margin-bottom: 8rpx;
}

.card-algorithm {
  display: inline-flex;
  align-items: center;
  gap: 6rpx;
}

.algorithm-text {
  font-size: 24rpx;
  color: #3b82f6;
}

.card-actions {
  display: flex;
  flex-direction: column;
  gap: 12rpx;
}

.action-btn {
  width: 64rpx;
  height: 64rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12rpx;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.9);
  }

  &.load-btn {
    background: #eff6ff;
  }

  &.delete-btn {
    background: #fef2f2;
  }
}
</style>
