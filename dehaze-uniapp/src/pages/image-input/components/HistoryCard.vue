<template>
  <view class="history-card" @click="handleLoad">
    <view class="card-thumbnail">
      <up-image
        v-if="thumbnailUrl"
        :src="thumbnailUrl"
        mode="aspectFill"
        width="120rpx"
        height="120rpx"
        :lazy-load="true"
        :fade="true"
      />
      <view v-else class="thumbnail-placeholder">
        <SvgIcon name="photo" size="28" color="#d1d5db" />
      </view>
      <view v-if="record.status === 1" class="result-badge">
        <text>已处理</text>
      </view>
    </view>

    <view class="card-content">
      <text class="card-name">{{ displayName }}</text>
      <text class="card-time">{{ formatTimestamp(record.createTime) }}</text>
      <text v-if="record.algorithmName" class="card-algo">{{ record.algorithmName }}</text>
    </view>

    <view class="card-arrow">
      <SvgIcon name="arrow-right" size="16" color="#d1d5db" />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { InputHistoryVO } from "dehaze-sdk-js";
import SvgIcon from "@/components/SvgIcon/index.vue";
import { formatTimestamp } from "../services/historyService";

interface Props {
  record: InputHistoryVO;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "load", record: InputHistoryVO): void;
}>();

const thumbnailUrl = computed(() => {
  return props.record.originalThumbnailUrl || props.record.originalImageUrl || "";
});

const displayName = computed(() => {
  const url = props.record.originalImageUrl || "";
  const path = url.split("?")[0];
  const segments = path.split("/");
  return segments[segments.length - 1] || "未命名图片";
});

const handleLoad = () => {
  emit("load", props.record);
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

  &:active {
    background: #f9fafb;
  }
}

.card-thumbnail {
  position: relative;
  width: 120rpx;
  height: 120rpx;
  border-radius: 12rpx;
  overflow: hidden;
  flex-shrink: 0;
  background: #f3f4f6;
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
  bottom: 4rpx;
  left: 4rpx;
  padding: 4rpx 10rpx;
  background: #10b981;
  border-radius: 6rpx;

  text {
    font-size: 18rpx;
    color: #ffffff;
  }
}

.card-content {
  flex: 1;
  min-width: 0;
}

.card-name {
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
  color: #6b7280;
  margin-bottom: 4rpx;
}

.card-algo {
  font-size: 22rpx;
  color: #3b82f6;
}

.card-arrow {
  flex-shrink: 0;
}
</style>
