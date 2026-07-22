<template>
  <view class="history-card" @click="handleLoad">
    <view class="card-thumbnail">
      <up-image
        v-if="record.predUrl"
        :src="record.predUrl"
        mode="aspectFill"
        width="120rpx"
        height="120rpx"
        :lazy-load="true"
        :fade="true"
      />
    </view>

    <view class="card-content">
      <text class="card-algo">{{ record.algorithmName || "未知算法" }}</text>
      <text class="card-time"
        >耗时 {{ record.time != null ? record.time + "s" : "-" }}</text
      >
      <text class="card-date">{{ formatTime(record.createTime || "") }}</text>
    </view>

    <view class="card-arrow">
      <u-icon name="arrow-right" size="16" color="#d1d5db" />
    </view>
  </view>
</template>

<script lang="ts" setup>
import type { PredLogVO } from "@/api/prediction";
import { formatTime } from "../data/imageInputData";

interface Props {
  record: PredLogVO;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "load", record: PredLogVO): void;
}>();

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

.card-content {
  flex: 1;
  min-width: 0;
}

.card-algo {
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

.card-date {
  font-size: 22rpx;
  color: #9ca3af;
}

.card-arrow {
  flex-shrink: 0;
}
</style>
