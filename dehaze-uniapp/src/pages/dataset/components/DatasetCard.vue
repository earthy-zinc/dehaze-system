<template>
  <view class="dataset-card" @click="handleClick">
    <view class="card-thumbnail">
      <up-image
        :src="dataset.thumbnail"
        mode="aspectFill"
        width="100%"
        height="100%"
        :lazy-load="true"
      />
    </view>
    <view class="card-content">
      <text class="card-title">{{ dataset.name }}</text>
      <text class="card-desc">{{ dataset.description || "暂无描述" }}</text>
      <view class="card-stats">
        <view class="stat-item">
          <u-icon name="photo" size="14" color="#14b8a6" />
          <text class="stat-text">{{ dataset.total_images }}</text>
        </view>
        <view class="stat-item">
          <u-icon name="clock" size="14" color="#9ca3af" />
          <text class="stat-text">{{ formattedDate }}</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { Dataset } from "../data/datasetData";
import { formatDate } from "../data/datasetData";

interface Props {
  dataset: Dataset;
}

interface Emits {
  (e: "click", dataset: Dataset): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const formattedDate = computed(() => formatDate(props.dataset.created_at));

const handleClick = () => {
  emit("click", props.dataset);
};
</script>

<style lang="scss" scoped>
.dataset-card {
  display: flex;
  background: #ffffff;
  border-radius: 24rpx;
  overflow: hidden;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  cursor: pointer;

  &:active {
    transform: scale(0.98);
    box-shadow: 0 2rpx 8rpx rgba(0, 0, 0, 0.12);
  }
}

.card-thumbnail {
  width: 240rpx;
  height: 240rpx;
  flex-shrink: 0;
  background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
}

.card-content {
  flex: 1;
  padding: 24rpx;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-width: 0;
}

.card-title {
  font-size: 32rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 12rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-desc {
  font-size: 26rpx;
  color: #6b7280;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-bottom: 16rpx;
}

.card-stats {
  display: flex;
  align-items: center;
  gap: 32rpx;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 8rpx;
}

.stat-text {
  font-size: 24rpx;
  color: #9ca3af;
}

/* PC端悬停效果 */
@media (min-width: 1024px) {
  .dataset-card:hover {
    transform: translateY(-4rpx);
    box-shadow: 0 8rpx 24rpx rgba(0, 0, 0, 0.12);
  }
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .card-thumbnail {
    width: 200rpx;
    height: 200rpx;
  }

  .card-content {
    padding: 20rpx;
  }

  .card-title {
    font-size: 28rpx;
  }

  .card-desc {
    font-size: 24rpx;
  }
}
</style>
