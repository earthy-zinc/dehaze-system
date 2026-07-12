<template>
  <scroll-view class="type-filter" scroll-x :show-scrollbar="false">
    <view class="filter-content">
      <view
        v-for="item in filterItems"
        :key="item.type"
        class="filter-btn"
        :class="{ active: activeFilter === item.type }"
        @click="handleChange(item.type)"
      >
        <text class="filter-text">{{ item.label }}</text>
        <text class="filter-count">{{ item.count }}</text>
      </view>
    </view>
  </scroll-view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import type { AnnotationFilter, AnnotationCounts } from "../data/datasetData";
import { ANNOTATION_FILTER_LABELS } from "../data/datasetData";

interface Props {
  activeFilter: AnnotationFilter;
  counts: AnnotationCounts;
}

interface Emits {
  (e: "change", filter: AnnotationFilter): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

const filterItems = computed(() => {
  const filters: AnnotationFilter[] = ["annotated", "unannotated"];
  return filters.map((type) => ({
    type,
    label: ANNOTATION_FILTER_LABELS[type],
    count: props.counts[type],
  }));
});

const handleChange = (type: AnnotationFilter) => {
  if (type !== props.activeFilter) {
    emit("change", type);
  }
};
</script>

<style lang="scss" scoped>
.type-filter {
  width: 100%;
  white-space: nowrap;
}

.filter-content {
  display: inline-flex;
  gap: 16rpx;
  padding: 8rpx 0;
}

.filter-btn {
  display: inline-flex;
  align-items: center;
  gap: 8rpx;
  padding: 16rpx 24rpx;
  border-radius: 40rpx;
  background: #ffffff;
  border: 2rpx solid #e5e7eb;
  color: #6b7280;
  font-size: 28rpx;
  font-weight: 500;
  transition: all 0.2s;
  flex-shrink: 0;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
    border-color: #14b8a6;
    color: #ffffff;
    box-shadow: 0 8rpx 24rpx rgba(20, 184, 166, 0.3);
  }
}

.filter-text {
  font-size: 28rpx;
}

.filter-count {
  display: inline-block;
  padding: 4rpx 12rpx;
  border-radius: 20rpx;
  background: rgba(255, 255, 255, 0.2);
  font-size: 24rpx;
}

.filter-btn:not(.active) .filter-count {
  background: #f3f4f6;
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .filter-btn {
    padding: 12rpx 20rpx;
  }

  .filter-text {
    font-size: 26rpx;
  }

  .filter-count {
    font-size: 22rpx;
  }
}
</style>
