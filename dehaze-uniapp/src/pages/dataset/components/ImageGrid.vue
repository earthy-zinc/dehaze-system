<template>
  <view class="image-grid-container">
    <!-- 展示模式切换 -->
    <view class="display-mode-switch">
      <view
        class="mode-btn"
        :class="{ active: displayMode === 'grid' }"
        @click="handleModeChange('grid')"
      >
        <SvgIcon
          name="grid"
          size="18"
          :color="displayMode === 'grid' ? '#ffffff' : '#6b7280'"
        />
      </view>
      <view
        class="mode-btn"
        :class="{ active: displayMode === 'waterfall' }"
        @click="handleModeChange('waterfall')"
      >
        <SvgIcon
          name="list"
          size="18"
          :color="displayMode === 'waterfall' ? '#ffffff' : '#6b7280'"
        />
      </view>
    </view>

    <!-- 网格模式 -->
    <view v-if="displayMode === 'grid'" class="grid-view">
      <ImageCard
        v-for="image in images"
        :key="image.id"
        :image="image"
        :is-waterfall="false"
        @click="handleImageClick"
      />
    </view>

    <!-- 瀑布流模式 -->
    <view v-else class="waterfall-view">
      <view
        class="waterfall-column"
        v-for="(column, index) in waterfallColumns"
        :key="index"
      >
        <ImageCard
          v-for="image in column"
          :key="image.id"
          :image="image"
          :is-waterfall="true"
          @click="handleImageClick"
        />
      </view>
    </view>

    <!-- 加载状态 -->
    <view v-if="loading" class="loading-container">
      <view class="loading-spinner" style="border-top-color: #14b8a6" />
      <text class="loading-text">加载中...</text>
    </view>

    <!-- 空状态 -->
    <view v-else-if="images.length === 0" class="empty-container">
      <view class="empty-tip">暂无图片</view>
    </view>

    <!-- 加载更多触发器 -->
    <view
      v-if="hasMore && !loading && images.length > 0"
      class="load-more-trigger"
      id="loadMoreTrigger"
    >
      <text class="load-more-text">上滑加载更多</text>
    </view>

    <!-- 全部加载完成 -->
    <view v-if="!hasMore && images.length > 0" class="load-complete">
      <text class="load-complete-text"
        >已加载全部 {{ images.length }} 张图片</text
      >
    </view>
  </view>
</template>

<script lang="ts" setup>
import {
  ref,
  computed,
  watch,
  onMounted,
  onUnmounted,
  getCurrentInstance,
} from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import ImageCard from "./ImageCard.vue";
import type { DatasetImageItem, DisplayMode } from "../data/datasetData";

interface Props {
  images: DatasetImageItem[];
  loading?: boolean;
  hasMore?: boolean;
  /** 初始展示模式 */
  initialMode?: DisplayMode;
}

interface Emits {
  (e: "image-click", image: DatasetImageItem): void;
  (e: "load-more"): void;
  (e: "mode-change", mode: DisplayMode): void;
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  hasMore: true,
  initialMode: "grid",
});

const emit = defineEmits<Emits>();

const instance = getCurrentInstance();
const displayMode = ref<DisplayMode>(props.initialMode);

// 瀑布流列数（响应式）
const columnCount = ref(2);

// 计算瀑布流列数据
const waterfallColumns = computed(() => {
  const columns: DatasetImageItem[][] = Array.from(
    { length: columnCount.value },
    () => []
  );
  const columnHeights = Array(columnCount.value).fill(0);

  props.images.forEach((image) => {
    // 找到最短的列
    const minHeightIndex = columnHeights.indexOf(Math.min(...columnHeights));
    const targetColumn = columns[minHeightIndex];
    if (!targetColumn) return;
    targetColumn.push(image);
    // 累加高度（使用图片宽高比估算）
    columnHeights[minHeightIndex] += image.height / image.width;
  });

  return columns;
});

// 切换展示模式
const handleModeChange = (mode: DisplayMode) => {
  if (mode !== displayMode.value) {
    displayMode.value = mode;
    emit("mode-change", mode);
  }
};

// 图片点击
const handleImageClick = (image: DatasetImageItem) => {
  emit("image-click", image);
};

// 更新列数
const updateColumnCount = () => {
  try {
    const sysInfo = uni.getSystemInfoSync();
    const width = sysInfo.windowWidth || 375;

    if (width >= 1024) {
      columnCount.value = 4;
    } else if (width >= 768) {
      columnCount.value = 3;
    } else {
      columnCount.value = 2;
    }
  } catch (error) {
    columnCount.value = 2;
  }
};

// 无限滚动观察器
let observer: UniApp.IntersectionObserver | null = null;

const setupInfiniteScroll = () => {
  if (!instance) return;

  observer = uni.createIntersectionObserver(instance.proxy, {
    thresholds: [0.1],
  });

  observer
    .relativeToViewport({ bottom: 100 })
    .observe("#loadMoreTrigger", (res) => {
      if (res.intersectionRatio > 0 && props.hasMore && !props.loading) {
        emit("load-more");
      }
    });
};

const destroyObserver = () => {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
};

onMounted(() => {
  updateColumnCount();

  // 延迟设置无限滚动，确保DOM已渲染
  setTimeout(() => {
    setupInfiniteScroll();
  }, 500);

  // #ifdef H5
  window.addEventListener("resize", updateColumnCount);
  // #endif
});

onUnmounted(() => {
  destroyObserver();

  // #ifdef H5
  window.removeEventListener("resize", updateColumnCount);
  // #endif
});

// 监听hasMore变化，重新设置观察器
watch(
  () => props.hasMore,
  (newVal) => {
    if (newVal) {
      setTimeout(() => {
        destroyObserver();
        setupInfiniteScroll();
      }, 100);
    }
  }
);
</script>

<style lang="scss" scoped>
.image-grid-container {
  width: 100%;
}

/* 展示模式切换 */
.display-mode-switch {
  display: flex;
  justify-content: flex-end;
  gap: 16rpx;
  margin-bottom: 24rpx;
}

.mode-btn {
  width: 64rpx;
  height: 64rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12rpx;
  background: $color-bg-secondary;
  transition: all 0.2s;

  &:active {
    transform: scale(0.95);
  }

  &.active {
    background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
  }
}

/* 网格视图 */
.grid-view {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24rpx;
}

/* 瀑布流视图 */
.waterfall-view {
  display: flex;
  gap: 24rpx;
}

.waterfall-column {
  flex: 1;
  display: flex;
  flex-direction: column;
}

/* 加载状态 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 64rpx 0;
}

.loading-text {
  margin-top: 16rpx;
  font-size: 28rpx;
  color: $color-text-placeholder;
}

/* 空状态 */
.empty-container {
  padding: 80rpx 0;
}

/* 加载更多触发器 */
.load-more-trigger {
  padding: 32rpx 0;
  text-align: center;
}

.load-more-text {
  font-size: 26rpx;
  color: $color-text-placeholder;
}

/* 加载完成 */
.load-complete {
  padding: 32rpx 0;
  text-align: center;
}

.load-complete-text {
  font-size: 26rpx;
  color: $color-text-placeholder;
}

/* 响应式网格 */
@media (min-width: 768px) {
  .grid-view {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 1024px) {
  .grid-view {
    grid-template-columns: repeat(4, 1fr);
  }
}

@media (min-width: 1280px) {
  .grid-view {
    grid-template-columns: repeat(5, 1fr);
  }
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .grid-view {
    gap: 16rpx;
  }

  .waterfall-view {
    gap: 16rpx;
  }

  .mode-btn {
    width: 56rpx;
    height: 56rpx;
  }
}
</style>
