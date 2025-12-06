<template>
  <view class="image-card" :class="{ waterfall: isWaterfall }" @click="handleClick">
    <view class="image-wrapper" :style="wrapperStyle">
      <up-image
        :src="image.image_url"
        :mode="isWaterfall ? 'widthFix' : 'aspectFill'"
        width="100%"
        :height="isWaterfall ? 'auto' : '100%'"
        :lazy-load="true"
        :fade="true"
        @load="onImageLoad"
      />
      <view class="type-badge" :class="image.image_type">
        {{ typeLabel }}
      </view>
    </view>
    <view class="image-info">
      <text class="image-filename">{{ image.filename }}</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed, ref } from "vue";
import type { DatasetImage } from "../data/datasetData";
import { IMAGE_TYPE_LABELS } from "../data/datasetData";

interface Props {
  image: DatasetImage;
  /** 是否为瀑布流模式 */
  isWaterfall?: boolean;
}

interface Emits {
  (e: "click", image: DatasetImage): void;
}

const props = withDefaults(defineProps<Props>(), {
  isWaterfall: false,
});
const emit = defineEmits<Emits>();

const imageLoaded = ref(false);

const typeLabel = computed(() => IMAGE_TYPE_LABELS[props.image.image_type]);

const wrapperStyle = computed(() => {
  if (props.isWaterfall) {
    // 瀑布流模式：根据图片宽高比计算高度
    const ratio = props.image.height / props.image.width;
    return {};
  }
  // 网格模式：正方形
  return {
    paddingBottom: "100%",
  };
});

const onImageLoad = () => {
  imageLoaded.value = true;
};

const handleClick = () => {
  emit("click", props.image);
};
</script>

<style lang="scss" scoped>
.image-card {
  background: #ffffff;
  border-radius: 16rpx;
  overflow: hidden;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  cursor: pointer;

  &:active {
    transform: scale(0.95);
  }

  &.waterfall {
    break-inside: avoid;
    margin-bottom: 24rpx;
  }
}

.image-wrapper {
  position: relative;
  width: 100%;
  background: #f3f4f6;

  /* 网格模式：正方形容器 */
  .image-card:not(.waterfall) & {
    position: relative;
    height: 0;
    padding-bottom: 100%;

    :deep(.u-image) {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }
  }
}

.type-badge {
  position: absolute;
  top: 12rpx;
  right: 12rpx;
  padding: 6rpx 16rpx;
  border-radius: 24rpx;
  font-size: 22rpx;
  font-weight: 500;
  backdrop-filter: blur(8px);
  color: #ffffff;

  &.foggy {
    background: rgba(107, 114, 128, 0.9);
  }

  &.clear {
    background: rgba(59, 130, 246, 0.9);
  }

  &.annotated {
    background: rgba(16, 185, 129, 0.9);
  }
}

.image-info {
  padding: 16rpx;
}

.image-filename {
  display: block;
  font-size: 24rpx;
  color: #4b5563;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* PC端悬停效果 */
@media (min-width: 1024px) {
  .image-card:hover {
    transform: translateY(-8rpx);
    box-shadow: 0 12rpx 32rpx rgba(0, 0, 0, 0.12);
  }
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .type-badge {
    padding: 4rpx 12rpx;
    font-size: 20rpx;
  }

  .image-info {
    padding: 12rpx;
  }

  .image-filename {
    font-size: 22rpx;
  }
}
</style>
