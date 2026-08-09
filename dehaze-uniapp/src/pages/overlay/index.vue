<template>
  <ImmersiveLayout title="重叠对比">
    <view v-if="hasImages" class="main-content">
      <view
        class="image-container"
        @touchstart="handleTouchStart"
        @touchmove="handleTouchMove"
        @touchend="handleTouchEnd"
      >
        <!-- 底层：处理后图片 -->
        <image :src="resultUrl" class="base-image" mode="widthFix" lazy-load />
        <!-- 上层：原图，通过 clip-path 控制显示区域 -->
        <view class="overlay-image-wrapper" :style="{ clipPath: clipStyle }">
          <image
            :src="originUrl"
            class="overlay-image"
            mode="widthFix"
            lazy-load
          />
        </view>
        <!-- 滑动分隔线 -->
        <view class="slider-divider" :style="dividerStyle">
          <view class="slider-line" />
          <view class="slider-handle">
            <text class="handle-icon">⟷</text>
          </view>
        </view>
        <!-- 标签 -->
        <view class="image-labels">
          <view
            class="label-tag label-original"
            :style="{ opacity: sliderPos > 10 ? 1 : 0 }"
          >
            <text>原图</text>
          </view>
          <view
            class="label-tag label-result"
            :style="{ opacity: sliderPos < 90 ? 1 : 0 }"
          >
            <text>处理后</text>
          </view>
        </view>
      </view>

      <view class="overlay-hint">
        <text>拖动分隔线对比效果</text>
      </view>

      <!-- 控制按钮 -->
      <view class="control-row">
        <view
          class="ctrl-btn"
          :class="{ active: isVertical }"
          @click="toggleDirection"
        >
          <text>{{ isVertical ? "垂直" : "水平" }}</text>
        </view>
        <view
          class="ctrl-btn"
          :class="{ active: snapToCenter }"
          @click="snapToCenter = !snapToCenter"
        >
          <text>吸附{{ snapToCenter ? "开" : "关" }}</text>
        </view>
        <view
          class="ctrl-btn"
          :class="{ active: isLocked }"
          @click="isLocked = !isLocked"
        >
          <text>{{ isLocked ? "已锁定" : "锁定" }}</text>
        </view>
        <view class="ctrl-btn" @click="handleAutoPlay">
          <text>{{ playing ? "停止" : "动画" }}</text>
        </view>
      </view>

      <!-- 算法信息 -->
      <view v-if="algorithm" class="info-card">
        <text class="card-title">算法信息</text>
        <view class="info-row">
          <text class="info-label">算法名称</text>
          <text class="info-value">{{ algorithm.name }}</text>
        </view>
        <view v-if="result?.time !== undefined" class="info-row">
          <text class="info-label">处理耗时</text>
          <text class="info-value">{{ result.time }}s</text>
        </view>
      </view>
    </view>

    <CompareEmptyState v-else text="暂无对比数据" btn-color="#8b5cf6" />

    <template #toolbar>
      <view class="toolbar-grid">
        <view
          v-for="m in modes"
          :key="m.key"
          class="toolbar-item"
          :class="{ active: m.key === 'overlay' }"
          @click="switchPage(m.path)"
        >
          <SvgIcon :name="m.icon" size="20" color="#8b5cf6" />
          <text>{{ m.label }}</text>
        </view>
      </view>
      <view class="toolbar-actions">
        <view class="action-item" @click="handleSave">
          <SvgIcon name="download" size="18" color="rgba(255,255,255,0.7)" />
          <text>保存</text>
        </view>
        <view class="action-item" @click="handleReprocess">
          <SvgIcon name="refresh" size="18" color="rgba(255,255,255,0.7)" />
          <text>重新处理</text>
        </view>
        <view class="action-item" @click="handleChangeAlgorithm">
          <SvgIcon name="swap" size="18" color="rgba(255,255,255,0.7)" />
          <text>换算法</text>
        </view>
        <view class="action-item" @click="handleFavorite">
          <SvgIcon
            :name="favorited ? 'star-fill' : 'star'"
            size="18"
            :color="favorited ? '#f59e0b' : 'rgba(255,255,255,0.7)'"
          />
          <text :style="{ color: favorited ? '#f59e0b' : '' }">{{
            favorited ? "已收藏" : "收藏"
          }}</text>
        </view>
      </view>
    </template>
  </ImmersiveLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import ImmersiveLayout from "@/layout/ImmersiveLayout.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";
import { FavoriteAPI } from "dehaze-sdk-js";

const store = useProcessingStore();
const sliderPos = ref(50);
const favorited = ref(false);
const favoriteLoading = ref(false);
const isVertical = ref(true);
const snapToCenter = ref(true);
const isLocked = ref(false);
const playing = ref(false);
let playTimer: number | null = null;
let playDirection = 1;

const originUrl = computed(() => store.originUrl);
const resultUrl = computed(() => store.result?.resultUrl || "");
const algorithm = computed(() => store.selectedAlgorithm);
const result = computed(() => store.result);
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));
const resultId = computed(() => store.result?.logId);

const clipStyle = computed(() => {
  if (isVertical.value) {
    return `inset(0 ${100 - sliderPos.value}% 0 0)`;
  }
  return `inset(${100 - sliderPos.value}% 0 0 0)`;
});

const dividerStyle = computed(() => {
  if (isVertical.value) {
    return {
      left: `${sliderPos.value}%`,
      top: "0",
      bottom: "0",
      width: "4rpx",
      height: "auto",
    };
  }
  return {
    top: `${sliderPos.value}%`,
    left: "0",
    right: "0",
    height: "4rpx",
    width: "auto",
  };
});

const modes = [
  {
    key: "side-by-side",
    label: "并排",
    path: "/pages/side-by-side/index",
    icon: "grid",
  },
  {
    key: "overlay",
    label: "重叠",
    path: "/pages/overlay/index",
    icon: "photo",
  },
  {
    key: "magnifier",
    label: "放大镜",
    path: "/pages/magnifier/index",
    icon: "search",
  },
  {
    key: "filter",
    label: "滤镜",
    path: "/pages/filter/index",
    icon: "setting",
  },
  {
    key: "metrics",
    label: "指标",
    path: "/pages/metrics/index",
    icon: "integral",
  },
];

function getTouchPos(e: TouchEvent): number {
  const touch = e.touches[0] || e.changedTouches[0];
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  if (!rect || !touch) return sliderPos.value;
  if (isVertical.value) {
    return ((touch.clientX - rect.left) / rect.width) * 100;
  }
  return ((touch.clientY - rect.top) / rect.height) * 100;
}

function handleTouchStart(e: TouchEvent) {
  if (isLocked.value) return;
  sliderPos.value = Math.max(5, Math.min(95, getTouchPos(e)));
}

function handleTouchMove(e: TouchEvent) {
  if (isLocked.value) return;
  sliderPos.value = Math.max(5, Math.min(95, getTouchPos(e)));
}

function handleTouchEnd() {
  if (isLocked.value) return;
  if (snapToCenter.value && sliderPos.value > 40 && sliderPos.value < 60) {
    sliderPos.value = 50;
  }
}

function toggleDirection() {
  isVertical.value = !isVertical.value;
  sliderPos.value = 50;
}

function handleAutoPlay() {
  if (playing.value) {
    playing.value = false;
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
    }
    return;
  }
  playing.value = true;
  playTimer = setInterval(() => {
    sliderPos.value += playDirection * 1;
    if (sliderPos.value >= 95 || sliderPos.value <= 5) {
      playDirection *= -1;
    }
  }, 30) as unknown as number;
}

function switchPage(url: string) {
  uni.redirectTo({ url });
}

function handleSave() {
  if (!resultUrl.value) {
    uni.showToast({ title: "无结果图片可保存", icon: "none" });
    return;
  }
  uni.downloadFile({
    url: resultUrl.value,
    success(res) {
      if (res.statusCode === 200) {
        uni.saveImageToPhotosAlbum({
          filePath: res.tempFilePath,
          success: () =>
            uni.showToast({ title: "已保存到相册", icon: "success" }),
          fail: () => uni.showToast({ title: "保存失败", icon: "none" }),
        });
      }
    },
  });
}

function handleReprocess() {
  uni.redirectTo({ url: "/pages/processing/index" });
}

function handleChangeAlgorithm() {
  uni.redirectTo({ url: "/pages/algorithm-select/index" });
}

async function handleFavorite() {
  if (!resultId.value) {
    uni.showToast({ title: "暂不支持收藏", icon: "none" });
    return;
  }
  if (favoriteLoading.value) return;
  favoriteLoading.value = true;
  try {
    if (favorited.value) {
      await FavoriteAPI.deleteByIds([resultId.value]);
      favorited.value = false;
      uni.showToast({ title: "已取消收藏", icon: "success" });
    } else {
      await FavoriteAPI.add({ targetType: "result", targetId: resultId.value });
      favorited.value = true;
      uni.showToast({ title: "已收藏", icon: "success" });
    }
  } catch {
    uni.showToast({ title: "操作失败", icon: "none" });
  } finally {
    favoriteLoading.value = false;
  }
}

onMounted(() => {
  if (!hasImages.value)
    uni.showToast({ title: "请先完成去雾处理", icon: "none" });
  if (resultId.value) {
    FavoriteAPI.getStatus("result", resultId.value)
      .then((res) => {
        favorited.value = res.favorited;
      })
      .catch(() => {});
  }
});

onUnmounted(() => {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
});
</script>

<style lang="scss" scoped>
.main-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  padding: 32rpx;
  overflow-y: auto;
}

.image-container {
  position: relative;
  width: 100%;
  overflow: hidden;
  touch-action: none;
  background: #000;
  border-radius: 24rpx;

  .base-image {
    display: block;
    width: 100%;
  }

  .overlay-image-wrapper {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;

    .overlay-image {
      display: block;
      width: 100%;
    }
  }

  .slider-divider {
    position: absolute;
    z-index: 10;
    background: #fff;
    box-shadow: 0 0 16rpx rgba(0, 0, 0, 0.3);

    .slider-line {
      width: 100%;
      height: 100%;
    }

    .slider-handle {
      position: absolute;
      top: 50%;
      left: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 72rpx;
      height: 72rpx;
      background: #fff;
      border-radius: 50%;
      box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.2);
      transform: translate(-50%, -50%);

      .handle-icon {
        font-size: 20rpx;
        color: #8b5cf6;
      }
    }
  }

  .image-labels {
    position: absolute;
    top: 16rpx;
    left: 16rpx;
    right: 16rpx;
    display: flex;
    justify-content: space-between;
    pointer-events: none;

    .label-tag {
      padding: 6rpx 16rpx;
      font-size: 22rpx;
      border-radius: 8rpx;
      transition: opacity 0.2s;

      &.label-original {
        color: #fff;
        background: rgba(251, 191, 36, 0.9);
      }

      &.label-result {
        color: #fff;
        background: rgba(139, 92, 246, 0.9);
      }
    }
  }
}

.overlay-hint {
  margin-top: 32rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.3);
}

.control-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12rpx;
  margin-top: 24rpx;
  width: 100%;
}

.ctrl-btn {
  text-align: center;
  padding: 16rpx 8rpx;
  border-radius: 16rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.08);

  &.active {
    background: rgba(139, 92, 246, 0.2);
    color: #8b5cf6;
    font-weight: 600;
  }
}

.info-card {
  width: 100%;
  padding: 28rpx 32rpx;
  margin-top: 24rpx;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 24rpx;

  .card-title {
    font-size: 28rpx;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    margin-bottom: 16rpx;
    display: block;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 12rpx 0;
    border-bottom: 1rpx solid rgba(255, 255, 255, 0.05);

    &:last-child {
      border-bottom: none;
    }

    .info-label {
      font-size: 26rpx;
      color: rgba(255, 255, 255, 0.5);
    }

    .info-value {
      font-size: 26rpx;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.8);
    }
  }
}

.toolbar-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4rpx;
  padding: 16rpx 16rpx 8rpx;
}

.toolbar-actions {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 4rpx;
  padding: 0 16rpx 16rpx;
}

.toolbar-item,
.action-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8rpx;
  padding: 20rpx 8rpx;
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.5);

  &:active {
    background: rgba(139, 92, 246, 0.15);
    border-radius: 12rpx;
  }

  &.active {
    color: #8b5cf6;
    background: rgba(139, 92, 246, 0.12);
    border-radius: 12rpx;
  }
}
</style>
