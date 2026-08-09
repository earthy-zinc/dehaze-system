<template>
  <ImmersiveLayout title="放大镜对比">
    <view v-if="hasImages" class="main-content">
      <view
        class="magnifier-wrapper"
        @touchstart="handleTouchStart"
        @touchmove="handleTouchMove"
        @touchend="handleTouchEnd"
      >
        <image :src="resultUrl" class="base-image" mode="widthFix" lazy-load />

        <!-- 原图放大镜（双窗口模式左） -->
        <view
          v-if="showOriginLens"
          class="lens lens-origin"
          :style="lensStyle(originUrl, 'left')"
        >
          <view class="lens-image" :style="lensImageStyle(originUrl)" />
        </view>

        <!-- 处理后放大镜（双窗口模式右） -->
        <view
          v-if="showResultLens"
          class="lens lens-result"
          :style="lensStyle(resultUrl, 'right')"
        >
          <view class="lens-image" :style="lensImageStyle(resultUrl)" />
        </view>

        <!-- 放大倍数标签 -->
        <view class="zoom-badge">{{ zoom }}x</view>
      </view>

      <!-- 提示 -->
      <view class="magnifier-hint">
        <text>拖动移动放大镜 · 双指捏合调整倍数 · 点击切换模式</text>
      </view>

      <!-- 控制面板 -->
      <view class="control-panel">
        <!-- 显示模式 -->
        <view class="control-group">
          <text class="control-label">显示模式</text>
          <view class="control-options">
            <view
              v-for="opt in displayModes"
              :key="opt.value"
              class="control-option"
              :class="{ active: displayMode === opt.value }"
              @click="displayMode = opt.value"
            >
              <text>{{ opt.label }}</text>
            </view>
          </view>
        </view>

        <!-- 放大倍数 -->
        <view class="control-group">
          <text class="control-label">放大倍数</text>
          <view class="control-options">
            <view
              v-for="z in zoomOptions"
              :key="z"
              class="control-option"
              :class="{ active: zoom === z }"
              @click="zoom = z"
            >
              <text>{{ z }}x</text>
            </view>
          </view>
        </view>

        <!-- 放大镜大小 -->
        <view class="control-group">
          <text class="control-label">放大镜大小</text>
          <view class="control-options">
            <view
              v-for="sz in sizeOptions"
              :key="sz.value"
              class="control-option"
              :class="{ active: lensSize === sz.value }"
              @click="lensSize = sz.value"
            >
              <text>{{ sz.label }}</text>
            </view>
          </view>
        </view>

        <!-- 边框样式 -->
        <view class="control-group">
          <text class="control-label">边框样式</text>
          <view class="control-options">
            <view
              v-for="bs in borderStyles"
              :key="bs.value"
              class="control-option"
              :class="{ active: borderStyle === bs.value }"
              @click="borderStyle = bs.value"
            >
              <text>{{ bs.label }}</text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <CompareEmptyState v-else text="暂无对比数据" btn-color="#f59e0b" />

    <template #toolbar>
      <view class="toolbar-grid">
        <view
          v-for="m in modes"
          :key="m.key"
          class="toolbar-item"
          :class="{ active: m.key === 'magnifier' }"
          @click="switchPage(m.path)"
        >
          <SvgIcon :name="m.icon" size="20" color="#f59e0b" />
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
import { ref, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import ImmersiveLayout from "@/layout/ImmersiveLayout.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";
import { FavoriteAPI } from "dehaze-sdk-js";

const store = useProcessingStore();
const favorited = ref(false);
const favoriteLoading = ref(false);

const zoomOptions = [2, 3, 5] as const;
const sizeOptions = [
  { value: 100, label: "小" },
  { value: 150, label: "中" },
  { value: 200, label: "大" },
] as const;
const displayModes = [
  { value: "origin" as const, label: "原图" },
  { value: "result" as const, label: "处理后" },
  { value: "compare" as const, label: "对比" },
];
const borderStyles = [
  { value: "circle" as const, label: "圆形" },
  { value: "square" as const, label: "方形" },
  { value: "rounded" as const, label: "圆角" },
];

type DisplayMode = "origin" | "result" | "compare";
type BorderStyle = "circle" | "square" | "rounded";
type ZoomValue = 2 | 3 | 5;

const zoom = ref<ZoomValue>(2);
const lensSize = ref<number>(150);
const displayMode = ref<DisplayMode>("compare");
const borderStyle = ref<BorderStyle>("circle");
const posX = ref(150);
const posY = ref(200);
const active = ref(false);
const wrapperRect = ref<{
  left: number;
  top: number;
  width: number;
  height: number;
} | null>(null);

// 双指捏合
let lastPinchDist = 0;

const originUrl = computed(() => store.originUrl);
const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));
const resultId = computed(() => store.result?.logId);

const showOriginLens = computed(
  () => displayMode.value === "origin" || displayMode.value === "compare"
);
const showResultLens = computed(
  () => displayMode.value === "result" || displayMode.value === "compare"
);

const borderRadius = computed(() => {
  if (borderStyle.value === "circle") return "50%";
  if (borderStyle.value === "rounded") return "16rpx";
  return "0";
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

function lensStyle(_imgUrl: string, side: "left" | "right" = "left") {
  const half = lensSize.value / 2;
  let left = posX.value - half;
  if (displayMode.value === "compare") {
    left = side === "left" ? posX.value - lensSize.value - 8 : posX.value + 8;
  }
  return {
    width: `${lensSize.value}px`,
    height: `${lensSize.value}px`,
    left: `${left}px`,
    top: `${posY.value - half}px`,
    borderRadius: borderRadius.value,
    borderColor: side === "left" ? "#3b82f6" : "#34d399",
  };
}

function lensImageStyle(imgUrl: string) {
  const rect = wrapperRect.value;
  const base: Record<string, string> = {
    width: "100%",
    height: "100%",
    backgroundImage: `url(${imgUrl})`,
    backgroundRepeat: "no-repeat",
    borderRadius: borderRadius.value,
  };
  if (!rect) {
    return { ...base, backgroundSize: "cover", backgroundPosition: "center" };
  }
  const relX = posX.value - rect.left;
  const relY = posY.value - rect.top;
  const bgW = rect.width * zoom.value;
  const bgH = rect.height * zoom.value;
  const bgX = -(relX * zoom.value - lensSize.value / 2);
  const bgY = -(relY * zoom.value - lensSize.value / 2);
  return {
    ...base,
    backgroundSize: `${bgW}px ${bgH}px`,
    backgroundPosition: `${bgX}px ${bgY}px`,
  };
}

function updateRect(e: any) {
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect?.();
  if (rect) {
    wrapperRect.value = {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
  }
}

function handleTouchStart(e: any) {
  updateRect(e);
  active.value = true;
  const touches = e.touches || [];
  if (touches.length === 2) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    lastPinchDist = Math.sqrt(dx * dx + dy * dy);
  }
}

function handleTouchMove(e: any) {
  const touches = e.touches || [];
  if (touches.length === 2) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (lastPinchDist > 0 && Math.abs(dist - lastPinchDist) > 20) {
      const curIdx = zoomOptions.indexOf(zoom.value);
      if (dist > lastPinchDist && curIdx < zoomOptions.length - 1) {
        zoom.value = zoomOptions[curIdx + 1] as ZoomValue;
      } else if (dist < lastPinchDist && curIdx > 0) {
        zoom.value = zoomOptions[curIdx - 1] as ZoomValue;
      }
      lastPinchDist = dist;
    }
    return;
  }
  active.value = true;
  updateRect(e);
  const touch = touches[0] || e.changedTouches?.[0];
  if (!touch) return;
  posX.value = touch.clientX;
  posY.value = touch.clientY;
}

function handleTouchEnd() {
  lastPinchDist = 0;
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
</script>

<style lang="scss" scoped>
.main-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.magnifier-wrapper {
  position: relative;
  flex: 1;
  overflow: hidden;
  background: #000;
}

.base-image {
  width: 100%;
  display: block;
}

.lens {
  position: fixed;
  border: 4rpx solid;
  box-shadow:
    0 4rpx 24rpx rgba(0, 0, 0, 0.5),
    0 0 0 9999rpx rgba(0, 0, 0, 0.3);
  overflow: hidden;
  pointer-events: none;
  z-index: 999;

  &.lens-origin {
    border-color: #3b82f6;
  }
  &.lens-result {
    border-color: #34d399;
  }
}

.lens-image {
  width: 100%;
  height: 100%;
}

.zoom-badge {
  position: absolute;
  top: 16rpx;
  left: 16rpx;
  padding: 4rpx 16rpx;
  border-radius: 32rpx;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  font-size: 22rpx;
  font-weight: 600;
}

.magnifier-hint {
  padding: 16rpx 32rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.3);
  text-align: center;
  background: rgba(255, 255, 255, 0.03);
}

.control-panel {
  max-height: 480rpx;
  padding: 24rpx 32rpx;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.03);
}

.control-group {
  margin-bottom: 24rpx;

  &:last-child {
    margin-bottom: 0;
  }
}

.control-label {
  display: block;
  margin-bottom: 16rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
}

.control-options {
  display: flex;
  flex-wrap: wrap;
  gap: 16rpx;

  .control-option {
    padding: 12rpx 32rpx;
    font-size: 26rpx;
    color: rgba(255, 255, 255, 0.5);
    background: rgba(255, 255, 255, 0.08);
    border: 2rpx solid rgba(255, 255, 255, 0.1);
    border-radius: 32rpx;

    &.active {
      color: #f59e0b;
      background: rgba(245, 158, 11, 0.15);
      border-color: #f59e0b;
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
    background: rgba(245, 158, 11, 0.15);
    border-radius: 12rpx;
  }

  &.active {
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.12);
    border-radius: 12rpx;
  }
}
</style>
