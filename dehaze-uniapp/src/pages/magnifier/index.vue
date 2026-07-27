<template>
  <PageLayout class="page">
    <view class="main-content">
      <PageHeaderCard
        icon="search"
        icon-color="#f59e0b"
        icon-bg="#fef3c7"
        title="放大镜对比"
        subtitle="触控移动查看局部细节"
        variant="dark"
      />

      <view v-if="hasImages" class="content-area">
        <!-- 原图背景 + 放大镜 -->
        <view
          class="magnifier-wrapper"
          @touchmove="handleMove"
          @touchstart="handleMove"
        >
          <image :src="originUrl" class="base-image" mode="widthFix" />
          <view v-if="active" class="lens" :style="lensStyle">
            <view class="lens-image" :style="lensImageStyle" />
          </view>
        </view>

        <!-- 模式切换 -->
        <view class="mode-row">
          <view
            v-for="m in modes"
            :key="m.key"
            class="mode-btn"
            :class="{ active: currentMode === m.key }"
            @click="currentMode = m.key"
          >
            {{ m.label }}
          </view>
        </view>

        <!-- 镜片大小调节 -->
        <view class="size-control">
          <text class="control-label">镜片大小</text>
          <slider
            :value="lensSize"
            :min="60"
            :max="160"
            :step="10"
            active-color="#f59e0b"
            block-size="20"
            @change="(e: any) => (lensSize = e.detail.value)"
          />
        </view>

        <!-- 导航 -->
        <view class="nav-row">
          <view class="nav-item" @click="switchPage('/pages/overlay/index')">
            <u-icon name="photo" size="20" color="#f59e0b" /><text
              >重叠对比</text
            >
          </view>
          <view
            class="nav-item"
            @click="switchPage('/pages/side-by-side/index')"
          >
            <u-icon name="grid" size="20" color="#f59e0b" /><text
              >并排对比</text
            >
          </view>
        </view>
      </view>

      <CompareEmptyState v-else text="暂无对比数据" btn-color="#f59e0b" />
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";

/** 放大倍数 */
const ZOOM = 2.5;

const store = useProcessingStore();
const active = ref(false);
const posX = ref(150);
const posY = ref(200);
const lensSize = ref(100);
const currentMode = ref<"result" | "origin">("result");
/** 底图层（magnifier-wrapper）的视口矩形，用于换算触摸点相对位置 */
const wrapperRect = ref<{
  left: number;
  top: number;
  width: number;
  height: number;
} | null>(null);

const originUrl = computed(() => store.originUrl);
const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));

const lensImg = computed(() =>
  currentMode.value === "result" ? resultUrl.value : originUrl.value
);

const modes = [
  { key: "result" as const, label: "处理结果" },
  { key: "origin" as const, label: "原图" },
];

const lensStyle = computed(() => {
  const half = lensSize.value / 2;
  return {
    width: `${lensSize.value}px`,
    height: `${lensSize.value}px`,
    left: `${posX.value - half}px`,
    top: `${posY.value - half}px`,
    borderRadius: "50%",
  };
});

/**
 * 放大镜内部图像样式：以触摸点为中心裁剪放大底图。
 * 通过 background-size 放大底图，再用 background-position 把触摸点平移到镜片中心。
 */
const lensImageStyle = computed(() => {
  const rect = wrapperRect.value;
  const base: Record<string, string> = {
    width: "100%",
    height: "100%",
    backgroundImage: `url(${lensImg.value})`,
    backgroundRepeat: "no-repeat",
    borderRadius: "50%",
  };
  if (!rect) {
    return { ...base, backgroundSize: "cover", backgroundPosition: "center" };
  }
  // 触摸点相对于底图的坐标
  const relX = posX.value - rect.left;
  const relY = posY.value - rect.top;
  // 放大后的背景尺寸
  const bgW = rect.width * ZOOM;
  const bgH = rect.height * ZOOM;
  // 让触摸点对齐镜片中心：偏移 = -(触摸点 * 放大倍数 - 镜片半径)
  const bgX = -(relX * ZOOM - lensSize.value / 2);
  const bgY = -(relY * ZOOM - lensSize.value / 2);
  return {
    ...base,
    backgroundSize: `${bgW}px ${bgH}px`,
    backgroundPosition: `${bgX}px ${bgY}px`,
  };
});

function handleMove(e: any) {
  active.value = true;
  const touch = e.touches?.[0] || e.changedTouches?.[0];
  if (!touch) return;
  // 实时获取底图区域，兼容滚动/缩放
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect?.();
  if (rect) {
    wrapperRect.value = {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
  }
  posX.value = touch.clientX;
  posY.value = touch.clientY;
}

function switchPage(url: string) {
  uni.redirectTo({ url });
}

onMounted(() => {
  if (!hasImages.value)
    uni.showToast({ title: "请先完成去雾处理", icon: "none" });
});
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: #000;
}
.main-content {
  padding: 24rpx;
}

.magnifier-wrapper {
  position: relative;
  width: 100%;
}
.base-image {
  width: 100%;
  display: block;
  border-radius: 16rpx;
}

.lens {
  position: fixed;
  border: 4rpx solid #f59e0b;
  box-shadow:
    0 4rpx 24rpx rgba(0, 0, 0, 0.5),
    0 0 0 9999rpx rgba(0, 0, 0, 0.3);
  overflow: hidden;
  pointer-events: none;
  z-index: 999;
}
.lens-image {
  width: 100%;
  height: 100%;
}

.mode-row {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
.mode-btn {
  flex: 1;
  text-align: center;
  padding: 20rpx;
  border-radius: 16rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.08);
  &.active {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
    font-weight: 600;
  }
}

.size-control {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 20rpx;
  padding: 24rpx 28rpx;
  margin-top: 24rpx;
}
.control-label {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 8rpx;
  display: block;
}

.nav-row {
  display: flex;
  gap: 20rpx;
  margin-top: 32rpx;
}
.nav-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12rpx;
  padding: 28rpx;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
  &:active {
    background: rgba(245, 158, 11, 0.15);
  }
}
</style>
