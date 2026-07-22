<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="search" size="28" color="#f59e0b" />
        </view>
        <view class="header-text">
          <text class="header-title">放大镜对比</text>
          <text class="header-subtitle">触控移动查看局部细节</text>
        </view>
      </view>

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

      <view v-else class="empty-state">
        <up-empty mode="image" text="暂无对比数据" />
        <button class="back-btn" @click="handleBack">返回处理页</button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";

const store = useProcessingStore();
const active = ref(false);
const posX = ref(150);
const posY = ref(200);
const lensSize = ref(100);
const currentMode = ref<"result" | "origin">("result");

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

const lensImageStyle = computed(() => ({
  width: "100%",
  height: "100%",
  backgroundImage: `url(${lensImg.value})`,
  backgroundSize: "cover",
  backgroundPosition: "center",
  transform: "scale(2)",
  borderRadius: "50%",
}));

function handleMove(e: any) {
  active.value = true;
  const touch = e.touches?.[0] || e.changedTouches?.[0];
  if (touch) {
    posX.value = touch.clientX;
    posY.value = touch.clientY;
  }
}

function switchPage(url: string) {
  uni.navigateTo({ url });
}
function handleBack() {
  uni.navigateBack();
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
.page-header-card {
  display: flex;
  align-items: center;
  gap: 24rpx;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 24rpx;
  padding: 32rpx;
  margin-bottom: 24rpx;
}
.header-icon {
  width: 80rpx;
  height: 80rpx;
  background: #fef3c7;
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}
.header-title {
  font-size: 36rpx;
  font-weight: 700;
  color: #1f2937;
}
.header-subtitle {
  font-size: 26rpx;
  color: #6b7280;
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

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.back-btn {
  margin-top: 32rpx;
  padding: 16rpx 48rpx;
  background: #f59e0b;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
