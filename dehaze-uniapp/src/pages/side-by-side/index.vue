<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="grid" size="28" color="#3b82f6" />
        </view>
        <view class="header-text">
          <text class="header-title">并排对比</text>
          <text class="header-subtitle">左右滑动查看去雾前后差异</text>
        </view>
      </view>

      <!-- 对比容器 -->
      <view v-if="hasImages" class="compare-wrapper">
        <view
          class="compare-container"
          @touchmove="handleTouchMove"
          @touchend="handleTouchEnd"
        >
          <!-- 底部：原图 -->
          <image
            :src="originUrl"
            class="compare-image base-image"
            mode="widthFix"
          />
          <!-- 顶部：结果图（clip 裁剪） -->
          <image
            :src="resultUrl"
            class="compare-image overlay-image"
            mode="widthFix"
            :style="{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }"
          />
          <!-- 分割线 -->
          <view class="divider" :style="{ left: sliderPos + '%' }">
            <view class="divider-line" />
            <view class="divider-handle">
              <text class="handle-arrow">⟷</text>
            </view>
          </view>
        </view>

        <!-- 标签 -->
        <view class="label-row">
          <text class="label left-label">处理后</text>
          <text class="label right-label">原图</text>
        </view>
      </view>

      <!-- 空状态 -->
      <view v-else class="empty-state">
        <up-empty mode="image" text="暂无对比数据" />
        <text class="empty-hint">请先完成去雾处理</text>
        <button class="back-btn" @click="handleBack">返回处理页</button>
      </view>

      <!-- 操作按钮 -->
      <view v-if="hasImages" class="action-section">
        <view class="action-grid">
          <view class="action-item" @click="handleOverlay">
            <u-icon name="photo" size="20" color="#3b82f6" />
            <text>重叠对比</text>
          </view>
          <view class="action-item" @click="handleMagnifier">
            <u-icon name="search" size="20" color="#3b82f6" />
            <text>放大镜</text>
          </view>
          <view class="action-item" @click="handleFilter">
            <u-icon name="setting" size="20" color="#3b82f6" />
            <text>滤镜调节</text>
          </view>
          <view class="action-item" @click="handleMetrics">
            <u-icon name="integral" size="20" color="#3b82f6" />
            <text>指标评估</text>
          </view>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";

const store = useProcessingStore();
const sliderPos = ref(50);

const originUrl = computed(() => store.originUrl);
const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));

function handleTouchMove(e: any) {
  const touch = e.touches?.[0] || e.changedTouches?.[0];
  if (!touch) return;
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect?.();
  if (!rect) return;
  let pos = ((touch.clientX - rect.left) / rect.width) * 100;
  pos = Math.max(5, Math.min(95, pos));
  sliderPos.value = pos;
}

function handleTouchEnd() {}

function handleOverlay() {
  uni.navigateTo({ url: "/pages/overlay/index" });
}
function handleMagnifier() {
  uni.navigateTo({ url: "/pages/magnifier/index" });
}
function handleFilter() {
  uni.navigateTo({ url: "/pages/filter/index" });
}
function handleMetrics() {
  uni.navigateTo({ url: "/pages/metrics/index" });
}
function handleBack() {
  uni.navigateBack();
}

onMounted(() => {
  if (!hasImages.value) {
    uni.showToast({ title: "请先完成去雾处理", icon: "none", duration: 2000 });
  }
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
  background: #dbeafe;
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}
.header-title {
  font-size: 36rpx;
  font-weight: 700;
  color: #fff;
}
.header-subtitle {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.6);
}
.page-header-card .header-title {
  color: #1f2937;
}
.page-header-card .header-subtitle {
  color: #6b7280;
}

.compare-wrapper {
  position: relative;
}
.compare-container {
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 16rpx;
  background: #000;
  touch-action: none;
}
.compare-image {
  width: 100%;
  display: block;
}
.overlay-image {
  position: absolute;
  top: 0;
  left: 0;
}

.divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 4rpx;
  transform: translateX(-50%);
  pointer-events: none;
}
.divider-line {
  width: 100%;
  height: 100%;
  background: #fff;
  box-shadow: 0 0 8rpx rgba(0, 0, 0, 0.5);
}
.divider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 64rpx;
  height: 64rpx;
  background: #fff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.3);
}
.handle-arrow {
  font-size: 24rpx;
  color: #3b82f6;
}

.label-row {
  display: flex;
  justify-content: space-between;
  padding: 16rpx 8rpx;
}
.label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.5);
}
.left-label {
  color: #3b82f6;
}

.action-section {
  margin-top: 40rpx;
}
.action-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20rpx;
}
.action-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12rpx;
  padding: 32rpx;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 20rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.7);
  &:active {
    background: rgba(59, 130, 246, 0.15);
  }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.empty-hint {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.4);
  margin: 16rpx 0 32rpx;
}
.back-btn {
  padding: 16rpx 48rpx;
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
