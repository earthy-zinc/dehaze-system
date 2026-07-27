<template>
  <PageLayout class="page">
    <view class="main-content">
      <PageHeaderCard
        icon="photo"
        icon-color="#8b5cf6"
        icon-bg="#ede9fe"
        title="重叠对比"
        subtitle="透明叠加查看去雾效果"
        variant="dark"
      />

      <view v-if="hasImages" class="content-area">
        <!-- 叠加容器 -->
        <view class="overlay-container">
          <image
            :src="originUrl"
            class="overlay-image bottom-image"
            mode="widthFix"
          />
          <image
            :src="resultUrl"
            class="overlay-image top-image"
            mode="widthFix"
            :style="{ opacity: opacity / 100 }"
          />
        </view>

        <!-- 透明度滑块 -->
        <view class="control-panel">
          <view class="slider-row">
            <text class="slider-label">原图</text>
            <slider
              :value="opacity"
              :min="0"
              :max="100"
              :step="1"
              active-color="#8b5cf6"
              block-size="22"
              @change="(e: any) => (opacity = e.detail.value)"
            />
            <text class="slider-label">处理后</text>
          </view>
          <view class="opacity-badge">{{ opacity }}%</view>
        </view>

        <!-- 预设按钮 -->
        <view class="preset-row">
          <view
            class="preset-btn"
            :class="{ active: opacity === 0 }"
            @click="opacity = 0"
            >仅原图</view
          >
          <view
            class="preset-btn"
            :class="{ active: opacity === 50 }"
            @click="opacity = 50"
            >半透明</view
          >
          <view
            class="preset-btn"
            :class="{ active: opacity === 100 }"
            @click="opacity = 100"
            >仅结果</view
          >
        </view>

        <!-- 导航 -->
        <view class="nav-row">
          <view
            class="nav-item"
            @click="switchPage('/pages/side-by-side/index')"
          >
            <u-icon name="grid" size="20" color="#8b5cf6" /><text
              >并排对比</text
            >
          </view>
          <view class="nav-item" @click="switchPage('/pages/magnifier/index')">
            <u-icon name="search" size="20" color="#8b5cf6" /><text
              >放大镜</text
            >
          </view>
        </view>
      </view>

      <CompareEmptyState v-else text="暂无对比数据" btn-color="#8b5cf6" />
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";

const store = useProcessingStore();
const opacity = ref(50);

const originUrl = computed(() => store.originUrl);
const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));

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

.overlay-container {
  position: relative;
  width: 100%;
  border-radius: 16rpx;
  overflow: hidden;
  background: #000;
}
.overlay-image {
  width: 100%;
  display: block;
}
.top-image {
  position: absolute;
  top: 0;
  left: 0;
}

.control-panel {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 20rpx;
  padding: 28rpx;
  margin-top: 24rpx;
}
.slider-row {
  display: flex;
  align-items: center;
  gap: 16rpx;
}
.slider-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.5);
  flex-shrink: 0;
}
slider {
  flex: 1;
}
.opacity-badge {
  text-align: center;
  margin-top: 12rpx;
  font-size: 32rpx;
  font-weight: 700;
  color: #8b5cf6;
}

.preset-row {
  display: flex;
  gap: 16rpx;
  margin-top: 24rpx;
}
.preset-btn {
  flex: 1;
  text-align: center;
  padding: 20rpx;
  border-radius: 16rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.08);
  &.active {
    background: rgba(139, 92, 246, 0.2);
    color: #8b5cf6;
    font-weight: 600;
  }
  &:active {
    opacity: 0.7;
  }
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
    background: rgba(139, 92, 246, 0.15);
  }
}
</style>
