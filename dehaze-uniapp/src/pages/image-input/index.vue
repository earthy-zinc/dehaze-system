<template>
  <PageLayout class="image-input-page">
    <view class="main-content">
      <!-- 页面标题卡片 -->
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="camera" size="28" color="#3b82f6" />
        </view>
        <view class="header-text">
          <text class="header-title">图像输入</text>
          <text class="header-subtitle">选择图片开始去雾处理</text>
        </view>
      </view>

      <!-- 输入方式选择 -->
      <view class="input-methods">
        <view class="methods-grid">
          <InputMethodCard
            v-for="method in INPUT_METHODS"
            :key="method.method"
            :icon="method.icon"
            :title="method.title"
            :subtitle="method.subtitle"
            :method="method.method"
            :active="currentMethod === method.method"
            @click="handleMethodChange"
          />
        </view>
      </view>

      <!-- 内容区域 -->
      <view class="content-area">
        <!-- 上传区域 -->
        <view v-show="currentMethod === 'upload'" class="content-section">
          <UploadArea @select="handleImageSelect" />
        </view>

        <!-- 拍照区域 -->
        <view v-show="currentMethod === 'camera'" class="content-section">
          <CameraArea @capture="handleImageSelect" />
        </view>

        <!-- 样例图片库 -->
        <view v-show="currentMethod === 'sample'" class="content-section">
          <SampleGallery @select="handleImageSelect" />
        </view>

        <!-- 历史记录 -->
        <view v-show="currentMethod === 'history'" class="content-section">
          <HistoryList ref="historyListRef" @select="handleImageSelect" />
        </view>
      </view>

      <!-- 图片预览 -->
      <ImagePreview
        v-if="currentImage"
        :image="currentImage"
        @remove="handleRemoveImage"
        @next="handleNextStep"
      />

      <!-- 快速体验卡片 -->
      <view v-if="!currentImage" class="quick-start-section">
        <QuickStartCard @start="handleImageSelect" />
      </view>

      <!-- 使用提示 -->
      <view v-if="!currentImage" class="tips-card">
        <view class="tips-header">
          <u-icon name="info-circle" size="18" color="#3b82f6" />
          <text class="tips-title">使用提示</text>
        </view>
        <view class="tips-list">
          <text class="tips-item">• 支持 JPG、PNG、WEBP、HEIC 格式图片</text>
          <text class="tips-item">• 单张图片最大 20MB，超过 5MB 自动压缩</text>
          <text class="tips-item">• 推荐使用有雾场景的图片以获得最佳效果</text>
          <text class="tips-item">• 可使用样例图片快速体验系统功能</text>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import PageLayout from "@/layout/index.vue";
import InputMethodCard from "./components/InputMethodCard.vue";
import UploadArea from "./components/UploadArea.vue";
import CameraArea from "./components/CameraArea.vue";
import SampleGallery from "./components/SampleGallery.vue";
import HistoryList from "./components/HistoryList.vue";
import ImagePreview from "./components/ImagePreview.vue";
import QuickStartCard from "./components/QuickStartCard.vue";
import type { InputMethod, ImageData } from "./data/imageInputData";
import { INPUT_METHODS } from "./data/imageInputData";

// ==================== 状态定义 ====================

/** 当前输入方式 */
const currentMethod = ref<InputMethod>("upload");

/** 当前选中的图片 */
const currentImage = ref<ImageData | null>(null);

/** 历史记录列表引用 */
const historyListRef = ref<InstanceType<typeof HistoryList> | null>(null);

// ==================== 方法定义 ====================

/** 切换输入方式 */
const handleMethodChange = (method: InputMethod) => {
  currentMethod.value = method;

  // 切换到历史记录时刷新列表
  if (method === "history" && historyListRef.value) {
    historyListRef.value.refresh();
  }
};

/** 选择图片 */
const handleImageSelect = (data: ImageData) => {
  currentImage.value = data;

  // 滚动到预览区域
  setTimeout(() => {
    uni.pageScrollTo({
      selector: ".image-preview",
      duration: 300,
    });
  }, 100);
};

/** 移除图片 */
const handleRemoveImage = () => {
  currentImage.value = null;
};

/** 下一步：跳转到算法选择 */
const handleNextStep = () => {
  if (!currentImage.value) {
    uni.showToast({
      title: "请先选择图片",
      icon: "none",
    });
    return;
  }

  // 保存当前图片到全局状态
  const app = getApp<{ globalData: { currentImage?: ImageData } }>();
  if (app.globalData) {
    app.globalData.currentImage = currentImage.value;
  }

  // 也存储到本地，以防页面刷新丢失
  try {
    uni.setStorageSync("current_image", JSON.stringify(currentImage.value));
  } catch (e) {
    console.warn("存储图片数据失败:", e);
  }

  // 跳转到算法选择页面
  uni.navigateTo({
    url: "/pages/algorithm-select/index",
    fail: () => {
      uni.showToast({
        title: "页面跳转失败",
        icon: "none",
      });
    },
  });
};
</script>

<style lang="scss" scoped>
.image-input-page {
  width: 100%;
  min-height: 100vh;
  background: #f9fafb;
}

.main-content {
  padding: 24rpx;
  padding-bottom: calc(120rpx + constant(safe-area-inset-bottom));
  padding-bottom: calc(120rpx + env(safe-area-inset-bottom));
}

/* 页面标题卡片 */
.page-header-card {
  display: flex;
  align-items: center;
  gap: 24rpx;
  background: #ffffff;
  border-radius: 24rpx;
  padding: 32rpx;
  margin-bottom: 24rpx;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
}

.header-icon {
  width: 80rpx;
  height: 80rpx;
  background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
  border-radius: 20rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.header-text {
  flex: 1;
}

.header-title {
  display: block;
  font-size: 36rpx;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 8rpx;
}

.header-subtitle {
  display: block;
  font-size: 26rpx;
  color: #6b7280;
}

/* 输入方式选择 */
.input-methods {
  margin-bottom: 24rpx;
}

.methods-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16rpx;
}

/* 内容区域 */
.content-area {
  margin-bottom: 24rpx;
}

.content-section {
  background: #ffffff;
  border-radius: 24rpx;
  padding: 24rpx;
  box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.06);
}

/* 快速体验区域 */
.quick-start-section {
  margin-top: 32rpx;
}

/* 使用提示 */
.tips-card {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border: 2rpx solid #bfdbfe;
  border-radius: 24rpx;
  padding: 32rpx;
  margin-top: 32rpx;
}

.tips-header {
  display: flex;
  align-items: center;
  gap: 12rpx;
  margin-bottom: 20rpx;
}

.tips-title {
  font-size: 30rpx;
  font-weight: 600;
  color: #1e40af;
}

.tips-list {
  display: flex;
  flex-direction: column;
  gap: 12rpx;
}

.tips-item {
  font-size: 26rpx;
  color: #1e3a8a;
  line-height: 1.5;
}

/* 小屏幕适配 */
@media (max-width: 375px) {
  .main-content {
    padding: 16rpx;
  }

  .page-header-card {
    padding: 24rpx;
  }

  .header-title {
    font-size: 32rpx;
  }

  .content-section {
    padding: 20rpx;
  }

  .tips-card {
    padding: 24rpx;
  }
}

/* 平板适配 */
@media (min-width: 768px) {
  .methods-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
</style>
