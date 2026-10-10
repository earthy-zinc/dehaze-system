<template>
  <view class="recommend-section">
    <SectionHeader title="模型推荐" subtitle="精选去雾模型，点击立即体验" />

    <!-- 加载中 -->
    <view v-if="loading" class="state-wrap">
      <view class="loading-spinner" />
      <text class="state-text">加载模型中...</text>
    </view>

    <!-- 加载失败 -->
    <view v-else-if="error" class="state-wrap">
      <text class="state-text">{{ error }}</text>
      <text class="retry-link" @click="loadModels">重新加载</text>
    </view>

    <!-- 模型卡片列表 -->
    <view v-else-if="models.length > 0" class="model-grid">
      <view
        v-for="model in models"
        :key="model.id"
        class="model-card"
        @click="handleClick(model)"
      >
        <view class="model-icon">
          <SvgIcon name="gift" size="24" color="#8b5cf6" />
        </view>
        <text class="model-name">{{ model.displayName }}</text>
        <text class="model-type">{{ model.modelId }}</text>
      </view>
    </view>

    <!-- 空状态 -->
    <view v-else class="state-wrap">
      <text class="state-text">暂无可用模型</text>
    </view>

    <view v-if="models.length > 0" class="more-row">
      <text class="more-link" @click="handleMore">查看全部模型 →</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import SectionHeader from "@/components/common/SectionHeader.vue";
import { AiModelAPI } from "dehaze-sdk-js";
import type { AiModelVO } from "dehaze-sdk-js";
import { getErrorMessage } from "@/utils/error";

/** 展示模型数量 */
const DISPLAY_COUNT = 4;

const loading = ref(false);
const error = ref("");
const models = ref<AiModelVO[]>([]);

async function loadModels() {
  if (loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    const list = await AiModelAPI.listEnabledModels("image_edit");
    models.value = list.slice(0, DISPLAY_COUNT);
  } catch (e) {
    error.value = getErrorMessage(e, "模型加载失败");
  } finally {
    loading.value = false;
  }
}

interface Emits {
  (e: "select", model: AiModelVO): void;
  (e: "more"): void;
}
const emit = defineEmits<Emits>();

function handleClick(model: AiModelVO) {
  emit("select", model);
}

function handleMore() {
  emit("more");
}

onMounted(loadModels);
</script>

<style lang="scss" scoped>
.recommend-section {
  padding: 80rpx 40rpx;
  background: $color-white;
}

.state-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16rpx;
  padding: 60rpx 0;
}

.state-text {
  font-size: 26rpx;
  color: $color-text-placeholder;
}

.retry-link {
  font-size: 26rpx;
  color: $color-accent;
  font-weight: 600;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24rpx;
  margin-top: 16rpx;
}

.model-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8rpx;
  padding: 28rpx;
  background: linear-gradient(135deg, #faf5ff 0%, #ede9fe 100%);
  border-radius: 20rpx;
  border: 2rpx solid transparent;
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.97);
    border-color: #8b5cf6;
  }
}

.model-icon {
  width: 64rpx;
  height: 64rpx;
  border-radius: 16rpx;
  background: $color-white;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8rpx;
  box-shadow: 0 4rpx 12rpx rgba(139, 92, 246, 0.15);
}

.model-name {
  font-size: 28rpx;
  font-weight: 700;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.model-type {
  font-size: 22rpx;
  color: #8b5cf6;
  background: rgba(255, 255, 255, 0.7);
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}

.model-desc {
  font-size: 22rpx;
  color: #6b7280;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.more-row {
  display: flex;
  justify-content: center;
  margin-top: 32rpx;
}

.more-link {
  font-size: 26rpx;
  color: $color-accent;
  font-weight: 600;
  padding: 12rpx 32rpx;

  &:active {
    opacity: 0.7;
  }
}

@media screen and (max-width: 768rpx) {
  .model-grid {
    grid-template-columns: 1fr;
  }
}
</style>
