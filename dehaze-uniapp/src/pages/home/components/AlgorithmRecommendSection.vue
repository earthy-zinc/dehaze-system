<template>
  <view class="recommend-section">
    <SectionHeader
      title="算法推荐"
      subtitle="精选去雾算法，点击立即体验"
    />

    <!-- 加载中 -->
    <view v-if="loading" class="state-wrap">
      <up-loading-icon mode="circle" size="32" color="#8b5cf6" />
      <text class="state-text">加载算法中...</text>
    </view>

    <!-- 加载失败 -->
    <view v-else-if="error" class="state-wrap">
      <text class="state-text">{{ error }}</text>
      <text class="retry-link" @click="loadAlgorithms">重新加载</text>
    </view>

    <!-- 算法卡片列表 -->
    <view v-else-if="algorithms.length > 0" class="algo-grid">
      <view
        v-for="algo in algorithms"
        :key="algo.id"
        class="algo-card"
        @click="handleClick(algo)"
      >
        <view class="algo-icon">
          <u-icon name="gift" size="24" color="#8b5cf6" />
        </view>
        <text class="algo-name">{{ algo.name }}</text>
        <text class="algo-type">{{ algo.type || "去雾算法" }}</text>
        <text class="algo-desc">{{ algo.description || "高效去雾算法" }}</text>
      </view>
    </view>

    <!-- 空状态 -->
    <view v-else class="state-wrap">
      <text class="state-text">暂无可用算法</text>
    </view>

    <view v-if="algorithms.length > 0" class="more-row">
      <text class="more-link" @click="handleMore">查看全部算法 →</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import SectionHeader from "@/components/common/SectionHeader.vue";
import { getAlgorithmList } from "@/api/algorithm";
import type { Algorithm } from "@/api/algorithm";

/** 展示算法数量 */
const DISPLAY_COUNT = 4;

const loading = ref(false);
const error = ref("");
const algorithms = ref<Algorithm[]>([]);

async function loadAlgorithms() {
  if (loading.value) return;
  loading.value = true;
  error.value = "";
  try {
    const list = await getAlgorithmList();
    algorithms.value = list.slice(0, DISPLAY_COUNT);
  } catch (e) {
    error.value = (e as { message?: string }).message || "算法加载失败";
  } finally {
    loading.value = false;
  }
}

interface Emits {
  (e: "select", algorithm: Algorithm): void;
  (e: "more"): void;
}
const emit = defineEmits<Emits>();

function handleClick(algorithm: Algorithm) {
  emit("select", algorithm);
}

function handleMore() {
  emit("more");
}

onMounted(loadAlgorithms);
</script>

<style lang="scss" scoped>
.recommend-section {
  padding: 80rpx 40rpx;
  background: #ffffff;
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
  color: #9ca3af;
}

.retry-link {
  font-size: 26rpx;
  color: #8b5cf6;
  font-weight: 600;
}

.algo-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24rpx;
  margin-top: 16rpx;
}

.algo-card {
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

.algo-icon {
  width: 64rpx;
  height: 64rpx;
  border-radius: 16rpx;
  background: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8rpx;
  box-shadow: 0 4rpx 12rpx rgba(139, 92, 246, 0.15);
}

.algo-name {
  font-size: 28rpx;
  font-weight: 700;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.algo-type {
  font-size: 22rpx;
  color: #8b5cf6;
  background: rgba(255, 255, 255, 0.7);
  padding: 4rpx 12rpx;
  border-radius: 8rpx;
}

.algo-desc {
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
  color: #8b5cf6;
  font-weight: 600;
  padding: 12rpx 32rpx;

  &:active {
    opacity: 0.7;
  }
}

@media screen and (max-width: 768rpx) {
  .algo-grid {
    grid-template-columns: 1fr;
  }
}
</style>
