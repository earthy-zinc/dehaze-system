<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="info-circle" size="28" color="#6366f1" />
        </view>
        <view class="header-text">
          <text class="header-title">算法信息</text>
          <text class="header-subtitle">查看当前算法详细信息</text>
        </view>
      </view>

      <view v-if="algorithm" class="content-area">
        <!-- 基本信息 -->
        <view class="section">
          <text class="section-title">基本信息</text>
          <view class="info-card">
            <view class="info-row"><text class="label">名称</text><text class="value">{{ algorithm.name }}</text></view>
            <view class="info-row"><text class="label">类型</text><text class="value type-tag">{{ algorithm.type || "未知" }}</text></view>
            <view class="info-row"><text class="label">版本</text><text class="value">v{{ algorithm.version || "-" }}</text></view>
            <view class="info-row"><text class="label">状态</text><text class="value">{{ algorithm.status === 1 ? "✅ 启用" : "⏸ 停用" }}</text></view>
          </view>
        </view>

        <!-- 描述 -->
        <view v-if="algorithm.description" class="section">
          <text class="section-title">算法描述</text>
          <view class="desc-card">
            <text class="desc-text">{{ algorithm.description }}</text>
          </view>
        </view>

        <!-- 技术指标 -->
        <view class="section">
          <text class="section-title">技术指标</text>
          <view class="specs-grid">
            <view class="spec-card">
              <text class="spec-label">计算量</text>
              <text class="spec-value">{{ algorithm.flops || "-" }}</text>
            </view>
            <view class="spec-card">
              <text class="spec-label">模型大小</text>
              <text class="spec-value">{{ algorithm.size || "-" }}</text>
            </view>
            <view class="spec-card">
              <text class="spec-label">路径</text>
              <text class="spec-value small">{{ algorithm.path || "-" }}</text>
            </view>
            <view class="spec-card">
              <text class="spec-label">创建时间</text>
              <text class="spec-value small">{{ formatTime(algorithm.createTime) }}</text>
            </view>
          </view>
        </view>

        <!-- 参数说明 -->
        <view v-if="algorithm.params" class="section">
          <text class="section-title">参数说明</text>
          <view class="params-card">
            <text class="params-text">{{ algorithm.params }}</text>
          </view>
        </view>
      </view>

      <view v-else-if="loading" class="loading-state">
        <up-loading-icon mode="circle" size="40" color="#6366f1" />
        <text class="loading-text">加载算法信息...</text>
      </view>

      <view v-else class="empty-state">
        <up-empty mode="search" text="暂无算法信息" />
        <text class="empty-hint">请先选择算法</text>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";
import { getAlgorithmDetail, type Algorithm } from "@/api/algorithm";

const store = useProcessingStore();
const loading = ref(false);
const algorithm = ref<Algorithm | null>(null);

async function loadDetail() {
  if (!store.selectedAlgorithm?.id) return;
  loading.value = true;
  try {
    const detail = await getAlgorithmDetail(store.selectedAlgorithm.id);
    algorithm.value = detail;
  } catch {
    algorithm.value = store.selectedAlgorithm;
  } finally {
    loading.value = false;
  }
}

function formatTime(time?: string): string {
  if (!time) return "-";
  return new Date(time).toLocaleDateString("zh-CN");
}

onMounted(() => {
  if (store.selectedAlgorithm) {
    algorithm.value = store.selectedAlgorithm;
    loadDetail();
  }
});
</script>

<style lang="scss" scoped>
.page { width: 100%; min-height: 100vh; background: #f9fafb; }
.main-content { padding: 24rpx; padding-bottom: calc(80rpx + constant(safe-area-inset-bottom)); }
.page-header-card {
  display: flex; align-items: center; gap: 24rpx;
  background: #fff; border-radius: 24rpx; padding: 32rpx; margin-bottom: 24rpx; box-shadow: 0 4rpx 16rpx rgba(0,0,0,0.06);
}
.header-icon { width: 80rpx; height: 80rpx; background: linear-gradient(135deg, #e0e7ff, #c7d2fe); border-radius: 20rpx; display: flex; align-items: center; justify-content: center; }
.header-title { font-size: 36rpx; font-weight: 700; color: #1f2937; display: block; margin-bottom: 8rpx; }
.header-subtitle { font-size: 26rpx; color: #6b7280; display: block; }

.section { margin-bottom: 24rpx; }
.section-title { font-size: 28rpx; font-weight: 600; color: #374151; margin-bottom: 12rpx; display: block; }

.info-card, .desc-card, .params-card {
  background: #fff; border-radius: 20rpx; padding: 28rpx; box-shadow: 0 2rpx 12rpx rgba(0,0,0,0.04);
}
.info-row { display: flex; justify-content: space-between; align-items: center; padding: 14rpx 0;
  & + & { border-top: 1rpx solid #f3f4f6; }
}
.label { font-size: 26rpx; color: #6b7280; }
.value { font-size: 26rpx; font-weight: 500; color: #1f2937; }
.type-tag { color: #6366f1; background: #e0e7ff; padding: 4rpx 12rpx; border-radius: 8rpx; font-size: 22rpx; }

.desc-text { font-size: 26rpx; color: #4b5563; line-height: 1.6; }
.params-text { font-size: 24rpx; color: #4b5563; line-height: 1.6; font-family: monospace; white-space: pre-wrap; }

.specs-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16rpx; }
.spec-card {
  background: #fff; border-radius: 16rpx; padding: 24rpx; box-shadow: 0 2rpx 8rpx rgba(0,0,0,0.03); text-align: center;
}
.spec-label { display: block; font-size: 22rpx; color: #9ca3af; margin-bottom: 8rpx; }
.spec-value { display: block; font-size: 28rpx; font-weight: 700; color: #1f2937; &.small { font-size: 22rpx; font-weight: 500; } }

.loading-state { display: flex; flex-direction: column; align-items: center; padding: 120rpx 0; }
.loading-text { margin-top: 24rpx; font-size: 28rpx; color: #9ca3af; }

.empty-state { display: flex; flex-direction: column; align-items: center; padding: 120rpx 0; }
.empty-hint { font-size: 26rpx; color: #9ca3af; margin-top: 16rpx; }
</style>
