<template>
  <view class="history-list">
    <!-- 头部 -->
    <view class="list-header">
      <text class="header-title">最近处理的图片</text>
    </view>

    <!-- 加载状态 -->
    <view v-if="loading" class="loading-container">
      <up-loading-icon mode="circle" size="32" color="#9ca3af" />
      <text class="loading-text">加载中...</text>
    </view>

    <!-- 历史记录列表 -->
    <view v-else-if="records.length > 0" class="history-list-content">
      <HistoryCard
        v-for="record in records"
        :key="record.id"
        :record="record"
        @load="handleLoad"
      />
    </view>

    <!-- 空状态 -->
    <view v-else class="empty-state">
      <view class="empty-icon">
        <u-icon name="clock" size="48" color="#d1d5db" />
      </view>
      <text class="empty-text">暂无历史记录</text>
      <text class="empty-hint">处理过的图片会显示在这里</text>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref, onMounted } from "vue";
import HistoryCard from "./HistoryCard.vue";
import type { ImageData } from "../data/imageInputData";
import { ModelAPI } from "dehaze-sdk-js";
import type { PredLogVO } from "dehaze-sdk-js";

const emit = defineEmits<{
  (e: "select", data: ImageData): void;
}>();

const loading = ref(false);
const records = ref<PredLogVO[]>([]);

/** 加载历史记录（调用后端预测日志 API） */
const loadHistory = async () => {
  loading.value = true;
  try {
    const result = await ModelAPI.getPredLogs({ pageNum: 1, pageSize: 20 });
    records.value = result.list;
  } catch {
    records.value = [];
  } finally {
    loading.value = false;
  }
};

/** 点击历史记录，加载为当前图片（用于重新处理） */
const handleLoad = (record: PredLogVO) => {
  // 优先使用原图（有雾输入）便于重新处理；无原图时回退到结果图
  const url = record.originUrl || record.predUrl || "";
  if (!url) {
    uni.showToast({ title: "该记录无可用图片", icon: "none" });
    return;
  }
  const imageData: ImageData = {
    url,
    name: record.algorithmName
      ? `${record.algorithmName}-历史记录`
      : "历史记录",
  };
  emit("select", imageData);
};

onMounted(() => {
  loadHistory();
});

// 暴露刷新方法（切换到该 Tab 时父组件调用）
defineExpose({
  refresh: loadHistory,
});
</script>

<style lang="scss" scoped>
.history-list {
  min-height: 300rpx;
}

.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24rpx;
}

.header-title {
  font-size: 28rpx;
  color: #6b7280;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 80rpx 0;
}

.loading-text {
  margin-top: 16rpx;
  font-size: 26rpx;
  color: #9ca3af;
}

.history-list-content {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80rpx 0;
}

.empty-icon {
  width: 120rpx;
  height: 120rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f3f4f6;
  border-radius: 50%;
  margin-bottom: 24rpx;
}

.empty-text {
  font-size: 30rpx;
  font-weight: 600;
  color: #6b7280;
  margin-bottom: 8rpx;
}

.empty-hint {
  font-size: 26rpx;
  color: #9ca3af;
}
</style>
