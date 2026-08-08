<template>
  <ImmersiveLayout title="并排对比">
    <scroll-view v-if="hasImages" class="main-content" scroll-y>
      <!-- 原图 -->
      <view class="image-section">
        <view class="image-label">
          <view class="label-tag label-original">
            <text>原图</text>
          </view>
          <text class="image-name">{{ originName }}</text>
        </view>
        <view class="image-wrapper">
          <image
            :src="originUrl"
            class="compare-image"
            mode="widthFix"
            lazy-load
          />
        </view>
      </view>

      <!-- 分隔线 -->
      <view class="image-divider">
        <view class="divider-line" />
      </view>

      <!-- 处理后 -->
      <view class="image-section">
        <view class="image-label">
          <view class="label-tag label-result">
            <text>处理后</text>
          </view>
          <text class="image-name">{{ algorithmName || "去雾结果" }}</text>
        </view>
        <view class="image-wrapper">
          <image
            :src="resultUrl"
            class="compare-image"
            mode="widthFix"
            lazy-load
          />
        </view>
      </view>

      <!-- 算法信息 -->
      <view v-if="algorithm" class="info-card">
        <text class="card-title">算法信息</text>
        <view class="info-row">
          <text class="info-label">算法名称</text>
          <text class="info-value">{{ algorithm.name }}</text>
        </view>
        <view v-if="algorithm.type" class="info-row">
          <text class="info-label">类型</text>
          <text class="info-value">{{ algorithm.type }}</text>
        </view>
        <view v-if="algorithm.version" class="info-row">
          <text class="info-label">版本</text>
          <text class="info-value">{{ algorithm.version }}</text>
        </view>
        <view v-if="result?.time !== undefined" class="info-row">
          <text class="info-label">处理耗时</text>
          <text class="info-value">{{ result.time }}s</text>
        </view>
        <view v-if="result?.fromCache" class="cache-tag">缓存命中</view>
      </view>

      <!-- 导出报告 -->
      <view class="export-section">
        <button
          class="export-btn"
          :loading="exporting"
          :disabled="exporting"
          @click="handleExportReport"
        >
          {{ exporting ? "生成中..." : "导出报告" }}
        </button>
      </view>
    </scroll-view>

    <CompareEmptyState
      v-else
      text="暂无对比数据"
      hint="请先完成去雾处理"
      btn-color="#3b82f6"
    />

    <template #toolbar>
      <view class="toolbar-grid">
        <view
          v-for="m in modes"
          :key="m.key"
          class="toolbar-item"
          :class="{ active: m.key === 'side-by-side' }"
          @click="switchPage(m.path)"
        >
          <SvgIcon :name="m.icon" size="20" color="#3b82f6" />
          <text>{{ m.label }}</text>
        </view>
      </view>
      <view class="toolbar-actions">
        <view class="action-item" @click="handleSave">
          <SvgIcon name="download" size="18" color="rgba(255,255,255,0.7)" />
          <text>保存</text>
        </view>
        <view class="action-item" @click="handleShare">
          <SvgIcon name="share" size="18" color="rgba(255,255,255,0.7)" />
          <text>分享</text>
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
          <SvgIcon :name="favorited ? 'star-fill' : 'star'" size="18" :color="favorited ? '#f59e0b' : 'rgba(255,255,255,0.7)'" />
          <text :style="{ color: favorited ? '#f59e0b' : '' }">{{ favorited ? '已收藏' : '收藏' }}</text>
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
import { ModelAPI, FavoriteAPI } from "dehaze-sdk-js";

type TaskStatus = 1 | 2 | 3;

const store = useProcessingStore();
const exporting = ref(false);
const favorited = ref(false);
const favoriteLoading = ref(false);

const originUrl = computed(() => store.originUrl);
const originName = computed(() => store.currentImage?.name || "原图");
const resultUrl = computed(() => store.result?.resultUrl || "");
const algorithm = computed(() => store.selectedAlgorithm);
const algorithmName = computed(() => store.selectedAlgorithm?.name);
const result = computed(() => store.result);
const hasImages = computed(() => !!(originUrl.value && resultUrl.value));
const resultId = computed(() => store.result?.id);

const modes = [
  { key: "side-by-side", label: "并排对比", path: "/pages/side-by-side/index", icon: "grid" },
  { key: "overlay", label: "重叠对比", path: "/pages/overlay/index", icon: "photo" },
  { key: "magnifier", label: "放大镜", path: "/pages/magnifier/index", icon: "search" },
  { key: "filter", label: "滤镜", path: "/pages/filter/index", icon: "setting" },
  { key: "metrics", label: "指标", path: "/pages/metrics/index", icon: "integral" },
];

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
          success: () => uni.showToast({ title: "已保存到相册", icon: "success" }),
          fail: () => uni.showToast({ title: "保存失败", icon: "none" }),
        });
      }
    },
    fail: () => uni.showToast({ title: "下载失败", icon: "none" }),
  });
}

function handleShare() {
  if (!resultUrl.value) {
    uni.showToast({ title: "无结果图片可分享", icon: "none" });
    return;
  }
  // #ifdef MP-WEIXIN
  uni.showShareImageMenu({ path: resultUrl.value });
  // #endif
  // #ifdef H5
  uni.setClipboardData({
    data: resultUrl.value,
    success: () => uni.showToast({ title: "链接已复制", icon: "success" }),
  });
  // #endif
}

function handleReprocess() {
  uni.redirectTo({ url: "/pages/processing/index" });
}

function handleChangeAlgorithm() {
  uni.redirectTo({ url: "/pages/algorithm-select/index" });
}

async function handleExportReport() {
  if (!resultUrl.value) {
    uni.showToast({ title: "缺少必要参数", icon: "none" });
    return;
  }
  exporting.value = true;
  try {
    const res = await ModelAPI.generateReport({ logId: 0, format: "pdf" });
    const taskId = res.taskId;
    if (!taskId) throw new Error("未返回任务ID");
    while (true) {
      await new Promise((r) => setTimeout(r, 2000));
      const statusRes = await ModelAPI.getReportStatus(taskId);
      const status = statusRes.status as TaskStatus;
      if (status === 2) {
        if (statusRes.downloadUrl) {
          uni.downloadFile({
            url: statusRes.downloadUrl,
            success(dlRes) {
              if (dlRes.statusCode === 200 && dlRes.tempFilePath) {
                uni.openDocument({ filePath: dlRes.tempFilePath, showMenu: true });
              }
            },
          });
        }
        break;
      }
      if (status === 3) throw new Error(statusRes.errorMessage || "报告生成失败");
    }
  } catch (e: any) {
    uni.showToast({ title: e.message || "报告生成失败", icon: "none" });
  } finally {
    exporting.value = false;
  }
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
  if (!hasImages.value) {
    uni.showToast({ title: "请先完成去雾处理", icon: "none", duration: 2000 });
  }
  if (resultId.value) {
    FavoriteAPI.getStatus("result", resultId.value)
      .then((res) => { favorited.value = res.favorited; })
      .catch(() => {});
  }
});
</script>

<style lang="scss" scoped>
.main-content {
  height: 100%;
  padding: 24rpx;
  overflow: hidden;
}

.image-section {
  margin-bottom: 16rpx;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 24rpx;

  .image-label {
    display: flex;
    gap: 16rpx;
    align-items: center;
    padding: 20rpx 28rpx;
    border-bottom: 2rpx solid rgba(255, 255, 255, 0.05);

    .label-tag {
      padding: 4rpx 16rpx;
      font-size: 22rpx;
      border-radius: 8rpx;

      &.label-original {
        color: #fbbf24;
        background: rgba(251, 191, 36, 0.15);
      }

      &.label-result {
        color: #34d399;
        background: rgba(52, 211, 153, 0.15);
      }
    }

    .image-name {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 26rpx;
      color: rgba(255, 255, 255, 0.5);
      white-space: nowrap;
    }
  }

  .image-wrapper {
    width: 100%;

    .compare-image {
      width: 100%;
    }
  }
}

.image-divider {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8rpx 0;

  .divider-line {
    width: 80rpx;
    height: 6rpx;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4rpx;
  }
}

.info-card {
  padding: 28rpx 32rpx;
  margin-top: 16rpx;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 24rpx;

  .card-title {
    font-size: 28rpx;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    margin-bottom: 16rpx;
    display: block;
  }

  .info-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12rpx 0;
    border-bottom: 1rpx solid rgba(255, 255, 255, 0.05);

    &:last-child {
      border-bottom: none;
    }

    .info-label {
      font-size: 26rpx;
      color: rgba(255, 255, 255, 0.5);
    }

    .info-value {
      font-size: 26rpx;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.8);
    }
  }

  .cache-tag {
    display: inline-block;
    margin-top: 12rpx;
    font-size: 22rpx;
    color: #34d399;
    background: rgba(52, 211, 153, 0.15);
    padding: 4rpx 12rpx;
    border-radius: 8rpx;
  }
}

.export-section {
  padding: 24rpx 32rpx;
  margin-top: 16rpx;
}

.export-btn {
  width: 100%;
  padding: 24rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 30rpx;
  font-weight: 600;
}

.toolbar-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4rpx;
  padding: 16rpx 16rpx 8rpx;
}

.toolbar-actions {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
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
    background: rgba(59, 130, 246, 0.15);
    border-radius: 12rpx;
  }

  &.active {
    color: #3b82f6;
    background: rgba(59, 130, 246, 0.12);
    border-radius: 12rpx;
  }
}
</style>
