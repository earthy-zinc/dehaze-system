<template>
  <ImmersiveLayout title="滤镜调节">
    <view v-if="hasImages" class="main-content">
      <!-- 预览图 -->
      <view class="preview-wrapper">
        <image
          :src="currentPreviewUrl"
          class="preview-image"
          mode="widthFix"
          :style="{ filter: filterString }"
          lazy-load
        />
        <view class="preview-toggle" @click="showOrigin = !showOrigin">
          <text>{{ showOrigin ? "原图" : "滤镜效果" }}</text>
        </view>
      </view>

      <scroll-view class="filter-scroll" scroll-y>
        <!-- 预设方案 -->
        <view class="panel">
          <view class="panel-header">
            <text class="panel-title">预设方案</text>
            <text class="save-btn" @click="handleSavePreset">保存当前</text>
          </view>
          <view class="preset-list">
            <view
              v-for="p in builtinPresets"
              :key="p.name"
              class="preset-item"
              :class="{ active: activePreset === p.name }"
              @click="applyPreset(p)"
            >
              <text>{{ p.name }}</text>
            </view>
            <view
              v-for="(p, idx) in customPresets"
              :key="'c' + idx"
              class="preset-item custom"
              :class="{ active: activePreset === p.name }"
              @click="applyPreset(p)"
              @longpress="handleDeletePreset(idx)"
            >
              <text>{{ p.name }}</text>
            </view>
          </view>
        </view>

        <!-- 滤镜参数 -->
        <view class="panel">
          <view class="panel-header">
            <text class="panel-title">参数调节</text>
            <text v-if="hasChanges" class="reset-btn" @click="handleReset"
              >重置</text
            >
          </view>
          <view v-for="item in filters" :key="item.key" class="slider-item">
            <view class="slider-label-row">
              <text class="slider-label">{{ item.label }}</text>
              <text class="slider-value">{{ item.value }}</text>
            </view>
            <slider
              :value="item.value"
              :min="item.min"
              :max="item.max"
              :step="1"
              :active-color="item.color"
              block-size="20"
              @change="(e: any) => updateFilter(item.key, e.detail.value)"
            />
          </view>
        </view>
      </scroll-view>
    </view>

    <CompareEmptyState v-else text="暂无处理结果" btn-color="#10b981" />

    <template #toolbar>
      <view class="toolbar-grid">
        <view
          v-for="m in modes"
          :key="m.key"
          class="toolbar-item"
          :class="{ active: m.key === 'filter' }"
          @click="switchPage(m.path)"
        >
          <SvgIcon :name="m.icon" size="20" color="#10b981" />
          <text>{{ m.label }}</text>
        </view>
      </view>
      <view class="toolbar-actions">
        <view class="action-item" @click="handleSave">
          <SvgIcon name="download" size="18" color="rgba(255,255,255,0.7)" />
          <text>保存</text>
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
          <SvgIcon
            :name="favorited ? 'star-fill' : 'star'"
            size="18"
            :color="favorited ? '#f59e0b' : 'rgba(255,255,255,0.7)'"
          />
          <text :style="{ color: favorited ? '#f59e0b' : '' }">{{
            favorited ? "已收藏" : "收藏"
          }}</text>
        </view>
      </view>
    </template>
  </ImmersiveLayout>
</template>

<script lang="ts" setup>
import { ref, reactive, computed, onMounted } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import ImmersiveLayout from "@/layout/ImmersiveLayout.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";
import { FavoriteAPI } from "dehaze-sdk-js";

const CUSTOM_PRESETS_KEY = "uniapp_custom_filter_presets";

const store = useProcessingStore();
const showOrigin = ref(false);
const activePreset = ref("");
const favorited = ref(false);
const favoriteLoading = ref(false);

const resultUrl = computed(() => store.result?.resultUrl || "");
const originUrl = computed(() => store.originUrl);
const hasImages = computed(() => !!resultUrl.value);
const currentPreviewUrl = computed(() =>
  showOrigin.value ? originUrl.value : resultUrl.value
);
const resultId = computed(() => store.result?.logId);

interface FilterItem {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  color: string;
}

interface Preset {
  name: string;
  params: Record<string, number>;
}

const filters = reactive<FilterItem[]>([
  {
    key: "brightness",
    label: "亮度",
    value: 0,
    min: -100,
    max: 100,
    color: "#fbbf24",
  },
  {
    key: "contrast",
    label: "对比度",
    value: 0,
    min: -100,
    max: 100,
    color: "#f59e0b",
  },
  {
    key: "saturation",
    label: "饱和度",
    value: 0,
    min: -100,
    max: 100,
    color: "#34d399",
  },
  {
    key: "temperature",
    label: "色温",
    value: 0,
    min: -100,
    max: 100,
    color: "#f87171",
  },
  {
    key: "sharpen",
    label: "锐化",
    value: 0,
    min: 0,
    max: 100,
    color: "#60a5fa",
  },
  {
    key: "denoise",
    label: "降噪",
    value: 0,
    min: 0,
    max: 100,
    color: "#a78bfa",
  },
]);

const defaultParams: Record<string, number> = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  sharpen: 0,
  denoise: 0,
};

const builtinPresets: Preset[] = [
  {
    name: "自然",
    params: {
      brightness: 5,
      contrast: 10,
      saturation: 5,
      temperature: 0,
      sharpen: 0,
      denoise: 0,
    },
  },
  {
    name: "鲜艳",
    params: {
      brightness: 0,
      contrast: 30,
      saturation: 40,
      temperature: 0,
      sharpen: 0,
      denoise: 0,
    },
  },
  {
    name: "柔和",
    params: {
      brightness: 0,
      contrast: -20,
      saturation: 0,
      temperature: 0,
      sharpen: -10,
      denoise: 0,
    },
  },
  {
    name: "清晰",
    params: {
      brightness: 0,
      contrast: 20,
      saturation: 0,
      temperature: 0,
      sharpen: 40,
      denoise: 0,
    },
  },
  {
    name: "复古",
    params: {
      brightness: 0,
      contrast: 0,
      saturation: -20,
      temperature: 30,
      sharpen: 0,
      denoise: 0,
    },
  },
];

const customPresets = ref<Preset[]>([]);

const filterString = computed(() => {
  const b = 1 + getVal("brightness") / 100;
  const c = 1 + getVal("contrast") / 100;
  const s = 1 + getVal("saturation") / 100;
  const t = getVal("temperature");
  const sepia = Math.abs(t) / 100;
  const hue = t * 0.5;
  const sb = 1 + getVal("sharpen") / 200;
  const blur = getVal("denoise") / 200;
  return `brightness(${b}) contrast(${c * sb}) saturate(${s}) sepia(${sepia}) hue-rotate(${hue}deg) blur(${blur}px)`;
});

function getVal(key: string): number {
  return filters.find((f) => f.key === key)?.value ?? 0;
}

const hasChanges = computed(() =>
  filters.some((f) => f.value !== (defaultParams[f.key] ?? 0))
);

const modes = [
  {
    key: "side-by-side",
    label: "并排",
    path: "/pages/side-by-side/index",
    icon: "grid",
  },
  {
    key: "overlay",
    label: "重叠",
    path: "/pages/overlay/index",
    icon: "photo",
  },
  {
    key: "magnifier",
    label: "放大镜",
    path: "/pages/magnifier/index",
    icon: "search",
  },
  {
    key: "filter",
    label: "滤镜",
    path: "/pages/filter/index",
    icon: "setting",
  },
  {
    key: "metrics",
    label: "指标",
    path: "/pages/metrics/index",
    icon: "integral",
  },
];

function updateFilter(key: string, value: number) {
  const item = filters.find((f) => f.key === key);
  if (item) {
    item.value = value;
    activePreset.value = "";
  }
}

function applyPreset(p: Preset) {
  filters.forEach((f) => {
    const v = p.params[f.key];
    if (v !== undefined) f.value = v;
  });
  activePreset.value = p.name;
}

function handleReset() {
  filters.forEach((f) => {
    f.value = defaultParams[f.key] ?? 0;
  });
  activePreset.value = "";
}

function handleSavePreset() {
  uni.showModal({
    title: "保存预设",
    editable: true,
    placeholderText: "请输入预设名称",
    success(res: UniApp.ShowModalRes) {
      const name = (res.content || "").trim();
      if (res.confirm && name) {
        const params: Record<string, number> = {};
        filters.forEach((f) => {
          params[f.key] = f.value;
        });
        customPresets.value.push({ name, params });
        uni.setStorageSync(
          CUSTOM_PRESETS_KEY,
          JSON.stringify(customPresets.value)
        );
        uni.showToast({ title: "预设已保存", icon: "success" });
      }
    },
  } as any);
}

function handleDeletePreset(index: number) {
  uni.showModal({
    title: "确认删除",
    content: "确定要删除此自定义预设吗？",
    success(res) {
      if (res.confirm) {
        customPresets.value.splice(index, 1);
        uni.setStorageSync(
          CUSTOM_PRESETS_KEY,
          JSON.stringify(customPresets.value)
        );
        uni.showToast({ title: "已删除", icon: "success" });
      }
    },
  });
}

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
          success: () =>
            uni.showToast({ title: "已保存到相册", icon: "success" }),
          fail: () => uni.showToast({ title: "保存失败", icon: "none" }),
        });
      }
    },
  });
}

function handleReprocess() {
  uni.redirectTo({ url: "/pages/processing/index" });
}

function handleChangeAlgorithm() {
  uni.redirectTo({ url: "/pages/algorithm-select/index" });
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
  try {
    const stored = uni.getStorageSync(CUSTOM_PRESETS_KEY);
    if (stored) customPresets.value = JSON.parse(stored);
  } catch {
    /* ignore */
  }
  if (!hasImages.value)
    uni.showToast({ title: "请先完成去雾处理", icon: "none" });
  if (resultId.value) {
    FavoriteAPI.getStatus("result", resultId.value)
      .then((res) => {
        favorited.value = res.favorited;
      })
      .catch(() => {});
  }
});
</script>

<style lang="scss" scoped>
.main-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.preview-wrapper {
  position: relative;
  max-height: 40vh;
  overflow: hidden;
  background: #000;
}

.preview-image {
  width: 100%;
  display: block;
}

.preview-toggle {
  position: absolute;
  right: 24rpx;
  bottom: 24rpx;
  padding: 12rpx 28rpx;
  font-size: 24rpx;
  color: #fff;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 32rpx;
}

.filter-scroll {
  flex: 1;
  overflow-y: auto;
}

.panel {
  margin: 24rpx 32rpx;
  padding: 24rpx 32rpx;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 16rpx;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24rpx;
}

.panel-title {
  font-size: 28rpx;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.8);
}

.save-btn,
.reset-btn {
  font-size: 26rpx;
  color: #10b981;
}

.preset-list {
  display: flex;
  flex-wrap: wrap;
  gap: 16rpx;
}

.preset-item {
  padding: 12rpx 32rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.08);
  border: 2rpx solid rgba(255, 255, 255, 0.1);
  border-radius: 32rpx;

  &.active {
    color: #10b981;
    background: rgba(16, 185, 129, 0.15);
    border-color: #10b981;
  }

  &.custom {
    color: #a78bfa;
    border-color: #a78bfa;
  }
}

.slider-item {
  margin-bottom: 32rpx;

  &:last-child {
    margin-bottom: 0;
  }
}

.slider-label-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16rpx;
}

.slider-label {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.5);
}

.slider-value {
  font-size: 26rpx;
  font-weight: 500;
  color: #10b981;
}

.toolbar-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4rpx;
  padding: 16rpx 16rpx 8rpx;
}

.toolbar-actions {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
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
    background: rgba(16, 185, 129, 0.15);
    border-radius: 12rpx;
  }

  &.active {
    color: #10b981;
    background: rgba(16, 185, 129, 0.12);
    border-radius: 12rpx;
  }
}
</style>
