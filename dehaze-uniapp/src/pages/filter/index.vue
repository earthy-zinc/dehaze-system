<template>
  <PageLayout level="L3" class="page">
    <view class="main-content">
      <PageHeaderCard
        icon="setting"
        icon-color="#10b981"
        icon-bg="#d1fae5"
        title="滤镜调节"
        subtitle="实时调节画面效果"
        variant="dark"
      />

      <view v-if="hasImages" class="content-area">
        <!-- 预览图 -->
        <view class="preview-wrapper">
          <image
            :src="resultUrl"
            class="preview-image"
            mode="widthFix"
            :style="{ filter: filterString }"
          />
        </view>

        <!-- 滤镜参数 -->
        <view class="filter-panel">
          <view v-for="item in filters" :key="item.key" class="filter-item">
            <view class="filter-label">
              <text>{{ item.label }}</text>
              <text class="filter-value">{{ item.value }}{{ item.unit }}</text>
            </view>
            <slider
              :value="item.value"
              :min="item.min"
              :max="item.max"
              :step="item.step"
              :active-color="item.color"
              block-size="20"
              @change="
                (e: SliderChangeEvent) => updateFilter(item.key, e.detail.value)
              "
            />
          </view>
        </view>

        <!-- 预设 -->
        <view class="preset-row">
          <view
            v-for="p in presets"
            :key="p.label"
            class="preset-btn"
            :class="{ active: activePreset === p.label }"
            @click="applyPreset(p)"
          >
            {{ p.label }}
          </view>
        </view>

        <view class="nav-row">
          <view
            class="nav-item"
            @click="switchPage('/pages/side-by-side/index')"
          >
            <u-icon name="grid" size="20" color="#10b981" /><text
              >并排对比</text
            >
          </view>
          <view class="nav-item" @click="switchPage('/pages/overlay/index')">
            <u-icon name="photo" size="20" color="#10b981" /><text
              >重叠对比</text
            >
          </view>
        </view>
      </view>

      <CompareEmptyState v-else text="暂无处理结果" btn-color="#10b981" />
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, reactive, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import PageHeaderCard from "@/components/common/PageHeaderCard.vue";
import CompareEmptyState from "@/components/common/CompareEmptyState.vue";
import { useProcessingStore } from "@/store/processing";
import type { SliderChangeEvent } from "@/types/uni-events";

type FilterKey = "brightness" | "contrast" | "saturate" | "warmth";

interface FilterConfig {
  key: FilterKey;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  color: string;
}

const store = useProcessingStore();

const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!resultUrl.value);

const filters = reactive<FilterConfig[]>([
  {
    key: "brightness",
    label: "亮度",
    value: 100,
    min: 50,
    max: 200,
    step: 1,
    unit: "%",
    color: "#fbbf24",
  },
  {
    key: "contrast",
    label: "对比度",
    value: 100,
    min: 50,
    max: 200,
    step: 1,
    unit: "%",
    color: "#f59e0b",
  },
  {
    key: "saturate",
    label: "饱和度",
    value: 100,
    min: 0,
    max: 300,
    step: 1,
    unit: "%",
    color: "#34d399",
  },
  {
    key: "warmth",
    label: "色温",
    value: 0,
    min: -30,
    max: 30,
    step: 1,
    unit: "",
    color: "#f87171",
  },
]);

const activePreset = ref("原始");

function getFilter(key: FilterKey): number {
  return filters.find((f) => f.key === key)?.value ?? 0;
}

const filterString = computed(() => {
  const warmth = getFilter("warmth");
  return `brightness(${getFilter("brightness")}%) contrast(${getFilter("contrast")}%) saturate(${getFilter("saturate")}%) sepia(${Math.max(warmth, 0)}%) hue-rotate(${Math.min(warmth, 0)}deg)`;
});

interface Preset {
  label: string;
  values: Record<FilterKey, number>;
}

const presets: Preset[] = [
  {
    label: "原始",
    values: { brightness: 100, contrast: 100, saturate: 100, warmth: 0 },
  },
  {
    label: "鲜艳",
    values: { brightness: 110, contrast: 120, saturate: 150, warmth: 0 },
  },
  {
    label: "柔和",
    values: { brightness: 95, contrast: 90, saturate: 80, warmth: 5 },
  },
  {
    label: "冷调",
    values: { brightness: 100, contrast: 105, saturate: 90, warmth: -15 },
  },
  {
    label: "暖调",
    values: { brightness: 105, contrast: 100, saturate: 110, warmth: 15 },
  },
  {
    label: "复古",
    values: { brightness: 90, contrast: 85, saturate: 60, warmth: 20 },
  },
];

function updateFilter(key: FilterKey, value: number) {
  const item = filters.find((f) => f.key === key);
  if (item) {
    item.value = value;
    activePreset.value = "自定义";
  }
}

function applyPreset(p: Preset) {
  (Object.keys(p.values) as FilterKey[]).forEach((key) => {
    const item = filters.find((f) => f.key === key);
    if (item) item.value = p.values[key];
  });
  activePreset.value = p.label;
}

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
  background: #1a1a2e;
}
.main-content {
  padding: 24rpx;
}

.preview-wrapper {
  border-radius: 16rpx;
  overflow: hidden;
}
.preview-image {
  width: 100%;
  display: block;
}

.filter-panel {
  background: rgba(255, 255, 255, 0.06);
  border-radius: 20rpx;
  padding: 28rpx;
  margin-top: 24rpx;
}
.filter-item {
  margin-bottom: 24rpx;
  &:last-child {
    margin-bottom: 0;
  }
}
.filter-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8rpx;
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.7);
}
.filter-value {
  font-weight: 600;
  color: #fff;
}

.preset-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12rpx;
  margin-top: 24rpx;
}
.preset-btn {
  padding: 14rpx 24rpx;
  border-radius: 16rpx;
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.5);
  background: rgba(255, 255, 255, 0.08);
  &.active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
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
    background: rgba(16, 185, 129, 0.15);
  }
}
</style>
