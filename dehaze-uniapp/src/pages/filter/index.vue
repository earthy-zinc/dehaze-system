<template>
  <PageLayout class="page">
    <view class="main-content">
      <view class="page-header-card">
        <view class="header-icon">
          <u-icon name="slider" size="28" color="#10b981" />
        </view>
        <view class="header-text">
          <text class="header-title">滤镜调节</text>
          <text class="header-subtitle">实时调节画面效果</text>
        </view>
      </view>

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
              @change="(e:any) => updateFilter(item.key, e.detail.value)"
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
          <view class="nav-item" @click="switchPage('/pages/side-by-side/index')">
            <u-icon name="grid" size="20" color="#10b981" /><text>并排对比</text>
          </view>
          <view class="nav-item" @click="switchPage('/pages/overlay/index')">
            <u-icon name="photo" size="20" color="#10b981" /><text>重叠对比</text>
          </view>
        </view>
      </view>

      <view v-else class="empty-state">
        <up-empty mode="image" text="暂无处理结果" />
        <button class="back-btn" @click="handleBack">返回处理页</button>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, reactive, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { useProcessingStore } from "@/store/processing";

const store = useProcessingStore();

const resultUrl = computed(() => store.result?.resultUrl || "");
const hasImages = computed(() => !!resultUrl.value);

const filters = reactive([
  { key: "brightness", label: "亮度", value: 100, min: 50, max: 200, step: 1, unit: "%", color: "#fbbf24" },
  { key: "contrast", label: "对比度", value: 100, min: 50, max: 200, step: 1, unit: "%", color: "#f59e0b" },
  { key: "saturate", label: "饱和度", value: 100, min: 0, max: 300, step: 1, unit: "%", color: "#34d399" },
  { key: "warmth", label: "色温", value: 0, min: -30, max: 30, step: 1, unit: "", color: "#f87171" },
]);

const activePreset = ref("原始");

const filterString = computed(() => {
  const b = filters[0].value;
  const c = filters[1].value;
  const s = filters[2].value;
  const w = filters[3].value;
  return `brightness(${b}%) contrast(${c}%) saturate(${s}%) sepia(${w > 0 ? w : 0}%) hue-rotate(${w < 0 ? w : 0}deg)`;
});

const presets = [
  { label: "原始", values: [100, 100, 100, 0] },
  { label: "鲜艳", values: [110, 120, 150, 0] },
  { label: "柔和", values: [95, 90, 80, 5] },
  { label: "冷调", values: [100, 105, 90, -15] },
  { label: "暖调", values: [105, 100, 110, 15] },
  { label: "复古", values: [90, 85, 60, 20] },
];

function updateFilter(key: string, value: number) {
  const item = filters.find((f) => f.key === key);
  if (item) { item.value = value; activePreset.value = "自定义"; }
}

function applyPreset(p: (typeof presets)[number]) {
  filters[0].value = p.values[0];
  filters[1].value = p.values[1];
  filters[2].value = p.values[2];
  filters[3].value = p.values[3];
  activePreset.value = p.label;
}

function switchPage(url: string) { uni.navigateTo({ url }); }
function handleBack() { uni.navigateBack(); }

onMounted(() => { if (!hasImages.value) uni.showToast({ title: "请先完成去雾处理", icon: "none" }); });
</script>

<style lang="scss" scoped>
.page { width: 100%; min-height: 100vh; background: #1a1a2e; }
.main-content { padding: 24rpx; }
.page-header-card {
  display: flex; align-items: center; gap: 24rpx;
  background: rgba(255,255,255,0.95); border-radius: 24rpx; padding: 32rpx; margin-bottom: 24rpx;
}
.header-icon { width: 80rpx; height: 80rpx; background: #d1fae5; border-radius: 20rpx; display: flex; align-items: center; justify-content: center; }
.header-title { font-size: 36rpx; font-weight: 700; color: #1f2937; }
.header-subtitle { font-size: 26rpx; color: #6b7280; }

.preview-wrapper { border-radius: 16rpx; overflow: hidden; }
.preview-image { width: 100%; display: block; }

.filter-panel { background: rgba(255,255,255,0.06); border-radius: 20rpx; padding: 28rpx; margin-top: 24rpx; }
.filter-item { margin-bottom: 24rpx; &:last-child { margin-bottom: 0; } }
.filter-label { display: flex; justify-content: space-between; margin-bottom: 8rpx; font-size: 26rpx; color: rgba(255,255,255,0.7); }
.filter-value { font-weight: 600; color: #fff; }

.preset-row { display: flex; flex-wrap: wrap; gap: 12rpx; margin-top: 24rpx; }
.preset-btn {
  padding: 14rpx 24rpx; border-radius: 16rpx; font-size: 24rpx;
  color: rgba(255,255,255,0.5); background: rgba(255,255,255,0.08);
  &.active { background: rgba(16,185,129,0.2); color: #10b981; }
}

.nav-row { display: flex; gap: 20rpx; margin-top: 32rpx; }
.nav-item {
  flex: 1; display: flex; flex-direction: column; align-items: center; gap: 12rpx;
  padding: 28rpx; background: rgba(255,255,255,0.08); border-radius: 20rpx;
  font-size: 24rpx; color: rgba(255,255,255,0.6);
  &:active { background: rgba(16,185,129,0.15); }
}

.empty-state { display: flex; flex-direction: column; align-items: center; padding: 120rpx 0; }
.back-btn { margin-top: 32rpx; padding: 16rpx 48rpx; background: #10b981; color: #fff; border: none; border-radius: 16rpx; font-size: 28rpx; }
</style>
