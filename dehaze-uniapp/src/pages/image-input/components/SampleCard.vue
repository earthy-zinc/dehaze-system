<template>
  <view class="sample-card" @click="handleClick">
    <view class="card-image">
      <up-image
        :src="sample.url"
        mode="aspectFill"
        width="100%"
        height="200rpx"
        :lazy-load="true"
        :fade="true"
      />
      <view class="difficulty-badge" :style="{ background: difficultyBgColor }">
        <text class="difficulty-text" :style="{ color: difficultyColor }">
          {{ sample.difficulty }}
        </text>
      </view>
    </view>
    <view class="card-info">
      <text class="card-name">{{ sample.name }}</text>
      <view class="card-meta">
        <text v-if="sample.scene" class="meta-scene">{{ sample.scene }}</text>
        <SvgIcon name="arrow-right" size="14" color="#9ca3af" />
      </view>
    </view>
  </view>
</template>

<script lang="ts" setup>
import { computed } from "vue";
import SvgIcon from "@/components/SvgIcon/index.vue";
import type { SampleImage } from "../data/imageInputData";
import {
  DIFFICULTY_COLORS,
  DIFFICULTY_BG_COLORS,
} from "../data/imageInputData";

interface Props {
  sample: SampleImage;
}

const props = defineProps<Props>();

const emit = defineEmits<{
  (e: "click", sample: SampleImage): void;
}>();

const difficultyColor = computed(
  () => DIFFICULTY_COLORS[props.sample.difficulty]
);
const difficultyBgColor = computed(
  () => DIFFICULTY_BG_COLORS[props.sample.difficulty]
);

const handleClick = () => {
  emit("click", props.sample);
};
</script>

<style lang="scss" scoped>
.sample-card {
  background: #ffffff;
  border-radius: 16rpx;
  overflow: hidden;
  box-shadow: 0 2rpx 12rpx rgba(0, 0, 0, 0.06);
  transition: all 0.2s ease;

  &:active {
    transform: scale(0.98);
    box-shadow: 0 4rpx 16rpx rgba(0, 0, 0, 0.1);
  }
}

.card-image {
  position: relative;
  width: 100%;
  height: 200rpx;
  overflow: hidden;
}

.difficulty-badge {
  position: absolute;
  top: 12rpx;
  right: 12rpx;
  padding: 6rpx 16rpx;
  border-radius: 20rpx;
}

.difficulty-text {
  font-size: 22rpx;
  font-weight: 600;
}

.card-info {
  padding: 20rpx;
}

.card-name {
  display: block;
  font-size: 26rpx;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 8rpx;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.meta-scene {
  font-size: 22rpx;
  color: #6b7280;
}
</style>
