<template>
  <view class="compare-empty-state">
    <up-empty mode="image" :text="text" />
    <text v-if="hint" class="empty-hint">{{ hint }}</text>
    <button
      class="back-btn"
      :style="{ background: btnColor }"
      @click="handleBack"
    >
      返回处理页
    </button>
  </view>
</template>

<script lang="ts" setup>
interface Props {
  text?: string;
  hint?: string;
  btnColor?: string;
}
withDefaults(defineProps<Props>(), {
  text: "暂无对比数据",
  hint: "",
  btnColor: "#3b82f6",
});

function handleBack() {
  uni.navigateBack();
}
</script>

<style lang="scss" scoped>
.compare-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 120rpx 0;
}
.empty-hint {
  font-size: 26rpx;
  color: rgba(255, 255, 255, 0.4);
  margin: 16rpx 0 32rpx;
}
.back-btn {
  margin-top: 32rpx;
  padding: 16rpx 48rpx;
  color: #fff;
  border: none;
  border-radius: 16rpx;
  font-size: 28rpx;
}
</style>
