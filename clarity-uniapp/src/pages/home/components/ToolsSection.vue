<template>
  <view class="tools-section">
    <view class="tools-grid">
      <FeatureCard
        v-for="tool in tools"
        :key="tool.id"
        :tool="tool"
        @click="handleToolClick"
      />
    </view>
  </view>
</template>

<script lang="ts" setup>
import { ref } from "vue";
import FeatureCard from "@/components/business/FeatureCard.vue";
import type { ToolItem } from "../data/homeData";
import { homeData } from "../data/homeData";

interface Emits {
  (e: "tool-click", tool: ToolItem): void;
}

const emit = defineEmits<Emits>();
const tools = ref(homeData.tools);

const handleToolClick = (tool: ToolItem) => {
  emit("tool-click", tool);
};
</script>

<style lang="scss" scoped>
.tools-section {
  padding: 80rpx 40rpx;
  background: #f9fafb;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24rpx;
  max-width: 100%;
  margin: 0 auto;
}

@media screen and (max-width: 768rpx) {
  .tools-grid {
    grid-template-columns: 1fr;
  }
}
</style>
