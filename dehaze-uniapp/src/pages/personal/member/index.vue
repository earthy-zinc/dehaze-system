<template>
  <PageLayout level="L2" title="我的会员" class="page">
    <view class="main-content">
      <view class="member-card">
        <view class="level-badge">{{ profile.levelName || "普通用户" }}</view>
        <text class="growth-label">成长值</text>
        <text class="growth-value">{{ profile.growthValue }}</text>
        <view class="growth-bar">
          <view class="growth-fill" :style="{ width: profile.progressPercent + '%' }" />
        </view>
        <text class="next-hint" v-if="profile.nextLevelGrowth && profile.nextLevelGrowth > 0">
          距下一级还需 {{ profile.nextLevelGrowth }} 成长值
        </text>
      </view>

      <view class="usage-card">
        <text class="section-title">本月用量</text>
        <view class="usage-row">
          <view class="usage-item">
            <text class="usage-num">{{ profile.monthlyDehazeUsed }}/{{ profile.monthlyDehazeQuota }}</text>
            <text class="usage-label">去雾处理</text>
          </view>
          <view class="usage-item">
            <text class="usage-num">{{ profile.monthlyEvaluateUsed }}/{{ profile.monthlyEvaluateQuota }}</text>
            <text class="usage-label">评估分析</text>
          </view>
        </view>
      </view>

      <view class="benefit-card" v-if="benefitItems.length > 0">
        <text class="section-title">会员权益</text>
        <view class="benefit-list">
          <view class="benefit-item" v-for="item in benefitItems" :key="item.key">
            <text class="benefit-value">{{ item.value }}</text>
            <text class="benefit-name">{{ item.label }}</text>
          </view>
        </view>
      </view>

      <view class="non-vip-guide" v-if="profile.levelCode === 'level_0'">
        <view class="guide-btn" @click="goPackage">
          <text class="guide-btn-text">开通 VIP</text>
        </view>
      </view>
    </view>
  </PageLayout>
</template>

<script lang="ts" setup>
import { ref, computed, onMounted } from "vue";
import PageLayout from "@/layout/index.vue";
import { MemberAPI } from "dehaze-sdk-js";
import type { MemberProfileVO } from "dehaze-sdk-js";

const profile = ref<MemberProfileVO>({
  userId: 0,
  username: "",
  nickname: "",
  levelCode: "level_0",
  levelName: "普通用户",
  growthValue: 0,
  nextLevelGrowth: undefined,
  progressPercent: 0,
  monthlyDehazeQuota: 0,
  monthlyDehazeUsed: 0,
  monthlyEvaluateQuota: 0,
  monthlyEvaluateUsed: 0,
  benefits: {} as any,
  status: 1,
});

const BENEFIT_LABELS: Record<string, string> = {
  monthlyDehazeQuota: "每月去雾次数",
  monthlyEvaluateQuota: "每月评估次数",
  historyRetention: "历史保留(天)",
  batchLimit: "批量上限",
  priority: "优先级",
  advancedParams: "高级参数",
  hdExport: "高清导出",
  reportExport: "报告导出",
  batchDownload: "批量下载",
};

const benefitItems = computed(() => {
  const b = profile.value.benefits;
  if (!b) return [];
  return Object.entries(BENEFIT_LABELS)
    .filter(([key]) => (b as any)[key] !== undefined)
    .map(([key, label]) => ({
      key,
      label,
      value: (b as any)[key],
    }));
});

function goPackage() {
  uni.navigateTo({ url: "/pages/personal/package/index" });
}

onMounted(async () => {
  try {
    profile.value = await MemberAPI.getProfile();
  } catch {
    uni.showToast({ title: "获取会员信息失败", icon: "none" });
  }
});
</script>

<style lang="scss" scoped>
.page {
  width: 100%;
  min-height: 100vh;
  background: $color-bg-primary;
}
.main-content {
  padding: $spacing-md;
}

.member-card {
  background: $gradient-primary;
  border-radius: $radius-xl;
  padding: 48rpx 32rpx;
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: $spacing-md;
  color: #fff;
}
.level-badge {
  font-size: $font-lg;
  font-weight: 700;
  margin-bottom: 24rpx;
  background: rgba(255, 255, 255, 0.2);
  padding: 12rpx 32rpx;
  border-radius: 12rpx;
}
.growth-label {
  font-size: $font-xs;
  opacity: 0.8;
}
.growth-value {
  font-size: 48rpx;
  font-weight: 700;
  margin: 8rpx 0 16rpx;
}
.growth-bar {
  width: 100%;
  height: 12rpx;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 6rpx;
  overflow: hidden;
  margin-bottom: 12rpx;
}
.growth-fill {
  height: 100%;
  background: #fff;
  border-radius: 6rpx;
  transition: width 0.3s;
}
.next-hint {
  font-size: $font-xs;
  opacity: 0.8;
}

.usage-card,
.benefit-card {
  background: #fff;
  border-radius: $radius-xl;
  padding: 28rpx;
  box-shadow: $shadow-sm;
  margin-bottom: $spacing-md;
}
.section-title {
  font-size: $font-md;
  font-weight: 600;
  color: $color-text-primary;
  display: block;
  margin-bottom: 20rpx;
}

.usage-row {
  display: flex;
}
.usage-item {
  flex: 1;
  text-align: center;
}
.usage-num {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-primary;
  display: block;
  margin-bottom: 4rpx;
}
.usage-label {
  font-size: $font-xs;
  color: $color-text-placeholder;
}

.benefit-list {
  display: flex;
  flex-direction: column;
  gap: 16rpx;
}
.benefit-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.benefit-value {
  font-size: $font-lg;
  font-weight: 700;
  color: $color-primary;
}
.benefit-name {
  font-size: $font-md;
  color: $color-text-primary;
}

.non-vip-guide {
  margin-top: $spacing-md;
  display: flex;
  justify-content: center;
}
.guide-btn {
  padding: 20rpx 80rpx;
  background: $gradient-primary;
  border-radius: 48rpx;
}
.guide-btn-text {
  font-size: $font-md;
  font-weight: 600;
  color: #fff;
}

.empty-text {
  font-size: $font-sm;
  color: $color-text-placeholder;
}
</style>
